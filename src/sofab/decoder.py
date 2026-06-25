"""SofaBuffers pull decoder (Go-style ``IStream`` equivalent).

The decoder reads from any object exposing ``read(n) -> bytes`` (a file, a
socket made file-like, an ``io.BytesIO``, or a chunk-feeding wrapper). It pulls
exactly what it needs, so it satisfies the format's streaming requirement for
blocking readers; large blob/string/array payloads are read in bulk.

Typical use::

    dec = Decoder(reader)
    while (field := dec.next()) is not None:
        if field.id == 1:
            value = dec.unsigned()
        else:
            dec.skip()
"""

from __future__ import annotations

from typing import Any, Protocol

from . import _core
from ._varint import decode_varint, zigzag_decode
from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    Field,
    FixlenSubtype,
    SofaDecodeError,
    SofaStateError,
    WireType,
)

# Pending-value kinds the consume methods dispatch on.
_SCALAR = 0
_FIXLEN = 1
_VARRAY = 2
_FARRAY = 3


class _Reader(Protocol):
    """Read protocol: an object with ``read(n) -> bytes``."""

    def read(self, n: int) -> bytes: ...


class Decoder:
    def __init__(self, reader: _Reader, *, chunk_size: int = 65536) -> None:
        self._read = reader.read
        self._chunk = chunk_size
        self._buf = b""
        self._pos = 0
        self._depth = 0
        self._cur: Field | None = None
        # pending unconsumed value: tuple keyed by the _* constants above
        self._pending: tuple[Any, ...] | None = None

    # --- low-level byte sourcing --------------------------------------------

    def _fill(self) -> bool:
        data = self._read(self._chunk)
        if not data:
            return False
        if self._pos:
            self._buf = self._buf[self._pos :] + data
        else:
            self._buf = self._buf + data if self._buf else data
        self._pos = 0
        return True

    def _read_byte(self) -> int | None:
        if self._pos >= len(self._buf):
            self._buf = b""
            self._pos = 0
            if not self._fill():
                return None
        byte = self._buf[self._pos]
        self._pos += 1
        return byte

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            if self._pos >= len(self._buf):
                self._buf = b""
                self._pos = 0
                if not self._fill():
                    raise SofaDecodeError("truncated payload")
            take = min(n - len(out), len(self._buf) - self._pos)
            out += self._buf[self._pos : self._pos + take]
            self._pos += take
        return bytes(out)

    def _varint(self, first: int | None = None) -> int:
        return decode_varint(self._read_byte, first)

    # --- field iteration ----------------------------------------------------

    @property
    def field(self) -> Field | None:
        """The most recently returned :class:`Field`."""
        return self._cur

    def next(self) -> Field | None:
        """Advance to the next field. Returns ``None`` at clean EOF.

        Any value left unconsumed from the previous field is skipped first.
        """
        if self._pending is not None:
            self._skip_pending()

        first = self._read_byte()
        if first is None:
            if self._depth != 0:
                raise SofaDecodeError("truncated: unbalanced sequence")
            return None

        header = self._varint(first)
        wtype = header & 0x07
        field_id = header >> 3

        if wtype == WireType.SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            self._cur = Field(0, WireType.SEQUENCE_END)
            return self._cur

        if field_id > ID_MAX:
            raise SofaDecodeError(f"id {field_id} out of range")

        if wtype == WireType.SEQUENCE_START:
            self._depth += 1
            self._cur = Field(field_id, WireType.SEQUENCE_START)
            return self._cur

        if wtype in (WireType.UNSIGNED, WireType.SIGNED):
            self._cur = Field(field_id, WireType(wtype))
            self._pending = (_SCALAR, wtype)
            return self._cur

        if wtype == WireType.FIXLEN:
            length_header = self._varint()
            length = length_header >> 3
            subtype = length_header & 0x07
            if subtype > FixlenSubtype.BLOB:
                raise SofaDecodeError(f"invalid fixlen subtype {subtype}")
            if length > FIXLEN_MAX:
                raise SofaDecodeError("fixlen length out of range")
            self._cur = Field(
                field_id, WireType.FIXLEN, size=length, subtype=FixlenSubtype(subtype)
            )
            self._pending = (_FIXLEN, subtype, length)
            return self._cur

        if wtype in (WireType.ARRAY_UNSIGNED, WireType.ARRAY_SIGNED):
            count = self._varint()
            if count < 1 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            self._cur = Field(field_id, WireType(wtype), count=count)
            self._pending = (_VARRAY, wtype, count)
            return self._cur

        # wtype == ARRAY_FIXLEN
        count = self._varint()
        if count < 1 or count > ARRAY_MAX:
            raise SofaDecodeError(f"array count {count} out of range")
        elem_header = self._varint()
        elem_size = elem_header >> 3
        subtype = elem_header & 0x07
        if subtype > FixlenSubtype.FP64:
            raise SofaDecodeError(f"invalid fixlen-array subtype {subtype}")
        self._cur = Field(
            field_id,
            WireType.ARRAY_FIXLEN,
            count=count,
            size=elem_size,
            subtype=FixlenSubtype(subtype),
        )
        self._pending = (_FARRAY, subtype, count, elem_size)
        return self._cur

    # --- skipping -----------------------------------------------------------

    def _skip_pending(self) -> None:
        pending = self._pending
        assert pending is not None
        self._pending = None
        kind = pending[0]
        if kind == _SCALAR:
            self._varint()
        elif kind == _FIXLEN:
            self._read_exact(pending[2])
        elif kind == _VARRAY:
            for _ in range(pending[2]):
                self._varint()
        else:  # _FARRAY
            self._read_exact(pending[2] * pending[3])

    def skip(self) -> None:
        """Skip the current field's value, or an entire (nested) sequence if the
        current field is a sequence start."""
        if self._cur is not None and self._cur.type == WireType.SEQUENCE_START:
            target = self._depth - 1
            while self._depth > target:
                if self.next() is None:
                    raise SofaDecodeError("truncated sequence")
            return
        if self._pending is not None:
            self._skip_pending()

    # --- scalar reads -------------------------------------------------------

    def _take_scalar(self, wtype: WireType) -> int:
        pending = self._pending
        if pending is None or pending[0] != _SCALAR or pending[1] != wtype:
            raise SofaStateError("no matching scalar value for the current field")
        self._pending = None
        return self._varint()

    def unsigned(self) -> int:
        return self._take_scalar(WireType.UNSIGNED)

    def signed(self) -> int:
        return zigzag_decode(self._take_scalar(WireType.SIGNED))

    def bool(self) -> bool:
        return self._take_scalar(WireType.UNSIGNED) != 0

    def _take_fixlen(self, subtype: FixlenSubtype) -> bytes:
        pending = self._pending
        if pending is None or pending[0] != _FIXLEN:
            raise SofaStateError("current field is not a fixlen value")
        if pending[1] != subtype:
            raise SofaStateError("fixlen subtype does not match the requested read")
        self._pending = None
        return self._read_exact(pending[2])

    def float32(self) -> float:
        data = self._take_fixlen(FixlenSubtype.FP32)
        if len(data) != 4:
            raise SofaDecodeError("fp32 payload must be 4 bytes")
        return _core.unpack_f32(data)

    def float64(self) -> float:
        data = self._take_fixlen(FixlenSubtype.FP64)
        if len(data) != 8:
            raise SofaDecodeError("fp64 payload must be 8 bytes")
        return _core.unpack_f64(data)

    def string(self) -> str:
        return self._take_fixlen(FixlenSubtype.STRING).decode("utf-8")

    def bytes(self) -> bytes:
        return self._take_fixlen(FixlenSubtype.BLOB)

    # --- array reads --------------------------------------------------------

    def _take_varray(self, wtype: WireType) -> int:
        pending = self._pending
        if pending is None or pending[0] != _VARRAY or pending[1] != wtype:
            raise SofaStateError("current field is not a matching varint array")
        self._pending = None
        return int(pending[2])

    def read_unsigned_array(self) -> list[int]:
        count = self._take_varray(WireType.ARRAY_UNSIGNED)
        return [self._varint() for _ in range(count)]

    def read_signed_array(self) -> list[int]:
        count = self._take_varray(WireType.ARRAY_SIGNED)
        return [zigzag_decode(self._varint()) for _ in range(count)]

    def _take_farray(self, subtype: FixlenSubtype) -> tuple[int, int]:
        pending = self._pending
        if pending is None or pending[0] != _FARRAY:
            raise SofaStateError("current field is not a fixlen array")
        if pending[1] != subtype:
            raise SofaStateError("fixlen-array subtype does not match the requested read")
        self._pending = None
        return int(pending[2]), int(pending[3])  # count, elem_size

    def read_float32_array(self) -> list[float]:
        count, elem_size = self._take_farray(FixlenSubtype.FP32)
        data = self._read_exact(count * elem_size)
        return [_core.unpack_f32(data[i : i + 4]) for i in range(0, len(data), 4)]

    def read_float64_array(self) -> list[float]:
        count, elem_size = self._take_farray(FixlenSubtype.FP64)
        data = self._read_exact(count * elem_size)
        return [_core.unpack_f64(data[i : i + 8]) for i in range(0, len(data), 8)]
