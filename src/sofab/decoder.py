"""SofaBuffers pull decoder (Go-style ``IStream`` equivalent).

The decoder reads from any object exposing ``read(n) -> bytes`` (a file, a
socket made file-like, an ``io.BytesIO``, or a chunk-feeding wrapper). It pulls
exactly what it needs, so it satisfies the format's streaming requirement for
blocking readers; large blob/string/array payloads are read in bulk.

**Hot-path model — "advance a cursor over a contiguous buffer" (protobuf's
trick).** Incoming bytes are accumulated into a single contiguous buffer
(``self._buf``) and parsed by advancing an integer cursor (``self._pos``) with
direct indexing — no per-byte function call, no intermediate copies. When the
cursor reaches the end mid-item the decoder transparently refills from the
reader and continues, so the same code path serves both a fully-buffered
message and a reader that dribbles one byte at a time. See ``_varint`` /
``_read_varints`` / ``_read_exact`` below.

Typical use::

    dec = Decoder(reader)
    while (field := dec.next()) is not None:
        if field.id == 1:
            value = dec.unsigned()
        else:
            dec.skip()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from . import _core
from ._varint import zigzag_decode
from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    MASK64,
    MAX_DEPTH,
    Field,
    FixlenSubtype,
    SofaDecodeError,
    SofaStateError,
    WireType,
)

if TYPE_CHECKING:
    from .visitor import Visitor

# Pending-value kinds the consume methods dispatch on.
_SCALAR = 0
_FIXLEN = 1
_VARRAY = 2
_FARRAY = 3

# Wire-type members indexed by their integer value, so the per-field hot path
# can recover the enum member by index (``_WT[wtype]``) instead of paying the
# full ``WireType(wtype)`` coercion (IntEnum.__call__/__new__) on every field.
_WT = tuple(WireType)


class _Reader(Protocol):
    """Read protocol: an object with ``read(n) -> bytes``."""

    def read(self, n: int) -> bytes: ...


class Decoder:
    """Pull-decodes a SofaBuffers stream field by field.

    Call :meth:`next` to advance to each field, then one of the typed read
    methods (:meth:`unsigned`, :meth:`string`, :meth:`read_float64_array`, …)
    to consume its value, or :meth:`skip` to discard it. Alternatively hand a
    :class:`sofab.Visitor` to :meth:`drive` for callback-style decoding.
    """

    def __init__(self, reader: _Reader, *, chunk_size: int = 65536) -> None:
        """Wrap ``reader`` (any object with ``read(n) -> bytes``).

        ``chunk_size`` is how many bytes each refill pulls from the reader.
        """
        self._read = reader.read
        self._chunk = chunk_size
        self._buf = b""
        self._pos = 0
        self._depth = 0
        self._cur: Field | None = None
        # pending unconsumed value: tuple keyed by the _* constants above
        self._pending: tuple[Any, ...] | None = None

    # --- low-level byte sourcing --------------------------------------------
    #
    # The buffer is never sliced per byte: ``_pos`` advances over ``_buf`` and
    # the consumed prefix is dropped only when a refill is actually needed.

    def _need(self, n: int) -> bool:
        """Ensure at least ``n`` bytes are available at ``_pos``, pulling more
        from the reader (and compacting the consumed prefix) as required.
        Returns ``False`` if the stream ends with fewer than ``n`` available."""
        buf = self._buf
        pos = self._pos
        if len(buf) - pos >= n:
            return True
        if pos:
            buf = buf[pos:]
            self._pos = 0
        read = self._read
        chunk = self._chunk
        while len(buf) < n:
            data = read(chunk)
            if not data:
                self._buf = buf
                return False
            buf = buf + data if buf else data
        self._buf = buf
        return True

    def _varint(self) -> int:
        """Decode one base-128 varint by advancing the cursor over the buffer,
        refilling only if it runs off the end mid-value."""
        buf = self._buf
        pos = self._pos
        if pos >= len(buf):
            if not self._need(1):
                raise SofaDecodeError("truncated varint")
            buf = self._buf
            pos = self._pos
        b = buf[pos]
        pos += 1
        if b < 0x80:  # one-byte fast path (ids, small counts, small values)
            self._pos = pos
            return b
        result = b & 0x7F
        shift = 7
        n = len(buf)
        while True:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise SofaDecodeError("truncated varint")
                buf = self._buf
                pos = self._pos
                n = len(buf)
            b = buf[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            if b < 0x80:
                self._pos = pos
                return result & MASK64
            shift += 7
            if shift >= 64:
                raise SofaDecodeError("varint overflow")

    def _read_exact(self, n: int) -> bytes:
        """Return the next ``n`` bytes. Fast path is a single buffer slice; the
        slow path accumulates across refills for a chunk-fed reader."""
        buf = self._buf
        pos = self._pos
        end = pos + n
        if end <= len(buf):
            self._pos = end
            return buf[pos:end]
        out = bytearray(buf[pos:])
        self._buf = b""
        self._pos = 0
        read = self._read
        chunk = self._chunk
        while len(out) < n:
            data = read(max(chunk, n - len(out)))
            if not data:
                raise SofaDecodeError("truncated payload")
            out += data
        if len(out) > n:  # keep the overshoot for the next read
            self._buf = bytes(out[n:])
        return bytes(out[:n])

    def _read_varints(self, count: int) -> list[int]:
        """Decode ``count`` consecutive varints in one tight loop that advances
        the cursor over the buffer — the whole varint codec is inlined here (no
        per-element call) and refills only when it runs off the end."""
        out = [0] * count
        buf = self._buf
        pos = self._pos
        n = len(buf)
        i = 0
        while i < count:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise SofaDecodeError("truncated varint")
                buf = self._buf
                pos = self._pos
                n = len(buf)
            b = buf[pos]
            pos += 1
            if b < 0x80:  # one-byte element
                out[i] = b
                i += 1
                continue
            result = b & 0x7F
            shift = 7
            while True:
                if pos >= n:
                    self._pos = pos
                    if not self._need(1):
                        raise SofaDecodeError("truncated varint")
                    buf = self._buf
                    pos = self._pos
                    n = len(buf)
                b = buf[pos]
                pos += 1
                result |= (b & 0x7F) << shift
                if b < 0x80:
                    break
                shift += 7
                if shift >= 64:
                    raise SofaDecodeError("varint overflow")
            out[i] = result & MASK64
            i += 1
        self._pos = pos
        return out

    def _skip_varints(self, count: int) -> None:
        """Advance the cursor past ``count`` varints without materialising them."""
        buf = self._buf
        pos = self._pos
        n = len(buf)
        i = 0
        while i < count:
            if pos < n:
                if buf[pos] < 0x80:
                    pos += 1
                    i += 1
                    continue
            self._pos = pos
            self._varint()
            buf = self._buf
            pos = self._pos
            n = len(buf)
            i += 1
        self._pos = pos

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

        if not self._need(1):
            if self._depth != 0:
                raise SofaDecodeError("truncated: unbalanced sequence")
            return None

        header = self._varint()
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
            if self._depth >= MAX_DEPTH:
                raise SofaDecodeError(f"nesting exceeds MAX_DEPTH={MAX_DEPTH}")
            self._depth += 1
            self._cur = Field(field_id, WireType.SEQUENCE_START)
            return self._cur

        if wtype == WireType.UNSIGNED or wtype == WireType.SIGNED:
            self._cur = Field(field_id, _WT[wtype])
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

        if wtype == WireType.ARRAY_UNSIGNED or wtype == WireType.ARRAY_SIGNED:
            count = self._varint()
            if count < 0 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            self._cur = Field(field_id, _WT[wtype], count=count)
            self._pending = (_VARRAY, wtype, count)
            return self._cur

        # wtype == ARRAY_FIXLEN
        count = self._varint()
        if count < 0 or count > ARRAY_MAX:
            raise SofaDecodeError(f"array count {count} out of range")
        if count == 0:
            # §4.8: a zero-count fixlen array carries no fixlen_word and no
            # payload — do not read further. The subtype is unknown / absent.
            self._cur = Field(field_id, WireType.ARRAY_FIXLEN, count=0, size=0)
            self._pending = (_FARRAY, None, 0, 0)
            return self._cur
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
            self._skip_varints(pending[2])
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

    # --- visitor driver -----------------------------------------------------

    def drive(self, visitor: Visitor) -> None:
        """Pull the whole stream, dispatching each field to ``visitor``'s typed
        hooks (see :class:`sofab.Visitor`). A visitor may decline a field via
        ``on_field`` / ``on_sequence_begin`` returning ``False`` to skip it
        without paying the decode cost."""
        while (f := self.next()) is not None:
            t = f.type
            if t == WireType.SEQUENCE_END:
                visitor.on_sequence_end()
            elif t == WireType.SEQUENCE_START:
                if visitor.on_sequence_begin(f.id) is False:
                    self.skip()
            elif visitor.on_field(f) is False:
                self.skip()
            elif t == WireType.UNSIGNED:
                visitor.on_unsigned(f.id, self.unsigned())
            elif t == WireType.SIGNED:
                visitor.on_signed(f.id, self.signed())
            elif t == WireType.FIXLEN:
                st = f.subtype
                if st == FixlenSubtype.FP32:
                    visitor.on_float32(f.id, self.float32())
                elif st == FixlenSubtype.FP64:
                    visitor.on_float64(f.id, self.float64())
                elif st == FixlenSubtype.STRING:
                    visitor.on_string(f.id, self.string())
                else:
                    visitor.on_bytes(f.id, self.bytes())
            elif t == WireType.ARRAY_UNSIGNED:
                visitor.on_unsigned_array(f.id, self.read_unsigned_array())
            elif t == WireType.ARRAY_SIGNED:
                visitor.on_signed_array(f.id, self.read_signed_array())
            else:  # ARRAY_FIXLEN
                if f.subtype == FixlenSubtype.FP32:
                    visitor.on_float32_array(f.id, self.read_float32_array())
                else:
                    visitor.on_float64_array(f.id, self.read_float64_array())

    # --- scalar reads -------------------------------------------------------

    def _take_scalar(self, wtype: WireType) -> int:
        pending = self._pending
        if pending is None or pending[0] != _SCALAR or pending[1] != wtype:
            raise SofaStateError("no matching scalar value for the current field")
        self._pending = None
        return self._varint()

    def unsigned(self) -> int:
        """Consume the current field as an unsigned integer.

        Raises :class:`SofaStateError` if the current field is not unsigned.
        """
        return self._take_scalar(WireType.UNSIGNED)

    def signed(self) -> int:
        """Consume the current field as a ZigZag-decoded signed integer.

        Raises :class:`SofaStateError` if the current field is not signed.
        """
        return zigzag_decode(self._take_scalar(WireType.SIGNED))

    def bool(self) -> bool:
        """Consume the current unsigned field as a boolean (non-zero is true)."""
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
        """Consume the current fixlen field as a 32-bit IEEE-754 float.

        Raises :class:`SofaStateError` if the field is not an fp32 fixlen, or
        :class:`SofaDecodeError` if its payload is not 4 bytes.
        """
        data = self._take_fixlen(FixlenSubtype.FP32)
        if len(data) != 4:
            raise SofaDecodeError("fp32 payload must be 4 bytes")
        return _core.unpack_f32(data)

    def float64(self) -> float:
        """Consume the current fixlen field as a 64-bit IEEE-754 float.

        Raises :class:`SofaStateError` if the field is not an fp64 fixlen, or
        :class:`SofaDecodeError` if its payload is not 8 bytes.
        """
        data = self._take_fixlen(FixlenSubtype.FP64)
        if len(data) != 8:
            raise SofaDecodeError("fp64 payload must be 8 bytes")
        return _core.unpack_f64(data)

    def string(self) -> str:
        """Consume the current fixlen field as a UTF-8 decoded string.

        Raises :class:`SofaStateError` if the field is not a STRING fixlen, or
        :class:`SofaDecodeError` if the payload is not valid UTF-8.
        """
        raw = self._take_fixlen(FixlenSubtype.STRING)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def bytes(self) -> bytes:
        """Consume the current fixlen field as a raw byte blob.

        Raises :class:`SofaStateError` if the field is not a BLOB fixlen.
        """
        return self._take_fixlen(FixlenSubtype.BLOB)

    # --- array reads --------------------------------------------------------

    def _take_varray(self, wtype: WireType) -> int:
        pending = self._pending
        if pending is None or pending[0] != _VARRAY or pending[1] != wtype:
            raise SofaStateError("current field is not a matching varint array")
        self._pending = None
        return int(pending[2])

    def read_unsigned_array(self) -> list[int]:
        """Consume the current field as a list of unsigned integers.

        Raises :class:`SofaStateError` if the field is not an unsigned array.
        """
        count = self._take_varray(WireType.ARRAY_UNSIGNED)
        return self._read_varints(count)

    def read_signed_array(self) -> list[int]:
        """Consume the current field as a list of ZigZag-decoded signed integers.

        Raises :class:`SofaStateError` if the field is not a signed array.
        """
        count = self._take_varray(WireType.ARRAY_SIGNED)
        return [zigzag_decode(v) for v in self._read_varints(count)]

    def _take_farray(self, subtype: FixlenSubtype) -> tuple[int, int]:
        pending = self._pending
        if pending is None or pending[0] != _FARRAY:
            raise SofaStateError("current field is not a fixlen array")
        count = int(pending[2])
        # A zero-count fixlen array carries no subtype on the wire (§4.8), so any
        # typed read accepts it and yields an empty list.
        if count != 0 and pending[1] != subtype:
            raise SofaStateError("fixlen-array subtype does not match the requested read")
        self._pending = None
        return count, int(pending[3])  # count, elem_size

    def read_float32_array(self) -> list[float]:
        """Consume the current field as a list of 32-bit IEEE-754 floats.

        Raises :class:`SofaStateError` if the field is not an fp32 array.
        """
        count, elem_size = self._take_farray(FixlenSubtype.FP32)
        data = self._read_exact(count * elem_size)
        return _core.unpack_f32_array(data, count)

    def read_float64_array(self) -> list[float]:
        """Consume the current field as a list of 64-bit IEEE-754 floats.

        Raises :class:`SofaStateError` if the field is not an fp64 array.
        """
        count, elem_size = self._take_farray(FixlenSubtype.FP64)
        data = self._read_exact(count * elem_size)
        return _core.unpack_f64_array(data, count)
