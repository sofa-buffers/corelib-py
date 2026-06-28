"""SofaBuffers encoder (``OStream`` equivalent).

Two construction models, mirroring the ecosystem:

* ``Encoder(writer)`` / ``Encoder()`` — Go-style. Bytes accumulate in an
  internal buffer; ``flush()`` drains them to ``writer`` (if given). With no
  writer the encoder is an in-memory buffer — read it with :meth:`getvalue`.
* ``Encoder.over_buffer(buf, offset, flush)`` — Rust/C/Java-style. Writes into
  a fixed caller buffer, reserving ``offset`` bytes at the front for a
  lower-layer header, draining via the ``flush`` sink when full.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from . import _core
from ._varint import encode_varint, zigzag_encode
from .types import (
    ARRAY_MAX,
    ID_MAX,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaBufferError,
    SofaError,
    SofaRangeError,
    SofaStateError,
    WireType,
)

FlushSink = Callable[[bytes], None]
Writer = object  # anything with .write(bytes)


class Encoder:
    """Encodes SofaBuffers fields to a byte stream."""

    def __init__(self, writer: Writer | None = None, *, sticky: bool = False) -> None:
        self._writer = writer
        self._buf = bytearray()
        # fixed-buffer mode (unused here):
        self._fixed: memoryview | None = None
        self._cap = 0
        self._cursor = 0
        self._flush_sink: FlushSink | None = None
        self._sticky = sticky
        self._error: SofaError | None = None
        self._depth = 0

    @classmethod
    def over_buffer(
        cls,
        buffer: bytearray,
        offset: int = 0,
        flush: FlushSink | None = None,
        *,
        sticky: bool = False,
    ) -> Encoder:
        self = cls.__new__(cls)
        self._writer = None
        self._buf = bytearray()
        self._fixed = None
        self._cap = 0
        self._cursor = 0
        self._flush_sink = flush
        self._sticky = sticky
        self._error = None
        self._depth = 0
        self.buffer_set(buffer, offset)
        return self

    def buffer_set(self, buffer: bytearray, offset: int = 0) -> None:
        """Install a new fixed output buffer mid-stream.

        Mirrors C ``sofab_ostream_buffer_set`` / Rust ``buffer_set`` / Java
        ``bufferSet``: typically called from inside the flush sink to hand the
        encoder a fresh buffer so encoding continues without interruption.
        ``offset`` bytes are reserved at the front (e.g. for a framing header).
        """
        if not 0 <= offset < len(buffer):
            raise SofaRangeError("offset must be within the buffer")
        self._fixed = memoryview(buffer)
        self._cap = len(buffer)
        self._cursor = offset

    # --- error / output handling --------------------------------------------

    @property
    def error(self) -> SofaError | None:
        """The first error recorded in sticky mode, or ``None``."""
        return self._error

    def _put(self, data: bytes) -> None:
        if self._fixed is None:
            self._buf += data
            return
        mv = self._fixed
        cap = self._cap
        pos = 0
        n = len(data)
        while pos < n:
            if self._cursor >= cap:
                self._drain()
                if self._cursor >= cap:
                    raise SofaBufferError("encoder buffer full")
            take = min(cap - self._cursor, n - pos)
            mv[self._cursor : self._cursor + take] = data[pos : pos + take]
            self._cursor += take
            pos += take

    def _drain(self) -> None:
        if self._flush_sink is None:
            raise SofaBufferError("encoder buffer full")
        self._flush_sink(bytes(self._fixed[0 : self._cursor]))  # type: ignore[index]
        self._cursor = 0

    def bytes_used(self) -> int:
        """Bytes written to the current buffer since construction/last flush."""
        return self._cursor if self._fixed is not None else len(self._buf)

    def flush(self) -> int:
        """Drain buffered bytes to the writer / flush sink; return the count."""
        if self._fixed is not None:
            used = self._cursor
            if self._flush_sink is not None and used:
                self._drain()
            return used
        used = len(self._buf)
        if self._writer is not None and used:
            self._writer.write(bytes(self._buf))  # type: ignore[attr-defined]
            self._buf.clear()
        return used

    def getvalue(self) -> bytes:
        """Return the accumulated bytes (in-memory writer model only)."""
        if self._fixed is not None:
            raise SofaStateError("getvalue() is only valid for the in-memory model")
        return bytes(self._buf)

    # --- internal write helpers ---------------------------------------------

    def _emit_varint(self, value: int) -> None:
        """Append a varint straight into the in-memory buffer with no
        intermediate ``bytes`` object (the hot path). Fixed-buffer mode falls
        back to the shared codec + the chunk-aware ``_put``."""
        if self._fixed is None:
            buf = self._buf
            while True:
                b = value & 0x7F
                value >>= 7
                if value:
                    buf.append(b | 0x80)
                else:
                    buf.append(b)
                    return
        else:
            self._put(encode_varint(value))

    def _header(self, field_id: int, wtype: WireType) -> None:
        if field_id < 0 or field_id > ID_MAX:
            raise SofaRangeError(f"id {field_id} out of range 0..{ID_MAX}")
        self._emit_varint((field_id << 3) | wtype)

    def _begin(self) -> bool:
        """Sticky-mode gate. Returns ``False`` if the op should be skipped."""
        return not (self._sticky and self._error is not None)

    def _fail(self, exc: SofaError) -> None:
        if self._sticky:
            if self._error is None:
                self._error = exc
        else:
            raise exc

    # --- scalars ------------------------------------------------------------

    def write_unsigned(self, field_id: int, value: int) -> None:
        if not self._begin():
            return
        try:
            if value < 0 or value > UNSIGNED_MAX:
                raise SofaRangeError(f"unsigned value {value} out of range")
            self._header(field_id, WireType.UNSIGNED)
            self._emit_varint(value)
        except SofaError as exc:
            self._fail(exc)

    def write_signed(self, field_id: int, value: int) -> None:
        if not self._begin():
            return
        try:
            if value < SIGNED_MIN or value > SIGNED_MAX:
                raise SofaRangeError(f"signed value {value} out of range")
            self._header(field_id, WireType.SIGNED)
            self._emit_varint(zigzag_encode(value))
        except SofaError as exc:
            self._fail(exc)

    def write_bool(self, field_id: int, value: bool) -> None:
        self.write_unsigned(field_id, 1 if value else 0)

    def write_float32(self, field_id: int, value: float) -> None:
        self._write_fixlen(field_id, _core.pack_f32(value), FixlenSubtype.FP32)

    def write_float64(self, field_id: int, value: float) -> None:
        self._write_fixlen(field_id, _core.pack_f64(value), FixlenSubtype.FP64)

    def write_string(self, field_id: int, text: str) -> None:
        self._write_fixlen(field_id, text.encode("utf-8"), FixlenSubtype.STRING)

    def write_bytes(self, field_id: int, data: bytes | bytearray | memoryview) -> None:
        self._write_fixlen(field_id, bytes(data), FixlenSubtype.BLOB)

    def _write_fixlen(self, field_id: int, data: bytes, subtype: FixlenSubtype) -> None:
        if not self._begin():
            return
        try:
            self._header(field_id, WireType.FIXLEN)
            self._emit_varint((len(data) << 3) | subtype)
            self._put(data)
        except SofaError as exc:
            self._fail(exc)

    # --- arrays -------------------------------------------------------------

    def write_unsigned_array(self, field_id: int, values: Iterable[int]) -> None:
        if not self._begin():
            return
        try:
            seq = list(values)
            self._array_header(field_id, WireType.ARRAY_UNSIGNED, len(seq))
            emit = self._emit_varint
            for v in seq:
                if v < 0 or v > UNSIGNED_MAX:
                    raise SofaRangeError(f"unsigned array value {v} out of range")
                emit(v)
        except SofaError as exc:
            self._fail(exc)

    def write_signed_array(self, field_id: int, values: Iterable[int]) -> None:
        if not self._begin():
            return
        try:
            seq = list(values)
            self._array_header(field_id, WireType.ARRAY_SIGNED, len(seq))
            emit = self._emit_varint
            for v in seq:
                if v < SIGNED_MIN or v > SIGNED_MAX:
                    raise SofaRangeError(f"signed array value {v} out of range")
                emit(zigzag_encode(v))
        except SofaError as exc:
            self._fail(exc)

    def write_float32_array(self, field_id: int, values: Iterable[float]) -> None:
        self._write_float_array(field_id, values, FixlenSubtype.FP32, _core.pack_f32_array, 4)

    def write_float64_array(self, field_id: int, values: Iterable[float]) -> None:
        self._write_float_array(field_id, values, FixlenSubtype.FP64, _core.pack_f64_array, 8)

    def _write_float_array(
        self,
        field_id: int,
        values: Iterable[float],
        subtype: FixlenSubtype,
        pack_array: Callable[[list[float]], bytes],
        elem_size: int,
    ) -> None:
        if not self._begin():
            return
        try:
            seq = [float(v) for v in values]
            self._array_header(field_id, WireType.ARRAY_FIXLEN, len(seq))
            self._emit_varint((elem_size << 3) | subtype)
            self._put(pack_array(seq))  # one struct.pack for the whole array
        except SofaError as exc:
            self._fail(exc)

    def _array_header(self, field_id: int, wtype: WireType, count: int) -> None:
        if count < 1 or count > ARRAY_MAX:
            raise SofaRangeError(f"array count {count} out of range 1..{ARRAY_MAX}")
        self._header(field_id, wtype)
        self._emit_varint(count)

    # --- sequences ----------------------------------------------------------

    def write_sequence_begin(self, field_id: int) -> None:
        if not self._begin():
            return
        try:
            self._header(field_id, WireType.SEQUENCE_START)
            self._depth += 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end(self) -> None:
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            self._emit_varint(WireType.SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)
