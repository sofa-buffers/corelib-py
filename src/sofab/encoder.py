"""SofaBuffers encoder (``OStream`` equivalent).

Two construction models, mirroring the ecosystem:

* ``Encoder(writer)`` / ``Encoder()`` — Go-style. Bytes accumulate in an
  internal buffer; ``flush()`` drains them to ``writer`` (if given). With no
  writer the encoder is an in-memory buffer — read it with :meth:`getvalue`.
* ``Encoder.over_buffer(buf, offset, flush)`` — Rust/C/Java-style. Writes into
  a fixed caller buffer, reserving ``offset`` bytes at the front for a
  lower-layer header, draining via the ``flush`` sink when full.

Sequences are framed **lazily**: :meth:`Encoder.write_sequence_begin_lazy` holds
the header back until the sequence receives content, so a sequence-typed field
whose value equals its declared default is omitted rather than emitted as an
empty ``begin``/``end`` frame (MESSAGE_SPEC §2). The closer picks the outcome —
:meth:`Encoder.write_sequence_end` drops a contentless sequence,
:meth:`Encoder.write_sequence_end_keep` forces the frame out (wrapper-array
elements, explicit empty arrays). Held-back ids are encoder state, never buffer
content, so a flush cannot split a pending run *by construction*: a pending header
occupies no buffer space, and the buffer only fills through a write — which
commits the whole run before its first byte goes out. A tiny output buffer
therefore produces exactly the one-shot bytes.

The run itself has no fixed window: it grows on demand, so the hold-back reaches
the full :data:`sofab.MAX_DEPTH` and every depth is canonical (CORELIB_PLAN §6 —
only a heap-free profile may bound the run and frame eagerly beyond the bound).
It is allocated on the first hold-back, so an encoder that never opens a sequence
never pays for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from operator import index as _index
from typing import Callable, SupportsIndex

from . import _core
from ._varint import encode_varint, zigzag_encode
from .types import (
    ARRAY_MAX,
    ID_MAX,
    MAX_DEPTH,
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


def _as_int(value: object, what: str) -> int:
    """Coerce ``value`` to the ``int`` an integer field will be written from.

    The rule is Python's own: a type that considers itself *losslessly* an
    integer implements ``__index__`` — ``int``, ``bool``, ``IntEnum``, NumPy
    integers do; ``float`` deliberately does not, because ``3.7`` cannot become
    an integer without discarding information. This is the same line CPython
    draws wherever an integer is required (``seq[i]``, ``range(x)``,
    ``bytes(n)``), and the same values both engines already agreed on.

    Anything else is refused with :class:`SofaRangeError` (§6.3
    ``InvalidArgument``) rather than truncated: writing ``3`` for a caller's
    ``3.7`` would be a value change the receiver has no way to detect.
    """
    try:
        # The TypeError this raises *is* the check — asking the object whether it
        # is an integer is cheaper and more accurate than any isinstance test.
        return _index(value)  # type: ignore[arg-type]
    except TypeError:
        raise SofaRangeError(
            f"{what} must be an integer, not {type(value).__name__}"
        ) from None


class Encoder:
    """Encodes SofaBuffers fields to a byte stream."""

    def __init__(self, writer: Writer | None = None, *, sticky: bool = False) -> None:
        """Create a Go-style encoder backed by an internal growable buffer.

        Bytes accumulate in memory; :meth:`flush` drains them to ``writer`` (any
        object with ``write(bytes)``) if one is given, otherwise read them back
        with :meth:`getvalue`. Pass ``sticky=True`` to latch the first error
        instead of raising on every call (inspect it via :attr:`error`).
        """
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
        # Ids of the innermost open sequences whose header has not been written
        # yet (MESSAGE_SPEC §2 lazy framing). Always a contiguous suffix of the
        # open sequences: writing any field commits the whole run at once, so
        # :meth:`write_sequence_end` can simply pop the last entry.
        #
        # ``None`` until the first hold-back: the list grows on demand (CORELIB_PLAN
        # §6 — an implementation that can allocate holds back to the full MAX_DEPTH,
        # so there is no fixed window and no eager-framing fallback), and an encoder
        # that never opens a sequence never allocates it at all.
        self._pending: list[int] | None = None

    @classmethod
    def over_buffer(
        cls,
        buffer: bytearray,
        offset: int = 0,
        flush: FlushSink | None = None,
        *,
        sticky: bool = False,
    ) -> Encoder:
        """Create an encoder that writes into a fixed caller-owned buffer.

        Rust/C/Java-style construction: bytes are written directly into
        ``buffer``, reserving ``offset`` bytes at the front for a lower-layer
        header. When the buffer fills, the encoder calls ``flush`` with the
        bytes written so far; ``flush`` is expected to drain them and (via
        :meth:`buffer_set`) hand back a fresh buffer so encoding continues. With
        no ``flush`` sink a full buffer raises :class:`SofaBufferError`. Pass
        ``sticky=True`` to latch the first error instead of raising per call
        (inspect it via :attr:`error`).
        """
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
        self._pending = None
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
        # Hoisted for the common no-drain path, but they must be re-read after any
        # drain: the flush sink may call buffer_set() to hand back a *different*
        # buffer (the documented pattern — see over_buffer), which replaces both.
        # Writing through a stale view sent everything past the first flush into the
        # orphaned buffer while the fresh one was emitted zeroed. A sink that drains
        # and reuses the same buffer was unaffected, which is why this survived.
        mv = self._fixed
        cap = self._cap
        pos = 0
        n = len(data)
        while pos < n:
            if self._cursor >= cap:
                self._drain()
                mv = self._fixed
                cap = self._cap
                # Defensive: _drain either raises (no sink) or resets the cursor
                # to 0, so a still-full buffer here is unreachable in practice.
                if self._cursor >= cap:  # pragma: no cover
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

    def _header(self, field_id: SupportsIndex, wtype: WireType) -> None:
        """Write a field header — the single choke point every field write passes
        through, and therefore where a held-back sequence run is committed.

        The field about to be written is *content*, which proves every enclosing
        sequence differs from its declared default and must be framed after all.
        Only genuine field writes reach here: a sequence is opened by
        :meth:`write_sequence_begin_lazy` (which never writes) and closed by
        :meth:`write_sequence_end` / :meth:`write_sequence_end_keep` (which emit
        the bare ``0x07`` end marker themselves), so no gate on ``wtype`` is
        needed and no writer can bypass the commit.
        """
        if not isinstance(field_id, int):
            field_id = _as_int(field_id, "id")
        if field_id < 0 or field_id > ID_MAX:
            raise SofaRangeError(f"id {field_id} out of range 0..{ID_MAX}")
        if self._pending:
            self._commit_pending()
        self._emit_varint((field_id << 3) | wtype)

    def _commit_pending(self) -> None:
        """Emit the held-back sequence headers, outermost first.

        Cold: it runs at most once per non-default sequence, never per field.
        The run is detached before the first byte goes out, so a flush sink that
        re-enters the encoder cannot see a half-committed run.
        """
        run = self._pending
        self._pending = None
        for field_id in run or ():
            self._emit_varint((field_id << 3) | WireType.SEQUENCE_START)

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

    def write_unsigned(self, field_id: SupportsIndex, value: SupportsIndex) -> None:
        """Write an unsigned integer field as a base-128 varint.

        ``value`` must be an integer in ``0..UNSIGNED_MAX`` (64-bit), else
        :class:`SofaRangeError`. "Integer" is Python's own rule — anything with
        ``__index__`` (``int``, ``bool``, ``IntEnum``, NumPy integers). A
        ``float`` is refused rather than truncated, ``3.0`` included; write
        ``int(x)`` if that is what you mean.
        """
        if not self._begin():
            return
        try:
            if not isinstance(value, int):
                value = _as_int(value, "unsigned value")
            if value < 0 or value > UNSIGNED_MAX:
                raise SofaRangeError(f"unsigned value {value} out of range")
            self._header(field_id, WireType.UNSIGNED)
            self._emit_varint(value)
        except SofaError as exc:
            self._fail(exc)

    def write_signed(self, field_id: SupportsIndex, value: SupportsIndex) -> None:
        """Write a signed integer field, ZigZag-encoded into a varint.

        ``value`` must be an integer in ``SIGNED_MIN..SIGNED_MAX`` (64-bit),
        else :class:`SofaRangeError` — see :meth:`write_unsigned` for what counts
        as an integer.
        """
        if not self._begin():
            return
        try:
            if not isinstance(value, int):
                value = _as_int(value, "signed value")
            if value < SIGNED_MIN or value > SIGNED_MAX:
                raise SofaRangeError(f"signed value {value} out of range")
            self._header(field_id, WireType.SIGNED)
            self._emit_varint(zigzag_encode(value))
        except SofaError as exc:
            self._fail(exc)

    def write_bool(self, field_id: SupportsIndex, value: bool) -> None:
        """Write a boolean as an unsigned field (``1``/``0``)."""
        self.write_unsigned(field_id, 1 if value else 0)

    def write_float32(self, field_id: SupportsIndex, value: float) -> None:
        """Write a 32-bit IEEE-754 float as a little-endian fixlen field."""
        self._write_fixlen(field_id, _core.pack_f32(value), FixlenSubtype.FP32)

    def write_float64(self, field_id: SupportsIndex, value: float) -> None:
        """Write a 64-bit IEEE-754 float as a little-endian fixlen field."""
        self._write_fixlen(field_id, _core.pack_f64(value), FixlenSubtype.FP64)

    def write_string(self, field_id: SupportsIndex, text: str) -> None:
        r"""Write a UTF-8 string as a fixlen field (STRING subtype).

        Encoding is strict UTF-8 (``str.encode("utf-8")`` with no ``errors=``).
        Python ``str`` is a Unicode string type, so per CORELIB_PLAN §6.4 it is
        **always strict**: ``SOFAB_STRICT_UTF8`` is a no-op for it and is
        omitted entirely (documented as always-ON). A ``str`` that cannot be
        encoded as valid UTF-8 — a lone/unpaired surrogate such as ``'\ud800'``
        — is refused with :class:`SofaRangeError` (the encode-side
        ``InvalidArgument`` outcome, MESSAGE_SPEC §8 producer-side MUST NOT),
        never silently replaced. Embedded ``U+0000`` is valid UTF-8 and
        round-trips unchanged.
        """
        if not self._begin():
            return
        try:
            data = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            self._fail(SofaRangeError(f"string field is not valid UTF-8: {exc}"))
            return
        self._write_fixlen(field_id, data, FixlenSubtype.STRING)

    def write_bytes(self, field_id: SupportsIndex,
                    data: bytes | bytearray | memoryview) -> None:
        """Write a raw byte blob as a fixlen field (BLOB subtype)."""
        self._write_fixlen(field_id, bytes(data), FixlenSubtype.BLOB)

    def _write_fixlen(self, field_id: SupportsIndex, data: bytes,
                      subtype: FixlenSubtype) -> None:
        if not self._begin():
            return
        try:
            self._header(field_id, WireType.FIXLEN)
            self._emit_varint((len(data) << 3) | subtype)
            self._put(data)
        except SofaError as exc:
            self._fail(exc)

    # --- arrays -------------------------------------------------------------

    def write_unsigned_array(self, field_id: SupportsIndex,
                             values: Iterable[SupportsIndex]) -> None:
        """Write an array of unsigned integers, each as a varint.

        The element count must be ``0..ARRAY_MAX`` and every element an integer
        in ``0..UNSIGNED_MAX``, else :class:`SofaRangeError` (see
        :meth:`write_unsigned` for what counts as an integer). A zero-count array
        is a valid, fully-specified empty array on the wire
        (``[header][count=0]``).
        """
        if not self._begin():
            return
        try:
            seq = list(values)
            self._array_header(field_id, WireType.ARRAY_UNSIGNED, len(seq))
            if self._fixed is None:
                # Hot path: the varint codec is inlined over the whole array so
                # each element costs a loop iteration rather than a Python call.
                buf = self._buf
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "unsigned array value")
                    if v < 0 or v > UNSIGNED_MAX:
                        raise SofaRangeError(f"unsigned array value {v} out of range")
                    while v >= 0x80:
                        buf.append((v & 0x7F) | 0x80)
                        v >>= 7
                    buf.append(v)
            else:
                emit = self._emit_varint
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "unsigned array value")
                    if v < 0 or v > UNSIGNED_MAX:
                        raise SofaRangeError(f"unsigned array value {v} out of range")
                    emit(v)
        except SofaError as exc:
            self._fail(exc)

    def write_signed_array(self, field_id: SupportsIndex,
                           values: Iterable[SupportsIndex]) -> None:
        """Write an array of signed integers, each ZigZag-encoded into a varint.

        The element count must be ``0..ARRAY_MAX`` and every element an integer
        in ``SIGNED_MIN..SIGNED_MAX``, else :class:`SofaRangeError` (see
        :meth:`write_unsigned` for what counts as an integer). A zero-count array
        is a valid, fully-specified empty array (``[header][count=0]``).
        """
        if not self._begin():
            return
        try:
            seq = list(values)
            self._array_header(field_id, WireType.ARRAY_SIGNED, len(seq))
            if self._fixed is None:
                buf = self._buf   # see write_unsigned_array: codec inlined
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "signed array value")
                    if v < SIGNED_MIN or v > SIGNED_MAX:
                        raise SofaRangeError(f"signed array value {v} out of range")
                    u = (v << 1) ^ (v >> 63)
                    while u >= 0x80:
                        buf.append((u & 0x7F) | 0x80)
                        u >>= 7
                    buf.append(u)
            else:
                emit = self._emit_varint
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "signed array value")
                    if v < SIGNED_MIN or v > SIGNED_MAX:
                        raise SofaRangeError(f"signed array value {v} out of range")
                    emit(zigzag_encode(v))
        except SofaError as exc:
            self._fail(exc)

    def write_float32_array(self, field_id: SupportsIndex, values: Iterable[float]) -> None:
        """Write an array of 32-bit floats as a packed little-endian fixlen array.

        The element count must be ``0..ARRAY_MAX``, else :class:`SofaRangeError`.
        A zero-count array emits ``[header][count=0][fixlen_word]`` — the
        ``fixlen_word`` is always present (so empty fp32/fp64 arrays stay
        distinguishable) but there is no payload (§4.8).
        """
        self._write_float_array(field_id, values, FixlenSubtype.FP32, _core.pack_f32_array, 4)

    def write_float64_array(self, field_id: SupportsIndex, values: Iterable[float]) -> None:
        """Write an array of 64-bit floats as a packed little-endian fixlen array.

        The element count must be ``0..ARRAY_MAX``, else :class:`SofaRangeError`.
        A zero-count array emits ``[header][count=0][fixlen_word]`` — the
        ``fixlen_word`` is always present (so empty fp32/fp64 arrays stay
        distinguishable) but there is no payload (§4.8).
        """
        self._write_float_array(field_id, values, FixlenSubtype.FP64, _core.pack_f64_array, 8)

    def _write_float_array(
        self,
        field_id: SupportsIndex,
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
            # §4.8: a fixlen array ALWAYS carries its fixlen_word (the shared
            # element subtype/width), even when empty, so an empty fp32 and fp64
            # array stay distinguishable on the wire. The payload loop then runs
            # zero times for a zero-count array.
            self._emit_varint((elem_size << 3) | subtype)
            self._put(pack_array(seq))  # one struct.pack for the whole array
        except SofaError as exc:
            self._fail(exc)

    def _array_header(self, field_id: SupportsIndex, wtype: WireType, count: int) -> None:
        # Defensive: count is always len() of a materialized list, so it is
        # non-negative and can't exceed ARRAY_MAX without exhausting memory first.
        if count < 0 or count > ARRAY_MAX:  # pragma: no cover
            raise SofaRangeError(f"array count {count} out of range 0..{ARRAY_MAX}")
        self._header(field_id, wtype)
        self._emit_varint(count)

    # --- sequences ----------------------------------------------------------

    def write_sequence_begin_lazy(self, field_id: SupportsIndex) -> None:
        """Open a nested sequence (sub-message) under ``field_id``, **holding its
        header back** until the sequence turns out to have content.

        MESSAGE_SPEC §2 omits a sequence-typed *field* whose value equals its
        declared default, and "not one child was written" is exactly that
        condition — evaluated per child field, recursively, for free, because the
        message layer already omits every child equal to its own default. A
        sequence closed with nothing in it therefore emits **nothing** instead of
        a two-byte empty frame, and an all-default message becomes the empty byte
        string. The predicate is never a byte image of the object, so in-memory
        padding cannot influence it.

        This is the only way to open a sequence. How it closes decides whether a
        contentless one survives: :meth:`write_sequence_end` drops it,
        :meth:`write_sequence_end_keep` forces the frame out.

        Must be balanced by a later :meth:`write_sequence_end` /
        :meth:`write_sequence_end_keep`. Refuses to open a sequence nested deeper
        than :data:`sofab.MAX_DEPTH` (255), raising :class:`SofaRangeError`.
        """
        if not self._begin():
            return
        try:
            if self._depth >= MAX_DEPTH:
                raise SofaRangeError(f"nesting exceeds MAX_DEPTH={MAX_DEPTH}")
            # This is the one write that does not go through _header — the id is
            # held back rather than emitted — so it applies the same rule itself.
            if not isinstance(field_id, int):
                field_id = _as_int(field_id, "id")
            if field_id < 0 or field_id > ID_MAX:
                raise SofaRangeError(f"id {field_id} out of range 0..{ID_MAX}")
            # No hold-back window to exhaust: the pending run is a Python list
            # that grows on demand, so it reaches the full MAX_DEPTH (CORELIB_PLAN
            # §6: only a heap-free profile may bound the run and frame eagerly
            # beyond the bound). There is therefore no eager-framing fallback,
            # and the "pending is a contiguous suffix of the open sequences"
            # invariant holds unconditionally. The list itself is allocated here,
            # on the first hold-back, not in the constructor — an encoder that
            # never opens a sequence never pays for one.
            if self._pending is None:
                self._pending = [field_id]
            else:
                self._pending.append(field_id)
            self._depth += 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end(self) -> None:
        """Close the innermost open sequence, letting it **vanish** if it received
        no content.

        Use it wherever absence encodes the same value as an empty frame: a
        ``struct``/``union`` field, and an array field whose declared ``default``
        is the empty collection (MESSAGE_SPEC §2). Where the frame must be
        visible, close with :meth:`write_sequence_end_keep` instead.

        Raises :class:`SofaStateError` if no sequence is currently open.
        """
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            if self._pending:
                # The innermost open sequence is the last held-back one (the
                # pending run is a suffix), so dropping it is a plain pop: no
                # header and no end marker ever reach the wire.
                self._pending.pop()
                self._depth -= 1
                return
            self._emit_varint(WireType.SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end_keep(self) -> None:
        """Close the innermost open sequence, **keeping** its frame even when it
        received no content.

        Behaves like a write: it first emits any held-back headers — this frame's
        and every enclosing one's — and then the end marker, so an empty sequence
        reaches the wire as ``begin`` + ``end``.

        Required wherever the frame carries information beyond its contents:

        * a **wrapper-array element** (``struct``/``union``/nested row): element
          presence is what carries a dynamic array's length — *highest present id
          + 1* (MESSAGE_SPEC §5.1) — so dropping an all-default element would
          change the decoded length, not just the bytes;
        * an array field already known to **differ from a non-empty declared
          default**: absence would reconstruct that default, so the empty frame is
          the only encoding of "explicitly empty" (§2, §3).

        The two failure directions are not symmetric, which is why this is the
        safe choice when in doubt: using it where :meth:`write_sequence_end` would
        do costs one non-canonical empty frame that every decoder normalizes away,
        while the reverse silently changes an array's length.

        Raises :class:`SofaStateError` if no sequence is currently open.
        """
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            if self._pending:
                self._commit_pending()
            self._emit_varint(WireType.SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)
