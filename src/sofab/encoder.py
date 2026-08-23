"""SofaBuffers encoder (``OStream`` equivalent).

**One buffer-ownership model** (CORELIB_PLAN §5.1): the encoder writes into a
**fixed** output buffer and drains it through a flush sink when that buffer
fills. It never grows or reallocates the buffer it writes into — "what was
handed over is what gets written". Three construction shapes express that one
model:

* ``Encoder.over_buffer(buf, offset, flush)`` — the primitive, and the only one
  that takes a caller-supplied buffer. Writes into ``buf``, reserving ``offset``
  bytes at the front for a lower-layer header, draining via the ``flush`` sink
  when full; *without* a sink a full buffer reports :class:`SofaBufferError`,
  which is the shape a caller sizes from a generated ``MAX_SIZE``.
* ``Encoder(writer)`` — the same primitive over a scratch buffer of
  :data:`_SCRATCH_SIZE` bytes installed **with** a sink that forwards to
  ``writer.write``. This is §5.1's "unbounded schema" shape, so a message of any
  size streams out through bounded memory *as it is written*, rather than
  accumulating until :meth:`Encoder.flush`.
* ``Encoder()`` — the same again, with the sink appending into the *result* the
  encoder hands back from :meth:`Encoder.getvalue`. What grows there is the
  message being returned, not a buffer the encoder writes into.

The scratch buffer of the latter two is one allocation, made once at
construction and never resized; §5.1 puts even that in the generated layer,
which knows the schema — so generated code that can bound its message should
prefer ``over_buffer`` with a ``MAX_SIZE``-sized buffer and no sink.

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
    FIXLEN_MAX,
    ID_MAX,
    MAX_DEPTH,
    MIN_OUTPUT_BUFFER,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaBufferError,
    SofaError,
    SofaRangeError,
    WireType,
)

#: Size of the scratch buffer the convenience constructors install (CORELIB_PLAN
#: §5.1 "unbounded schema" shape). One allocation per encoder, made once at
#: construction and never resized — the encoder drains it through its sink
#: whenever it fills, so the memory an encode holds is this constant and not the
#: message. A kibibyte is large enough that the drain is amortized to nothing
#: (one sink call per ~1 kB of output) and small enough to stay off the cost of
#: a per-message encoder.
_SCRATCH_SIZE = 1024

# The wire types and fixlen subtypes as plain ints. Every one of these is used
# as a *number* -- OR-ed into a header, passed to _header -- and reaching it
# through the enum class costs a global load plus an attribute lookup on every
# write call. The enum members stay in the public signatures; only the arithmetic
# uses these.
_WT_UNSIGNED = int(WireType.UNSIGNED)
_WT_SIGNED = int(WireType.SIGNED)
_WT_FIXLEN = int(WireType.FIXLEN)
_WT_ARRAY_UNSIGNED = int(WireType.ARRAY_UNSIGNED)
_WT_ARRAY_SIGNED = int(WireType.ARRAY_SIGNED)
_WT_ARRAY_FIXLEN = int(WireType.ARRAY_FIXLEN)
_WT_SEQUENCE_START = int(WireType.SEQUENCE_START)
_WT_SEQUENCE_END = int(WireType.SEQUENCE_END)
_ST_FP32 = int(FixlenSubtype.FP32)
_ST_FP64 = int(FixlenSubtype.FP64)
_ST_STRING = int(FixlenSubtype.STRING)
_ST_BLOB = int(FixlenSubtype.BLOB)

#: Bytes a varint can occupy (§4.1), and therefore the room the in-place
#: fast path in :meth:`Encoder._emit_varint` requires before it writes.
_VARINT_MAX = 10

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
        """Create an encoder over a scratch buffer the library installs for you.

        This is §5.1's "unbounded schema" shape: a **fixed** buffer of
        :data:`_SCRATCH_SIZE` bytes installed **with** a flush sink, never a
        buffer that grows. With ``writer`` (any object with ``write(bytes)``) the
        sink forwards to it, so the message streams out while it is written and
        the encoder holds at most one scratch buffer of it; with no writer the
        sink appends into the result :meth:`getvalue` hands back.

        Pass ``sticky=True`` to latch the first error instead of raising on every
        call (inspect it via :attr:`error`).

        A caller that wants to own the buffer — the conformant shape for
        generated code, which knows the schema — uses :meth:`over_buffer`
        instead.
        """
        self._writer = writer
        # The in-memory model's sink is the *result* it hands back — the "growing
        # result" §5.1 names for the unbounded shape, which is the message, not a
        # buffer the encoder writes into. It is the list of drained chunks
        # :meth:`getvalue` joins, and it stays ``None`` until the first drain: a
        # message that fits in the scratch buffer never allocates one at all.
        self._in_memory = writer is None
        self._result: list[bytes] | None = None
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
        # The one allocation, made here and never resized. Installed exactly as
        # :meth:`buffer_set` would (the checks it makes are constants here: a
        # zero offset into a buffer of _SCRATCH_SIZE >= MIN_OUTPUT_BUFFER).
        scratch = bytearray(_SCRATCH_SIZE)
        self._fixed = memoryview(scratch)
        self._fixed_ba = scratch
        self._cap = _SCRATCH_SIZE
        self._installs = 1

    def _has_sink(self) -> bool:
        """Whether a flush can occur — i.e. whether the installed buffer is a
        sink-installed one, which is what :data:`~sofab.MIN_OUTPUT_BUFFER` binds
        (§5.1). All three shapes of sink count: the caller's callback, the
        writer, and the in-memory result."""
        return (
            self._in_memory or self._writer is not None or self._flush_sink is not None
        )

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

        With a ``flush`` sink the buffer must leave at least
        :data:`~sofab.MIN_OUTPUT_BUFFER` bytes past ``offset``; without one there
        is no minimum. See :meth:`buffer_set`, which enforces both.
        """
        self = cls.__new__(cls)
        self._writer = None
        self._in_memory = False
        self._result = None
        self._cap = 0
        self._cursor = 0
        self._installs = 0
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

        The offset belongs to *this installation*, not to the buffer
        (CORELIB_PLAN §5.1): the cursor starts at ``offset`` and the offset is
        then consumed, so a later flush the sink returns from without installing
        anything resumes at 0. Re-installing — even the *same* buffer — is what
        re-arms the reservation, which is how a sink gets fresh header room in
        every flushed unit rather than only in the first.

        :data:`~sofab.MIN_OUTPUT_BUFFER` binds here, and only for a buffer that
        is installed **with** a flush sink: ``len(buffer) - offset`` must be at
        least that many bytes, checked at installation and at every mid-stream
        set, so an unusable buffer is refused where it is handed over rather than
        partway through a message. Without a sink no flush can occur and no
        minimum applies — the buffer holds the message or reports
        :class:`SofaBufferError` — which is what keeps a caller sizing from a
        generated ``MAX_SIZE`` exact, down to a zero-byte remainder.
        """
        if not 0 <= offset <= len(buffer):
            raise SofaRangeError("offset must be within the buffer")
        usable = len(buffer) - offset
        if usable < MIN_OUTPUT_BUFFER and self._has_sink():
            raise SofaRangeError(
                f"a buffer installed with a flush sink needs at least "
                f"MIN_OUTPUT_BUFFER={MIN_OUTPUT_BUFFER} usable byte(s), got {usable}"
            )
        self._fixed = memoryview(buffer)
        # The same storage under both views: the memoryview for slice writes
        # (twice as fast as a bytearray's, and it cannot resize the caller's
        # buffer by accident), the bytearray for the single-byte writes of the
        # inlined varint loops (a third faster than the memoryview's).
        self._fixed_ba = buffer
        self._cap = len(buffer)
        self._cursor = offset
        # Counted so _drain can tell whether the sink took the buffer (installed
        # a replacement) or merely copied it and returned.
        self._installs += 1

    # --- error / output handling --------------------------------------------

    @property
    def error(self) -> SofaError | None:
        """The first error recorded in sticky mode, or ``None``."""
        return self._error

    def _put(self, data: bytes) -> None:
        n = len(data)
        if self._in_memory and n >= self._cap:
            # A divisible run at least as long as the whole buffer, in the model
            # whose sink is the result: hand it over as its own chunk instead of
            # copying it through the buffer a bufferful at a time. Every caller
            # passes a fresh, immutable ``bytes`` (an encoded ``str``, a copied
            # blob, a packed float array), so the result may keep it as-is, and
            # draining first is what keeps the wire order.
            if self._cursor:
                self._drain()
            if self._result is None:
                self._result = [data]
            else:
                self._result.append(data)
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
        while pos < n:
            if self._cursor >= cap:
                self._drain()
                mv = self._fixed
                cap = self._cap
                # Defensive: _drain either raises (no sink) or leaves the cursor
                # inside the buffer — at 0, or at the offset a sink installed,
                # which buffer_set bounds — so a still-full buffer is unreachable.
                if self._cursor >= cap:  # pragma: no cover
                    raise SofaBufferError("encoder buffer full")
            take = min(cap - self._cursor, n - pos)
            mv[self._cursor : self._cursor + take] = data[pos : pos + take]
            self._cursor += take
            pos += take

    def _drain(self) -> None:
        if self._in_memory:
            # The sink is the result the encoder hands back: the drained bytes are
            # kept as a chunk for :meth:`getvalue` to join, which costs one copy
            # here and none there (``b"".join`` of a single chunk returns it
            # unchanged). The buffer is never taken, so the cursor resumes at 0.
            chunk = bytes(self._fixed[0 : self._cursor])
            if self._result is None:
                self._result = [chunk]
            else:
                self._result.append(chunk)
            self._cursor = 0
            return
        if self._writer is None and self._flush_sink is None:
            raise SofaBufferError("encoder buffer full")
        # CORELIB_PLAN §5.1 "what a returning flush callback leaves behind": a sink
        # that returns without installing a buffer *copied*, so the active buffer
        # stays active and encoding resumes at 0. A sink that *took* the buffer must
        # install a replacement before returning, and that installation's offset is
        # the new cursor — resetting to 0 here would silently drop the header room it
        # just reserved and overwrite it with payload in every packet but the first.
        installs = self._installs
        snapshot = bytes(self._fixed[0 : self._cursor])
        if self._writer is not None:
            self._writer.write(snapshot)  # type: ignore[attr-defined]
        else:
            self._flush_sink(snapshot)  # type: ignore[misc]
        if self._installs == installs:
            self._cursor = 0

    def bytes_used(self) -> int:
        """Bytes standing in the output buffer, i.e. written since it was
        installed and not yet drained.

        The buffer is fixed, so this never exceeds its size — for the
        convenience models, :data:`_SCRATCH_SIZE`. It is *not* the length of the
        message: bytes already drained to the writer/sink are no longer here.
        """
        return self._cursor

    def flush(self) -> int:
        """Drain buffered bytes to the writer / flush sink; return the count."""
        used = self._cursor
        if used and self._has_sink():
            self._drain()
        return used

    def getvalue(self) -> bytes:
        """Return the encoded message (in-memory model only).

        Only ``Encoder()`` retains one: with a writer the bytes have already
        been handed over, and with :meth:`over_buffer` they are in the caller's
        buffer — returning the undrained tail of either would be partial output
        dressed up as a whole message (CORELIB_PLAN §5.1), so both raise
        :class:`SofaRangeError`.
        """
        if not self._in_memory:
            raise SofaRangeError("getvalue() is only valid for the in-memory model")
        chunks = self._result
        if chunks is None:  # never drained: the message is the buffer prefix
            return bytes(self._fixed[0 : self._cursor])
        if not self._cursor:  # fully drained (the usual case, after flush())
            return b"".join(chunks)
        return b"".join([*chunks, bytes(self._fixed[0 : self._cursor])])

    # --- internal write helpers ---------------------------------------------

    def _emit_varint(self, value: int) -> None:
        """Write a varint into the output buffer (the hot path).

        With a whole varint's room left it is encoded straight into the buffer,
        with no intermediate ``bytes`` object; on the last few bytes of the
        buffer it goes through the shared codec and the chunk-aware
        :meth:`_put`, which splits it across the drain. Both paths produce the
        same bytes — the split is what ``MIN_OUTPUT_BUFFER == 1`` asserts.
        """
        cursor = self._cursor
        if cursor + _VARINT_MAX <= self._cap:
            buf = self._fixed_ba
            # By width -- see write_unsigned_array. A field header is one byte
            # while the id is below 16 and two up to 4096, so those two are the
            # ones this site sees.
            if value < 0x80:
                buf[cursor] = value
                self._cursor = cursor + 1
            elif value < 0x4000:
                buf[cursor] = (value & 0x7F) | 0x80
                buf[cursor + 1] = value >> 7
                self._cursor = cursor + 2
            else:
                while value >= 0x4000:
                    buf[cursor] = (value & 0x7F) | 0x80
                    buf[cursor + 1] = ((value >> 7) & 0x7F) | 0x80
                    cursor += 2
                    value >>= 14
                if value >= 0x80:
                    buf[cursor] = (value & 0x7F) | 0x80
                    cursor += 1
                    value >>= 7
                buf[cursor] = value
                self._cursor = cursor + 1
        else:
            self._put(encode_varint(value))

    def _header(self, field_id: SupportsIndex, wtype: int) -> None:
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
            self._emit_varint((field_id << 3) | _WT_SEQUENCE_START)

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
            self._header(field_id, _WT_UNSIGNED)
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
            self._header(field_id, _WT_SIGNED)
            self._emit_varint(zigzag_encode(value))
        except SofaError as exc:
            self._fail(exc)

    def write_bool(self, field_id: SupportsIndex, value: bool) -> None:
        """Write a boolean as an unsigned field (``1``/``0``)."""
        self.write_unsigned(field_id, 1 if value else 0)

    def write_float32(self, field_id: SupportsIndex, value: float) -> None:
        """Write a 32-bit IEEE-754 float as a little-endian fixlen field."""
        self._write_fixlen(field_id, _core.pack_f32(value), _ST_FP32)

    def write_float64(self, field_id: SupportsIndex, value: float) -> None:
        """Write a 64-bit IEEE-754 float as a little-endian fixlen field."""
        self._write_fixlen(field_id, _core.pack_f64(value), _ST_FP64)

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
        self._write_fixlen(field_id, data, _ST_STRING)

    def write_bytes(self, field_id: SupportsIndex,
                    data: bytes | bytearray | memoryview) -> None:
        """Write a raw byte blob as a fixlen field (BLOB subtype).

        A blob longer than :data:`sofab.FIXLEN_MAX` is refused with
        :class:`SofaRangeError` (see :meth:`_write_fixlen`) — on the *declared*
        length, before the copy, so an oversized payload is never duplicated
        just to be rejected.
        """
        if not self._begin():
            return
        n = len(data)
        if n > FIXLEN_MAX:
            self._fail(SofaRangeError(
                f"fixlen payload of {n} bytes exceeds FIXLEN_MAX={FIXLEN_MAX}"))
            return
        self._write_fixlen(field_id, bytes(data), _ST_BLOB)

    def _write_fixlen(self, field_id: SupportsIndex, data: bytes,
                      subtype: int) -> None:
        if not self._begin():
            return
        try:
            n = len(data)
            # §6.2: FIXLEN_MAX is a format-wide ceiling, so a longer payload
            # could only be framed by a fixlen word (§4.6, length range
            # 0..2,147,483,647) that every conformant decoder rejects — this
            # port's own included. Refusing it here is §6.3's InvalidArgument;
            # emitting it would hand the caller an unreadable message while
            # reporting success (the encode-side form of §5.1). The check
            # precedes the field header, so a refused field leaves nothing
            # behind on the wire.
            if n > FIXLEN_MAX:
                raise SofaRangeError(
                    f"fixlen payload of {n} bytes exceeds FIXLEN_MAX={FIXLEN_MAX}")
            self._header(field_id, _WT_FIXLEN)
            self._emit_varint((n << 3) | subtype)
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
            self._array_header(field_id, _WT_ARRAY_UNSIGNED, len(seq))
            # Hot path: the varint codec is inlined over the whole array so each
            # element costs a loop iteration rather than a Python call, and the
            # cursor lives in a local until the loop ends or has to drain. The
            # view and capacity are re-read after every drain — a sink may
            # install a different buffer (see _put).
            buf = self._fixed_ba
            limit = self._cap - _VARINT_MAX   # last cursor an inline varint fits at
            cursor = self._cursor
            try:
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "unsigned array value")
                    if v < 0 or v > UNSIGNED_MAX:
                        raise SofaRangeError(f"unsigned array value {v} out of range")
                    if cursor > limit:
                        # Too close to the end for the inline path: _put splits
                        # the element across the drain and may land in a fresh
                        # buffer, so everything it touches is re-read after it.
                        self._cursor = cursor
                        try:
                            self._put(encode_varint(v))
                        finally:
                            # Whatever _put reached is authoritative, including
                            # when it failed partway: the outer finally must not
                            # rewind the cursor over bytes it already wrote.
                            cursor = self._cursor
                        buf = self._fixed_ba
                        limit = self._cap - _VARINT_MAX
                        continue
                    # Varints are emitted by width, not one group at a time.
                    # One and two bytes get a straight line each -- between them
                    # that is nearly every id, length and count on the wire --
                    # and anything longer runs a loop that does TWO 7-bit groups
                    # a turn: five steps where two single-group turns cost eight.
                    # ``>= 0x4000`` means at least three groups remain, so both
                    # bytes that loop writes are certain to need a continuation
                    # bit. The array elements this loop encodes are the case that
                    # gets there -- 8.4 groups per element on the u64 workload.
                    if v < 0x80:
                        buf[cursor] = v
                        cursor += 1
                    elif v < 0x4000:
                        buf[cursor] = (v & 0x7F) | 0x80
                        buf[cursor + 1] = v >> 7
                        cursor += 2
                    else:
                        while v >= 0x4000:
                            buf[cursor] = (v & 0x7F) | 0x80
                            buf[cursor + 1] = ((v >> 7) & 0x7F) | 0x80
                            cursor += 2
                            v >>= 14
                        if v >= 0x80:
                            buf[cursor] = (v & 0x7F) | 0x80
                            cursor += 1
                            v >>= 7
                        buf[cursor] = v
                        cursor += 1
            finally:
                # Also on the way out of a rejected element: what was written
                # stays written, exactly as it did when the buffer was growable.
                self._cursor = cursor
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
            self._array_header(field_id, _WT_ARRAY_SIGNED, len(seq))
            buf = self._fixed_ba   # see write_unsigned_array: codec inlined
            limit = self._cap - _VARINT_MAX
            cursor = self._cursor
            try:
                for v in seq:
                    if not isinstance(v, int):
                        v = _as_int(v, "signed array value")
                    if v < SIGNED_MIN or v > SIGNED_MAX:
                        raise SofaRangeError(f"signed array value {v} out of range")
                    u = (v << 1) ^ (v >> 63)
                    if cursor > limit:
                        self._cursor = cursor
                        try:
                            self._put(encode_varint(u))
                        finally:
                            cursor = self._cursor   # see write_unsigned_array
                        buf = self._fixed_ba
                        limit = self._cap - _VARINT_MAX
                        continue
                    if u < 0x80:
                        buf[cursor] = u
                        cursor += 1
                    elif u < 0x4000:
                        buf[cursor] = (u & 0x7F) | 0x80
                        buf[cursor + 1] = u >> 7
                        cursor += 2
                    else:
                        while u >= 0x4000:
                            buf[cursor] = (u & 0x7F) | 0x80
                            buf[cursor + 1] = ((u >> 7) & 0x7F) | 0x80
                            cursor += 2
                            u >>= 14
                        if u >= 0x80:
                            buf[cursor] = (u & 0x7F) | 0x80
                            cursor += 1
                            u >>= 7
                        buf[cursor] = u
                        cursor += 1
            finally:
                self._cursor = cursor
        except SofaError as exc:
            self._fail(exc)

    def write_float32_array(self, field_id: SupportsIndex, values: Iterable[float]) -> None:
        """Write an array of 32-bit floats as a packed little-endian fixlen array.

        The element count must be ``0..ARRAY_MAX``, else :class:`SofaRangeError`.
        A zero-count array emits ``[header][count=0][fixlen_word]`` — the
        ``fixlen_word`` is always present (so empty fp32/fp64 arrays stay
        distinguishable) but there is no payload (§4.8).
        """
        self._write_float_array(field_id, values, _ST_FP32, _core.pack_f32_array, 4)

    def write_float64_array(self, field_id: SupportsIndex, values: Iterable[float]) -> None:
        """Write an array of 64-bit floats as a packed little-endian fixlen array.

        The element count must be ``0..ARRAY_MAX``, else :class:`SofaRangeError`.
        A zero-count array emits ``[header][count=0][fixlen_word]`` — the
        ``fixlen_word`` is always present (so empty fp32/fp64 arrays stay
        distinguishable) but there is no payload (§4.8).
        """
        self._write_float_array(field_id, values, _ST_FP64, _core.pack_f64_array, 8)

    def _write_float_array(
        self,
        field_id: SupportsIndex,
        values: Iterable[float],
        subtype: int,
        pack_array: Callable[[list[float]], bytes],
        elem_size: int,
    ) -> None:
        if not self._begin():
            return
        try:
            seq = [float(v) for v in values]
            self._array_header(field_id, _WT_ARRAY_FIXLEN, len(seq))
            # §4.8: a fixlen array ALWAYS carries its fixlen_word (the shared
            # element subtype/width), even when empty, so an empty fp32 and fp64
            # array stay distinguishable on the wire. The payload loop then runs
            # zero times for a zero-count array.
            self._emit_varint((elem_size << 3) | subtype)
            self._put(pack_array(seq))  # one struct.pack for the whole array
        except SofaError as exc:
            self._fail(exc)

    def _array_header(self, field_id: SupportsIndex, wtype: int, count: int) -> None:
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

        Raises :class:`SofaRangeError` if no sequence is currently open.
        """
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaRangeError("sequence_end without matching begin")
            if self._pending:
                # The innermost open sequence is the last held-back one (the
                # pending run is a suffix), so dropping it is a plain pop: no
                # header and no end marker ever reach the wire.
                self._pending.pop()
                self._depth -= 1
                return
            self._emit_varint(_WT_SEQUENCE_END)
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

        Raises :class:`SofaRangeError` if no sequence is currently open.
        """
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaRangeError("sequence_end without matching begin")
            if self._pending:
                self._commit_pending()
            self._emit_varint(_WT_SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)
