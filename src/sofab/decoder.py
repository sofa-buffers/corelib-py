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

**Suspend and resume (CORELIB_PLAN §5.2).** A reader that runs dry mid-field —
a socket with nothing buffered yet — makes the call raise
:class:`SofaIncompleteError`. That is a first-class outcome, not an error, so
the call consumes nothing: the cursor goes back to where it started, the bytes
already read stay buffered, and re-issuing the same call once more bytes have
arrived parses the field from its first byte. See the "resume transactions"
section in :class:`Decoder`.

Typical use::

    dec = Decoder(reader)
    while (field := dec.next()) is not None:
        if field.id == 1:
            value = dec.unsigned()
        else:
            dec.skip()
"""

from __future__ import annotations

import sys
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
    SofaIncompleteError,
    SofaLimitError,
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

    def __init__(
        self,
        reader: _Reader,
        *,
        chunk_size: int = 65536,
        max_array_count: int | None = None,
        max_string_len: int | None = None,
        max_blob_len: int | None = None,
    ) -> None:
        """Wrap ``reader`` (any object with ``read(n) -> bytes``).

        ``chunk_size`` is how many bytes each refill pulls from the reader.

        ``max_array_count`` / ``max_string_len`` / ``max_blob_len`` are optional
        **receiver-side** decode limits: a field whose wire-declared array count
        or fixlen string/blob length exceeds the configured cap is rejected with
        :class:`SofaLimitError` at header-decode time — *before* any allocation
        or payload buffering, so a hostile claim fails even if the payload never
        arrives. ``None`` (the default) means "no limit" — today's behaviour. The
        limits are policy, not schema: the generator bakes the configured values
        into generated code and passes them here; this runtime only enforces them
        and never invents a default cap of its own.
        """
        self._read = reader.read
        self._chunk = chunk_size
        self._max_array_count = max_array_count
        self._max_string_len = max_string_len
        self._max_blob_len = max_blob_len
        self._buf = b""
        self._pos = 0
        self._depth = 0
        self._cur: Field | None = None
        # pending unconsumed value: tuple keyed by the _* constants above
        self._pending: tuple[Any, ...] | None = None
        # Resume transaction (§5.2): the buffer offset the call in flight
        # started at, and -1 or the floor a multi-field walk pins. See _suspend.
        self._keep = 0
        self._floor = -1

    # --- resume transactions (CORELIB_PLAN §5.2) ----------------------------
    #
    # §5.2 requires the decoder to "suspend and resume at **any** byte boundary
    # without losing state": running out of bytes mid-construct is INCOMPLETE,
    # a first-class outcome the caller answers by supplying more bytes — not an
    # error that may consume anything. For this pull decoder that means every
    # public call is a transaction: it either consumes the whole construct or
    # consumes nothing at all. On the suspension path the cursor goes back to
    # where the call began and the bytes already parsed stay buffered, so
    # re-issuing the same call once more bytes have arrived re-parses the
    # construct from its **first** byte. Without that, a resumed call would
    # restart mid-construct and silently decode fabricated fields — the one
    # outcome §5.2 forbids (folding a merely-split message into INVALID, or
    # worse, into a wrong COMPLETE).
    #
    # Two rules keep the whole mechanism down to one integer — ``_keep``, the
    # cursor position the call in flight started at — so the guarantee costs the
    # hot path a single store per call rather than a state snapshot:
    #
    # * **the cursor is the only thing that moves while a call can still
    #   suspend.** ``_pending`` is cleared, ``_depth`` stepped and ``_cur``
    #   published only after the construct's last byte is in hand, so there is
    #   nothing else to undo. Every ``_take_*`` below therefore consumes first
    #   and commits after — the one ordering rule this file depends on.
    # * **every call is one field.** ``_keep`` is re-armed by ``next()`` and by
    #   each typed read, so it is always meaningful and never has to be cleared:
    #   it simply names the start of the most recent call. It doubles as the
    #   floor :meth:`_need` may compact to, which is what keeps a suspended
    #   construct's bytes in the buffer for the retry.
    #
    # ``skip()`` over a whole *sequence* is the one call that spans many fields.
    # It pins ``_floor`` at its first byte so the refill path cannot drop the
    # sequence behind the walk, and restores ``_pos``/``_depth``/``_cur``/
    # ``_pending`` itself if the walk suspends — a re-issued ``skip()`` then
    # replays the sequence from its start.
    #
    # INVALID is deliberately *not* rewound: it is terminal (§5.2), so no
    # continuation of bytes can make the stream valid again and there is nothing
    # to resume.

    def _suspend(self, msg: str) -> SofaIncompleteError:
        """Rewind to the current call's first byte and build its ``INCOMPLETE``.

        Called at every truncation site, so the rewind happens whether the raise
        is nested inside a field walk or not. ``_keep`` is maintained by
        :meth:`_need` across compaction, so it names the call's start byte in the
        *current* buffer even if the buffer was rebased while the call ran.
        """
        self._pos = self._keep
        return SofaIncompleteError(msg)

    # --- low-level byte sourcing --------------------------------------------
    #
    # The buffer is never sliced per byte: ``_pos`` advances over ``_buf`` and
    # the consumed prefix is dropped only when a refill is actually needed.

    def _need(self, n: int) -> bool:
        """Ensure at least ``n`` bytes are available at ``_pos``, pulling more
        from the reader (and compacting the consumed prefix) as required.
        Returns ``False`` if the stream ends with fewer than ``n`` available.

        Everything the reader hands over is kept: on failure the bytes read so
        far stay in ``_buf``, which is what makes a suspension non-destructive.
        """
        buf = self._buf
        pos = self._pos
        if len(buf) - pos >= n:
            return True
        # Compaction may only drop what can never be read again. With a resume
        # transaction open that floor is the transaction's start, not the
        # cursor: the bytes in between belong to the construct being parsed and
        # a suspension has to be able to replay them (§5.2).
        base = self._keep if self._floor < 0 else self._floor
        if base:
            buf = buf[base:]
            pos -= base
            self._pos = pos
            self._keep -= base
            if self._floor > 0:
                self._floor -= base
        want = pos + n
        if len(buf) < want:
            read = self._read
            chunk = self._chunk
            short = want - len(buf)
            data = read(chunk if chunk > short else short)
            if not data:
                self._buf = buf
                return False
            buf = buf + data if buf else data
            if len(buf) < want:
                # More than one read needed: accumulate in a bytearray from here,
                # since repeated ``bytes + bytes`` would make a chunk-fed large
                # payload quadratic. (One read is the common case and stays a
                # plain concatenation — often not even that, on the first fill.)
                acc = bytearray(buf)
                while len(acc) < want:
                    short = want - len(acc)
                    data = read(chunk if chunk > short else short)
                    if not data:
                        self._buf = bytes(acc)
                        return False
                    acc += data
                buf = bytes(acc)
        self._buf = buf
        return True

    def _varint(self) -> int:
        """Decode one base-128 varint by advancing the cursor over the buffer,
        refilling only if it runs off the end mid-value."""
        buf = self._buf
        pos = self._pos
        if pos >= len(buf):
            if not self._need(1):
                raise self._suspend("truncated varint")
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
                    raise self._suspend("truncated varint")
                buf = self._buf
                pos = self._pos
                n = len(buf)
            b = buf[pos]
            pos += 1
            # Reject an overlong (>64-bit) varint before OR-ing: if this byte's
            # 7 payload bits would spill past bit 63 they are unrepresentable in
            # u64 and must be INVALID, not silently masked away on return
            # (§4.1/§6.3, issue #43). ``room`` is the bits left below 64; only
            # when fewer than 7 remain can a payload bit overflow.
            room = 64 - shift
            if room < 7 and (b & 0x7F) >> room:
                raise SofaDecodeError("overlong varint")
            result |= (b & 0x7F) << shift
            if b < 0x80:
                self._pos = pos
                return result & MASK64
            shift += 7
            if shift >= 64:
                raise SofaDecodeError("overlong varint")

    def _read_exact(self, n: int) -> bytes:
        """Return the next ``n`` bytes. Fast path is a single buffer slice; the
        slow path accumulates across refills for a chunk-fed reader.

        The accumulation happens inside ``_buf`` rather than in a local, so a
        payload that stops halfway is still buffered when the truncation is
        reported and the next attempt continues from it (§5.2)."""
        buf = self._buf
        pos = self._pos
        end = pos + n
        if end <= len(buf):
            self._pos = end
            return buf[pos:end]
        if not self._need(n):
            raise self._suspend("truncated payload")
        buf = self._buf
        pos = self._pos
        end = pos + n
        self._pos = end
        return buf[pos:end]

    def _elem_bound_error(
        self,
        out: list[int],
        lo: int | None,
        hi: int | None,
        zigzag: bool,
    ) -> Exception:
        """The exception a truncated array should actually raise.

        :class:`SofaDecodeError` when an element already decoded falls outside
        the field's declared width, otherwise the truncation that was about to
        be reported. §5.2 makes INVALID dominate INCOMPLETE, and an element
        outside its declared width is INVALID by §7.1 — it is established by its
        own bytes, so the array running out behind it cannot downgrade the
        verdict (generator#267, Crucible F-0043).

        Applied HERE, at the truncation, rather than per element: an array that
        completes is decided by the caller's own scan over the returned list,
        which sees exactly the same elements and reaches the same answer. That
        keeps the decode loop above a pure decode — the native engine, where a
        typed compare is free, does check at the element (``_speedups.pyx``);
        both produce the same verdict, which is what the shared vectors pin.
        """
        if hi is not None:
            for v in out:
                x = zigzag_decode(v) if zigzag else v
                if x < lo or x > hi:  # type: ignore[operator]
                    return SofaDecodeError("array element outside declared width")
        return self._suspend("truncated varint")

    def _read_varints(
        self,
        count: int,
        lo: int | None = None,
        hi: int | None = None,
        zigzag: bool = False,
    ) -> list[int]:
        """Decode ``count`` consecutive varints in one tight loop that advances
        the cursor over the buffer — the whole varint codec is inlined here (no
        per-element call) and refills only when it runs off the end.

        ``lo``/``hi`` are the field's declared element width, when it declares
        one; a truncation is then reported through :meth:`_elem_bound_error`,
        which turns it into INVALID if an element already in hand breaches it.

        The result is built incrementally with ``append`` rather than pre-sized
        to ``count``: ``count`` comes straight off the wire and is capped only at
        ``ARRAY_MAX`` (INT32_MAX), so pre-sizing (``[0] * count``) would let a
        tiny hostile message claiming ``count = 2**31`` force a ~16 GB list
        allocation before a single element byte is read (amplification DoS,
        issue #31). Growing as elements are actually decoded bounds the
        allocation by the payload really present — a truncated oversize claim
        runs the reader dry and raises :class:`SofaIncompleteError` promptly. The
        float path already reads its payload first (``read_float32_array``); this
        brings the varint path in line."""
        out: list[int] = []
        append = out.append
        buf = self._buf
        pos = self._pos
        n = len(buf)
        i = 0
        while i < count:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise self._elem_bound_error(out, lo, hi, zigzag)
                buf = self._buf
                pos = self._pos
                n = len(buf)
            b = buf[pos]
            pos += 1
            if b < 0x80:  # one-byte element
                append(b)
                i += 1
                continue
            result = b & 0x7F
            shift = 7
            while True:
                if pos >= n:
                    self._pos = pos
                    if not self._need(1):
                        raise self._elem_bound_error(out, lo, hi, zigzag)
                    buf = self._buf
                    pos = self._pos
                    n = len(buf)
                b = buf[pos]
                pos += 1
                # An element value is a varint like any other, so the 64-bit
                # bound of §4.1 applies to it: if this byte's payload bits would
                # land at bit >= 64 they are unrepresentable and the encoding is
                # INVALID — masking them off on return would silently corrupt
                # the value instead (issue #64). Same guard, same wording as
                # ``_varint`` above; this loop inlines the codec for speed and
                # so has to carry it itself. ``room`` is the bits left below 64.
                room = 64 - shift
                if room < 7 and (b & 0x7F) >> room:
                    raise SofaDecodeError("overlong varint")
                result |= (b & 0x7F) << shift
                if b < 0x80:
                    break
                shift += 7
                if shift >= 64:
                    raise SofaDecodeError("overlong varint")
            append(result & MASK64)
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

        If the bytes run out inside the header (or inside the value being
        skipped), :class:`SofaIncompleteError` is raised and the decoder is left
        untouched: call ``next()`` again once more bytes are available and the
        field is parsed from its first byte (§5.2).
        """
        self._keep = self._pos  # opens this field's resume transaction (§5.2)
        if self._pending is not None:
            self._skip_pending()

        if not self._need(1):
            if self._depth != 0:
                raise self._suspend("truncated: unbalanced sequence")
            return None

        header = self._varint()
        wtype = header & 0x07
        field_id = header >> 3

        # The id is bounded by ID_MAX on every header without exception (§6.2),
        # including a sequence end whose id is otherwise discarded (§4.9): the
        # bound is on the id's value, so this must run before the wire-type
        # dispatch below, not inside the branches that use the id.
        if field_id > ID_MAX:
            raise SofaDecodeError(f"id {field_id} out of range")

        if wtype == WireType.SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            self._cur = Field(0, WireType.SEQUENCE_END)
            return self._cur

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
            # A wrong-width fp field is malformed regardless of what bytes
            # follow, so this INVALID verdict must be reached at header time —
            # before any payload read — so it takes precedence over the
            # INCOMPLETE a truncated payload would otherwise raise (§7). Mirrors
            # the eager element-width check on the fixlen-array path below. Do
            # not eager-check STRING/BLOB: those are variable-length, so a
            # truncated string/blob is legitimately INCOMPLETE.
            if subtype == FixlenSubtype.FP32 and length != 4:
                raise SofaDecodeError("fp32 fixlen length must be 4")
            if subtype == FixlenSubtype.FP64 and length != 8:
                raise SofaDecodeError("fp64 fixlen length must be 8")
            # Receiver-configured limits (policy, not malformation): reject an
            # oversize string/blob here — before its payload is read or buffered.
            if (
                subtype == FixlenSubtype.STRING
                and self._max_string_len is not None
                and length > self._max_string_len
            ):
                raise SofaLimitError(
                    f"string length {length} exceeds max_string_len {self._max_string_len}"
                )
            if (
                subtype == FixlenSubtype.BLOB
                and self._max_blob_len is not None
                and length > self._max_blob_len
            ):
                raise SofaLimitError(
                    f"blob length {length} exceeds max_blob_len {self._max_blob_len}"
                )
            self._cur = Field(
                field_id, WireType.FIXLEN, size=length, subtype=FixlenSubtype(subtype)
            )
            self._pending = (_FIXLEN, subtype, length)
            return self._cur

        if wtype == WireType.ARRAY_UNSIGNED or wtype == WireType.ARRAY_SIGNED:
            count = self._varint()
            if count < 0 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            if self._max_array_count is not None and count > self._max_array_count:
                raise SofaLimitError(
                    f"array count {count} exceeds max_array_count {self._max_array_count}"
                )
            self._cur = Field(field_id, _WT[wtype], count=count)
            self._pending = (_VARRAY, wtype, count)
            return self._cur

        # wtype == ARRAY_FIXLEN
        count = self._varint()
        if count < 0 or count > ARRAY_MAX:
            raise SofaDecodeError(f"array count {count} out of range")
        if self._max_array_count is not None and count > self._max_array_count:
            raise SofaLimitError(
                f"array count {count} exceeds max_array_count {self._max_array_count}"
            )
        # §4.8: a fixlen array ALWAYS carries its fixlen_word (the shared element
        # subtype/width), even when empty — so read it unconditionally to recover
        # the true subtype. A zero-count array simply has no payload after it.
        elem_header = self._varint()
        elem_size = elem_header >> 3
        subtype = elem_header & 0x07
        if subtype > FixlenSubtype.FP64:
            raise SofaDecodeError(f"invalid fixlen-array subtype {subtype}")
        # §4.8/§5.2: a fixlen array carries fp32 (element size 4) or fp64
        # (element size 8) — any other width is malformed. This INVALID verdict
        # must be reached at header time, before any payload read, so it takes
        # precedence over the INCOMPLETE a truncated payload would raise (§7).
        # Mirrors the eager element-width check on the scalar fixlen path above.
        # subtype is already narrowed to fp32/fp64, so these exact-width checks
        # bound elem_size completely — no separate FIXLEN_MAX check is needed.
        if subtype == FixlenSubtype.FP32 and elem_size != 4:
            raise SofaDecodeError("fp32 fixlen-array element size must be 4")
        if subtype == FixlenSubtype.FP64 and elem_size != 8:
            raise SofaDecodeError("fp64 fixlen-array element size must be 8")
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

    def _farray_nbytes(self, count: int, elem_size: int) -> int:
        """On-wire payload size of a fixlen array (``count * elem_size``).

        ``elem_size`` is unbounded on the wire, so the product can exceed any
        addressable size. A payload that cannot fit a ``Py_ssize_t`` can never be
        satisfied by a real buffer, so surface it as a truncated read rather than
        letting a downstream ``read(n)`` raise a raw ``OverflowError`` (which
        would leak an implementation detail and diverge from the native engine).
        """
        total = count * elem_size
        if total > sys.maxsize:
            raise self._suspend("truncated payload")
        return total

    def _skip_pending(self) -> None:
        pending = self._pending
        assert pending is not None
        kind = pending[0]
        if kind == _SCALAR:
            self._varint()
        elif kind == _FIXLEN:
            self._read_exact(pending[2])
        elif kind == _VARRAY:
            self._skip_varints(pending[2])
        else:  # _FARRAY
            self._read_exact(self._farray_nbytes(pending[2], pending[3]))
        # Cleared only now: had the value run out mid-skip, the field has to
        # stay pending so the retry skips it again from its first byte (§5.2).
        self._pending = None

    def skip(self) -> None:
        """Skip the current field's value, or an entire (nested) sequence if the
        current field is a sequence start.

        Suspends as a unit: if the bytes run out part-way, nothing is consumed
        and the same ``skip()`` can be re-issued when more arrive (§5.2).
        """
        self._keep = self._pos
        if self._cur is not None and self._cur.type == WireType.SEQUENCE_START:
            # Walking a whole sequence spans many fields, so unlike every other
            # call this one moves the field state — and lets ``next()`` re-arm
            # ``_keep`` — before it can suspend. ``_floor`` pins the refill
            # path's compaction at the first byte *inside* the sequence for the
            # duration, and the field state is put back here, so a re-issued
            # ``skip()`` replays the whole sequence (§5.2).
            self._floor = self._pos
            depth, cur, pending = self._depth, self._cur, self._pending
            try:
                target = depth - 1
                while self._depth > target:
                    # Defensive: at EOF with an open sequence, next() itself
                    # raises "truncated: unbalanced sequence", so it never
                    # returns None here.
                    if self.next() is None:  # pragma: no cover
                        raise self._suspend("truncated sequence")
            except SofaIncompleteError:
                self._pos = self._keep = self._floor
                self._depth, self._cur, self._pending = depth, cur, pending
                raise
            finally:
                self._floor = -1
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
        self._keep = self._pos
        value = self._varint()
        self._pending = None  # committed only once the value is in hand (§5.2)
        return value

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
        self._keep = self._pos
        data = self._read_exact(pending[2])
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return data

    def float32(self) -> float:
        """Consume the current fixlen field as a 32-bit IEEE-754 float.

        Raises :class:`SofaStateError` if the field is not an fp32 fixlen, or
        :class:`SofaDecodeError` if its payload is not 4 bytes.
        """
        data = self._take_fixlen(FixlenSubtype.FP32)
        if len(data) != 4:  # pragma: no cover - header validates width eagerly (see next())
            raise SofaDecodeError("fp32 payload must be 4 bytes")
        return _core.unpack_f32(data)

    def float64(self) -> float:
        """Consume the current fixlen field as a 64-bit IEEE-754 float.

        Raises :class:`SofaStateError` if the field is not an fp64 fixlen, or
        :class:`SofaDecodeError` if its payload is not 8 bytes.
        """
        data = self._take_fixlen(FixlenSubtype.FP64)
        if len(data) != 8:  # pragma: no cover - header validates width eagerly (see next())
            raise SofaDecodeError("fp64 payload must be 8 bytes")
        return _core.unpack_f64(data)

    def fixlen_len(self) -> int:
        """Return the current fixlen field's payload byte length without consuming it.

        The length is read straight from the field's length header, so this is a
        pure peek: it does not advance the decoder and a following
        :meth:`string`/:meth:`bytes`/:meth:`float32`/:meth:`float64` still reads
        the same field. It lets a caller bound a string or blob against its schema
        ``maxlen`` using the exact wire byte length the decoder already parsed —
        checked before allocation, and without re-encoding a decoded ``str`` just
        to measure it (the string field is UTF-8 on the wire, so the payload byte
        length is the length ``maxlen`` bounds).

        Raises :class:`SofaStateError` if the current field is not a fixlen value.
        """
        pending = self._pending
        if pending is None or pending[0] != _FIXLEN:
            raise SofaStateError("current field is not a fixlen value")
        return int(pending[2])

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
        """Validate the pending array and return its count.

        The pending value is *not* cleared here — the caller clears it once the
        payload has actually been decoded, so a suspension leaves the array
        re-readable from its first element (§5.2).
        """
        pending = self._pending
        if pending is None or pending[0] != _VARRAY or pending[1] != wtype:
            raise SofaStateError("current field is not a matching varint array")
        return int(pending[2])

    def read_unsigned_array(self, elem_max: int | None = None) -> list[int]:
        """Consume the current field as a list of unsigned integers.

        Pass the schema's declared element width as ``elem_max`` (``255`` for a
        ``u8`` array, and so on) so an element outside it keeps the message
        INVALID even when the array behind it is truncated — a caller scanning
        the returned list can only decide an array that arrives, and §5.2 makes
        INVALID dominate the INCOMPLETE that one which does not would otherwise
        report (§7.1, generator#267). Omit for ``u64``, whose range is the value
        domain, and for an unbounded consumer.

        Raises :class:`SofaStateError` if the field is not an unsigned array.
        """
        count = self._take_varray(WireType.ARRAY_UNSIGNED)
        self._keep = self._pos
        out = self._read_varints(count, 0, elem_max)
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return out

    def read_signed_array(
        self,
        elem_min: int | None = None,
        elem_max: int | None = None,
    ) -> list[int]:
        """Consume the current field as a list of ZigZag-decoded signed integers.

        ``elem_min``/``elem_max`` bound each element to its declared width — see
        :meth:`read_unsigned_array`.

        Raises :class:`SofaStateError` if the field is not a signed array.
        """
        count = self._take_varray(WireType.ARRAY_SIGNED)
        self._keep = self._pos
        # ZigZag inlined rather than calling zigzag_decode per element: the
        # transform is two operations, the call around it was the expensive part.
        out = [
            (v >> 1) ^ -(v & 1)
            for v in self._read_varints(count, elem_min, elem_max, True)
        ]
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return out

    def _take_farray(self, subtype: FixlenSubtype) -> tuple[int, int]:
        pending = self._pending
        if pending is None or pending[0] != _FARRAY:
            raise SofaStateError("current field is not a fixlen array")
        # §4.8: a fixlen array always carries its fixlen_word, so the subtype is
        # known even for a zero-count array — check it like any other read.
        if pending[1] != subtype:
            raise SofaStateError("fixlen-array subtype does not match the requested read")
        # Like _take_varray, the pending value is cleared by the caller only
        # after the payload has been read (§5.2).
        return int(pending[2]), int(pending[3])  # count, elem_size

    def read_float32_array(self) -> list[float]:
        """Consume the current field as a list of 32-bit IEEE-754 floats.

        Raises :class:`SofaStateError` if the field is not an fp32 array.
        """
        count, elem_size = self._take_farray(FixlenSubtype.FP32)
        self._keep = self._pos
        data = self._read_exact(self._farray_nbytes(count, elem_size))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        # The fixlen_word must declare a 4-byte element width for fp32; reject a
        # mismatch as malformed instead of letting struct.unpack raise a raw
        # struct.error (which would leak an implementation detail and diverge
        # from the native engine, which raises SofaDecodeError here).
        if len(data) != count * 4:
            raise SofaDecodeError("fixlen-array element width does not match its subtype")
        return _core.unpack_f32_array(data, count)

    def read_float64_array(self) -> list[float]:
        """Consume the current field as a list of 64-bit IEEE-754 floats.

        Raises :class:`SofaStateError` if the field is not an fp64 array.
        """
        count, elem_size = self._take_farray(FixlenSubtype.FP64)
        self._keep = self._pos
        data = self._read_exact(self._farray_nbytes(count, elem_size))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        # fp64 elements are 8 bytes wide; see read_float32_array.
        if len(data) != count * 8:
            raise SofaDecodeError("fixlen-array element width does not match its subtype")
        return _core.unpack_f64_array(data, count)
