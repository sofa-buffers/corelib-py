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
    SofaRangeError,
    WireType,
)

if TYPE_CHECKING:
    from .visitor import Visitor

# Pending-value kinds the consume methods dispatch on.
_SCALAR = 0
_FIXLEN = 1
_VARRAY = 2
_FARRAY = 3
# A pending value a receiver-side cap has rejected (§6.2.1). The real pending
# tuple is parked inside it — ``(_LIMIT, message, pending)`` — so the rejection
# can still be waived by :meth:`Decoder.schema_bounded` and so a peek can read
# through it. Every consume path re-raises it; see ``_mismatch``.
_LIMIT = 4

# Wire-type members indexed by their integer value, so the per-field hot path
# can recover the enum member by index (``_WT[wtype]``) instead of paying the
# full ``WireType(wtype)`` coercion (IntEnum.__call__/__new__) on every field.
_WT = tuple(WireType)

# The lowest value a signed element can carry, used as the open lower side when
# a field declares only the upper half of its element width (``_read_varints``).
_I64_MIN = -(1 << 63)


class _Reader(Protocol):
    """Read protocol: an object with ``read(n) -> bytes``."""

    def read(self, n: int) -> bytes: ...


class Decoder:
    """Pull-decodes a SofaBuffers stream field by field.

    Call :meth:`next` to advance to each field, then one of the typed read
    methods (:meth:`unsigned`, :meth:`string`, :meth:`read_float64_array`, …)
    to consume its value, or :meth:`skip` to discard it. Alternatively hand a
    :class:`sofab.Visitor` to :meth:`drive` for callback-style decoding.

    **A typed read whose type contradicts the field on the wire is not an
    error** (MESSAGE_SPEC §7.3, CORELIB_PLAN §6.3). It returns ``None`` and
    consumes nothing, so the field is skipped by the following :meth:`next` just
    like a field with an unknown id, and the decode stays COMPLETE — reading a
    STRING field with :meth:`float64` yields ``None``, not an exception.
    Generated code tests :attr:`Field.type` / :attr:`Field.subtype` before
    reading and so never sees this; hand-written callers should either do the
    same or treat ``None`` as "not my field". A read issued when there is **no**
    pending value at all — before the first :meth:`next`, twice for one field, or
    on a sequence start/end — is a caller mistake and raises
    :class:`sofab.SofaRangeError`.
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
        :class:`SofaLimitError`. The verdict is reached at the count/length
        header — *before* any allocation or payload buffering, so a hostile claim
        fails even if the payload never arrives — and is raised by the call that
        would consume the field (a typed read, :meth:`skip`, or the auto-skip in
        the following :meth:`next`), which is what leaves room for
        :meth:`schema_bounded` in between. ``None`` (the default) means "no
        limit". The limits are policy, not schema: the generator bakes the
        configured values into generated code and passes them here; this runtime
        only enforces them and never invents a default cap of its own, and it
        never applies one to a field the caller declares schema-bounded (§6.2.1).
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

    def _skip_exact(self, n: int) -> None:
        """Consume the next ``n`` bytes without materialising them (§5.2: a skip
        *consumes and discards*; it must not pay for a copy of what it throws
        away). Same sourcing and same suspension as :meth:`_read_exact` — only
        the result object is missing.

        The bytes are still buffered on the slow path, because they are not free
        to drop: a suspension has to be able to replay the skipped construct from
        its first byte, and ``skip()`` over a sequence replays a whole nested
        walk (see ``_floor``). Retention is therefore the resume contract's,
        while the copy was pure waste — this drops the copy and keeps the
        contract, so nothing observable changes."""
        buf = self._buf
        pos = self._pos
        end = pos + n
        if end <= len(buf):
            self._pos = end
            return
        if not self._need(n):
            raise self._suspend("truncated payload")
        self._pos += n

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

        ``lo``/``hi`` are the field's declared element width, when the field
        declares one, and either side may be omitted. An element outside it is
        INVALID by §7.1, and the check happens AT the element — as soon as its
        own bytes have been decoded, before the loop looks at anything else.
        That is what makes the verdict a property of the element rather than of
        the message around it: §7.1 forbids enforcement from being "an emergent
        property of the memory model", so an array that completes and one that is
        truncated behind the same bad element have to be rejected alike (issue
        #67). It also gives §5.2's precedence for free — INVALID is raised before
        the truncation behind it is ever reached (generator#267, Crucible F-0043)
        — and matches the native engine element for element (``_speedups.pyx``).

        The bound costs the one-byte fast path nothing in the usual case: a
        one-byte element is 0..127 raw, -64..63 ZigZagged, so when the declared
        width already spans that range (every width from u8/i8 up) the test is
        hoisted out of the loop entirely and only multi-byte elements are
        compared.

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
        # Normalise the declared width once: an omitted side is the widest value
        # its domain can hold, so a one-sided bound binds its own side and leaves
        # the other open instead of faulting on the missing half (issue #67).
        bounded = lo is not None or hi is not None
        blo = _I64_MIN if lo is None else lo
        bhi = MASK64 if hi is None else hi
        # A one-byte element spans 0..127 raw (-64..63 ZigZagged); when the bound
        # covers all of it the fast path can skip the compare altogether.
        check_fast = bounded and (
            (blo > -65 or bhi < 63) if zigzag else (blo > 0 or bhi < 127)
        )
        buf = self._buf
        pos = self._pos
        n = len(buf)
        i = 0
        while i < count:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise self._suspend("truncated varint")
                buf = self._buf
                pos = self._pos
                n = len(buf)
            b = buf[pos]
            pos += 1
            if b < 0x80:  # one-byte element
                if check_fast:
                    x = (b >> 1) ^ -(b & 1) if zigzag else b
                    if x < blo or x > bhi:
                        raise SofaDecodeError("array element outside declared width")
                append(b)
                i += 1
                continue
            result = b & 0x7F
            shift = 7
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
            result &= MASK64
            if bounded:
                x = (result >> 1) ^ -(result & 1) if zigzag else result
                if x < blo or x > bhi:
                    raise SofaDecodeError("array element outside declared width")
            append(result)
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
            self._cur = Field(
                field_id, WireType.FIXLEN, size=length, subtype=FixlenSubtype(subtype)
            )
            pending: tuple[Any, ...] = (_FIXLEN, subtype, length)
            # Receiver-configured caps (policy, not malformation): the verdict on
            # an oversize string/blob is reached here, on the length word alone —
            # before its payload is read or buffered, which is where §6.2.1 wants
            # it — and parked on the pending value rather than raised. Nothing is
            # read or allocated until the field is consumed, and every consume
            # path raises it (``_mismatch``), so deferring the raise by one
            # call costs the protection nothing; what it buys is the window
            # §6.2.1 requires, in which the caller can declare the field
            # schema-bounded and take the cap off it (:meth:`schema_bounded`).
            if subtype == FixlenSubtype.STRING:
                cap = self._max_string_len
                if cap is not None and length > cap:
                    pending = (
                        _LIMIT,
                        f"string length {length} exceeds max_string_len {cap}",
                        pending,
                    )
            elif subtype == FixlenSubtype.BLOB:
                cap = self._max_blob_len
                if cap is not None and length > cap:
                    pending = (
                        _LIMIT,
                        f"blob length {length} exceeds max_blob_len {cap}",
                        pending,
                    )
            self._pending = pending
            return self._cur

        if wtype == WireType.ARRAY_UNSIGNED or wtype == WireType.ARRAY_SIGNED:
            count = self._varint()
            if count < 0 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            self._cur = Field(field_id, _WT[wtype], count=count)
            pending = (_VARRAY, wtype, count)
            # Parked, not raised — see the fixlen branch above (§6.2.1).
            cap = self._max_array_count
            if cap is not None and count > cap:
                pending = (_LIMIT, f"array count {count} exceeds max_array_count {cap}", pending)
            self._pending = pending
            return self._cur

        # wtype == ARRAY_FIXLEN
        count = self._varint()
        if count < 0 or count > ARRAY_MAX:
            raise SofaDecodeError(f"array count {count} out of range")
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
        pending = (_FARRAY, subtype, count, elem_size)
        # Parked, not raised — see the fixlen branch above (§6.2.1).
        cap = self._max_array_count
        if cap is not None and count > cap:
            pending = (_LIMIT, f"array count {count} exceeds max_array_count {cap}", pending)
        self._pending = pending
        return self._cur

    def schema_bounded(self) -> None:
        """Declare that the **schema** bounds the size of the field :meth:`next`
        most recently returned — a ``count:`` on an array, a ``maxlen:`` on a
        string or blob — so the receiver-side caps (``max_array_count`` /
        ``max_string_len`` / ``max_blob_len``) are not applied to it.

        CORELIB_PLAN §6.2.1 requires exactly that: a cap is *capacity* the
        deployment is willing to commit where the **sender** picks the size
        freely, and it "MUST NOT be applied to a field the schema already
        bounds". There the schema bound governs, and an over-bound value is
        `INVALID` (:class:`SofaDecodeError`, MESSAGE_SPEC §7.1) rather than the
        cap's :class:`SofaLimitError`, which §6.3 says is "never raised for a
        field the schema bounds".

        Only the schema knows, so only the caller can answer — generated code
        calls this on exactly the fields whose declaration bounds them, right
        before the typed read. Declaring is therefore a **promise to enforce**:
        with the cap off, nothing else stands between an untrusted length word
        and the allocation it implies, so the caller must reject a count/length
        past its declared bound itself (:meth:`fixlen_len` gives the wire byte
        length for that, without consuming the field).

        The declaration covers the current field only — the next :meth:`next`
        starts an undeclared, and therefore capped, field again — and it is a
        no-op on a field no cap has rejected, so it is safe to call
        unconditionally. A :class:`sofab.Visitor` driven by :meth:`drive` can
        call it from ``on_field``, which is reached before the typed read.
        """
        pending = self._pending
        if pending is not None and pending[0] == _LIMIT:
            self._pending = pending[2]

    @staticmethod
    def _mismatch(pending: tuple[Any, ...] | None) -> None:
        """The answer a typed read owes a pending value whose wire tag
        contradicts it: ``None`` — see "typed reads and §7.3" below.

        Two conditions reach here that are not that. No pending value **at all**
        is a caller mistake, the §6.3 ``InvalidArgument`` outcome. And a parked
        receiver-cap rejection (§6.2.1) stands in the way of the skip as much as
        of the read — skipping the field still buffers its payload (see
        :meth:`_skip_pending`) — and is a terminal rejection of the *message*,
        not an answer about one read. Both raise.

        Reached only from paths that have already found the pending kind wrong,
        so the ordinary read costs nothing for it.
        """
        if pending is None:
            raise SofaRangeError("no value pending for the current field")
        if pending[0] == _LIMIT:
            raise SofaLimitError(pending[1])
        return None

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
            self._skip_exact(pending[2])
        elif kind == _VARRAY:
            self._skip_varints(pending[2])
        elif kind == _FARRAY:
            self._skip_exact(self._farray_nbytes(pending[2], pending[3]))
        else:  # _LIMIT — a skip still buffers the payload, so the cap binds it
            raise SofaLimitError(pending[1])
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
        # Every read below is dispatched on the field's *own* wire type, so its
        # tag matches by construction and none of them can return the §7.3
        # ``None`` — which is what the ignores below say to mypy. There is no
        # §7.3 mismatch to answer here at all: a visitor declares nothing about a
        # field's type, it simply declines the fields it does not want.
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
                visitor.on_unsigned(f.id, self.unsigned())  # type: ignore[arg-type]
            elif t == WireType.SIGNED:
                visitor.on_signed(f.id, self.signed())  # type: ignore[arg-type]
            elif t == WireType.FIXLEN:
                st = f.subtype
                if st == FixlenSubtype.FP32:
                    visitor.on_float32(f.id, self.float32())  # type: ignore[arg-type]
                elif st == FixlenSubtype.FP64:
                    visitor.on_float64(f.id, self.float64())  # type: ignore[arg-type]
                elif st == FixlenSubtype.STRING:
                    visitor.on_string(f.id, self.string())  # type: ignore[arg-type]
                else:
                    visitor.on_bytes(f.id, self.bytes())  # type: ignore[arg-type]
            elif t == WireType.ARRAY_UNSIGNED:
                visitor.on_unsigned_array(f.id, self.read_unsigned_array())  # type: ignore[arg-type]
            elif t == WireType.ARRAY_SIGNED:
                visitor.on_signed_array(f.id, self.read_signed_array())  # type: ignore[arg-type]
            else:  # ARRAY_FIXLEN
                if f.subtype == FixlenSubtype.FP32:
                    visitor.on_float32_array(f.id, self.read_float32_array())  # type: ignore[arg-type]
                else:
                    visitor.on_float64_array(f.id, self.read_float64_array())  # type: ignore[arg-type]

    # --- typed reads and §7.3 -----------------------------------------------
    #
    # Every typed read below compares the whole tag of the pending value — the
    # wire type plus, for the fixlen kinds, the subtype, since fp32/fp64/string/
    # blob share WireType.FIXLEN — against the type the read declares, and
    # answers a contradiction the way MESSAGE_SPEC §7.3 requires: **not** as an
    # error. The field is left pending and unconsumed, so the next `next()` (or
    # an explicit `skip()`) discards it exactly like a field with an unknown id,
    # the caller's destination is never written, and a decode that meets nothing
    # else stays COMPLETE. The read reports that by returning None — the one
    # answer that cannot be mistaken for a decoded value; returning a zero would
    # be writing the destination, which is what §7.3 forbids. Because the value
    # is still pending, a caller may immediately re-read the same field with the
    # type the wire actually carries.
    #
    # Having **no** pending value at all is the other half, and is a genuine
    # caller mistake — a read before next(), a second read of one field, or a
    # read on a sequence start/end. CORELIB_PLAN §6.3 has exactly one code for
    # that, InvalidArgument, so it raises SofaRangeError.

    def _take_scalar(self, wtype: WireType) -> int | None:
        pending = self._pending
        if pending is None or pending[0] != _SCALAR or pending[1] != wtype:
            return self._mismatch(pending)  # §7.3
        self._keep = self._pos
        value = self._varint()
        self._pending = None  # committed only once the value is in hand (§5.2)
        return value

    def unsigned(self) -> int | None:
        """Consume the current field as an unsigned integer.

        Returns ``None`` if the field is not unsigned (§7.3), leaving it for the
        skip; raises :class:`SofaRangeError` if no value is pending at all.
        """
        return self._take_scalar(WireType.UNSIGNED)

    def signed(self) -> int | None:
        """Consume the current field as a ZigZag-decoded signed integer.

        Returns ``None`` if the field is not signed (§7.3), leaving it for the
        skip; raises :class:`SofaRangeError` if no value is pending at all.
        """
        raw = self._take_scalar(WireType.SIGNED)
        return None if raw is None else zigzag_decode(raw)

    def bool(self) -> bool | None:
        """Consume the current unsigned field as a boolean (non-zero is true).

        Returns ``None`` if the field is not unsigned (§7.3), leaving it for the
        skip; raises :class:`SofaRangeError` if no value is pending at all. Test
        it with ``is True`` / ``is not None`` rather than truthiness, which cannot
        tell ``None`` apart from ``False``.
        """
        raw = self._take_scalar(WireType.UNSIGNED)
        return None if raw is None else raw != 0

    def _take_fixlen(self, subtype: FixlenSubtype) -> bytes | None:
        pending = self._pending
        if pending is None or pending[0] != _FIXLEN or pending[1] != subtype:
            return self._mismatch(pending)  # §7.3
        self._keep = self._pos
        data = self._read_exact(pending[2])
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return data

    def float32(self) -> float | None:
        """Consume the current fixlen field as a 32-bit IEEE-754 float.

        Returns ``None`` if the field is not an fp32 fixlen (§7.3), leaving it
        for the skip; raises :class:`SofaRangeError` if no value is pending at
        all. The payload is necessarily 4 bytes: §4.6's width rule is settled at the
        length header by :meth:`next`, which is where §5.2 wants that INVALID
        verdict reached — before any payload read, so it outranks a truncation
        behind it. Re-deciding the width here could only restate a check no
        input can reach (the native engine does not carry one either).
        """
        data = self._take_fixlen(FixlenSubtype.FP32)
        return None if data is None else _core.unpack_f32(data)

    def float64(self) -> float | None:
        """Consume the current fixlen field as a 64-bit IEEE-754 float.

        Returns ``None`` if the field is not an fp64 fixlen (§7.3), leaving it
        for the skip; raises :class:`SofaRangeError` if no value is pending at
        all. The payload is necessarily 8 bytes (see :meth:`float32`).
        """
        data = self._take_fixlen(FixlenSubtype.FP64)
        return None if data is None else _core.unpack_f64(data)

    def fixlen_len(self) -> int | None:
        """Return the current fixlen field's payload byte length without consuming it.

        The length is read straight from the field's length header, so this is a
        pure peek: it does not advance the decoder and a following
        :meth:`string`/:meth:`bytes`/:meth:`float32`/:meth:`float64` still reads
        the same field. It lets a caller bound a string or blob against its schema
        ``maxlen`` using the exact wire byte length the decoder already parsed —
        checked before allocation, and without re-encoding a decoded ``str`` just
        to measure it (the string field is UTF-8 on the wire, so the payload byte
        length is the length ``maxlen`` bounds).

        Returns ``None`` if the current field is not a fixlen value at all
        (§7.3) — there is no fixlen length to report, and nothing is consumed
        either way; raises :class:`SofaRangeError` if no value is pending.
        """
        pending = self._pending
        if pending is None:
            raise SofaRangeError("no value pending for the current field")
        if pending[0] != _FIXLEN:
            # A parked receiver-cap rejection (§6.2.1) keeps the real pending
            # value inside it, and this peek reads nothing and allocates
            # nothing — so it answers through the wrapper, and for the same
            # reason it does not re-raise the cap the way a consuming read
            # does. That is deliberate: this is the length generated code
            # measures the SCHEMA bound against, and that bound's INVALID
            # outranks the cap (§7.1), so it must be decidable whether or not
            # :meth:`schema_bounded` has been called yet.
            if pending[0] == _LIMIT and pending[2][0] == _FIXLEN:
                return int(pending[2][2])
            return None  # §7.3: not a fixlen field, so it has no fixlen length
        return int(pending[2])

    def string(self) -> str | None:
        """Consume the current fixlen field as a UTF-8 decoded string.

        Returns ``None`` if the field is not a STRING fixlen (§7.3), leaving it
        for the skip; raises :class:`SofaRangeError` if no value is pending at
        all, or :class:`SofaDecodeError` if the payload is not valid UTF-8.
        """
        raw = self._take_fixlen(FixlenSubtype.STRING)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def bytes(self) -> bytes | None:
        """Consume the current fixlen field as a raw byte blob.

        Returns ``None`` if the field is not a BLOB fixlen (§7.3), leaving it for
        the skip; raises :class:`SofaRangeError` if no value is pending at all.
        """
        return self._take_fixlen(FixlenSubtype.BLOB)

    # --- array reads --------------------------------------------------------

    def _take_varray(self, wtype: WireType) -> int | None:
        """Validate the pending array and return its count.

        The pending value is *not* cleared here — the caller clears it once the
        payload has actually been decoded, so a suspension leaves the array
        re-readable from its first element (§5.2).
        """
        pending = self._pending
        if pending is None or pending[0] != _VARRAY or pending[1] != wtype:
            return self._mismatch(pending)  # §7.3
        return int(pending[2])

    def read_unsigned_array(self, elem_max: int | None = None) -> list[int] | None:
        """Consume the current field as a list of unsigned integers.

        Pass the schema's declared element width as ``elem_max`` (``255`` for a
        ``u8`` array, and so on): an element outside it is INVALID (§7.1) and is
        rejected as its own bytes are decoded, so the verdict is the same whether
        the array completes or is truncated behind that element, and §5.2's
        precedence of INVALID over the INCOMPLETE such a truncation would
        otherwise report follows from the order alone (generator#267, issue #67).
        Omit for ``u64``, whose range is the value domain, and for an unbounded
        consumer.

        Returns ``None`` if the field is not an unsigned array (§7.3), leaving it
        for the skip; raises :class:`SofaRangeError` if no value is pending at
        all.
        """
        count = self._take_varray(WireType.ARRAY_UNSIGNED)
        if count is None:
            return None
        self._keep = self._pos
        # No lower bound: an unsigned element decodes to 0..2**64-1 by
        # construction, so passing 0 would only cost the loop a compare per
        # element (and would arm the bound check for an unbounded u64 array).
        out = self._read_varints(count, None, elem_max)
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return out

    def read_signed_array(
        self,
        elem_min: int | None = None,
        elem_max: int | None = None,
    ) -> list[int] | None:
        """Consume the current field as a list of ZigZag-decoded signed integers.

        ``elem_min``/``elem_max`` bound each element to its declared width — see
        :meth:`read_unsigned_array`. Either may be given on its own, which bounds
        that side and leaves the other open.

        Returns ``None`` if the field is not a signed array (§7.3), leaving it
        for the skip; raises :class:`SofaRangeError` if no value is pending at
        all.
        """
        count = self._take_varray(WireType.ARRAY_SIGNED)
        if count is None:
            return None
        self._keep = self._pos
        # ZigZag inlined rather than calling zigzag_decode per element: the
        # transform is two operations, the call around it was the expensive part.
        out = [
            (v >> 1) ^ -(v & 1)
            for v in self._read_varints(count, elem_min, elem_max, True)
        ]
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return out

    def _take_farray(self, subtype: FixlenSubtype) -> tuple[int, int] | None:
        pending = self._pending
        # §4.8: a fixlen array always carries its fixlen_word, so the subtype is
        # known even for a zero-count array — check it like any other read.
        if pending is None or pending[0] != _FARRAY or pending[1] != subtype:
            return self._mismatch(pending)  # §7.3
        # Like _take_varray, the pending value is cleared by the caller only
        # after the payload has been read (§5.2).
        return int(pending[2]), int(pending[3])  # count, elem_size

    def read_float32_array(self) -> list[float] | None:
        """Consume the current field as a list of 32-bit IEEE-754 floats.

        Returns ``None`` if the field is not an fp32 array (§7.3), leaving it for
        the skip; raises :class:`SofaRangeError` if no value is pending at all.
        """
        taken = self._take_farray(FixlenSubtype.FP32)
        if taken is None:
            return None
        count, elem_size = taken
        self._keep = self._pos
        # ``elem_size`` is necessarily 4 here: §4.8's width rule is settled at
        # the fixlen_word by ``next()``, which is where §5.2 wants that INVALID
        # verdict reached — before any payload read, so it outranks a truncation
        # behind it. The payload is therefore exactly ``count * 4`` bytes or this
        # read raises, and re-deciding the width afterwards could only restate a
        # check no input can reach (issue #75).
        data = self._read_exact(self._farray_nbytes(count, elem_size))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return _core.unpack_f32_array(data, count)

    def read_float64_array(self) -> list[float] | None:
        """Consume the current field as a list of 64-bit IEEE-754 floats.

        Returns ``None`` if the field is not an fp64 array (§7.3), leaving it for
        the skip; raises :class:`SofaRangeError` if no value is pending at all.
        """
        taken = self._take_farray(FixlenSubtype.FP64)
        if taken is None:
            return None
        count, elem_size = taken
        self._keep = self._pos
        # ``elem_size`` is necessarily 8 here; see read_float32_array.
        data = self._read_exact(self._farray_nbytes(count, elem_size))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return _core.unpack_f64_array(data, count)
