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
from array import array as _array
from typing import Any

from . import _core
from .binding import (
    K_ARRAY_FLOAT32,
    K_ARRAY_SIGNED,
    K_ARRAY_UNSIGNED,
    K_BYTES,
    K_FLOAT32,
    K_FLOAT64,
    K_SIGNED,
    K_STRING,
    K_UNSIGNED,
    Binding,
    Entry,
)
from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    MASK64,
    MAX_DEPTH,
    Field,
    FixlenSubtype,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaLimitError,
    SofaRangeError,
    Status,
    WireType,
)

# Imported at runtime, not only for typing: the driver compares a visitor's
# control hooks against the base class's to tell an override from the default.
from .visitor import Visitor

# Pending-value kinds the consume methods dispatch on.
_SCALAR = 0
_FIXLEN = 1
_VARRAY = 2
_FARRAY = 3
# A pending value a receiver-side cap has rejected (§6.2.1). The real pending
# tuple is parked inside it — ``(_LIMIT, message, pending)`` — so the rejection
# can still be waived by :meth:`Decoder.schema_bounded` and so a peek can read
# through it. Every path that would consume the field raises it instead.
_LIMIT = 4

# Wire-type members indexed by their integer value, so the per-field hot path
# can recover the enum member by index (``_WT[wtype]``) instead of paying the
# full ``WireType(wtype)`` coercion (IntEnum.__call__/__new__) on every field.
# What the push driver was doing when it ran out of bytes, so the next feed()
# resumes *that* call rather than restarting the field walk behind it (§5.2).
_R_NONE = 0
_R_SKIP = 1
_R_BOUND = 2
_R_VISIT = 3

_WT = tuple(WireType)

# The lowest value a signed element can carry, used as the open lower side when
# a field declares only the upper half of its element width (``_read_varints``).
_I64_MIN = -(1 << 63)


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
        *,
        binding: Binding | None = None,
        visitor: Visitor | None = None,
        words: Any = None,
        objects: list[Any] | None = None,
        max_array_count: int | None = None,
        max_string_len: int | None = None,
        max_blob_len: int | None = None,
    ) -> None:
        """Build a push decoder around a field handler (CORELIB_PLAN §5.2).

        The handler is a ``binding`` (:class:`sofab.Binding`, the fast path —
        fields land in caller-owned storage with no Python call per field), a
        ``visitor`` (:class:`sofab.Visitor`, the callback path), or both, in
        which case the binding takes each field it names and the visitor gets
        the rest. Bytes go in through :meth:`feed`, which returns the
        three-valued :class:`sofab.Status`.

        ``words`` and ``objects`` are the destinations a ``binding`` writes into
        and must be supplied with one: ``words`` a writable, C-contiguous buffer
        of ``binding.tree_words_required * 8`` bytes or more (a ``bytearray``),
        and ``objects`` a list of at least ``binding.tree_objects_required``
        entries. The decoder allocates neither and never sizes either from the
        wire.

        ``max_array_count`` / ``max_string_len`` / ``max_blob_len`` are optional
        **receiver-side** decode limits on fields the schema leaves unbounded: a
        field whose wire-declared count or length exceeds the cap is rejected
        with :class:`SofaLimitError`. The verdict is reached at the count/length
        header — before any allocation or payload buffering, so a hostile claim
        fails even if the payload never arrives. ``None`` (the default) means "no
        limit". They never apply to a field a binding declares a bound for: that
        declaration *is* the schema bound, and exceeding it is INVALID rather
        than a policy rejection (§6.2.1).
        """
        if binding is None and visitor is None:
            raise SofaRangeError("a decoder needs a field handler (binding / visitor)")
        self._max_array_count = max_array_count
        self._max_string_len = max_string_len
        self._max_blob_len = max_blob_len
        self._buf: bytes | bytearray = b""
        # len(self._buf), kept in step with it. The buffer only ever changes in
        # feed() and reset(), while the walk asks for its length constantly —
        # 5.2 len() calls per field on the composite workload, each a builtin
        # call. Holding the number is what the native engine already does.
        self._n = 0
        self._pos = 0
        self._depth = 0
        self._cur: Field | None = None
        # pending unconsumed value: tuple keyed by the _* constants above
        self._pending: tuple[Any, ...] | None = None
        # Resume transaction (§5.2): the buffer offset the call in flight
        # started at. See _suspend.
        self._keep = 0
        # The id and fixlen subtype of the header _next_wire last parsed, kept
        # unboxed for the caller that gets no Field.
        self._cur_id = 0
        self._cur_subtype = -1
        # The wire type _visit_value was last entered with, so a resumed visit
        # picks up the same dispatch without a Field to read it off.
        self._cur_wtype_resume = -1
        # Whether _next_wire maintains the two above. Only a binding resolves a
        # field by id without a Field to read it from.
        self._track_ids = binding is not None or visitor is not None

        self._binding = binding
        self._visitor = visitor
        # Whether the visitor overrides the two control hooks. Both default to a
        # no-op on the base class, and calling one that was never overridden
        # costs a Python call per field for nothing. ``_wants_field`` also
        # decides whether a Field object is built at all: it is the only thing
        # that receives one, and the typed hooks take an id.
        self._wants_field = (
            visitor is not None and type(visitor).on_field is not Visitor.on_field
        )
        self._wants_seq_begin = (
            visitor is not None
            and type(visitor).on_sequence_begin is not Visitor.on_sequence_begin
        )
        # The active table and the stack of enclosing ones. A sequence opens a
        # fresh id scope (§4.9), so descending *replaces* the table rather than
        # layering onto it — an id bound in the parent must not match inside.
        self._bmap: dict[int, Entry] | None = binding._by_id if binding is not None else None
        self._bstack: list[dict[int, Entry] | None] = []
        self._status = Status.COMPLETE
        self._error: SofaError | None = None
        self._resume_kind = _R_NONE
        self._resume_entry: Entry | None = None
        self._running = False
        self._objects = objects
        # One buffer, three views over the same bytes: the wire tells us which
        # to use per field, and a cast costs nothing at decode time.
        self._wq: Any = None
        self._wu: Any = None
        self._wd: Any = None
        if binding is not None:
            if words is None:
                raise SofaRangeError("a binding needs a words buffer")
            raw = memoryview(words)
            if raw.readonly:
                raise SofaRangeError("the words buffer must be writable")
            raw = raw.cast("B")
            if raw.nbytes % 8:
                raise SofaRangeError("the words buffer must be a multiple of 8 bytes")
            if raw.nbytes < binding.tree_words_required * 8:
                raise SofaRangeError(
                    f"words buffer holds {raw.nbytes // 8} slots, "
                    f"the binding needs {binding.tree_words_required}"
                )
            self._wq = raw.cast("q")
            self._wu = raw.cast("Q")
            self._wd = raw.cast("d")
            if objects is None:
                if binding.tree_objects_required:
                    raise SofaRangeError("a binding with string/blob fields needs objects")
            elif len(objects) < binding.tree_objects_required:
                raise SofaRangeError(
                    f"objects holds {len(objects)} entries, "
                    f"the binding needs {binding.tree_objects_required}"
                )

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
    #   point a suspension rewinds to, which is what keeps a suspended
    #   construct's bytes in the buffer for the retry.
    #
    # ``skip()`` over a whole *sequence* is the one call that spans many fields.
    # It remembers its first byte so a suspension can put ``_pos``/``_depth``/
    # ``_cur``/``_pending`` back — a re-issued skip then replays the sequence
    # from its start.
    #
    # INVALID is deliberately *not* rewound: it is terminal (§5.2), so no
    # continuation of bytes can make the stream valid again and there is nothing
    # to resume.

    def _suspend(self, msg: str) -> SofaIncompleteError:
        """Rewind to the current call's first byte and build its ``INCOMPLETE``.

        Called at every truncation site, so the rewind happens whether the raise
        is nested inside a field walk or not. ``_keep`` is maintained by
        :meth:`feed` across compaction, so it names the call's start byte in the
        *current* buffer even if the buffer was rebased between calls.
        """
        self._pos = self._keep
        return SofaIncompleteError(msg)

    # --- low-level byte sourcing --------------------------------------------
    #
    # The buffer is never sliced per byte: ``_pos`` advances over ``_buf`` and
    # the consumed prefix is dropped only when a refill is actually needed.

    def _varint(self) -> int:
        """Decode one base-128 varint by advancing the cursor over the buffer,
        refilling only if it runs off the end mid-value."""
        buf = self._buf
        pos = self._pos
        if pos >= self._n:
            raise self._suspend("truncated varint")
        b = buf[pos]
        pos += 1
        if b < 0x80:  # one-byte fast path (ids, small counts, small values)
            self._pos = pos
            return b
        result = b & 0x7F
        shift = 7
        n = self._n
        while True:
            if pos >= n:
                self._pos = pos
                raise self._suspend("truncated varint")
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

        A payload that stops halfway stays buffered when the truncation is
        reported, so the next attempt continues from it (§5.2)."""
        buf = self._buf
        pos = self._pos
        end = pos + n
        if end > self._n:
            raise self._suspend("truncated payload")
        self._pos = end
        # Always a real ``bytes``: while a construct is being accumulated the
        # buffer *is* a bytearray, and a slice of one would both be the wrong
        # type and alias storage the next feed can move. Slicing bytes already
        # gives bytes, so the memoryview detour — three objects to make one — is
        # only worth it for the bytearray case, where it saves the second copy.
        if type(buf) is bytes:
            return buf[pos:end]
        return bytes(memoryview(buf)[pos:end])

    def _skip_exact(self, n: int) -> None:
        """Consume the next ``n`` bytes without materialising them (§5.2: a skip
        *consumes and discards*; it must not pay for a copy of what it throws
        away). Same sourcing and same suspension as :meth:`_read_exact` — only
        the result object is missing.

        The bytes stay buffered either way, because they are not free to drop: a
        suspension has to be able to replay the skipped construct from its first
        byte, and a declined sequence replays a whole nested walk. Retention is
        the resume contract's; the copy was pure waste."""
        end = self._pos + n
        if end > self._n:
            raise self._suspend("truncated payload")
        self._pos = end

    def _read_varints(
        self,
        count: int,
        lo: int | None = None,
        hi: int | None = None,
        zigzag: bool = False,
        into: Any = None,
        base: int = 0,
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
        brings the varint path in line.

        ``into``/``base`` are the bound decode path (:class:`sofab.Binding`):
        elements go straight into the caller's slots as they are decoded, so the
        array costs no list and no second pass over it. The returned list is
        empty then — the caller already has the values where it wanted them. The
        DoS argument above does not apply to that path at all: the destination
        was sized by the caller from the schema, and a wire count past it was
        rejected before this was ever called."""
        out: list[int] = []
        append = out.append
        # ZigZag is folded in here for the bound path: the list path hands raw
        # values back and lets read_signed_array transform them, but with a
        # destination there is nowhere to do a second pass.
        store = into is not None
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
        n = self._n
        i = 0
        while i < count:
            if pos >= n:
                self._pos = pos
                raise self._suspend("truncated varint")
            b = buf[pos]
            pos += 1
            if b < 0x80:  # one-byte element
                if check_fast:
                    x = (b >> 1) ^ -(b & 1) if zigzag else b
                    if x < blo or x > bhi:
                        raise SofaDecodeError("array element outside declared width")
                if store:
                    into[base + i] = (b >> 1) ^ -(b & 1) if zigzag else b
                else:
                    append(b)
                i += 1
                continue
            result = b & 0x7F
            shift = 7
            while True:
                if pos >= n:
                    self._pos = pos
                    raise self._suspend("truncated varint")
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
            if store:
                into[base + i] = (result >> 1) ^ -(result & 1) if zigzag else result
            else:
                append(result)
            i += 1
        self._pos = pos
        return out

    def _skip_varints(self, count: int) -> None:
        """Advance the cursor past ``count`` varints without materialising them."""
        buf = self._buf
        pos = self._pos
        n = self._n
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
            n = self._n
            i += 1
        self._pos = pos

    # --- field iteration ----------------------------------------------------

    def _next_wire(self, want_field: bool) -> int:
        """Parse one field header; return its wire type, or ``-1`` at clean EOF.

        The body of :meth:`next`, plus the one thing :meth:`next` cannot express:
        whether the caller wants a :class:`Field` at all. ``skip()``'s sequence
        walk and a push decode bound to destinations do not, and building one is
        not free — a ``Field`` is ~250 ns here, so a 36-field message would spend
        ~9 us on objects nobody reads. With ``want_field`` false none are built:
        the id and fixlen subtype go to ``_cur_id`` / ``_cur_subtype`` instead
        (and only when a binding needs them), and the size/count stay in the
        pending tuple, where the consume paths already look for them.
        """
        self._keep = self._pos  # opens this field's resume transaction (§5.2)
        if self._pending is not None:
            self._skip_pending()
            # The auto-skip committed, so it must not be replayed: re-open the
            # transaction *after* it. Without this, a suspension later in this
            # same call (the EOF check below, or the header parse) rewinds to
            # before the skipped value, and the retry re-reads those bytes as a
            # new field. Only a push decoder can reach it — a reader-backed one
            # blocks inside its refill instead of returning to the caller here.
            self._keep = self._pos

        buf = self._buf
        pos = self._pos
        if pos >= self._n:
            if self._depth != 0:
                raise self._suspend("truncated: unbalanced sequence")
            # ``_cur`` deliberately keeps the last field past EOF: `field` is
            # documented as "the most recently returned Field".
            return -1

        # The header varint, read inline. A call into _varint costs more than
        # the byte it usually returns: ids below 16 with a wire type packed
        # beside them fit in one byte, which is the overwhelmingly common
        # header, and the EOF test above has already proved that byte is there.
        # Anything longer falls back to the real reader.
        header = buf[pos]
        if header < 0x80:
            self._pos = pos + 1
        else:
            header = self._varint()
        wtype = header & 0x07
        field_id = header >> 3
        if self._track_ids:
            # Only a decoder with a binding resolves fields by id without a
            # Field object. Everything else reads the id off the Field it is
            # handed, so it must not pay for maintaining a second copy.
            self._cur_id = field_id
            self._cur_subtype = -1

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
            if want_field:
                self._cur = Field(0, WireType.SEQUENCE_END)
            return wtype

        if wtype == WireType.SEQUENCE_START:
            if self._depth >= MAX_DEPTH:
                raise SofaDecodeError(f"nesting exceeds MAX_DEPTH={MAX_DEPTH}")
            self._depth += 1
            if want_field:
                self._cur = Field(field_id, WireType.SEQUENCE_START)
            return wtype

        if wtype == WireType.UNSIGNED or wtype == WireType.SIGNED:
            if want_field:
                self._cur = Field(field_id, _WT[wtype])
            self._pending = (_SCALAR, wtype)
            return wtype

        if wtype == WireType.FIXLEN:
            # Same one-byte fast path as the header above, but this byte is not
            # guaranteed to be buffered, so the bound is tested first. A length
            # word is one byte for any payload under 16 bytes.
            pos = self._pos
            if pos < self._n:
                length_header = buf[pos]
                if length_header < 0x80:
                    self._pos = pos + 1
                else:
                    length_header = self._varint()
            else:
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
            if want_field:
                self._cur = Field(
                    field_id, WireType.FIXLEN, size=length,
                    subtype=FixlenSubtype(subtype),
                )
            if self._track_ids:
                self._cur_subtype = subtype
            pending: tuple[Any, ...] = (_FIXLEN, subtype, length)
            # Receiver-configured caps (policy, not malformation): the verdict on
            # an oversize string/blob is reached here, on the length word alone —
            # before its payload is read or buffered, which is where §6.2.1 wants
            # it — and parked on the pending value rather than raised. Nothing is
            # read or allocated until the field is consumed, and every consume
            # path raises it, so deferring the raise by one step costs the
            # protection nothing; what it buys is the window
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
            return wtype

        if wtype == WireType.ARRAY_UNSIGNED or wtype == WireType.ARRAY_SIGNED:
            count = self._varint()
            if count < 0 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            if want_field:
                self._cur = Field(field_id, _WT[wtype], count=count)
            pending = (_VARRAY, wtype, count)
            # Parked, not raised — see the fixlen branch above (§6.2.1).
            cap = self._max_array_count
            if cap is not None and count > cap:
                pending = (_LIMIT, f"array count {count} exceeds max_array_count {cap}", pending)
            self._pending = pending
            return wtype

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
        if want_field:
            self._cur = Field(
                field_id,
                WireType.ARRAY_FIXLEN,
                count=count,
                size=elem_size,
                subtype=FixlenSubtype(subtype),
            )
        if self._track_ids:
            self._cur_subtype = subtype
        pending = (_FARRAY, subtype, count, elem_size)
        # Parked, not raised — see the fixlen branch above (§6.2.1).
        cap = self._max_array_count
        if cap is not None and count > cap:
            pending = (_LIMIT, f"array count {count} exceeds max_array_count {cap}", pending)
        self._pending = pending
        return wtype

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

    def _skip_sequence(self) -> None:
        """Skip an entire (nested) sequence, from the start marker just read to
        its matching end.

        The driver is the only caller and reaches it only on a sequence start —
        a declined value needs no skip at all, because it stays pending and the
        next header discards it.

        Suspends as a unit: if the bytes run out part-way, nothing is consumed
        and the same skip can be re-issued when more arrive (§5.2).
        """
        # Walking a whole sequence spans many fields, so unlike every other call
        # this one moves the field state — and lets _next_wire re-arm ``_keep`` —
        # before it can suspend. The field state is put back here, so a re-issued
        # skip replays the whole sequence (§5.2).
        self._keep = floor = self._pos
        depth, cur, pending = self._depth, self._cur, self._pending
        try:
            target = depth - 1
            while self._depth > target:
                # The walk discards every field it passes, so it asks for no
                # Field objects. Defensive: at EOF with an open sequence,
                # _next_wire itself raises "truncated: unbalanced sequence",
                # so it never returns -1 here.
                if self._next_wire(False) < 0:  # pragma: no cover
                    raise self._suspend("truncated sequence")
        except SofaIncompleteError:
            self._pos = self._keep = floor
            self._depth, self._cur, self._pending = depth, cur, pending
            raise
        # The walk built no Field, so ``_cur`` still names the sequence that was
        # just consumed. Publish the end marker the walk stopped on, exactly as a
        # Field-building walk would have left it, so a visitor that keeps the
        # last Field does not see a stale sequence start. One object per skip,
        # not one per field.
        self._cur = Field(0, WireType.SEQUENCE_END)

    # --- push-feed driver (CORELIB_PLAN §5.2) -------------------------------
    #
    # The other half of §5.2's "push-feed / pull-read" model. The caller hands
    # over chunks; this side walks the fields and binds each value straight into
    # the destination the handler declared — no callback per field when a
    # binding names it, and no callback per array *element* ever.
    #
    # Resumption is the pull decoder's transaction model, one level up. Each
    # individual call is already all-or-nothing (see _suspend), so the only
    # thing the driver has to add is *which* call was in flight when the bytes
    # ran out: restarting the field walk instead would skip the half-read value
    # the retry is supposed to finish. That is what _resume_kind records, and it
    # is why the value read below is never re-entered through next().

    @property
    def error(self) -> SofaError | None:
        """The failure that made :meth:`feed` return :attr:`Status.INVALID`, or
        ``None``. Mirrors :attr:`sofab.Encoder.error`: the status is the answer,
        this is the reason behind it."""
        return self._error

    @property
    def status(self) -> Status:
        """The outcome of the last :meth:`feed`."""
        return self._status

    def feed(self, data: Any) -> Status:
        """Consume ``data`` and report the outcome for the bytes so far (§5.2).

        Accepts anything with a buffer — ``bytes``, ``bytearray``, a
        ``memoryview`` over either. **The chunk is borrowed only for the
        duration of this call** (§6, chunk lifetime): whatever the decoder still
        needs afterwards — a construct split across the boundary, a decoded
        string or blob — is copied out before it returns, so the caller may
        reuse or overwrite that memory the moment ``feed`` comes back.

        Returns :attr:`Status.COMPLETE`, :attr:`Status.INCOMPLETE` or
        :attr:`Status.INVALID`. There is deliberately no ``finish``/``end``
        counterpart: an ``INCOMPLETE`` at end-of-input is truncation, and only
        the caller's framing can say so. ``INVALID`` is terminal — every later
        ``feed`` returns it again without consuming anything, and the reason
        stays on :attr:`error`.

        A **receiver-side limit** rejection (§6.2.1) is not one of the three
        outcomes: the message is well-formed and the receiver simply declined
        it, so it is raised as :class:`SofaLimitError` rather than folded into
        ``INVALID`` (§6.3).
        """
        if self._running:
            raise SofaRangeError("feed() is not re-entrant")
        if self._status is Status.INVALID:
            return Status.INVALID
        # Two shapes, because they want opposite things.
        #
        # Nothing carried — the whole previous feed was consumed, which is the
        # one-shot case and the steady state of a chunked one — and the chunk is
        # *adopted*: ``bytes`` is immutable, so there is nothing to copy.
        #
        # Something carried, i.e. a construct that could not complete: the
        # buffer becomes a bytearray and the chunk is *appended*. A construct
        # that cannot complete leaves ``_pos`` at 0 (the suspension rewinds it),
        # so every following feed only extends, at amortised O(len(chunk)).
        # Rebuilding ``carry + chunk`` instead would copy the whole carry per
        # chunk — a 1 MB blob fed in 4 KiB pieces costs ~122 MB of copying that
        # way.
        buf = self._buf
        if self._pos >= self._n:
            buf = data if isinstance(data, bytes) else bytes(data)
            self._buf = buf
        else:
            if not isinstance(buf, bytearray):
                buf = bytearray(buf)
                self._buf = buf
            if self._pos:
                del buf[: self._pos]
            buf += data
        self._n = len(buf)
        self._pos = 0
        self._keep = 0
        self._running = True
        try:
            if self._drive_push():
                self._status = Status.INCOMPLETE
                return Status.INCOMPLETE
        except SofaIncompleteError:
            self._status = Status.INCOMPLETE
            return Status.INCOMPLETE
        except SofaDecodeError as exc:
            self._error = exc
            self._status = Status.INVALID
            return Status.INVALID
        finally:
            self._running = False
        self._status = Status.COMPLETE
        return Status.COMPLETE

    def reset(self) -> None:
        """Forget the stream and start a new message, keeping the handler and
        its destinations. Lets one decoder serve many messages without rebuilding
        the binding — the destinations are the caller's to clear (or not: a slot
        the next message does not write keeps whatever is in it, which is how
        absence is reported)."""
        self._buf = b""
        self._n = 0
        self._pos = 0
        self._depth = 0
        self._cur = None
        self._pending = None
        self._keep = 0
        self._status = Status.COMPLETE
        self._error = None
        self._resume_kind = _R_NONE
        self._resume_entry = None
        self._bmap = self._binding._by_id if self._binding is not None else None
        self._bstack.clear()

    def _value_ready(self) -> bool:
        """Are all of the pending value's bytes buffered?

        Asked once per feed, on the resume path only, and only for the two kinds
        whose length is known from the header — a fixlen payload and a fixlen
        array. The *first* attempt at a value still just tries and catches; what
        this removes is the retry raising again on every later chunk. A 1 MB blob
        fed in 4 KiB pieces suspends 244 times, and 243 of those would otherwise
        spend more on the exception machinery than on the bytes. Putting the
        check in the field walk instead would charge every field for it, which on
        the pure engine costs more than it saves.
        """
        pending = self._pending
        assert pending is not None  # only asked while a value is pending
        kind = pending[0]
        have = self._n - self._pos
        if kind == _FIXLEN:
            return bool(have >= pending[2])
        if kind == _FARRAY:
            return bool(have >= pending[2] * pending[3])
        return True

    def _drive_push(self) -> bool:
        """Walk the fed bytes. Returns ``True`` if it stopped short of a value
        whose bytes have not all arrived — the cheap half of INCOMPLETE, with no
        exception raised. Every other suspension still comes through
        :class:`SofaIncompleteError`."""
        visitor = self._visitor
        rk = self._resume_kind
        if rk != _R_NONE:
            # A value read ran out of bytes last time. Finish *it* — walking on
            # to the next header would auto-skip the value the caller is owed.
            if rk != _R_SKIP and not self._value_ready():
                return True
            self._resume_kind = _R_NONE
            try:
                if rk == _R_BOUND:
                    entry = self._resume_entry
                    assert entry is not None
                    self._take_bound(entry)
                elif rk == _R_VISIT:
                    assert visitor is not None
                    self._visit_value(visitor, self._cur_wtype_resume)
                else:
                    self._skip_sequence()
            except SofaIncompleteError:
                self._resume_kind = rk
                raise

        # A Field is built only for a visitor that overrides ``on_field`` — the
        # one consumer that takes one. Not building it is most of what makes the
        # other paths fast (see _next_wire).
        want_field = self._wants_field
        while True:
            t = self._next_wire(want_field)
            if t < 0:
                return False

            if t == WireType.SEQUENCE_END:
                if self._bstack:
                    self._bmap = self._bstack.pop()
                if visitor is not None:
                    visitor.on_sequence_end()
                continue

            bmap = self._bmap
            entry = bmap.get(self._cur_id) if bmap is not None else None
            if entry is not None and (
                t != entry.wt
                or (entry.st is not None and self._cur_subtype != entry.st)
            ):
                # §7.3: the wire tag contradicts what the schema declared for
                # this id. Not an error — treat it exactly like an unknown id.
                entry = None
                if t != WireType.SEQUENCE_START:
                    continue

            if t == WireType.SEQUENCE_START:
                if entry is not None:
                    child = entry.child
                    assert child is not None
                    self._bstack.append(self._bmap)
                    self._bmap = child._by_id
                    c = entry.count_at
                    if c >= 0:
                        self._wu[c] = self._wu[c] + 1
                    continue
                if visitor is not None and (
                    not self._wants_seq_begin
                    or visitor.on_sequence_begin(self._cur_id) is not False
                ):
                    # §4.9 opens a fresh id scope, so the enclosing table must
                    # not match inside it.
                    self._bstack.append(self._bmap)
                    self._bmap = None
                    continue
                try:
                    self._skip_sequence()
                except SofaIncompleteError:
                    self._resume_kind = _R_SKIP
                    raise
                continue

            if entry is not None:
                try:
                    self._take_bound(entry)
                except SofaIncompleteError:
                    self._resume_kind = _R_BOUND
                    self._resume_entry = entry
                    raise
                continue

            if visitor is not None:
                if self._wants_field:
                    f = self._cur
                    assert f is not None
                    if visitor.on_field(f) is False:
                        continue
                try:
                    self._visit_value(visitor, t)
                except SofaIncompleteError:
                    self._resume_kind = _R_VISIT
                    raise
                continue
            # Nobody wants it. The value stays pending and the next next()
            # discards it, which is cheaper than skipping it here and suspends
            # in exactly the same place.

    def _take_bound(self, e: Entry) -> None:
        """Decode the current field into the destination ``e`` names.

        Consumes nothing on the suspension path, like every other read, so the
        retry redoes the whole value — including refilling a partly written
        array from element zero (§5.2).

        The driver matched the field's whole tag before calling, so the consume
        helpers here skip re-deriving it, and the size/count come from the
        pending tuple rather than from a :class:`Field` — which is what lets a
        bound decode run without one ever being built.
        """
        k = e.kind
        at = e.at
        got = 1
        if k == K_UNSIGNED:
            self._wu[at] = self._take_scalar_matched()
        elif k == K_SIGNED:
            raw = self._take_scalar_matched()
            self._wq[at] = (raw >> 1) ^ -(raw & 1)
        elif k == K_FLOAT64:
            self._wd[at] = _core.unpack_f64(self._take_fixlen_matched(8))
        elif k == K_FLOAT32:
            self._wd[at] = _core.unpack_f32(self._take_fixlen_matched(4))
        elif k == K_STRING or k == K_BYTES:
            pending = self._pending
            assert pending is not None
            if e.cap:
                # A declared maxlen makes the field schema-bounded: the
                # receiver-side cap stops applying to it (§6.2.1) and an
                # over-long payload is INVALID, not a policy rejection (§7.1).
                if pending[0] == _LIMIT:
                    pending = pending[2]
                    self._pending = pending
                if pending[2] > e.cap:
                    raise SofaDecodeError(
                        f"fixlen length {pending[2]} exceeds the {e.cap} "
                        f"the schema declares"
                    )
            elif pending[0] == _LIMIT:
                # No declared bound, so the configured cap still governs the
                # field and has already rejected it. The typed reads reach this
                # It has to be raised explicitly here, or the consume below
                # would walk straight past the verdict.
                raise SofaLimitError(pending[1])
            data = self._take_fixlen_matched(pending[2])
            if k == K_BYTES:
                self._objects[at] = data  # type: ignore[index]
            else:
                try:
                    self._objects[at] = data.decode("utf-8")  # type: ignore[index]
                except UnicodeDecodeError as exc:
                    raise SofaDecodeError("invalid UTF-8 in string field") from exc
        else:
            pending = self._pending
            assert pending is not None
            if pending[0] == _LIMIT:
                # Binding an array declares its bound, so the field is
                # schema-bounded and the receiver cap does not apply (§6.2.1).
                pending = pending[2]
                self._pending = pending
            got = pending[2]
            if got > e.cap:
                # The destination's size is the schema's bound, so a longer
                # array is a malformed message, not a receiver policy call
                # (MESSAGE_SPEC §7.1). Rejected here — at the count header,
                # before an element is read (§5.2).
                raise SofaDecodeError(
                    f"array count {got} exceeds the {e.cap} the schema declares"
                )
            self._keep = self._pos
            bounded = e.elem_bounded
            if k == K_ARRAY_UNSIGNED:
                # Straight into the caller's slots: no list, and no second pass
                # over one.
                self._read_varints(
                    got, None, e.elem_hi if bounded else None, False, self._wu, at
                )
            elif k == K_ARRAY_SIGNED:
                self._read_varints(
                    got,
                    e.elem_lo if bounded else None,
                    e.elem_hi if bounded else None,
                    True,
                    self._wq,
                    at,
                )
            else:
                # A float payload is fixed-width, so it is read whole and
                # unpacked in one call; handing that to ``array`` moves it into
                # the destination at C speed rather than element by element.
                width = 4 if k == K_ARRAY_FLOAT32 else 8
                data = self._read_exact(self._farray_nbytes(got, width))
                values = (
                    _core.unpack_f32_array(data, got)
                    if k == K_ARRAY_FLOAT32
                    else _core.unpack_f64_array(data, got)
                )
                self._wd[at : at + got] = _array("d", values)
            self._pending = None  # committed only once the payload is in hand

        c = e.count_at
        if c >= 0:
            self._wu[c] = got

    def _take_scalar_matched(self) -> int:
        """Consume the pending scalar. The caller has already matched the whole
        tag, so this only consumes — and the result is an ``int``, not
        ``int | None``."""
        self._keep = self._pos
        value = self._varint()
        self._pending = None  # committed only once the value is in hand (§5.2)
        return value

    def _take_fixlen_matched(self, length: int) -> bytes:
        """:meth:`_take_scalar_matched` for a fixlen payload."""
        self._keep = self._pos
        data = self._read_exact(length)
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return data

    # --- value reads for the visitor path -----------------------------------
    #
    # The driver dispatches on the field's own wire type before calling, so every
    # read here is reached with a tag that already matches. There is no §7.3
    # mismatch to answer: a *binding* can contradict the wire and is skipped by
    # the driver (see _drive_push), and a visitor declares nothing about a
    # field's type at all — it is simply handed what the wire carried.

    def _visit_value(self, visitor: Visitor, t: int) -> None:
        """Hand the current field's value to ``visitor``'s typed hook.

        The id and subtype come off the decoder's own state rather than a Field:
        the typed hooks take an id, so unless ``on_field`` is overridden there is
        no reason to have built one.
        """
        self._cur_wtype_resume = t
        pending = self._pending
        assert pending is not None
        fid = self._cur_id
        if t == WireType.UNSIGNED:
            visitor.on_unsigned(fid, self._take_scalar_matched())
        elif t == WireType.SIGNED:
            raw = self._take_scalar_matched()
            visitor.on_signed(fid, (raw >> 1) ^ -(raw & 1))
        elif t == WireType.FIXLEN:
            self._visit_fixlen(visitor, fid, pending)
        elif t == WireType.ARRAY_UNSIGNED:
            visitor.on_unsigned_array(fid, self._take_varints(pending[2], False))
        elif t == WireType.ARRAY_SIGNED:
            visitor.on_signed_array(fid, self._take_varints(pending[2], True))
        elif pending[1] == FixlenSubtype.FP32:
            visitor.on_float32_array(fid, self._take_farray_values(pending, 4))
        else:
            visitor.on_float64_array(fid, self._take_farray_values(pending, 8))

    def _visit_fixlen(self, visitor: Visitor, fid: int, pending: tuple[Any, ...]) -> None:
        if pending[0] == _LIMIT:
            # A parked receiver cap (§6.2.1): the field is being refused, and the
            # walk over its payload is exactly what the cap exists to prevent.
            raise SofaLimitError(pending[1])
        subtype = pending[1]
        if subtype == FixlenSubtype.FP32:
            visitor.on_float32(fid, _core.unpack_f32(self._take_fixlen_matched(4)))
        elif subtype == FixlenSubtype.FP64:
            visitor.on_float64(fid, _core.unpack_f64(self._take_fixlen_matched(8)))
        elif subtype == FixlenSubtype.STRING:
            data = self._take_fixlen_matched(pending[2])
            try:
                visitor.on_string(fid, data.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise SofaDecodeError("invalid UTF-8 in string field") from exc
        else:
            visitor.on_bytes(fid, self._take_fixlen_matched(pending[2]))

    def _take_varints(self, count: int, zigzag: bool) -> list[int]:
        """The whole pending integer array, as a list (the visitor's shape)."""
        if self._pending is not None and self._pending[0] == _LIMIT:
            raise SofaLimitError(self._pending[1])
        self._keep = self._pos
        out = self._read_varints(count, None, None, zigzag)
        if zigzag:
            out = [(v >> 1) ^ -(v & 1) for v in out]
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return out

    def _take_farray_values(self, pending: tuple[Any, ...], width: int) -> list[float]:
        if pending[0] == _LIMIT:
            raise SofaLimitError(pending[1])
        count = pending[2]
        self._keep = self._pos
        data = self._read_exact(self._farray_nbytes(count, pending[3]))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return (
            _core.unpack_f32_array(data, count)
            if width == 4
            else _core.unpack_f64_array(data, count)
        )
