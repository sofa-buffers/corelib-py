"""SofaBuffers push decoder (CORELIB_PLAN §5.2).

Bytes go **in** through :meth:`Decoder.feed`, in chunks of any size, and each
call returns the three-valued :class:`sofab.Status` for the bytes so far. There
is no ``finish``/``finalize`` step: an ``INCOMPLETE`` at end of input is
truncation, and only the caller's framing can say so (§5.2.4).

Fields come **out** through a caller-supplied handler — a
:class:`sofab.Visitor`, which §5.3.1 makes the decode surface, or a
:class:`sofab.Binding`, a table mapping field ids to slots in storage the caller
owns, or both at once. There is no pull API: no ``next()``, no cursor, no typed
read a caller issues.

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

    class Sink(sofab.Visitor):
        def on_unsigned(self, field_id, value):
            ...

    dec = Decoder(visitor=Sink())
    for chunk in stream:
        status = dec.feed(chunk)
"""

from __future__ import annotations

import sys
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
    DEFAULT_REASSEMBLY,
    FIXLEN_MAX,
    ID_MAX,
    MASK64,
    MAX_DEPTH,
    Field,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaLimitError,
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
# can still be waived by the schema bound a binding entry declares, and so a
# route that needs the count or the subtype can read through it.
#
# Parked rather than raised because the verdict is not final at the header: it
# depends on where the value is headed, which only the field's route knows
# (#128). A path that would put the value in storage of the DECODER's own —
# sized by the wire — raises it; a path that skips the field, or that fills a
# destination the handler supplied, unwraps it and walks on.
_LIMIT = 4

# Wire-type members indexed by their integer value, so the per-field hot path
# can recover the enum member by index (``_WT[wtype]``) instead of paying the
# full ``WireType(wtype)`` coercion (IntEnum.__call__/__new__) on every field.
# What the push driver was doing when it ran out of bytes, so the next feed()
# resumes *that* call rather than restarting the field walk behind it (§5.2).
_R_NONE = 0
_R_SKIP = 1
_R_VISIT = 2

_WT = tuple(WireType)
# The fixlen subtypes by index, for the same reason and at the same cost:
# ``on_schema_bound`` is told the tag the wire carried, and recovering the
# member by index is a tuple load rather than an ``IntEnum`` coercion.
_ST = tuple(FixlenSubtype)

# The two members ``on_schema_bound`` names outright, as single global loads:
# the kind is known from the pending tuple there, so there is nothing to index.
_WTM_FIXLEN = WireType.FIXLEN
_WTM_ARRAY_FIXLEN = WireType.ARRAY_FIXLEN

# The wire types and fixlen subtypes as plain ints. ``WireType.FIXLEN`` inside a
# comparison is a global load plus an attribute lookup on the enum class, paid
# on every field for a number that is a compile-time constant everywhere it is
# used. The names below are the same values with only the global load left.
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

# The lowest value a signed element can carry, used as the open lower side when
# a field declares only the upper half of its element width (``_read_varints``).
_I64_MIN = -(1 << 63)


def _writable(dst: Any, who: str) -> memoryview:
    """The caller's destination as a memoryview, or the §6.3 verdict on why not."""
    try:
        view = memoryview(dst)
    except TypeError as exc:
        raise SofaArgumentError(
            f"{who} returned a destination that is not a writable, "
            "contiguous buffer"
        ) from exc
    if view.readonly or not view.c_contiguous:
        raise SofaArgumentError(
            f"{who} returned a destination that is not a writable, "
            "contiguous buffer"
        )
    return view


def _width_fits(itemsize: int, zigzag: bool, lo: int | None, hi: int | None) -> bool:
    """Does the declared element width fit an ``itemsize``-byte slot?"""
    bits = itemsize * 8
    if zigzag:
        return (
            lo is not None
            and hi is not None
            and lo >= -(1 << (bits - 1))
            and hi < (1 << (bits - 1))
        )
    return hi is not None and hi < (1 << bits)


def _stated_limit(name: str, value: int | None, ceiling: int) -> int:
    """Take a receiver cap the **caller** stated, or refuse the call (§6.2.1).

    §6.2.1 fixes the provenance of the number even where the comparison runs
    inside the codec: a codec "**MUST NOT** hold a limit of its own, **MUST NOT**
    supply a default for one it was not given, **MUST NOT** read an omitted
    argument as *unlimited*, and **MUST NOT** clamp to one". A format ceiling
    (§6.2) reached because no cap was stated is the *format's* bound, not a
    receiver cap, so it cannot stand in for one either.

    An omitted cap is therefore a defect in the **call**, and §6.3 puts a defect
    in the call in the ``InvalidArgument`` tier — never ``LimitExceeded``, which
    would promise a limit to raise that was never configured.
    """
    if value is None:
        raise SofaArgumentError(
            f"{name} is required (§6.2.1): the codec holds no limit of its own "
            f"and reads no omitted argument as unlimited"
        )
    if value < 0 or value > ceiling:
        raise SofaArgumentError(f"{name}={value} is outside 0..{ceiling}")
    return value


class Decoder:
    """Decodes a SofaBuffers stream, pushing each field at a handler.

    Construct it with a ``visitor``, a ``binding``, or both, then hand it bytes
    with :meth:`feed`. Where both are given the binding takes every field it
    names and the visitor gets the rest.

    **A declared type that contradicts the field on the wire is not an error**
    (MESSAGE_SPEC §7.3, CORELIB_PLAN §6.3). A binding entry whose wire type or
    subtype does not match the field is skipped: its destination is left
    untouched and the decode stays COMPLETE. Nothing is raised, nothing is
    materialized, nothing is validated — and the fallback visitor is not offered
    it either, because a skipped field is skipped for the whole handler.
    """

    # Every attribute this decoder holds, declared rather than left to an
    # instance dict.
    #
    # Not a size micro-optimisation. CPython shares one key table between all
    # instances of a class, but only while the class stays within
    # SHARED_KEYS_MAX_SIZE == 30 attributes; the 31st drops every instance
    # onto its own combined dict and slows down *every* ``self._x`` in the
    # decoder. This class sat at exactly 30, so the next attribute anyone
    # added -- ``_wants_array_begin`` below is one -- cost 12.8% on
    # `decode: composite`, a message with no arrays in it at all. Slots remove
    # the cliff rather than stepping around it: attribute access becomes a
    # fixed offset and the count stops mattering.
    __slots__ = (
        "_buf",
        # One memoryview over ``_buf``, remade whenever the buffer changes and
        # dropped when the feed ends. It is how a string's payload is transcoded
        # without copying it out first, and §6.6.2's "language-forced handle" is
        # exactly what it is: it carries no message bytes of its own and costs
        # the same over ten bytes as over ten megabytes. Held per *feed*, never
        # across one — §6's chunk lifetime is what _retain enforces, and a view
        # left pointing at a returned chunk would break it.
        "_bufsrc",
        "_bufview",
        "_capped",
        "_cur_id",
        "_cur_subtype",
        "_cur_wtype_resume",
        "_depth",
        "_error",
        "_keep",
        "_max_dyn_array_count",
        "_max_dyn_blob_len",
        "_max_dyn_string_len",
        "_n",
        "_pending",
        "_pos",
        "_limit",
        "_rbuf",
        "_rend",
        "_rstart",
        "_resume_kind",
        "_running",
        "_status",
        "_visitor",
        "_vstack",
        "_vsp",
        # --- the destination map (§6.6.3) ------------------------------------
        # Where a mapped field's value goes, and what the schema declares for
        # it. Both are the *caller's* answers, settled once at construction;
        # neither is a decode rule, and no rule below branches on them.
        "_bmap",
        "_bstack",
        "_bsp",
        "_objects",
        "_wu",
        "_wq",
        "_wd",
        "_resume_entry",
        "_make_field",
        "_wants_array_begin",
        "_wants_bound",
        "_wants_blob_begin",
        "_wants_string_begin",
        "_wants_farray_begin",
        "_wants_f32_bits",
        "_wants_f32_array_bits",
        "_wants_field",
        "_wants_seq_begin",
    )


    def __init__(
        self,
        *,
        binding: Binding | None = None,
        visitor: Visitor | None = None,
        words: Any = None,
        objects: list[Any] | None = None,
        max_dyn_array_count: int | None = None,
        max_dyn_string_len: int | None = None,
        max_dyn_blob_len: int | None = None,
        reassembly: Any = None,
    ) -> None:
        """Build a push decoder around a field handler (CORELIB_PLAN §5.2).

        The handler is a ``visitor`` (:class:`sofab.Visitor`), a ``binding``
        (:class:`sofab.Binding` — a table of field id to destination, compiled
        into a visitor over that table), or both, in which case the binding
        takes each field it names and the visitor gets the rest. There is one
        decode surface either way (§5.3.1); a table is a way of saying where a
        field goes, not a second route for getting it there. Bytes go in through
        :meth:`feed`, which returns the three-valued :class:`sofab.Status`.

        ``words`` and ``objects`` are the destinations a ``binding`` writes into
        and must be supplied with one: ``words`` a writable, C-contiguous buffer
        of ``binding.tree_words_required * 8`` bytes or more (a ``bytearray``),
        and ``objects`` a list of at least ``binding.tree_objects_required``
        entries. The decoder allocates neither and never sizes either from the
        wire.

        ``max_dyn_array_count`` / ``max_dyn_string_len`` / ``max_dyn_blob_len`` are the
        **receiver-side** limits of §6.2.1, on the fields the schema leaves
        unbounded: a field whose wire-declared count or length exceeds one is
        rejected with :class:`SofaLimitError`. The verdict is reached at the
        count/length header — before any allocation or payload buffering, so a
        hostile claim fails even if the payload never arrives.

        **All three are required.** §6.2.1 lets the codec *perform* the
        comparison — "a corelib **MAY** take a limit as an argument and perform
        the check itself" — but the number stays the caller's: the codec "**MUST
        NOT** hold a limit of its own, **MUST NOT** supply a default for one it
        was not given, **MUST NOT** read an omitted argument as *unlimited*, and
        **MUST NOT** clamp to one". Omitting one — or passing ``None`` — is a
        defect in the call and raises :class:`SofaArgumentError` (§6.3's
        ``InvalidArgument`` tier), never :class:`SofaLimitError`, which would
        promise a limit to raise that was never configured. A caller that wants
        the widest limit the format admits states the ceiling itself
        (``max_dyn_array_count=sofab.ARRAY_MAX``); that is then the caller's
        number, not the codec's default.

        What they bound is what **this decoder** allocates, which §6.2.1 states
        as their whole purpose: an unbounded field "would let the *sender*
        dictate the *receiver's* allocation". So they govern the default route,
        where the decoder has to build a ``str``, a ``bytes`` or a list and the
        wire is the only size it could build one from. A field that is skipped
        allocates nothing and is not capped; neither is one read into a buffer
        the handler returned from :meth:`sofab.Visitor.on_blob_begin` or
        :meth:`sofab.Visitor.on_array_begin`, which those hooks size themselves
        after being told the announced length or count. There the destination's
        own size is the ceiling, and a short one is
        :class:`SofaArgumentError` rather than a policy rejection.

        **There is no unset state and no unlimited mode** (§6.2.1): "unbounded by
        the schema" is still bounded by the receiver. The numbers themselves
        belong to generated code, which knows the schema and the deployment; the
        codec neither invents a policy of its own nor clamps to one.

        They never apply to a field the handler declares a schema bound for —
        a ``maxlen``/``cap`` on a binding entry, or a
        :meth:`sofab.Visitor.on_schema_bound` answer. That declaration *is* the
        schema bound, and exceeding it is INVALID rather than a policy
        rejection (§6.2.1).

        ``reassembly`` is where a construct split across fed chunks is joined.
        Pass a ``bytearray`` to supply the storage, an ``int`` to have the
        decoder take that many bytes **at construction**, or leave it out for
        :data:`sofab.DEFAULT_REASSEMBLY` bytes. There is no other shape: §6.6.2
        says a codec "**MUST NOT** grow a private accumulator instead", so the
        buffer is sized once and never extended, and a construct that does not
        fit it is :class:`SofaArgumentError` — the §6.3 tier for a well-formed
        message that does not fit the storage this caller offered. What that
        buys is §6.6's whole point: a caller bounds a decode's memory **by
        construction**, and no sender can change the answer by sending different
        bytes.

        A message fed in **one** call never touches the buffer, whatever its
        size — nothing spans a chunk boundary when there is only one chunk. It
        is a chunked reader that has to size it for the largest ``string``,
        ``blob`` or array payload it will take across a boundary.
        """
        if binding is None and visitor is None:
            raise SofaArgumentError("a decoder needs a field handler (binding / visitor)")
        # §6.2.1: the numbers are the caller's, and there is no default to fall
        # back on -- omitting one is refused here rather than resolved to the
        # format ceiling, which is the format's bound and not a receiver cap.
        # Written out rather than looped: a decoder is constructed per message on
        # the one-shot path, and a Python-level loop over three tuples is a
        # measurable share of that.
        arr_cap = _stated_limit("max_dyn_array_count", max_dyn_array_count, ARRAY_MAX)
        str_cap = _stated_limit("max_dyn_string_len", max_dyn_string_len, FIXLEN_MAX)
        blob_cap = _stated_limit("max_dyn_blob_len", max_dyn_blob_len, FIXLEN_MAX)
        self._max_dyn_array_count = arr_cap
        self._max_dyn_string_len = str_cap
        self._max_dyn_blob_len = blob_cap
        # Whether any limit is tighter than the ceiling the format already
        # enforces. At the ceiling a limit cannot fire — a longer value is
        # INVALID before the check is reached — so the header walk can skip the
        # whole block, which otherwise costs a subtype test, an attribute load
        # and a comparison for every string, blob and array it passes.
        self._capped = (
            arr_cap < ARRAY_MAX or str_cap < FIXLEN_MAX or blob_cap < FIXLEN_MAX
        )
        # The reassembly buffer, and the span of it currently holding a
        # construct that spans a chunk boundary.
        # A receiver-limit rejection, latched. §6.3 calls LimitExceeded "a
        # terminal, receiver-local policy rejection", so once one is reached the
        # decode is over: every later feed re-raises it and consumes nothing.
        # It is not INVALID -- the bytes are well formed and decode under a
        # looser limit -- so it rides the error channel rather than the status,
        # which §6.3 names as one of the two permitted ways to surface it.
        self._limit: SofaLimitError | None = None
        self._rstart = 0
        self._rend = 0
        # There is exactly one reassembly shape, and it never grows (§6.6.2).
        # A bytearray, not any writable buffer: both engines index it directly,
        # and the accelerator reaches its bytes through PyByteArray_AS_STRING.
        # Widening this would mean two buffer protocols where §5.3 wants one
        # behaviour.
        if reassembly is None:
            self._rbuf: Any = bytearray(DEFAULT_REASSEMBLY)
        elif isinstance(reassembly, bytearray):
            self._rbuf = reassembly
        elif isinstance(reassembly, int) and not isinstance(reassembly, bool):
            if reassembly < 16:
                raise SofaArgumentError(
                    f"reassembly={reassembly} is too small; 16 bytes is the "
                    "least that can hold a construct spanning a chunk"
                )
            self._rbuf = bytearray(reassembly)
        else:
            raise SofaArgumentError(
                "reassembly must be a bytearray, a byte count, or omitted"
            )
        self._buf: bytes | bytearray = b""
        self._bufsrc: Any = None
        self._bufview: Any = None
        # len(self._buf), kept in step with it. The buffer only ever changes in
        # feed() and reset(), while the walk asks for its length constantly —
        # 5.2 len() calls per field on the composite workload, each a builtin
        # call. Holding the number is what the native engine already does.
        self._n = 0
        self._pos = 0
        self._depth = 0
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

        # §5.3.1: one decode surface, and the table is reached *through* it.
        # A handler declares its destinations once, from
        # :meth:`sofab.Visitor.destinations`; ``binding=`` is the constructor
        # shorthand for a handler that declares exactly that and nothing else.
        # Either way there is one handler object, one walk and one set of rules
        # — the map only says *where* a value goes, never how it is decoded.
        table: Binding | None = binding
        if (
            table is None
            and visitor is not None
            and type(visitor).destinations is not Visitor.destinations
        ):
            # Asked once, and only of a handler that overrides it — a decoder is
            # built per message on the one-shot path, and a call that always
            # answers None is a call for nothing.
            declared = visitor.destinations()
            if declared is not None:
                table, words, objects = declared
        self._visitor: Visitor | None = visitor
        self._bmap: dict[int, Entry] | None = None
        self._bstack: list[Any] = [None] * MAX_DEPTH
        self._bsp = 0
        self._objects = objects
        self._wu: Any = None
        self._wq: Any = None
        self._wd: Any = None
        self._resume_entry: Entry | None = None
        if table is not None:
            self._bind_words(table, words, objects)
        # Whether the visitor overrides the two control hooks. Both default to a
        # no-op on the base class, and calling one that was never overridden
        # costs a Python call per field for nothing. ``_wants_field`` also
        # decides whether a Field object is built at all: it is the only thing
        # that receives one, and the typed hooks take an id.
        # Visitors suspended by a descent: a handler may answer
        # on_sequence_begin with another Visitor, and the sub-tree's events go
        # there until its end marker.
        #
        # MAX_DEPTH slots, filled here rather than grown on descent: §6.6 makes
        # construction the one place a codec may allocate, and "growing it
        # afterwards is forbidden even where the ceiling it grows towards is
        # correct". _next_wire refuses a message nesting past MAX_DEPTH before
        # any of these is written, so the slots are the ceiling and the index is
        # the depth.
        self._vstack: list[Any] = [None] * MAX_DEPTH
        self._vsp = 0
        self._wants_field = False
        self._wants_bound = False
        self._make_field = False
        self._wants_seq_begin = False
        self._wants_array_begin = False
        self._wants_blob_begin = False
        self._wants_string_begin = False
        self._wants_farray_begin = False
        self._wants_f32_bits = False
        self._wants_f32_array_bits = False
        if visitor is not None:
            self._bind_visitor(visitor)
        self._status = Status.COMPLETE
        self._error: SofaError | None = None
        self._resume_kind = _R_NONE
        self._running = False

    # --- resume transactions (CORELIB_PLAN §5.2) ----------------------------
    #
    # §5.2 requires the decoder to "suspend and resume at **any** byte boundary
    # without losing state": running out of bytes mid-construct is INCOMPLETE,
    # a first-class outcome the caller answers by supplying more bytes — not an
    # error that may consume anything. For this decoder that means every
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
    # * **every call is one field.** ``_keep`` is re-armed by the header walk
    #   and by
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
        # An overlong (>64-bit) varint is INVALID, not something to mask away on
        # return (§4.1.3/§6.3, issue #43) -- but only the tenth byte can be the
        # one to overflow. The first nine carry bits 0..62, so the check belongs
        # on the tenth, where exactly one payload bit still fits, and not on
        # every byte. An eleventh cannot follow: a byte that small is already a
        # terminator. Same shape as the array reader's inner loop.
        result = b & 0x7F
        shift = 7
        n = self._n
        while shift < 63:
            if pos >= n:
                self._pos = pos
                raise self._suspend("truncated varint")
            b = buf[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            if b < 0x80:
                self._pos = pos
                return result
            shift += 7
        if pos >= n:
            self._pos = pos
            raise self._suspend("truncated varint")
        b = buf[pos]
        pos += 1
        if b > 0x01:
            raise SofaDecodeError("overlong varint")
        self._pos = pos
        return result | (b << 63)

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

    def _span_exact(self, n: int) -> tuple[Any, int]:
        """Claim the next ``n`` bytes **in place**: the buffer they live in and
        the offset they start at, with nothing copied out.

        For a consumer that is done with them before ``feed`` returns — §6's
        chunk lifetime is what makes that safe, and §6.6 is why it is worth
        having: a payload on its way into a destination the caller already owns
        must not be copied into anything the wire sizes on the way.
        """
        pos = self._pos
        end = pos + n
        if end > self._n:
            raise self._suspend("truncated payload")
        self._pos = end
        return self._buf, pos

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
        # ZigZag is folded in on **both** paths. The list path used to hand raw
        # values back for the caller to transform in a second pass, and that pass
        # was a second list the wire sized -- two wire-sized allocations to
        # deliver one array (§6.6). Decoding the element where it is already in
        # hand costs the same arithmetic and allocates nothing extra.
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
        # ``for``, not a hand-rolled counter: range() advances the index in the
        # interpreter's own loop instead of costing a compare and an add per
        # element.
        for i in range(count):
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
                elif zigzag:
                    append((b >> 1) ^ -(b & 1))
                else:
                    append(b)
                continue
            # A multi-byte element. The 64-bit bound of §4.1 applies to it like
            # to any other varint: payload bits landing at bit >= 64 are
            # unrepresentable and the encoding is INVALID, and masking them off
            # on return would silently corrupt the value instead (issue #64).
            #
            # Both of the tests that enforce it are out of the loop. A u64 fills
            # its first nine bytes with bits 0..62, so no byte before the tenth
            # can overflow, and the tenth carries only bit 63 — one payload bit,
            # which is what the `> 0x01` below tests. An eleventh byte cannot
            # follow, because a value that small is already a terminator.
            #
            # The per-byte bounds test is out of the loop too, on the outer test
            # that ten bytes are buffered — a varint is at most ten bytes, so
            # inside that window no read can run off the end. When fewer remain,
            # _varint takes the element and suspends properly if it is truncated.
            if n - pos >= 9:
                result = b & 0x7F
                shift = 7
                while shift < 63:
                    b = buf[pos]
                    pos += 1
                    result |= (b & 0x7F) << shift
                    if b < 0x80:
                        break
                    shift += 7
                else:
                    b = buf[pos]
                    pos += 1
                    if b > 0x01:
                        raise SofaDecodeError("overlong varint")
                    result |= b << 63
            else:
                self._pos = pos - 1  # replay this element from its first byte
                result = self._varint()
                buf = self._buf
                pos = self._pos
                n = self._n
            if bounded:
                x = (result >> 1) ^ -(result & 1) if zigzag else result
                if x < blo or x > bhi:
                    raise SofaDecodeError("array element outside declared width")
            if store:
                into[base + i] = (result >> 1) ^ -(result & 1) if zigzag else result
            elif zigzag:
                append((result >> 1) ^ -(result & 1))
            else:
                append(result)
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

    def _next_wire(self) -> int:
        """Parse one field header; return its wire type, or ``-1`` at clean EOF.

        The header half of the walk, and **no** :class:`Field` — building one is
        not free (~250 ns, so a 36-field message would spend ~9 us on them), and
        the walk cannot yet know whether anyone will be offered this field: a
        field the handler's destination map names never reaches ``on_field``.
        The one place that can know builds it, from ``_cur_id``,
        ``_cur_subtype`` and the pending tuple, where the consume paths already
        look for the size and count (:meth:`_build_field`).
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
        # the one or two bytes it usually returns, and the EOF test above has
        # already proved the first of them is there.
        #
        # Two bytes, not one: a header packs the wire type into the low 3 bits,
        # so it stays single-byte only while the id is below 16. Ids run past
        # that in any real schema — 53 of the 77 headers on the composite
        # workload are two bytes — and a fast path that misses two thirds of the
        # time is not one. Nothing longer than two bytes is handled here;
        # ``_varint`` owns the 64-bit bound and the overlong verdict (§4.1.3),
        # and a two-byte varint cannot reach either.
        header = buf[pos]
        if header < 0x80:
            self._pos = pos + 1
        elif pos + 1 < self._n and buf[pos + 1] < 0x80:
            header = (header & 0x7F) | (buf[pos + 1] << 7)
            self._pos = pos + 2
        else:
            header = self._varint()
        wtype = header & 0x07
        field_id = header >> 3
        # A decoder always has a binding or a visitor (the constructor refuses
        # one with neither), and both resolve fields by id without a Field, so
        # the id is always published here.
        self._cur_id = field_id
        self._cur_subtype = -1
        # The id is bounded by ID_MAX on every header without exception (§6.2),
        # including a sequence end whose id is otherwise discarded (§4.9): the
        # bound is on the id's value, so this must run before the wire-type
        # dispatch below, not inside the branches that use the id.
        if field_id > ID_MAX:
            raise SofaDecodeError(f"id {field_id} out of range")

        if wtype == _WT_FIXLEN:
            # Same one-byte fast path as the header above, but this byte is not
            # guaranteed to be buffered, so the bound is tested first. A length
            # word is one byte for any payload under 16 bytes.
            pos = self._pos
            n = self._n
            if pos < n:
                length_header = buf[pos]
                if length_header < 0x80:
                    self._pos = pos + 1
                elif pos + 1 < n and buf[pos + 1] < 0x80:
                    length_header = (length_header & 0x7F) | (buf[pos + 1] << 7)
                    self._pos = pos + 2
                else:
                    length_header = self._varint()
            else:
                length_header = self._varint()
            length = length_header >> 3
            subtype = length_header & 0x07
            # The two subtype families want different checks, and splitting on
            # them costs one comparison instead of four. STRING (2) and BLOB (3)
            # are variable-length, so only the format-wide ceiling binds them; a
            # truncated one is legitimately INCOMPLETE. FP32 (0) and FP64 (1)
            # carry one fixed width each, and a wrong one is malformed whatever
            # follows — so that INVALID must be reached here, at header time,
            # ahead of the INCOMPLETE a truncated payload would otherwise raise
            # (§7). Mirrors the eager element-width check on the fixlen-array
            # path below.
            if subtype >= _ST_STRING:
                if subtype > _ST_BLOB:
                    raise SofaDecodeError(f"invalid fixlen subtype {subtype}")
                if length > FIXLEN_MAX:
                    raise SofaDecodeError("fixlen length out of range")
            elif subtype == _ST_FP32:
                if length != 4:
                    raise SofaDecodeError("fp32 fixlen length must be 4")
            elif length != 8:
                raise SofaDecodeError("fp64 fixlen length must be 8")
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
            # schema-bounded — a binding entry's declared bound — and take
            # the cap off it.
            if self._capped:
                if subtype == _ST_STRING:
                    cap = self._max_dyn_string_len
                    if length > cap:
                        pending = (
                            _LIMIT,
                            f"string length {length} exceeds max_dyn_string_len {cap}",
                            pending,
                        )
                elif subtype == _ST_BLOB:
                    cap = self._max_dyn_blob_len
                    if length > cap:
                        pending = (
                            _LIMIT,
                            f"blob length {length} exceeds max_dyn_blob_len {cap}",
                            pending,
                        )
            self._pending = pending
            return wtype

        if wtype < _WT_FIXLEN:  # UNSIGNED (0) or SIGNED (1)
            self._pending = (_SCALAR, wtype)
            return wtype

        if wtype == _WT_SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            return wtype

        if wtype == _WT_SEQUENCE_START:
            if self._depth >= MAX_DEPTH:
                raise SofaDecodeError(f"nesting exceeds MAX_DEPTH={MAX_DEPTH}")
            self._depth += 1
            return wtype

        if wtype == _WT_ARRAY_UNSIGNED or wtype == _WT_ARRAY_SIGNED:
            count = self._varint()
            if count < 0 or count > ARRAY_MAX:
                raise SofaDecodeError(f"array count {count} out of range")
            pending = (_VARRAY, wtype, count)
            # Parked, not raised — see the fixlen branch above (§6.2.1).
            if self._capped and count > self._max_dyn_array_count:
                cap = self._max_dyn_array_count
                pending = (
                    _LIMIT,
                    f"array count {count} exceeds max_dyn_array_count {cap}",
                    pending,
                )
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
        if subtype > _ST_FP64:
            raise SofaDecodeError(f"invalid fixlen-array subtype {subtype}")
        # §4.8/§5.2: a fixlen array carries fp32 (element size 4) or fp64
        # (element size 8) — any other width is malformed. This INVALID verdict
        # must be reached at header time, before any payload read, so it takes
        # precedence over the INCOMPLETE a truncated payload would raise (§7).
        # Mirrors the eager element-width check on the scalar fixlen path above.
        # subtype is already narrowed to fp32/fp64, so these exact-width checks
        # bound elem_size completely — no separate FIXLEN_MAX check is needed.
        if subtype == _ST_FP32 and elem_size != 4:
            raise SofaDecodeError("fp32 fixlen-array element size must be 4")
        if subtype == _ST_FP64 and elem_size != 8:
            raise SofaDecodeError("fp64 fixlen-array element size must be 8")
        self._cur_subtype = subtype
        pending = (_FARRAY, subtype, count, elem_size)
        # Parked, not raised — see the fixlen branch above (§6.2.1).
        if self._capped and count > self._max_dyn_array_count:
            cap = self._max_dyn_array_count
            pending = (_LIMIT, f"array count {count} exceeds max_dyn_array_count {cap}", pending)
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
        if kind == _LIMIT:
            # A skipped field is not capped (#128). §6.2.1 enforces a receiver
            # limit "at the count/length header — before the allocation it is
            # meant to prevent", and a skip makes no allocation for it to
            # prevent: the payload is walked, never materialized, which is
            # exactly what §6.7.2's skip row means by "neither materializes nor
            # validates". So the parked verdict is dropped here and the real
            # value walked like any other — including a §7.3 tag mismatch, whose
            # payload "was never this field's value" and so cannot be measured
            # against a bound meant for one.
            #
            # Unwrapped before the walk rather than after it: should the payload
            # run out mid-skip the field stays pending, and the retry then
            # re-enters with the cap already gone instead of unwrapping twice.
            pending = pending[2]
            self._pending = pending
            kind = pending[0]
        if kind == _SCALAR:
            self._varint()
        elif kind == _FIXLEN:
            self._skip_exact(pending[2])
        elif kind == _VARRAY:
            self._skip_varints(pending[2])
        else:  # _FARRAY
            self._skip_exact(self._farray_nbytes(pending[2], pending[3]))
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
        depth, pending = self._depth, self._pending
        try:
            target = depth - 1
            while self._depth > target:
                # The walk discards every field it passes, so it asks for no
                # Field objects. Defensive: at EOF with an open sequence,
                # _next_wire itself raises "truncated: unbalanced sequence",
                # so it never returns -1 here.
                if self._next_wire() < 0:  # pragma: no cover
                    raise self._suspend("truncated sequence")
        except SofaIncompleteError:
            self._pos = self._keep = floor
            self._depth, self._pending = depth, pending
            raise

    # --- push-feed driver (CORELIB_PLAN §5.2) -------------------------------
    #
    # The other half of §5.2's "push-feed / pull-read" model. The caller hands
    # over chunks; this side walks the fields and binds each value straight into
    # the destination the handler declared — no callback per field when a
    # binding names it, and no callback per array *element* ever.
    #
    # Resumption is the decoder's transaction model, one level up. Each
    # individual call is already all-or-nothing (see _suspend), so the only
    # thing the driver has to add is *which* call was in flight when the bytes
    # ran out: restarting the field walk instead would skip the half-read value
    # the retry is supposed to finish. That is what _resume_kind records, and it
    # is why the value read below is never re-entered through the header walk.

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
            raise SofaArgumentError("feed() is not re-entrant")
        if self._limit is not None:
            raise self._limit
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
        # Sets _pos itself: with a carry the walk resumes where the held bytes
        # start, which is not the front of the buffer.
        self._reassemble(data)
        self._keep = self._pos
        self._running = True
        try:
            if self._drive_push():
                self._status = Status.INCOMPLETE
                return Status.INCOMPLETE
        except SofaIncompleteError:
            self._status = Status.INCOMPLETE
            return Status.INCOMPLETE
        except SofaLimitError as exc:
            # Raised by this decoder's own check, or by a handler deciding on an
            # index the codec surfaced (§6.2.1). Either way the decode is over.
            self._limit = exc
            self._error = exc
            raise
        except SofaDecodeError as exc:
            self._error = exc
            self._status = Status.INVALID
            return Status.INVALID
        finally:
            self._running = False
            self._retain()
        self._status = Status.COMPLETE
        return Status.COMPLETE

    def _reassemble(self, data: Any) -> None:
        """Put ``data`` where the walk can reach it, using only the caller's
        reassembly buffer (§6.6).

        With nothing carried the chunk is used where it lies, for the duration
        of the call; :meth:`_retain` copies out whatever survives it. With a
        carry the chunk is appended **into the caller's buffer**, which is never
        grown — a construct that does not fit is refused (§6.6.2), which is what
        lets a caller bound a decode's memory by construction.
        """
        held = self._rend - self._rstart
        if not held:
            buf = data if isinstance(data, bytes) else bytes(data)
            self._buf = buf
            self._n = len(buf)
            self._rstart = self._rend = 0
            self._pos = 0
            return
        r = self._rbuf
        n = len(data)
        if self._rend + n > len(r):
            # Slide what is held back to the front and try again; only then is
            # the buffer genuinely too small.
            if self._rstart:
                r[:held] = r[self._rstart : self._rend]
                self._rstart, self._rend = 0, held
            if held + n > len(r):
                raise SofaArgumentError(
                    f"reassembly buffer holds {len(r)} bytes; the construct "
                    f"spanning this chunk needs {held + n}"
                )
        r[self._rend : self._rend + n] = data
        self._rend += n
        self._buf = r
        self._n = self._rend
        self._pos = self._rstart

    def _retain(self) -> None:
        """Keep whatever this feed did not consume, and let the chunk go.

        Runs on the way out of every feed. The unconsumed tail is the caller's
        chunk until this copies it into the caller's *reassembly* buffer, which
        is what makes §6's chunk-lifetime promise true: once ``feed`` returns,
        the decoder holds nothing of what was handed to it.
        """
        # The view goes first, before any branch can return: it points into
        # ``_buf``, and §6 ends this chunk's life with the call.
        self._bufsrc = None
        self._bufview = None
        if self._status is Status.INVALID or self._limit is not None:
            # Terminal (§5.2.3, §6.3): nothing will resume, so there is nothing
            # to keep. Dropping also keeps the real verdict on the error channel
            # — a reassembly buffer complaining about the tail of a message the
            # decoder has already refused would bury it.
            self._rstart = self._rend = 0
            self._buf = b""
            self._n = 0
            self._pos = 0
            return
        r = self._rbuf
        carry = self._n - self._pos
        if not carry:
            self._rstart = self._rend = 0
            self._buf = b""
            self._n = 0
            self._pos = 0
            return
        if self._buf is r:
            # Already in place; remember where, so the next chunk appends
            # instead of re-copying what is held (a megabyte fed in kibibytes
            # would otherwise cost a copy of the whole carry per chunk).
            self._rstart = self._pos
            return
        if carry > len(r):
            raise SofaArgumentError(
                f"reassembly buffer holds {len(r)} bytes; the construct "
                f"spanning this chunk needs {carry}"
            )
        r[:carry] = self._buf[self._pos : self._n]
        self._rstart, self._rend = 0, carry
        self._buf = r
        self._n = carry
        self._pos = 0

    def reset(self) -> None:
        """Forget the stream and start a new message, keeping the handler and
        its destinations. Lets one decoder serve many messages without rebuilding
        the binding — the destinations are the caller's to clear (or not: a slot
        the next message does not write keeps whatever is in it, which is how
        absence is reported)."""
        self._bufsrc = None
        self._bufview = None
        self._buf = b""
        self._n = 0
        self._pos = 0
        self._rstart = self._rend = 0
        self._depth = 0
        self._pending = None
        self._keep = 0
        self._status = Status.COMPLETE
        self._error = None
        self._limit = None
        if self._vsp:
            # A descent left mid-message: the handler the caller gave us is the
            # one at the bottom of the stack. The slots themselves are kept —
            # they are construction-time state (§6.6), so reset rewinds the
            # index rather than dropping the list.
            self._visitor = self._vstack[0]
            self._vsp = 0
            self._bind_visitor(self._visitor)
        self._resume_kind = _R_NONE
        self._resume_entry = None
        if self._wu is not None:
            # A descent left mid-message: the map the caller declared is the one
            # at the bottom. The slots stay — they are construction-time state
            # (§6.6) — so reset rewinds the index rather than dropping the list.
            self._bmap = self._bstack[0] if self._bsp else self._bmap
            while self._bsp:
                self._bsp -= 1
                self._bmap = self._bstack[self._bsp]
                self._bstack[self._bsp] = None

    def _bind_words(self, table: Binding, words: Any, objects: Any) -> None:
        """Check the caller's storage against the map and take three views over
        it — done once, at construction, because §6.6 makes construction the one
        place a decode's storage is settled."""
        if words is None:
            raise SofaArgumentError("a binding needs a words buffer")
        raw = memoryview(words)
        if raw.readonly:
            raise SofaArgumentError("the words buffer must be writable")
        raw = raw.cast("B")
        if raw.nbytes % 8:
            raise SofaArgumentError("the words buffer must be a multiple of 8 bytes")
        if raw.nbytes < table.tree_words_required * 8:
            raise SofaArgumentError(
                f"words buffer holds {raw.nbytes // 8} slots, "
                f"the binding needs {table.tree_words_required}"
            )
        if objects is None:
            if table.tree_objects_required:
                raise SofaArgumentError("a binding with string/blob fields needs objects")
        elif len(objects) < table.tree_objects_required:
            raise SofaArgumentError(
                f"objects holds {len(objects)} entries, "
                f"the binding needs {table.tree_objects_required}"
            )
        table.freeze()
        self._wu = raw.cast("Q")
        self._wq = raw.cast("q")
        self._wd = raw.cast("d")
        self._bmap = table._by_id

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
                if rk == _R_VISIT:
                    if self._resume_entry is not None:
                        self._mapped_field(self._resume_entry)
                    else:
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
        #
        # It is a *per-handler* flag, so descending into a child handler and
        # returning from one both change it: kept in a local for the loop's
        # sake, it is re-read at each of the two places the handler changes.
        # Without that a child overriding ``on_field`` was never asked (and the
        # walk asserted on the Field nobody had built).
        make_field = self._make_field
        while True:
            t = self._next_wire()
            if t < 0:
                return False

            if t == _WT_SEQUENCE_END:
                if self._bsp:
                    self._bsp -= 1
                    self._bmap = self._bstack[self._bsp]
                    self._bstack[self._bsp] = None
                # The end belongs to whoever was handling the scope, so a child
                # hears its own scope close before it is popped.
                if visitor is not None:
                    visitor.on_sequence_end()
                if self._vsp:
                    self._vsp -= 1
                    visitor = self._visitor = self._vstack[self._vsp]
                    self._bind_visitor(visitor)
                    # The flags are the *handler's*, so they change with it.
                    make_field = self._make_field
                continue

            # --- the destination map, consulted once per field ---------------
            # Two questions, both of them the caller's: where does this value go,
            # and what does the schema declare for it. No rule below branches on
            # the answer — the walk, the cap, the bound, the §7.3 test, the UTF-8
            # check, the element width and the resume transaction are the same
            # code for a mapped field and an unmapped one, which is what makes
            # this one surface rather than two (§5.3.1).
            bmap = self._bmap
            entry = bmap.get(self._cur_id) if bmap is not None else None
            if entry is not None and (
                t != entry.wt
                or (entry.st is not None and self._cur_subtype != entry.st)
            ):
                # §7.3: the wire tag contradicts what the schema declared for
                # this id. Treated exactly like an unknown id — skipped, slot
                # untouched, decode stays COMPLETE.
                entry = None
                if t != _WT_SEQUENCE_START:
                    continue

            if t == _WT_SEQUENCE_START:
                if entry is not None:
                    child = entry.child
                    assert child is not None
                    self._bstack[self._bsp] = bmap
                    self._bsp += 1
                    self._bmap = child._by_id
                    c = entry.count_at
                    if c >= 0:
                        self._wu[c] = self._wu[c] + 1
                    continue
                if visitor is None:
                    try:
                        self._skip_sequence()
                    except SofaIncompleteError:
                        self._resume_kind = _R_SKIP
                        raise
                    continue
                answer = (
                    visitor.on_sequence_begin(self._cur_id)
                    if self._wants_seq_begin
                    else None
                )
                if answer is not False:
                    if bmap is not None:
                        # §4.9 opens a fresh id scope, so the enclosing map must
                        # not match inside it.
                        self._bstack[self._bsp] = bmap
                        self._bsp += 1
                        self._bmap = None
                    if isinstance(answer, Visitor):
                        # The handler named someone else for this sub-tree.
                        self._vstack[self._vsp] = visitor
                        self._vsp += 1
                        visitor = self._visitor = answer
                        self._bind_visitor(answer)
                        make_field = self._make_field
                    continue
                try:
                    self._skip_sequence()
                except SofaIncompleteError:
                    self._resume_kind = _R_SKIP
                    raise
                continue

            if entry is not None:
                # One call, so the walk keeps the shape it has for every other
                # field: a mapped field's bound and store live in _mapped_field.
                try:
                    self._mapped_field(entry)
                except SofaIncompleteError:
                    self._resume_kind = _R_VISIT
                    self._resume_entry = entry
                    raise
                continue
            if make_field:
                # ``_make_field`` is a handler's flag, so it is only ever set
                # where there is one. The Field is built HERE — a field the map
                # named has already `continue`d above, so this is the only place
                # one can be observed, and the only place one is made.
                assert visitor is not None
                if visitor.on_field(self._build_field(t)) is False:
                    # Skipped: the value stays pending and the next header walk
                    # discards it, which suspends in exactly the same place an
                    # explicit skip would and costs one call less. Nothing is
                    # materialized and nothing is validated (§6.7.2), so no cap
                    # and no schema bound is answered for it either (§6.2.1).
                    continue
            if self._wants_bound:
                # Independent of ``make_field`` now: this hook takes integers,
                # so a handler that declares schema bounds and nothing else
                # builds no Field at all.
                assert visitor is not None
                self._schema_bound(visitor, self._cur_id)
            if visitor is None:
                # Nobody wants it: the value stays pending and the next header
                # walk discards it.
                continue
            try:
                self._visit_value(visitor, t)
            except SofaIncompleteError:
                self._resume_kind = _R_VISIT
                self._resume_entry = None
                raise

    def _schema_bound(self, visitor: Visitor, fid: int) -> None:
        """Ask the handler what the **schema** bounds this field's count/length
        at, and settle the field against the answer (§6.2.1).

        One site, for one surface. A declared bound does two things and this is
        where both happen, at the count/length header, before a payload byte is
        read or any storage is written:

        * a wire count/length above it is INVALID (MESSAGE_SPEC §7.1) — the
          message contradicts the schema, which is a statement about validity;
        * the receiver-side cap stops applying, because §6.2.1 forbids applying
          one "to a field the schema already bounds".

        Only string, blob and array fields carry a count or a length to bound;
        a scalar has neither, so none is asked for.

        **The tag goes with it**, and that is what keeps this route level with
        the table one. Every other hook is reached only for the kind it names —
        ``on_string_begin`` for a string, ``on_array_begin`` for an integer
        array — so the decoder has matched the wire's tag before calling it.
        This hook spans string, blob and both array kinds, so it has not; an id
        the handler bounds can arrive under a tag the handler never declared for
        it, and MESSAGE_SPEC §7.3 says such a field is skipped like an unknown
        id. Told only the id, a handler answers its bound for someone else's
        field and turns a §7.3 skip into an INVALID — the divergence issue #133
        measured against a ``Binding``, which gets the tag test run for it above.
        Told the tag, it answers ``-1`` and the two routes agree again.

        Everything passed is an integer or an enum member recovered by index, so
        overriding this hook still costs no object per field.
        """
        pending = self._pending
        assert pending is not None  # every value field parks one at its header
        real = pending[2] if pending[0] == _LIMIT else pending
        kind = real[0]
        if kind == _SCALAR:
            return
        if kind == _FIXLEN:
            st = real[1]
            if st < _ST_STRING:
                return  # an fp32/fp64 payload: a fixed width, nothing to bound
            wt: WireType = _WTM_FIXLEN
            sub: FixlenSubtype | None = _ST[st]
        elif kind == _VARRAY:
            # (_VARRAY, wtype, count): the wire type IS the element signedness,
            # and an integer array carries no subtype word at all.
            wt = _WT[real[1]]
            sub = None
        else:  # _FARRAY -- (_FARRAY, subtype, count, elem_size)
            wt = _WTM_ARRAY_FIXLEN
            sub = _ST[real[1]]
        self._settle_bound(visitor.on_schema_bound(fid, real[2], wt, sub))

    def _settle_bound(self, declared: int) -> None:
        """**The** site the schema bound is applied at — the one place in this
        file that knows what a declared count/length means.

        Both sources reach it: a destination map's entry, and a handler's
        :meth:`sofab.Visitor.on_schema_bound`. That is what keeps the two from
        disagreeing about a field, which is exactly the defect §5.3.1's rationale
        names and the one `A2-0147` measured.

        A declared bound does two things, both here, at the count/length header,
        before a payload byte is read or any storage is written:

        * a wire count/length above it is INVALID (MESSAGE_SPEC §7.1) — the
          message contradicts the schema, which is a statement about validity;
        * the receiver-side cap stops applying, because §6.2.1 forbids applying
          one "to a field the schema already bounds".
        """
        if declared < 0:
            return
        pending = self._pending
        assert pending is not None
        capped = pending[0] == _LIMIT
        real = pending[2] if capped else pending
        # Only a count- or length-bearing field can reach here with a bound: the
        # hook half returns early for a scalar and for an fp32/fp64 payload, and
        # a table entry declares one only for an array, a string or a blob (see
        # ``Entry.declared``) — with the §7.3 tag test ahead of it, so the kind
        # on the wire is the kind the entry declared.
        what = "fixlen length" if real[0] == _FIXLEN else "array count"
        n = real[2]
        if n > declared:
            raise SofaDecodeError(
                f"{what} {n} exceeds the {declared} the schema declares"
            )
        if capped:
            # Spent: the schema bounds this field, so the cap never governed it.
            self._pending = real

    def _mapped_field(self, e: Entry) -> None:
        """A field the handler's declared destination map names.

        **Nothing is decided here that is not decided for every other field.**
        The schema bound comes off the map instead of off a hook, but it is the
        same number reaching the same rule (:meth:`_settle_bound`) — which is
        why a mapped field and a hooked one cannot disagree about it, the
        divergence ``A2-0147`` measured. The receiver cap, the §7.3 tag test,
        the element width and the resume transaction were settled before this is
        reached, by the code an unmapped field runs too. What is left is the
        assignment a typed hook would otherwise have made:
        ``words[at] = value`` instead of ``visitor.on_unsigned(id, value)``.

        Consumes nothing on the suspension path, like every other read, so the
        retry redoes the whole value — including refilling a partly written
        array from element zero (§5.2).
        """
        self._settle_bound(e.declared)
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
            if e.into:
                # §6.6.3's third shape: the destination was declared before the
                # decode began and the slot already holds it, so the payload is
                # copied in and nothing is sized from the wire.
                if pending[0] == _LIMIT:
                    # Spent, for the same reason on_string_begin spends it: the
                    # buffer is the caller's own storage, chosen after the schema
                    # was known, so there is no allocation of this decoder's left
                    # for the cap to prevent (§6.2.1). What bounds the field is
                    # the buffer, and a payload past it is refused below.
                    self._pending = pending = pending[2]
                got = pending[2]
                dst = self._objects[at]  # type: ignore[index]
                if k == K_BYTES:
                    self._take_blob_into(dst, got, "Binding.blob_into")
                else:
                    self._take_string_into(dst, got, "Binding.string_into")
            else:
                if pending[0] == _LIMIT:
                    # The schema left this field unbounded, so the cap still
                    # governs it and has already rejected it (§6.2.1). The hook
                    # path reaches the same verdict on its way past the parked
                    # tuple; the store goes straight to the payload, so it is
                    # raised here.
                    raise SofaLimitError(pending[1])
                if k == K_BYTES:
                    self._objects[at] = self._take_fixlen_matched(  # type: ignore[index]
                        pending[2]
                    )
                else:
                    self._objects[at] = self._take_text_matched(  # type: ignore[index]
                        pending[2]
                    )
        else:
            pending = self._pending
            assert pending is not None
            got = pending[2]
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
                width = 4 if k == K_ARRAY_FLOAT32 else 8
                buf, off = self._span_exact(self._farray_nbytes(got, pending[3]))
                _core.unpack_farray_into(self._wd, at, buf, got, width, off)
            self._pending = None  # committed only once the payload is in hand
        c = e.count_at
        if c >= 0:
            self._wu[c] = got

    def _take_text_matched(self, size: int) -> str:
        """The pending ``string`` payload as a ``str``, transcoded straight out
        of the buffer it was fed into.

        The ``str`` is the value a handler that asked for one gets, and the wire
        sizes it — §6.6.3's materialized aggregate, and the gap this port's
        README itemises. What is **not** here is the ``bytes`` copy that used to
        be made on the way to it: that was the codec's own scratch, sized by the
        wire, and no caller ever saw it. A megabyte string cost a megabyte twice.

        What is left is one ``memoryview``, which is §6.6.2's language-forced
        handle exactly: it carries no message bytes of its own, and it costs the
        same over ten bytes as over ten megabytes. One is kept per buffer rather
        than made per field, and :meth:`_retain` drops it when the feed ends, so
        it never outlives the chunk it points into (§6).

        The visitor's ``on_string`` branch in :meth:`_visit_value` carries these
        same lines **inlined**, for the reason the rest of that branch is inlined:
        a string field is the commonest thing on the wire and the delegation
        showed up in the instruction count (~3% of a fifty-string message). They
        are the same lines and must stay so — nothing here is a decode *rule*,
        which is what §5.3.1 requires a single implementation of; the bound, the
        cap, the tag test and the resume transaction all ran before either is
        reached.
        """
        self._keep = pos = self._pos
        end = pos + size
        if end > self._n:
            raise self._suspend("truncated payload")
        buf = self._buf
        if buf is not self._bufsrc:
            self._bufsrc = buf
            self._bufview = memoryview(buf)
        self._pos = end
        self._pending = None  # committed once the payload is in hand (§5.2)
        try:
            return str(self._bufview[pos:end], "utf-8")
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def _take_fixlen_matched(self, length: int) -> bytes:
        """:meth:`_take_scalar_matched` for a fixlen payload."""
        self._keep = self._pos
        data = self._read_exact(length)
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return data

    def _build_field(self, t: int) -> Field:
        """The :class:`sofab.Field` :meth:`sofab.Visitor.on_field` is handed.

        Built **here**, not in the header walk, and only for a field that is
        actually going to be offered: a field the handler's destination map
        names never reaches ``on_field``, so building one for it was an object
        the caller could not observe. Everything it carries is already on the
        decoder — the id, the fixlen subtype, and the pending tuple's size or
        count — so nothing is re-parsed to get it.
        """
        pending = self._pending
        assert pending is not None  # every value field parks one at its header
        real = pending[2] if pending[0] == _LIMIT else pending
        kind = real[0]
        if kind == _SCALAR:
            return Field(self._cur_id, _WT[t])
        if kind == _FIXLEN:
            return Field(
                self._cur_id, WireType.FIXLEN, size=real[2],
                subtype=FixlenSubtype(real[1]),
            )
        if kind == _VARRAY:
            return Field(self._cur_id, _WT[t], count=real[2])
        return Field(
            self._cur_id,
            WireType.ARRAY_FIXLEN,
            count=real[2],
            size=real[3],
            subtype=FixlenSubtype(real[1]),
        )

    def _take_scalar_matched(self) -> int:
        """Consume the pending scalar. The caller has already matched the whole
        tag, so this only consumes — and the result is an ``int``, not
        ``int | None``."""
        self._keep = self._pos
        value = self._varint()
        self._pending = None  # committed only once the value is in hand (§5.2)
        return value

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
        if t == _WT_UNSIGNED:
            visitor.on_unsigned(fid, self._take_scalar_matched())
        elif t == _WT_SIGNED:
            raw = self._take_scalar_matched()
            visitor.on_signed(fid, (raw >> 1) ^ -(raw & 1))
        elif t == _WT_FIXLEN:
            # Folded in rather than delegated. A string field is the commonest
            # thing on the wire and used to cost four nested calls to deliver —
            # _visit_value, _visit_fixlen, _take_fixlen_matched, _read_exact —
            # three of which only chose the next one. Everything they did is
            # here: the parked-cap check, the resume transaction, the bounds
            # test, and the one copy out of the buffer.
            capped = pending[0] == _LIMIT
            # Read through a parked cap rather than raising on sight (#128): the
            # subtype and the length below are facts about the field, and the
            # handler needs both before anyone can say whether the cap applies.
            real = pending[2] if capped else pending
            if self._wants_string_begin and real[1] == _ST_STRING:
                # Same bargain as on_blob_begin below, and the same reasoning:
                # the handler is told the announced byte length before a byte is
                # copied, and a buffer it hands back is one it sized itself.
                dst = visitor.on_string_begin(fid, real[2])
                if dst is not None:
                    pending = real
                    self._pending = real
                    self._take_string_into(dst, real[2])
                    return
            if self._wants_blob_begin and real[1] == _ST_BLOB:
                # Offered before the cap is answered, and on purpose. §6.2.1's
                # limit exists to stop the *sender* dictating the *receiver's*
                # allocation; a handler that hands back a buffer has chosen the
                # size itself, so there is no allocation of this decoder's left
                # for the cap to prevent. The handler is told the announced
                # length and is free to refuse it — that call is its own, and
                # this is where it gets to make it.
                dst = visitor.on_blob_begin(fid, real[2])
                if dst is not None:
                    # The cap is spent: unwrap, or the copy below and the resume
                    # behind it would look for the length in the wrapper.
                    pending = real
                    self._pending = real
                    self._take_blob_into(dst, real[2])
                    return
            if capped:
                # No destination came back, so this decoder is the one that
                # would build the ``bytes``/``str`` — and the only size it could
                # build one from is the wire's. That is the allocation §6.2.1 is
                # about, so the cap speaks.
                raise SofaLimitError(pending[1])
            subtype = pending[1]
            if subtype == _ST_STRING:
                # Folded in rather than delegated, like the rest of this branch:
                # a string field is the commonest thing on the wire, and the
                # ``bytes``-free transcode is four lines. It is the SAME four as
                # _take_text_matched's, and it must stay so -- see there for what
                # they are for and why the memoryview is the whole cost.
                self._keep = pos = self._pos
                end = pos + pending[2]
                if end > self._n:
                    raise self._suspend("truncated payload")
                buf = self._buf
                if buf is not self._bufsrc:
                    self._bufsrc = buf
                    self._bufview = memoryview(buf)
                self._pos = end
                self._pending = None  # committed once the payload is in hand
                try:
                    text = str(self._bufview[pos:end], "utf-8")
                except UnicodeDecodeError as exc:
                    raise SofaDecodeError("invalid UTF-8 in string field") from exc
                visitor.on_string(fid, text)
                return
            self._keep = pos = self._pos
            end = pos + pending[2]
            if end > self._n:
                raise self._suspend("truncated payload")
            buf = self._buf
            # Always a real ``bytes`` — see _read_exact for why the bytearray
            # case takes the memoryview.
            data = buf[pos:end] if type(buf) is bytes else bytes(memoryview(buf)[pos:end])
            self._pos = end
            self._pending = None  # committed once the payload is in hand (§5.2)
            if subtype == _ST_BLOB:
                visitor.on_bytes(fid, data)
            elif subtype == _ST_FP32:
                # _next_wire already refused any other width for these two, so
                # the payload is exactly 4 or 8 bytes.
                if self._wants_f32_bits:
                    # §6.5: a bit-exact consumer takes the wire bits, never the
                    # widened value.
                    visitor.on_float32_bits(fid, _core.unpack_u32(data))
                else:
                    visitor.on_float32(fid, _core.unpack_f32(data))
            else:
                visitor.on_float64(fid, _core.unpack_f64(data))
        else:
            # An array kind. Read through a parked cap for the count and the
            # subtype below — they are facts about the field, and both the
            # handler and the verdict need them (#128); _visit_varints and
            # _take_farray_values each answer the cap themselves, once they know
            # whose storage the elements are headed for.
            real = pending[2] if pending[0] == _LIMIT else pending
            if t == _WT_ARRAY_UNSIGNED:
                self._visit_varints(visitor, fid, t, real[2], False)
            elif t == _WT_ARRAY_SIGNED:
                self._visit_varints(visitor, fid, t, real[2], True)
            elif real[1] == _ST_FP32:
                if self._wants_f32_array_bits:
                    self._visit_farray_bits(visitor, fid, pending)
                elif not self._visit_farray_into(visitor, fid, pending, 4):
                    visitor.on_float32_array(fid, self._take_farray_values(pending, 4))
            elif not self._visit_farray_into(visitor, fid, pending, 8):
                visitor.on_float64_array(fid, self._take_farray_values(pending, 8))

    def _take_blob_into(self, dst: Any, size: int, who: str = "on_blob_begin") -> None:
        """Copy a blob's payload into the caller's buffer (§6.6.3).

        No ``bytes`` is built on the way -- which is the point: the only size a
        codec could build one from is the wire's, and a megabyte blob would cost
        a megabyte allocation per message.

        ``who`` names the route the buffer arrived by, so the §6.3 refusal reads
        the same whether the caller answered :meth:`sofab.Visitor.on_blob_begin`
        per field or declared the slot once with :meth:`sofab.Binding.blob_into`.
        It is the **only** difference between the two: one rule, one
        implementation, whichever way the destination was stated (§5.3.1).
        """
        view = _writable(dst, who)
        if view.itemsize != 1:
            raise SofaArgumentError(
                f"{who}'s destination must hold single bytes"
            )
        if view.nbytes < size:
            raise SofaArgumentError(
                f"{who} gave {view.nbytes} bytes for a blob of {size}"
            )
        self._keep = pos = self._pos
        end = pos + size
        if end > self._n:
            raise self._suspend("truncated payload")
        view[:size] = memoryview(self._buf)[pos:end]
        self._pos = end
        self._pending = None  # committed once the payload is in hand (§5.2)

    def _take_string_into(
        self, dst: Any, size: int, who: str = "on_string_begin"
    ) -> None:
        """Validate a string's payload and copy it into the caller's buffer.

        §6.6.3's destination route for the third aggregate. No ``str`` is built
        on the way — the only size a codec could build one from is the wire's —
        but the bytes are still validated, because §6.7.2 makes a field the
        handler *reads* both materialized and validated.

        ``who`` names the route the buffer arrived by; see :meth:`_take_blob_into`.
        """
        view = _writable(dst, who)
        if view.itemsize != 1:
            raise SofaArgumentError(
                f"{who}'s destination must hold single bytes"
            )
        if view.nbytes < size:
            raise SofaArgumentError(
                f"{who} gave {view.nbytes} bytes for a string of {size}"
            )
        self._keep = pos = self._pos
        end = pos + size
        if end > self._n:
            raise self._suspend("truncated payload")
        if not _core.utf8_valid(self._buf, pos, size):
            # INVALID before the destination is touched: a caller that asked for
            # the bytes never sees a half-written buffer behind a verdict.
            raise SofaDecodeError("invalid UTF-8 in string field")
        view[:size] = memoryview(self._buf)[pos:end]
        self._pos = end
        self._pending = None  # committed once the payload is in hand (§5.2)

    def _visit_farray_into(
        self, visitor: Visitor, fid: int, pending: tuple[Any, ...], width: int
    ) -> bool:
        """Offer a fixlen array's elements a destination, and fill it if one
        comes back. ``False`` means the handler wants the list instead.
        """
        if not self._wants_farray_begin:
            return False
        real = pending[2] if pending[0] == _LIMIT else pending
        count = real[2]
        dst = visitor.on_float_array_begin(
            fid, FixlenSubtype.FP32 if width == 4 else FixlenSubtype.FP64, count
        )
        if dst is None:
            return False
        # The cap is spent: the destination is the handler's own storage, so
        # there is no allocation of this decoder's left for it to prevent.
        self._pending = real
        view = _writable(dst, "on_float_array_begin")
        if view.itemsize != 8:
            raise SofaArgumentError(
                "on_float_array_begin's destination must hold 8-byte items; a "
                "Python float is a double, and so is every value written here"
            )
        if view.nbytes < count * 8:
            raise SofaArgumentError(
                f"on_float_array_begin returned {view.nbytes // 8} slots for an "
                f"array of {count}"
            )
        self._keep = self._pos
        buf, off = self._span_exact(self._farray_nbytes(count, real[3]))
        self._pending = None  # committed once the payload is in hand (§5.2)
        # Through bytes: memoryview.cast refuses to go between two non-byte
        # formats directly, and the caller may hand back any 8-byte item type.
        target = view.cast("B").cast("d")
        try:
            _core.unpack_farray_into(target, 0, buf, count, width, off)
        finally:
            target.release()
        return True

    def _bind_visitor(self, visitor: Visitor) -> None:
        """Take the hook flags off ``visitor``'s type.

        Computed once per handler rather than per field: which hooks are
        overridden cannot change between fields, and a descent into a child
        handler is the only thing that changes the answer.
        """
        cls = type(visitor)
        self._wants_field = cls.on_field is not Visitor.on_field
        self._wants_bound = cls.on_schema_bound is not Visitor.on_schema_bound
        # A Field is built for the ONE hook that takes one. Every other hook —
        # on_schema_bound included — takes integers and interned enum members,
        # so declaring a schema bound costs no object per field. That is what
        # lets generated code drop on_field entirely: with the wire's tag on
        # this hook, there is nothing left for on_field to pre-filter (#133).
        self._make_field = self._wants_field
        self._wants_seq_begin = cls.on_sequence_begin is not Visitor.on_sequence_begin
        self._wants_array_begin = cls.on_array_begin is not Visitor.on_array_begin
        self._wants_blob_begin = cls.on_blob_begin is not Visitor.on_blob_begin
        self._wants_string_begin = cls.on_string_begin is not Visitor.on_string_begin
        self._wants_farray_begin = (
            cls.on_float_array_begin is not Visitor.on_float_array_begin
        )
        # §6.5's raw fp32 channel, opt-in by override: a handler that overrides
        # it is a bit-exact consumer and gets the wire bits instead of the
        # widened value.
        self._wants_f32_bits = cls.on_float32_bits is not Visitor.on_float32_bits
        self._wants_f32_array_bits = (
            cls.on_float32_array_bits is not Visitor.on_float32_array_bits
        )

    def _visit_varints(
        self, visitor: Visitor, fid: int, wtype: int, count: int, zigzag: bool
    ) -> None:
        """Deliver an integer array to ``visitor``, by whichever route it asked
        for in :meth:`sofab.Visitor.on_array_begin` (§6.6.3).

        Two things can only be settled here, at the header: the element width
        the schema declares, and where the elements are to go. Both are gone by
        the time the typed hook holds the list — a width checked there cannot
        reject an array that never arrived (§7.1), and a destination offered
        there is a destination the values have already been built without.
        """
        pending = self._pending
        capped = pending is not None and pending[0] == _LIMIT
        dst: Any = None
        lo: int | None = None
        hi: int | None = None
        if self._wants_array_begin:
            # Asked before a parked receiver cap is answered (#128). §6.2.1's
            # limit is there to stop the *sender* dictating the *receiver's*
            # allocation, and a handler that hands back a buffer has already
            # chosen the size itself — there is no allocation of this decoder's
            # left for the cap to prevent. It was told the announced count and
            # may refuse it; that call belongs to the handler, not here.
            spec = visitor.on_array_begin(fid, _WT[wtype], count)
            if spec is not None:
                dst, lo, hi = spec
        if capped and dst is None:
            # Nowhere to put them but a list of this decoder's own, sized by the
            # wire. That is the allocation §6.2.1 exists to prevent, and this —
            # the count header, before an element is read — is where it says so.
            assert pending is not None
            raise SofaLimitError(pending[1])
        if capped:
            # Spent: unwrap so the resume transaction and the reads below see
            # the real pending value rather than the wrapper around it.
            assert pending is not None
            self._pending = pending[2]
        self._keep = self._pos
        if dst is None:
            # One list, not two: the ZigZag transform is folded into the element
            # loop, where the raw value is already in hand. The list itself is
            # the value the handler asked for and the wire sizes it -- that is
            # the gap §6.6.3 names -- but nothing else on the way to it is sized
            # by the wire any more.
            out = self._read_varints(count, lo, hi, zigzag)
            self._pending = None  # committed once the payload is in hand (§5.2)
            if zigzag:
                visitor.on_signed_array(fid, out)
            else:
                visitor.on_unsigned_array(fid, out)
            return
        # Straight into the caller's storage: no list, and on the native engine
        # not one element ever boxed. The buffer is the caller's, so a short one
        # is refused rather than grown (§6.6) -- and every verdict below is
        # reached here, at the count header, before an element is read.
        view = _writable(dst, "on_array_begin")
        isz = view.itemsize
        if isz not in (1, 2, 4, 8):
            raise SofaArgumentError(
                f"on_array_begin's destination holds {isz}-byte items; "
                "1, 2, 4 or 8 are supported"
            )
        # A destination narrower than 8 bytes is only safe if the declared width
        # says every element fits it. Asked once, here, so the fill can narrow
        # without a second test per element -- and refused rather than silently
        # truncated.
        if isz != 8 and not _width_fits(isz, zigzag, lo, hi):
            raise SofaArgumentError(
                "on_array_begin declared a width that does not fit its "
                f"{isz}-byte destination"
            )
        if view.nbytes < count * isz:
            raise SofaArgumentError(
                f"on_array_begin returned {view.nbytes // isz} slots "
                f"for an array of {count}"
            )
        self._read_varints(count, lo, hi, zigzag, view, 0)
        self._pending = None

    def _visit_farray_bits(
        self, visitor: Visitor, fid: int, pending: tuple[Any, ...]
    ) -> None:
        """Hand an ``fp32`` array's payload to the handler as raw wire bytes.

        §6.5's array half. Nothing is decoded and nothing is allocated: the
        payload is claimed in place and passed through the callback, which is
        §6.7's second route — the bytes are the caller's own input and their
        validity ends when the callback returns.
        """
        if pending[0] == _LIMIT:
            # No storage of the decoder's is at stake, but the handler has not
            # been asked and the wire still chose the length. The cap governs
            # the same route it always did.
            raise SofaLimitError(pending[1])
        count = pending[2]
        self._keep = self._pos
        buf, off = self._span_exact(self._farray_nbytes(count, pending[3]))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        raw = memoryview(buf)[off : off + count * 4]
        view = raw.toreadonly()
        try:
            visitor.on_float32_array_bits(fid, count, view)
        finally:
            # Released, not merely dropped: §6.7 ends the value's validity with
            # the callback, and releasing the view the handler was handed makes
            # that true rather than merely documented.
            view.release()
            raw.release()

    def _take_farray_values(self, pending: tuple[Any, ...], width: int) -> list[float]:
        """The fallback route for a fixlen array: a ``list`` of Python floats.

        The list is what the handler asked for and the wire sizes it, which is
        the gap §6.6.3 names and the README itemises. What is **not** here any
        more is the ``bytes`` copy of the payload that used to be made on the way
        to it: the values are unpacked straight out of the buffer they were fed
        into (:meth:`_span_exact`), so the route costs one wire-sized allocation
        rather than two. Everything the codec could stop sizing from the wire, it
        has stopped sizing from the wire.
        """
        if pending[0] == _LIMIT:
            raise SofaLimitError(pending[1])
        count = pending[2]
        self._keep = self._pos
        buf, off = self._span_exact(self._farray_nbytes(count, pending[3]))
        self._pending = None  # committed only once the payload is in hand (§5.2)
        return (
            _core.unpack_f32_array(buf, count, off)
            if width == 4
            else _core.unpack_f64_array(buf, count, off)
        )
