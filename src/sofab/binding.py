"""Field-id → destination table: decode straight into caller-owned storage.

CORELIB_PLAN §5.3 recommends the visitor pattern "because the primary consumer
of this library is *generated code* … those objects already exist at decode
time; the visitor pattern lets the decoder write each field straight into the
waiting member without an intermediate representation". A :class:`Binding` is
that idea with the last Python call taken out of it: instead of calling back per
field so the handler can store the value, the handler declares **once** where
every field belongs, and the decoder writes there itself.

What that buys is not a micro-optimisation. Measured on a 36-field / 12-array /
51-element message (issue #109), parsing costs ~220 instructions per field while
*every value made visible to Python* costs ~750–1100 — so the number of times a
decode crosses the Python boundary, not the parser, is what a Python port's
decode speed is made of. A bound decode crosses it zero times.

Two pieces of storage, both **supplied and sized by the caller** — the decoder
allocates neither and never sizes anything from the wire (documentation#54 §6.6,
CORELIB_PLAN §6.2.1):

``words``
    One writable, C-contiguous byte buffer whose length is a multiple of 8 — a
    ``bytearray`` is the obvious choice. Every numeric field lands in it as one
    64-bit slot: unsigned as ``uint64``, signed as ``int64``, ``fp32``/``fp64``
    both widened to a native ``double``, arrays as ``cap`` consecutive slots.
    Read the slots back through as many typed views over the *same* buffer as
    you need — ``memoryview(buf).cast("q")``, ``.cast("Q")``, ``.cast("d")`` —
    which costs no copy and no second buffer.

``objects``
    A pre-sized ``list``, for the two field kinds that have no fixed-width
    machine representation: ``string`` and ``blob``. Each lands at its own index.

A field the table does not name is not an error: it is dispatched to the
:class:`sofab.Visitor` the decoder was given, or skipped. So a binding covers
the schema's hot fields and everything else keeps working.

Example — the shape generated code would emit::

    b = Binding()
    b.unsigned(1, at=0).signed(2, at=1).string(3, at=0, count_at=2)
    b.unsigned_array(4, at=8, cap=16, count_at=3)

    words = bytearray(b.words_required * 8)
    objs = [None] * b.objects_required
    dec = Decoder(binding=b, words=words, objects=objs)
    st = dec.feed(chunk)

    u = memoryview(words).cast("Q")
    u[0]                      # field 1
    objs[0]                   # field 3, or untouched if it never arrived
    u[2]                      # 1 if field 3 arrived, else untouched
    u[8:8 + u[3]]             # field 4's elements, u[3] of them
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaArgumentError,
    WireType,
)

if TYPE_CHECKING:
    from typing import Any

# --- entry kinds -------------------------------------------------------------
#
# Plain module-level ints rather than an IntEnum: both engines compare them on
# the decode hot path, and the native one lowers them to C ``int`` switches.

K_UNSIGNED = 0
K_SIGNED = 1
K_FLOAT32 = 2
K_FLOAT64 = 3
K_STRING = 4
K_BYTES = 5
K_ARRAY_UNSIGNED = 6
K_ARRAY_SIGNED = 7
K_ARRAY_FLOAT32 = 8
K_ARRAY_FLOAT64 = 9
K_SEQUENCE = 10

#: For each kind, the wire tag it accepts: ``(wire type, fixlen subtype or None)``.
#: A field whose wire tag contradicts its binding is **not** an error — it is
#: skipped exactly like an unknown id and the decode stays COMPLETE
#: (MESSAGE_SPEC §7.3, CORELIB_PLAN §6.3).
KIND_TAG: tuple[tuple[WireType, FixlenSubtype | None], ...] = (
    (WireType.UNSIGNED, None),
    (WireType.SIGNED, None),
    (WireType.FIXLEN, FixlenSubtype.FP32),
    (WireType.FIXLEN, FixlenSubtype.FP64),
    (WireType.FIXLEN, FixlenSubtype.STRING),
    (WireType.FIXLEN, FixlenSubtype.BLOB),
    (WireType.ARRAY_UNSIGNED, None),
    (WireType.ARRAY_SIGNED, None),
    (WireType.ARRAY_FIXLEN, FixlenSubtype.FP32),
    (WireType.ARRAY_FIXLEN, FixlenSubtype.FP64),
    (WireType.SEQUENCE_START, None),
)

#: Kinds whose ``at`` indexes ``objects`` rather than ``words``.
_OBJECT_KINDS = frozenset((K_STRING, K_BYTES))
#: Kinds that consume ``cap`` consecutive slots instead of one.
_ARRAY_KINDS = frozenset(
    (K_ARRAY_UNSIGNED, K_ARRAY_SIGNED, K_ARRAY_FLOAT32, K_ARRAY_FLOAT64)
)


class Entry:
    """One row of a :class:`Binding`. Built by the binder methods, read by the
    engines; not something callers construct."""

    __slots__ = (
        "kind", "field_id", "at", "cap", "count_at", "child", "wt", "st",
        "elem_lo", "elem_hi", "elem_bounded",
    )

    def __init__(
        self,
        kind: int,
        field_id: int,
        at: int,
        cap: int,
        count_at: int,
        child: Binding | None,
        elem_lo: int = 0,
        elem_hi: int = 0,
        elem_bounded: bool = False,
    ) -> None:
        self.kind = kind
        self.field_id = field_id
        self.at = at
        self.cap = cap
        self.count_at = count_at
        self.child = child
        # The element width the schema declares for an array (§7.1), checked at
        # the element, before it is stored. Absent means "as wide as the wire
        # type allows".
        self.elem_lo = elem_lo
        self.elem_hi = elem_hi
        self.elem_bounded = elem_bounded
        # Precomputed so the §7.3 tag test on the hot path is two int compares.
        self.wt, self.st = KIND_TAG[kind]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Entry(kind={self.kind}, field_id={self.field_id}, at={self.at}, "
            f"cap={self.cap}, count_at={self.count_at})"
        )


class Binding:
    """Where each field id's value belongs. Build once, decode many times.

    Every binder method returns ``self``, so a table reads as one statement.
    ``at`` is a slot index — into ``words`` for the numeric kinds, into
    ``objects`` for :meth:`string` and :meth:`bytes`. ``count_at`` is an optional
    ``words`` slot the decoder writes the field's *arrival* into: ``1`` for a
    scalar that turned up, the element count for an array, the number of
    occurrences for a sequence. Slots the decoder never writes are left exactly
    as the caller prepared them, which is how a decode reports absence without
    inventing a sentinel.
    """

    __slots__ = (
        "_entries", "_by_id", "_words_required", "_objects_required",
        "_tree", "_compiled", "_frozen",
    )

    def __init__(self) -> None:
        self._entries: list[Entry] = []
        self._by_id: dict[int, Entry] = {}
        self._words_required = 0
        self._objects_required = 0
        # Derived once and reused: a Decoder is built per message in the
        # one-shot path, and walking the tree (or recompiling the native table)
        # per decode would cost more than the decode.
        self._tree: tuple[int, int] | None = None
        self._compiled: Any = None
        self._frozen = False

    # --- introspection ------------------------------------------------------

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The rows, in the order they were bound. The engines compile this."""
        return tuple(self._entries)

    @property
    def words_required(self) -> int:
        """Slots the ``words`` buffer must hold — i.e. it must be at least
        ``words_required * 8`` bytes. Counts this table only; a child
        :meth:`sequence` binding shares the same buffer, so take the maximum
        over the whole tree (or give every table disjoint slots, which is what
        generated code does)."""
        return self._words_required

    @property
    def objects_required(self) -> int:
        """Entries the ``objects`` list must hold."""
        return self._objects_required

    @property
    def tree_words_required(self) -> int:
        """:attr:`words_required` over this table *and* every table reachable
        through :meth:`sequence`. A child shares the parent's storage, so this
        is the size the one buffer has to have."""
        return self._tree_sizes()[0]

    @property
    def tree_objects_required(self) -> int:
        """:attr:`objects_required` over the whole tree; see
        :attr:`tree_words_required`."""
        return self._tree_sizes()[1]

    def _tree_sizes(self) -> tuple[int, int]:
        tree = self._tree
        if tree is None:
            reachable = self.freeze()
            tree = (
                max(b.words_required for b in reachable),
                max(b.objects_required for b in reachable),
            )
            self._tree = tree
        return tree

    def freeze(self) -> list[Binding]:
        """Close the table — this one and every child — and return the whole
        reachable set.

        A binding is a build-once artifact: a :class:`sofab.Decoder` derives its
        storage requirements and (in the native engine) a compiled lookup table
        from it, and caches both, so a table that changed afterwards would decode
        against a stale copy. Freezing at first use makes that a clear error
        instead. Called for you — building the decoder is what freezes the table
        — and idempotent, so calling it yourself is harmless.

        It is deliberately the *whole tree*: a child bound into a parent is
        reachable only downwards, so freezing the root is the only moment at
        which every table in it can be reached at once.
        """
        reachable = self._reachable()
        for b in reachable:
            b._frozen = True
        return reachable

    def _reachable(self) -> list[Binding]:
        """This table and every child, breadth-first, each visited once — a
        schema may legitimately be recursive."""
        seen = {id(self): self}
        out = [self]
        i = 0
        while i < len(out):
            for e in out[i]._entries:
                child = e.child
                if child is not None and id(child) not in seen:
                    seen[id(child)] = child
                    out.append(child)
            i += 1
        return out

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Binding {len(self._entries)} fields, "
            f"{self._words_required} words, {self._objects_required} objects>"
        )

    # --- binder methods -----------------------------------------------------

    def unsigned(self, field_id: int, at: int, count_at: int | None = None) -> Binding:
        """Bind an unsigned-integer field to ``words`` slot ``at`` (``uint64``)."""
        return self._add(K_UNSIGNED, field_id, at, 0, count_at, None)

    def signed(self, field_id: int, at: int, count_at: int | None = None) -> Binding:
        """Bind a signed-integer field to ``words`` slot ``at`` (``int64``)."""
        return self._add(K_SIGNED, field_id, at, 0, count_at, None)

    def boolean(self, field_id: int, at: int, count_at: int | None = None) -> Binding:
        """Bind a boolean field. Booleans have no wire type (§4.4): this is
        :meth:`unsigned`, and the slot receives ``0`` or the value the sender
        wrote — the caller tests it for truth."""
        return self._add(K_UNSIGNED, field_id, at, 0, count_at, None)

    def float32(self, field_id: int, at: int, count_at: int | None = None) -> Binding:
        """Bind an ``fp32`` field to ``words`` slot ``at``, widened to a native
        ``double`` (read it back through a ``.cast("d")`` view)."""
        return self._add(K_FLOAT32, field_id, at, 0, count_at, None)

    def float64(self, field_id: int, at: int, count_at: int | None = None) -> Binding:
        """Bind an ``fp64`` field to ``words`` slot ``at`` as a ``double``."""
        return self._add(K_FLOAT64, field_id, at, 0, count_at, None)

    def string(
        self, field_id: int, at: int, maxlen: int = 0, count_at: int | None = None
    ) -> Binding:
        """Bind a UTF-8 ``string`` field to ``objects[at]``.

        ``maxlen`` is the schema's declared byte length, or ``0`` for a field the
        schema leaves unbounded. Declaring it makes the field **schema-bounded**:
        a longer payload is INVALID (MESSAGE_SPEC §7.1) and the receiver-side
        ``max_dyn_string_len`` cap no longer applies to it (§6.2.1). Left at ``0``
        the cap applies as usual."""
        return self._add(K_STRING, field_id, at, maxlen, count_at, None)

    def bytes(
        self, field_id: int, at: int, maxlen: int = 0, count_at: int | None = None
    ) -> Binding:
        """Bind a ``blob`` field to ``objects[at]``; see :meth:`string` for
        ``maxlen``."""
        return self._add(K_BYTES, field_id, at, maxlen, count_at, None)

    def unsigned_array(
        self,
        field_id: int,
        at: int,
        cap: int,
        count_at: int | None = None,
        elem_max: int | None = None,
    ) -> Binding:
        """Bind an unsigned-integer array to ``words[at:at + cap]``.

        ``cap`` is the **schema's** maximum element count. A message declaring
        more is malformed against that schema, so it is rejected as INVALID
        (MESSAGE_SPEC §7.1) — the decoder never sizes storage from the wire.

        ``elem_max`` is the schema's declared element width (``0xFF`` for a
        ``u8`` array, and so on). Given, it is checked **at** each element,
        before the element is stored, so a too-wide value is INVALID at the
        element that carries it rather than after the array completes."""
        return self._add(
            K_ARRAY_UNSIGNED, field_id, at, cap, count_at, None, 0, elem_max
        )

    def signed_array(
        self,
        field_id: int,
        at: int,
        cap: int,
        count_at: int | None = None,
        elem_min: int | None = None,
        elem_max: int | None = None,
    ) -> Binding:
        """Bind a signed-integer array to ``words[at:at + cap]`` (``int64``).

        The two halves of the declared width are independent: either may be
        given on its own and bounds its own side (see :meth:`unsigned_array`)."""
        return self._add(
            K_ARRAY_SIGNED, field_id, at, cap, count_at, None, elem_min, elem_max
        )

    def float32_array(
        self, field_id: int, at: int, cap: int, count_at: int | None = None
    ) -> Binding:
        """Bind an ``fp32`` array to ``words[at:at + cap]``, widened to
        ``double`` per element."""
        return self._add(K_ARRAY_FLOAT32, field_id, at, cap, count_at, None)

    def float64_array(
        self, field_id: int, at: int, cap: int, count_at: int | None = None
    ) -> Binding:
        """Bind an ``fp64`` array to ``words[at:at + cap]``."""
        return self._add(K_ARRAY_FLOAT64, field_id, at, cap, count_at, None)

    def sequence(
        self, field_id: int, child: Binding, count_at: int | None = None
    ) -> Binding:
        """Descend into a nested sequence with ``child`` as its table (§4.9).

        The child writes into the *same* ``words`` / ``objects`` storage, so a
        whole message tree decodes into one flat pair of buffers. ``count_at``
        counts how many times the sequence occurred, which is what tells a
        caller whether an optional sub-message was present.

        A sequence with no binding is skipped whole, sub-tree and all — the
        auto-skip §5.2 requires — and costs nothing but the walk."""
        if not isinstance(child, Binding):
            raise SofaArgumentError("sequence child must be a Binding")
        if child._frozen and not self._frozen:
            # The child is already closed, so binding it here would extend a
            # frozen tree by the back door.
            raise SofaArgumentError("sequence child is already in use by a decoder")
        return self._add(K_SEQUENCE, field_id, 0, 0, count_at, child)

    # --- internals ----------------------------------------------------------

    def _add(
        self,
        kind: int,
        field_id: Any,
        at: Any,
        cap: Any,
        count_at: Any,
        child: Binding | None,
        elem_lo: Any = None,
        elem_hi: Any = None,
    ) -> Binding:
        if self._frozen:
            raise SofaArgumentError(
                "this binding is already in use by a decoder; build the table "
                "before the decoder, not after"
            )
        fid = _index(field_id, "field id")
        if fid < 0 or fid > ID_MAX:
            raise SofaArgumentError(f"field id {fid} out of range")
        if fid in self._by_id:
            raise SofaArgumentError(f"field id {fid} is already bound")
        slot = _index(at, "slot index")
        if slot < 0:
            raise SofaArgumentError(f"slot index {slot} out of range")
        n = _index(cap, "capacity")
        if n < 0 or n > (FIXLEN_MAX if kind in _OBJECT_KINDS else ARRAY_MAX):
            raise SofaArgumentError(f"capacity {n} out of range")
        if count_at is None:
            cnt = -1
        else:
            cnt = _index(count_at, "count slot")
            if cnt < 0:
                raise SofaArgumentError(f"count slot {cnt} out of range")
            self._words_required = max(self._words_required, cnt + 1)

        if kind in _OBJECT_KINDS:
            self._objects_required = max(self._objects_required, slot + 1)
        elif kind in _ARRAY_KINDS:
            self._words_required = max(self._words_required, slot + n)
        elif kind != K_SEQUENCE:
            self._words_required = max(self._words_required, slot + 1)

        lo = SIGNED_MIN if elem_lo is None else _index(elem_lo, "elem_min")
        hi = (SIGNED_MAX if kind == K_ARRAY_SIGNED else UNSIGNED_MAX) \
            if elem_hi is None else _index(elem_hi, "elem_max")
        if not (SIGNED_MIN <= lo <= SIGNED_MAX) or not (0 <= hi <= UNSIGNED_MAX):
            raise SofaArgumentError("declared element width out of range")
        entry = Entry(kind, fid, slot, n, cnt, child, lo, hi,
                      elem_lo is not None or elem_hi is not None)
        self._entries.append(entry)
        self._by_id[fid] = entry
        return self


def _index(value: Any, what: str) -> int:
    """``__index__`` or a §6.3 InvalidArgument — never a silent truncation."""
    try:
        index: int = value.__index__()
    except AttributeError as exc:
        raise SofaArgumentError(
            f"{what} must be an integer, got {type(value).__name__}"
        ) from exc
    return index
