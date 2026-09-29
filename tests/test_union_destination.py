"""A one-of table: a union's held option lands on the destination map (#164).

MESSAGE_SPEC §4.2 gives a union no tag but its child's id, and §7.4.1 settles
what a scope with several children means: the held option is the **last
correctly-typed occurrence of any option id**. A table built with
``which_at=`` records exactly that one piece of state — the id, in a ``words``
slot — which is all a reader needs to know which option it holds.

The options here are the kinds a later arrival replaces **whole** (a scalar, a
string), which is why they need no reset: whatever a discarded option left in
its slots is unreachable through ``which_at``, and an option that returns
overwrites its own storage. A ``struct``/``union`` option is the case that does
need §7.4.1's reset, and :meth:`sofab.Binding.sequence` refuses one for now.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, Recorder, Status, capped, encode

# The union field, and its three options. ``default_id`` is 0, the value the
# caller prepares ``which_at`` with -- absence then needs no sentinel (§4.2).
UNION_ID = 21
OPT_U16, OPT_STR, OPT_F32 = 0, 1, 2
DEFAULT_ID = OPT_U16

# words layout: 0 = which, 1 = u16 option, 2 = fp32 option, 3 = union occurrences
W_WHICH, W_U16, W_F32, W_COUNT = 0, 1, 2, 3


def _table() -> Binding:
    option = Binding(closed=True, which_at=W_WHICH)
    option.unsigned(OPT_U16, at=W_U16, max_value=0xFFFF)
    option.string(OPT_STR, at=0)
    option.float32(OPT_F32, at=W_F32)
    return Binding(closed=True).sequence(UNION_ID, option, count_at=W_COUNT)


def _decode(engine, message, chunk: int | None = None):
    root = _table()
    words = bytearray(root.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    view[W_WHICH] = DEFAULT_ID  # the caller prepares the default (§4.2)
    objects: list[object] = [None]
    data = encode(message)
    dec = engine(binding=root, words=words, objects=objects, **capped())
    step = chunk or len(data)
    status = None
    for i in range(0, len(data), step):
        status = dec.feed(data[i:i + step])
    held = view[W_WHICH]
    return status, held, view, memoryview(words).cast("d"), objects


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestHeldOption:
    def test_two_children_in_one_frame_the_last_one_holds(self, engine, chunk):
        # §7.4.1, first example: seq[21]( [0:u16] 7  [1:str] "x" ) -> option 1.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 7)
            e.write_string(OPT_STR, "x")
            e.write_sequence_end()

        status, held, view, _, objects = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == OPT_STR
        assert objects[0] == "x"
        # The discarded option's slot still holds what it received; it is
        # unreachable, because ``which_at`` does not name it.
        assert view[W_U16] == 7

    def test_an_option_that_returns_starts_over(self, engine, chunk):
        # §7.4.1's third example, with scalar options: a scalar is replaced
        # whole by its own arrival, so "starts from its own default" needs
        # nothing of the decoder here.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 7)
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_float32(OPT_F32, 1.5)
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 9)
            e.write_sequence_end()

        status, held, view, dbl, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == OPT_U16
        assert view[W_U16] == 9
        assert view[W_COUNT] == 3  # three occurrences of the union field

    def test_a_reopened_frame_continues_the_scope(self, engine, chunk):
        # §7.4: a sequence opened again continues its scope, so the held option
        # carries across the frames.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_float32(OPT_F32, 2.5)
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 3)
            e.write_sequence_end()

        status, held, view, dbl, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == OPT_U16
        assert view[W_U16] == 3
        # The discarded fp32 option still holds its value; ``which_at`` is what
        # makes it unreachable, not an erased slot.
        assert dbl[W_F32] == 2.5

    def test_a_mistyped_option_never_switches(self, engine, chunk):
        # §7.3 + §7.4.1: an occurrence the tag test skips is not an occurrence,
        # so a correctly-typed earlier option survives a mistyped later one.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_float32(OPT_F32, 4.5)
            e.write_string(OPT_U16, "not a u16")   # id 0 is bound unsigned
            e.write_sequence_end()

        status, held, view, dbl, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == OPT_F32
        assert dbl[W_F32] == 4.5
        assert view[W_U16] == 0  # the mistyped option stored nothing

    def test_an_unknown_option_id_never_switches(self, engine, chunk):
        # A child whose id names no option is skipped as an unknown id, and a
        # closed table is what keeps it off the visitor (§7.4.1).
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_float32(OPT_F32, 6.5)
            e.write_unsigned(9, 123)               # a newer sender's option
            e.write_sequence_end()

        status, held, _, _, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == OPT_F32

    def test_an_empty_union_frame_holds_default_id(self, engine, chunk):
        # §4.2: an empty union sequence carries no option, so it decodes
        # identically to the omitted field -- the slot the caller prepared.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_end_keep()

        status, held, view, dbl, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == DEFAULT_ID
        assert view[W_COUNT] == 1  # the frame did arrive

    def test_an_absent_union_leaves_the_slot_alone(self, engine, chunk):
        def m(e):
            e.write_unsigned(1, 5)  # some other field, no union at all

        status, held, view, dbl, _ = _decode(engine, m, chunk)
        assert status == Status.COMPLETE
        assert held == DEFAULT_ID
        assert view[W_COUNT] == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_the_table_and_the_visitor_agree_about_the_held_option(engine):
    """§5.3.1's test is *one implementation of every rule*: the table must not
    answer a question differently from the hooks. A handler that computes the
    held option from the hook stream is the second reading, and the two must
    match on every one of these messages."""
    def m(e):
        e.write_sequence_begin_lazy(UNION_ID)
        e.write_unsigned(OPT_U16, 7)
        e.write_string(OPT_STR, "x")
        e.write_sequence_end()
        e.write_sequence_begin_lazy(UNION_ID)
        e.write_float32(OPT_F32, 1.5)
        e.write_string(OPT_U16, "mistyped")
        e.write_sequence_end()

    data = encode(m)

    # The hook reading: every value hook inside the union's scope is an
    # occurrence, and the generated visitor path applies §7.4.1 itself.
    class Handler(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.held: int | None = None
            self.depth = 0

        def on_sequence_begin(self, field_id):
            self.depth += 1
            return None

        def on_sequence_end(self):
            self.depth -= 1

        def _arrived(self, field_id):
            if self.depth:
                self.held = field_id

        def on_unsigned(self, field_id, value):
            self._arrived(field_id)

        def on_string(self, field_id, value):
            self._arrived(field_id)

        def on_float32(self, field_id, value):
            self._arrived(field_id)

    h = Handler()
    dec = engine(visitor=h, **capped())
    assert dec.feed(data) == Status.COMPLETE

    # The table reading.
    root = _table()
    words = bytearray(root.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    view[W_WHICH] = DEFAULT_ID
    dec = engine(binding=root, words=words, objects=[None], **capped())
    assert dec.feed(data) == Status.COMPLETE

    # The visitor sees the mistyped id 0 as a string (it has no schema to test
    # it against); the table skips it under §7.3. So the two agree only up to
    # the last *correctly typed* option, which is what §7.4.1 defines.
    assert view[W_WHICH] == OPT_F32
    assert h.held == OPT_U16  # the visitor's own, unfiltered last arrival


# --- a struct option, and §7.4.1's reset (#167) ------------------------------
#
# The one option kind a later arrival does not rewrite whole: it carries only
# the children it sends, so the ones it leaves out must read as their own
# defaults rather than as whatever the option held before it was discarded.

S_WHICH, S_U16, S_X, S_Y, S_COUNT, S_INNER, S_INNER_V = 0, 1, 2, 3, 4, 5, 6
OPT_STRUCT = 2
X_DEFAULT, Y_DEFAULT = 7, 0


def _struct_table(inner: Binding | None = None) -> Binding:
    """A union with a scalar option (id 0) and a struct option (id 2)."""
    member = Binding(closed=True)
    member.signed(0, at=S_X, default=X_DEFAULT)
    member.signed(1, at=S_Y, default=Y_DEFAULT)
    if inner is not None:
        member.sequence(2, inner)
    option = Binding(closed=True, which_at=S_WHICH, default_id=DEFAULT_ID)
    option.unsigned(OPT_U16, at=S_U16, max_value=0xFFFF)
    option.sequence(OPT_STRUCT, member, count_at=S_COUNT)
    return Binding(closed=True).sequence(UNION_ID, option)


def _decode_struct(engine, message, table: Binding, chunk: int | None = None):
    words = bytearray(table.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    signed = memoryview(words).cast("q")
    view[S_WHICH] = DEFAULT_ID
    signed[S_X] = X_DEFAULT  # the caller prepares its declared defaults (§2)
    signed[S_Y] = Y_DEFAULT
    data = encode(message)
    dec = engine(binding=table, words=words, objects=[], **capped())
    step = chunk or len(data)
    status = None
    for i in range(0, len(data), step):
        status = dec.feed(data[i:i + step])
    return status, view, signed


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestStructOption:
    def test_a_struct_option_continues_its_scope(self, engine, chunk):
        # §7.4.1, second example: the same option twice continues under §7.4,
        # so a child set by the earlier opening is retained.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(0, 1)
            e.write_sequence_end()
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(1, 2)
            e.write_sequence_end()
            e.write_sequence_end()

        status, view, signed = _decode_struct(engine, m, _struct_table(), chunk)
        assert status == Status.COMPLETE
        assert view[S_WHICH] == OPT_STRUCT
        assert (signed[S_X], signed[S_Y]) == (1, 2)

    def test_a_struct_option_that_returns_starts_from_its_default(self, engine, chunk):
        # §7.4.1, THIRD example -- the whole reason #167 exists:
        #   seq(struct(x=1))  seq(u16 5)  seq(struct(y=2))  ->  {x=default, y=2}
        # The earlier x does not survive being discarded by option 0.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(0, 1)
            e.write_sequence_end()
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 5)
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(1, 2)
            e.write_sequence_end()
            e.write_sequence_end()

        status, view, signed = _decode_struct(engine, m, _struct_table(), chunk)
        assert status == Status.COMPLETE
        assert view[S_WHICH] == OPT_STRUCT
        assert signed[S_X] == X_DEFAULT, "the discarded option's x survived"
        assert signed[S_Y] == 2
        # The occurrence count restarts with the option, so it counts the
        # arrivals of the option that is held, not of the ones before it.
        assert view[S_COUNT] == 1

    def test_the_scalar_option_is_not_reset_by_the_struct_option(self, engine, chunk):
        # Only the option being SELECTED is touched (§7.4.1). The scalar's slot
        # keeps what it received -- unreachable, because which_at names the
        # struct -- which is what makes a reader that consults the slot safe.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 5)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(1, 2)
            e.write_sequence_end()
            e.write_sequence_end()

        status, view, signed = _decode_struct(engine, m, _struct_table(), chunk)
        assert status == Status.COMPLETE
        assert view[S_WHICH] == OPT_STRUCT
        assert view[S_U16] == 5
        assert (signed[S_X], signed[S_Y]) == (X_DEFAULT, 2)

    def test_an_empty_struct_option_frame_holds_the_option_at_its_default(
        self, engine, chunk
    ):
        # §4.2/§2: a held option that is not default_id is written even when
        # every child is at its own default, so this frame is what a conformant
        # encoder emits for it -- and it must switch the union.
        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 5)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_sequence_end_keep()
            e.write_sequence_end()

        status, view, signed = _decode_struct(engine, m, _struct_table(), chunk)
        assert status == Status.COMPLETE
        assert view[S_WHICH] == OPT_STRUCT
        assert (signed[S_X], signed[S_Y]) == (X_DEFAULT, Y_DEFAULT)

    def test_a_nested_union_starts_at_its_own_default_id(self, engine, chunk):
        # A union option may be another union (§4.2). The inner union's held
        # option is part of the outer option's storage, so re-selecting the
        # outer one puts the inner back at ITS default_id -- the value the
        # decoder cannot derive and the table therefore states.
        inner = Binding(closed=True, which_at=S_INNER, default_id=4)
        inner.signed(4, at=S_INNER_V, default=-1)
        inner.signed(5, at=S_INNER_V)
        table = _struct_table(inner)

        def m(e):
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_sequence_begin_lazy(2)      # the inner union
            e.write_signed(5, 9)                # holding option 5
            e.write_sequence_end()
            e.write_sequence_end()
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_unsigned(OPT_U16, 5)        # discard the struct option
            e.write_sequence_end()
            e.write_sequence_begin_lazy(UNION_ID)
            e.write_sequence_begin_lazy(OPT_STRUCT)
            e.write_signed(1, 2)                # back, without the inner union
            e.write_sequence_end()
            e.write_sequence_end()

        words = bytearray(table.tree_words_required * 8)
        view = memoryview(words).cast("Q")
        signed = memoryview(words).cast("q")
        view[S_WHICH] = DEFAULT_ID
        view[S_INNER] = 4
        signed[S_X] = X_DEFAULT
        data = encode(m)
        dec = engine(binding=table, words=words, objects=[], **capped())
        step = chunk or len(data)
        for i in range(0, len(data), step):
            status = dec.feed(data[i:i + step])
        assert status == Status.COMPLETE
        assert view[S_WHICH] == OPT_STRUCT
        assert view[S_INNER] == 4, "the inner union kept a discarded option"
        assert signed[S_INNER_V] == -1
        assert (signed[S_X], signed[S_Y]) == (X_DEFAULT, 2)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_string_option_starts_empty_again(engine):
    # §4.2 admits no non-empty default for a string option, so "" is what it
    # starts from -- and the objects slot is what has to say so.
    option = Binding(closed=True, which_at=0, default_id=1)
    option.string(1, at=0)
    option.unsigned(2, at=1)
    root = Binding(closed=True).sequence(UNION_ID, option)

    def m(e):
        e.write_sequence_begin_lazy(UNION_ID)
        e.write_string(1, "held")
        e.write_unsigned(2, 3)      # option 2 discards the string option
        e.write_string(1, "")       # ... and it comes back, empty on the wire
        e.write_sequence_end()

    words = bytearray(root.tree_words_required * 8)
    objects: list[object] = ["prepared"]
    dec = engine(binding=root, words=words, objects=objects, **capped())
    assert dec.feed(encode(m)) == Status.COMPLETE
    assert memoryview(words).cast("Q")[0] == 1
    assert objects[0] == ""


def test_an_option_row_may_not_declare_a_non_empty_default():
    # §4.2: a string/blob/array OPTION starts empty. Which rows are options is
    # what `which_at` says, so the refusal hangs off the table, not off the kind
    # -- a member of a struct option is an ordinary row (corelib-py#169).
    one_of = Binding(closed=True, which_at=0)
    with pytest.raises(Exception, match="must not declare a non-empty default"):
        one_of.string(1, at=0, default="abc")
    with pytest.raises(Exception, match="must not declare a non-empty default"):
        one_of.unsigned_array(2, at=1, cap=4, default=(1, 2))
    # The same rows in an ordinary table take it.
    plain = Binding(closed=True)
    plain.string(1, at=0, default="abc")
    plain.unsigned_array(2, at=1, cap=4, count_at=5, default=(1, 2))


def test_a_struct_union_row_takes_no_default():
    # Through `_add`: `sequence()` exposes no `default=`, and this is the guard
    # that keeps it that way. Kind 12 is K_SEQUENCE.
    with pytest.raises(Exception, match="starts from the rows of its child"):
        Binding()._add(12, 1, 0, 0, None, Binding(), default=3)


def test_an_into_default_is_prepared_as_bytes():
    # The slot holds the caller's buffer, so the default cannot be *stored* --
    # it is copied in, and what is copied is bytes whatever the kind: a string
    # default is UTF-8-encoded once at bind time, not per reset.
    b = Binding(closed=True, which_at=0)
    member = Binding(closed=True)
    member.string_into(0, at=0, maxlen=8, count_at=1, default="ab\u00e4")
    member.blob_into(1, at=1, maxlen=4, count_at=2, default=b"\x01\x02")
    b.unsigned(0, at=3)
    b.sequence(1, member)
    b.freeze()
    row = b._by_id[1]
    assert row.reset_into == ((0, "ab\u00e4".encode()), (1, b"\x01\x02"))
    # The length slot carries the default's BYTE length, not its character count.
    plan = dict(row.reset_words)
    assert plan[1] == 4, "'ab\u00e4' is four bytes"
    assert plan[2] == 2
    # An into row's own_objects stays empty: the caller's buffer is never
    # replaced, only written into.
    assert row.reset_objects == ()


def test_a_default_that_does_not_fit_the_declared_bound_is_refused():
    with pytest.raises(Exception, match="longer than the declared maxlen"):
        Binding().string(1, at=0, maxlen=2, default="abcd")
    with pytest.raises(Exception, match="longer than the declared capacity"):
        Binding().unsigned_array(1, at=0, cap=2, default=(1, 2, 3))
    # A string's maxlen counts BYTES, so a multi-byte character fills it faster.
    with pytest.raises(Exception, match="longer than the declared maxlen"):
        Binding().string(1, at=0, maxlen=2, default="\u00e4\u00f6")


def test_an_array_default_is_checked_element_by_element():
    with pytest.raises(Exception, match="outside the declared width"):
        Binding().unsigned_array(1, at=0, cap=4, elem_max=0xFF, default=(1, 256))
    with pytest.raises(Exception, match="must be iterable"):
        Binding().unsigned_array(1, at=0, cap=4, default=3)
    with pytest.raises(Exception, match="must be str"):
        Binding().string(1, at=0, default=b"abc")
    with pytest.raises(Exception, match="must be bytes, not str"):
        Binding().bytes(1, at=0, default="abc")


def test_an_array_default_normalizes_each_element_like_its_own_arrival():
    import struct as _struct
    b = Binding(closed=True, which_at=0)
    member = Binding(closed=True)
    member.boolean_array(0, at=1, cap=2, count_at=3, default=(42, 0))
    member.float32_array(1, at=4, cap=2, count_at=6, default=(0.1,))
    b.unsigned(0, at=7)
    b.sequence(1, member)
    b.freeze()
    plan = dict(b._by_id[1].reset_words)
    assert (plan[1], plan[2]) == (1, 0), "§4.4: anything other than 0 is true"
    assert plan[3] == 2, "the count is the default's own length"
    f32 = _struct.unpack("<Q", _struct.pack("<d", _struct.unpack(
        "<f", _struct.pack("<f", 0.1))[0]))[0]
    assert plan[4] == f32, "an fp32 element is the value an fp32 array can hold"
    assert plan[6] == 1


def test_a_float_default_that_is_not_a_number_is_refused():
    # A string that happens to parse is a mistake, not a default -- the same
    # line ``_index`` draws for the integer binders.
    with pytest.raises(Exception, match="must be a real number"):
        Binding().float64(1, at=0, default="1.5e3")
    assert Binding().float32(1, at=0, default=3).entries[0].reset_words == ()  # int ok


def test_a_default_outside_the_declared_width_is_refused():
    with pytest.raises(Exception, match="outside the declared width"):
        Binding().unsigned(1, at=0, max_value=0xFF, default=256)


def test_a_boolean_default_is_normalized_and_a_float_default_is_rounded():
    b = Binding(closed=True, which_at=0)
    b.boolean(1, at=1, default=42)
    b.float32(2, at=2, default=0.1)
    b.float64(3, at=3, default=0.1)
    b.freeze()
    rows = {e.field_id: e.reset_words for e in b.entries}
    assert rows[1] == ((1, 1),), "every value other than 0 is true (§4.4)"
    import struct as _struct
    f32 = _struct.unpack("<Q", _struct.pack("<d", _struct.unpack(
        "<f", _struct.pack("<f", 0.1))[0]))[0]
    f64 = _struct.unpack("<Q", _struct.pack("<d", 0.1))[0]
    assert rows[2] == ((2, f32),) and rows[3] == ((3, f64),)
    assert f32 != f64, "an fp32 default is the value an fp32 field can hold"


def test_the_default_id_is_range_checked_and_readable():
    assert Binding(which_at=0, default_id=5).default_id == 5
    assert Binding().default_id == 0
    with pytest.raises(Exception, match="default id"):
        Binding(which_at=0, default_id=-1)


# --- the table's own bookkeeping --------------------------------------------


def test_the_which_slot_is_counted_and_readable():
    option = Binding(closed=True, which_at=7).unsigned(0, at=3)
    assert option.which_at == 7
    # The slot is words storage like any other, so the caller's buffer has to
    # hold it -- even though no row names it.
    assert option.words_required == 8
    assert Binding().which_at == -1


def test_a_which_slot_out_of_range_is_refused():
    with pytest.raises(Exception, match="which slot"):
        Binding(which_at=-1)


def test_freeze_pushes_the_slot_onto_every_row_wherever_the_table_is_bound():
    # A child bound from two places is one table, so both bindings see the same
    # one-of behaviour -- the same argument ``closed`` is documented with.
    option = Binding(closed=True, which_at=4).unsigned(0, at=0).signed(1, at=1)
    root = Binding(closed=True).sequence(9, option).sequence(10, option)
    root.freeze()
    assert [e.which_at for e in option.entries] == [4, 4]
    # The union's own rows in the parent are not alternatives.
    assert [e.which_at for e in root.entries] == [-1, -1]


def test_a_cycle_through_a_one_of_table_freezes_once():
    # A recursive schema is legitimate, and freeze() walks each table once
    # (``_reachable`` dedupes by identity), so pushing the slot onto the rows
    # must not walk the cycle forever either.
    union = Binding(closed=True, which_at=2).unsigned(0, at=0).signed(1, at=1)
    outer = Binding(closed=True)
    outer.sequence(9, union)     # the union field
    outer.sequence(10, outer)    # a struct field of the message's own type
    assert len(outer.freeze()) == 2
    assert [e.which_at for e in union.entries] == [2, 2]
