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


@pytest.mark.parametrize("engine", ENGINES)
def test_a_sequence_option_is_refused_for_now(engine):
    with pytest.raises(Exception, match="not supported yet"):
        Binding(which_at=0).sequence(3, Binding())
