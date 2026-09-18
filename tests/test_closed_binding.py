"""A closed table skips the ids it does not name, even with a visitor (#150).

The decoder descends into a bound sequence without telling the visitor (#146),
so the visitor still believes the walk is in the parent's scope. An id an OPEN
child table does not name is handed to it under the parent's identity — the
misplacement the first test pins. ``Binding(closed=True)`` skips such an id the
way a decoder with no visitor already does: no hook, no Field, decode COMPLETE.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, Recorder, Status, capped, encode


def _message(e) -> None:
    e.write_unsigned(7, 111)             # a parent field
    e.write_sequence_begin_lazy(1)       # the bound child
    e.write_unsigned(0, 5)               # named by the child
    e.write_unsigned(7, 999)             # unknown to the child: a newer sender
    e.write_string(8, "new")
    e.write_sequence_begin_lazy(9)       # an unknown nested sequence
    e.write_unsigned(1, 1)
    e.write_sequence_end()
    e.write_sequence_end()
    e.write_unsigned(8, 222)             # a parent field after the child


DATA = encode(_message)


def _decode(engine, closed: bool, chunk: int | None = None):
    child = Binding(closed=closed).unsigned(0, at=0, count_at=1)
    root = Binding().sequence(1, child=child, count_at=2)

    class Handler(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.words = bytearray(root.tree_words_required * 8)
            self.seen_fields = 0

        def destinations(self):
            return (root, self.words, None)

        def on_field(self, field):
            self.seen_fields += 1

    h = Handler()
    dec = engine(visitor=h, **capped())
    step = chunk or len(DATA)
    status = None
    for i in range(0, len(DATA), step):
        status = dec.feed(DATA[i:i + step])
    return status, h, memoryview(h.words).cast("Q").tolist()


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestClosedChild:
    def test_open_child_hands_unknown_ids_to_the_visitor(self, engine, chunk):
        # The default is unchanged — and it is the misplacement: id 7 inside the
        # child arrives looking exactly like the parent's id 7.
        status, h, words = _decode(engine, False, chunk)
        assert status == Status.COMPLETE
        assert h.events == [
            ("u", 7, 111), ("u", 7, 999), ("str", 8, "new"),
            ("seq{", 9), ("u", 1, 1), ("seq}",), ("u", 8, 222),
        ]
        assert words == [5, 1, 1]

    def test_closed_child_skips_what_it_does_not_name(self, engine, chunk):
        status, h, words = _decode(engine, True, chunk)
        assert status == Status.COMPLETE
        assert h.events == [("u", 7, 111), ("u", 8, 222)]
        # No hook for a skipped field, not even on_field.
        assert h.seen_fields == 2
        assert words == [5, 1, 1]


@pytest.mark.parametrize("engine", ENGINES)
def test_closed_root_without_visitor_is_unchanged(engine):
    # With no visitor every unnamed id is skipped anyway; closing changes nothing.
    child = Binding(closed=True).unsigned(0, at=0)
    root = Binding(closed=True).unsigned(8, at=1).sequence(1, child=child)
    words = bytearray(root.tree_words_required * 8)
    dec = engine(binding=root, words=words, objects=[], **capped())
    assert dec.feed(DATA) == Status.COMPLETE
    assert memoryview(words).cast("Q").tolist() == [5, 222]


@pytest.mark.parametrize("engine", ENGINES)
def test_mismatched_tag_sequence_is_skipped_by_a_closed_table(engine):
    # §7.3: an id the table binds under another tag is treated as unknown — so a
    # closed table skips a SEQUENCE_START arriving for its scalar id, whole.
    child = Binding(closed=True).unsigned(9, at=0, count_at=1)
    root = Binding().sequence(1, child=child)

    class Handler(Recorder):
        words = bytearray(root.tree_words_required * 8)

        def destinations(self):
            return (root, self.words, None)

    h = Handler()
    dec = engine(visitor=h, **capped())
    assert dec.feed(DATA) == Status.COMPLETE
    assert h.events == [("u", 7, 111), ("u", 8, 222)]
    assert memoryview(h.words).cast("Q").tolist() == [0, 0]


def test_closed_is_a_table_property():
    assert Binding().closed is False
    assert Binding(closed=True).closed is True
