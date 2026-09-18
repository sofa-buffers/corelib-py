"""A handler hears both halves of a sequence scope, or neither (#146).

``on_sequence_begin`` and ``on_sequence_end`` are the only way a flat visitor
learns where it is, so they must pair up exactly. Three routes open a scope
without the handler being asked, or hand it to someone else, and each has to be
closed the same way it was opened:

* a sequence the destination map binds is the *table's* — the visitor hears
  neither the begin nor the end;
* a sequence the visitor declines is skipped whole — no end either;
* a sequence the visitor hands to a child visitor is the child's until *its*
  end — a sequence nested inside it closes back to the child, not to the
  parent.

Every case asserts the full event sequence, and replays the message split at
every byte so a chunk boundary inside a scope cannot lose or duplicate the
record of how it was opened.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import NO_CAPS

from sofab import Binding, Encoder, Status, Visitor


def _wire(*ops) -> bytes:
    """``("b", id)`` opens a sequence, ``("e",)`` closes one, ``("u", id, v)``
    writes an unsigned."""
    buf = bytearray(256)
    e = Encoder.over_buffer(buf)
    for op in ops:
        if op[0] == "b":
            e.write_sequence_begin_lazy(op[1])
        elif op[0] == "e":
            e.write_sequence_end()
        else:
            e.write_unsigned(op[1], op[2])
    return bytes(buf[: e.bytes_used()])


class Log(Visitor):
    """Records every scope and unsigned event, tagged with its own name.

    ``decline`` lists the ids answered ``False``; ``children`` maps an id to
    the visitor answered for it. ``table`` is declared through
    ``destinations()``, the route a generated backend takes."""

    def __init__(self, log, name="v", table=None, decline=(), children=None):
        self.log = log
        self.name = name
        self.table = table
        self.words = bytearray(8 * 16)
        self.decline = frozenset(decline)
        self.children = children or {}

    def destinations(self):
        if self.table is None:
            return None
        return (self.table, self.words, None)

    def on_sequence_begin(self, field_id):
        self.log.append((self.name, "begin", field_id))
        if field_id in self.decline:
            return False
        return self.children.get(field_id, True)

    def on_sequence_end(self):
        self.log.append((self.name, "end"))

    def on_unsigned(self, field_id, value):
        self.log.append((self.name, "u", field_id, value))


def _decode(dec_cls, msg, visitor, split=None):
    d = dec_cls(visitor=visitor, **NO_CAPS)
    if split is None:
        assert d.feed(msg) == Status.COMPLETE
    else:
        d.feed(msg[:split])
        assert d.feed(msg[split:]) == Status.COMPLETE
    return visitor


def _every_split(dec_cls, msg, make, expected):
    """One-shot and every two-chunk split agree on the events and the slots."""
    log: list = []
    v = _decode(dec_cls, msg, make(log))
    assert log == expected
    for cut in range(1, len(msg)):
        log2: list = []
        v2 = _decode(dec_cls, msg, make(log2), split=cut)
        assert log2 == expected, f"split at {cut}"
        assert v2.words == v.words, f"split at {cut}"
    return v


def _slots(v):
    return memoryview(v.words).cast("Q").tolist()


# { 1: { 0: 7 }, 2: 9 } -- the reproducer from the issue.
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_bound_sequence_is_silent_to_the_visitor(dec_cls):
    msg = _wire(("b", 1), ("u", 0, 7), ("e",), ("u", 2, 9))

    def make(log):
        child = Binding().unsigned(0, at=0, count_at=1)
        return Log(log, table=Binding().sequence(1, child=child, count_at=2))

    v = _every_split(dec_cls, msg, make, [("v", "u", 2, 9)])
    assert _slots(v)[:3] == [7, 1, 1]


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_bound_and_unbound_sequences_side_by_side(dec_cls):
    msg = _wire(
        ("b", 1), ("u", 0, 7), ("e",),
        ("b", 3), ("u", 0, 8), ("e",),
        ("b", 1), ("u", 0, 5), ("e",),
        ("u", 4, 9),
    )  # fmt: skip

    def make(log):
        child = Binding().unsigned(0, at=0, count_at=1)
        return Log(log, table=Binding().sequence(1, child=child, count_at=2))

    v = _every_split(dec_cls, msg, make, [
        ("v", "begin", 3), ("v", "u", 0, 8), ("v", "end"),
        ("v", "u", 4, 9),
    ])  # fmt: skip
    # A scalar's count_at is its arrival flag, not a tally; the sequence's
    # counts both occurrences.
    assert _slots(v)[:3] == [5, 1, 2]


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_an_unbound_sequence_inside_a_bound_one(dec_cls):
    # The child table binds id 0 only; the sequence 5 inside it is the visitor's
    # and is delivered balanced, and nothing of the bound scope around it is.
    msg = _wire(
        ("b", 1), ("u", 0, 7), ("b", 5), ("u", 0, 8), ("e",), ("e",),
        ("u", 2, 9),
    )  # fmt: skip

    def make(log):
        child = Binding().unsigned(0, at=0, count_at=1)
        return Log(log, table=Binding().sequence(1, child=child, count_at=2))

    v = _every_split(dec_cls, msg, make, [
        ("v", "begin", 5), ("v", "u", 0, 8), ("v", "end"),
        ("v", "u", 2, 9),
    ])  # fmt: skip
    assert _slots(v)[:3] == [7, 1, 1]


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_bound_sequence_inside_a_bound_one(dec_cls):
    msg = _wire(
        ("b", 1), ("b", 2), ("u", 0, 7), ("e",), ("u", 3, 4), ("e",),
        ("u", 6, 9),
    )  # fmt: skip

    def make(log):
        inner = Binding().unsigned(0, at=0)
        outer = Binding().sequence(2, child=inner)
        return Log(log, table=Binding().sequence(1, child=outer))

    v = _every_split(dec_cls, msg, make, [("v", "u", 3, 4), ("v", "u", 6, 9)])
    assert _slots(v)[0] == 7


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_declined_sequence_has_no_end(dec_cls):
    msg = _wire(
        ("b", 3), ("b", 4), ("u", 0, 8), ("e",), ("e",),
        ("b", 1), ("u", 0, 7), ("e",),
        ("u", 2, 9),
    )  # fmt: skip

    def make(log):
        child = Binding().unsigned(0, at=0)
        return Log(log, table=Binding().sequence(1, child=child), decline={3})

    v = _every_split(dec_cls, msg, make, [("v", "begin", 3), ("v", "u", 2, 9)])
    assert _slots(v)[0] == 7


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_child_visitor_keeps_its_scope_through_a_nested_end(dec_cls):
    # No table at all: the child named for sequence 1 handles everything up to
    # sequence 1's own end, including what follows a sequence nested inside it.
    msg = _wire(
        ("b", 1), ("b", 2), ("u", 0, 7), ("e",), ("u", 3, 8), ("e",),
        ("u", 4, 9),
    )  # fmt: skip

    def make(log):
        return Log(log, "a", children={1: Log(log, "b")})

    _every_split(dec_cls, msg, make, [
        ("a", "begin", 1),
        ("b", "begin", 2), ("b", "u", 0, 7), ("b", "end"),
        ("b", "u", 3, 8), ("b", "end"),
        ("a", "u", 4, 9),
    ])  # fmt: skip


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_child_visitor_beside_a_bound_sequence(dec_cls):
    msg = _wire(
        ("b", 3), ("b", 2), ("u", 0, 8), ("e",), ("e",),
        ("b", 1), ("u", 0, 7), ("e",),
        ("u", 4, 9),
    )  # fmt: skip

    def make(log):
        child = Binding().unsigned(0, at=0)
        return Log(log, "a", table=Binding().sequence(1, child=child), children={3: Log(log, "b")})

    v = _every_split(dec_cls, msg, make, [
        ("a", "begin", 3),
        ("b", "begin", 2), ("b", "u", 0, 8), ("b", "end"), ("b", "end"),
        ("a", "u", 4, 9),
    ])  # fmt: skip
    assert _slots(v)[0] == 7


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_reset_mid_scope_forgets_how_it_was_opened(dec_cls):
    # Stopped inside a child's nested scope, then reused: the next message is
    # the caller's handler's again, from the top.
    first = _wire(("b", 1), ("b", 2), ("u", 0, 7))
    second = _wire(("b", 5), ("u", 0, 1), ("e",), ("u", 4, 9))
    log: list = []
    a = Log(log, "a", children={1: Log(log, "b")})
    d = dec_cls(visitor=a, **NO_CAPS)
    d.feed(first)
    d.reset()
    del log[:]
    assert d.feed(second) == Status.COMPLETE
    assert log == [("a", "begin", 5), ("a", "u", 0, 1), ("a", "end"), ("a", "u", 4, 9)]
