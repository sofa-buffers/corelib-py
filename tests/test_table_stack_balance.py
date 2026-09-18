"""A visitor scope nested in a visitor scope keeps the table suppressed (#152).

Entering a sequence the visitor handles suppresses the table (§4.9 opens a fresh
id scope), and every SEQUENCE_END undoes one entry. The pure engine used to
suppress only when a map was live, so a second visitor scope inside the first
pushed nothing — yet its END still popped, and the root table matched ids inside
a scope that was the visitor's. Observable only two scopes deep: a visitor
sequence, a nested visitor sequence, then a sibling whose id the ROOT table names.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, Recorder, Status, capped, encode


def _message(e) -> None:
    e.write_sequence_begin_lazy(7)      # visitor scope A
    e.write_sequence_begin_lazy(0)      #   visitor scope B, nested
    e.write_unsigned(1, 0)
    e.write_sequence_end()              #   end of B
    e.write_sequence_begin_lazy(2)      #   A's own id 2 — which the ROOT names
    e.write_unsigned(0, 5)
    e.write_sequence_end()
    e.write_sequence_end()              # end of A
    e.write_sequence_begin_lazy(2)      # the root's id 2: the table's
    e.write_unsigned(0, 9)
    e.write_sequence_end()


DATA = encode(_message)


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
def test_nested_visitor_scope_keeps_the_table_out(engine, chunk):
    root = Binding().sequence(2, child=Binding().unsigned(0, at=0, count_at=1),
                              count_at=2)

    class Handler(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.words = bytearray(root.tree_words_required * 8)

        def destinations(self):
            return (root, self.words, None)

    h = Handler()
    dec = engine(visitor=h, **capped())
    step = chunk or len(DATA)
    status = None
    for i in range(0, len(DATA), step):
        status = dec.feed(DATA[i:i + step])
    assert status == Status.COMPLETE
    assert h.events == [
        ("seq{", 7), ("seq{", 0), ("u", 1, 0), ("seq}",),
        ("seq{", 2), ("u", 0, 5), ("seq}",), ("seq}",),
    ]
    # Only the root-level id 2 reached the table: value 9, one occurrence.
    assert memoryview(h.words).cast("Q").tolist() == [9, 1, 1]
