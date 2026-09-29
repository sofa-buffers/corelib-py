#!/usr/bin/env python3
"""What a union field costs: the visitor path vs. the destination table (#164).

A **supplementary, language-native view** (BENCH_SPEC), like
``bench/decode_shapes.py``: the shapes it compares are Python's, not one of the
shared comparison rows. It exists because a union is the one schema construct a
:class:`sofab.Binding` could not express until ``which_at``, so a union field
left the C loop for four Python calls per message — and that is a large enough
difference to decide where unions belong.

MESSAGE_SPEC §4.2: a union is a sequence carrying one child, and the child's id
is the only tag it has. §7.4.1: the held option is the last correctly-typed
occurrence of any option id. A one-of table records that id in a ``words`` slot.

The message is eight bound scalar fields plus one union field whose held option
is an ``fp32`` — the shape of the generator bench's ``aux_sensor`` (``$ref
SensorSample``, scalar options, ``default_id: 0``), which is where the
regression #164 reports was measured.

  ``plain_bound``          the same message WITHOUT the union field, every field
                           on the table. The control: what a message that has no
                           union pays for the feature existing.
  ``union_bound``          the union on the table, ``which_at`` holding the id.
  ``union_visitor``        the union off the table (an open scope), a handler
                           taking ``on_sequence_begin`` / ``on_float32`` /
                           ``on_sequence_end`` -- what the Python backend emits
                           today.
  ``union_visitor_field``  the same handler, also overriding ``on_field``: the
                           four calls #164 counts.

Usage:
    python bench/union_shapes.py <driver> [reps]   # for run_union_callgrind.sh
    python bench/union_shapes.py selftest          # both readings agree
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sofab import (  # noqa: E402
    ARRAY_MAX,
    FIXLEN_MAX,
    IMPL,
    Binding,
    Decoder,
    Encoder,
    Status,
    Visitor,
)

# CORELIB_PLAN §6.2.1: the receiver caps are the CALLER's numbers and a Decoder
# has no default for them, so every construction below states all three. They
# are the format ceilings (§6.2), at which a cap cannot fire -- a benchmark
# measures the decode, not a policy. Same reasoning as bench/decode_shapes.py.
CAPS = dict(
    max_dyn_array_count=ARRAY_MAX,
    max_dyn_string_len=FIXLEN_MAX,
    max_dyn_blob_len=FIXLEN_MAX,
    reassembly=1 << 21,
)

N_PLAIN = 8          # ordinary bound fields, ids 0..7
UNION_ID = 21        # the union field in the root scope
OPT_F32 = 1          # the held option's id inside the union's scope
W_WHICH, W_F32 = N_PLAIN, N_PLAIN + 1


def build_msg(with_union: bool) -> bytes:
    enc = Encoder()
    for i in range(N_PLAIN):
        enc.write_unsigned(i, 1000 + i)
    if with_union:
        enc.write_sequence_begin_lazy(UNION_ID)
        enc.write_float32(OPT_F32, 2.5)
        enc.write_sequence_end()
    enc.flush()
    return enc.getvalue()


def _plain_rows(b: Binding) -> Binding:
    for i in range(N_PLAIN):
        b.unsigned(i, at=i)
    return b


def bound_table(with_union: bool) -> Binding:
    root = _plain_rows(Binding(closed=True))
    if with_union:
        option = Binding(closed=True, which_at=W_WHICH).float32(OPT_F32, at=W_F32)
        root.sequence(UNION_ID, option)
    return root


def open_table() -> Binding:
    # The union's id is not named and the table is OPEN, so the union reaches
    # the visitor -- what the Python backend does for a union at any depth.
    return _plain_rows(Binding())


class UnionHandler(Visitor):
    """The generated-handler shape: the table takes the scalars, the hooks take
    the union, and the handler applies §7.4.1 itself."""

    def __init__(self, table: Binding, words: bytearray) -> None:
        self._table = table
        self._words = words
        self.held = -1
        self.value = 0.0

    def destinations(self):
        return (self._table, self._words, None)

    def on_sequence_begin(self, field_id):
        return None

    def on_float32(self, field_id, value):
        self.held = field_id
        self.value = value

    def on_sequence_end(self):
        pass


class UnionHandlerField(UnionHandler):
    def on_field(self, field):
        return None


def make_visitor(handler_cls):
    """Decoder and destinations built once, as in bench/decode_shapes.py: the
    row prices the decode, not the construction."""
    table = open_table()
    words = bytearray(table.tree_words_required * 8)
    handler = handler_cls(table, words)
    dec = Decoder(visitor=handler, **CAPS)

    def body(msg: bytes) -> int:
        dec.reset()
        dec.feed(msg)
        return 0

    return body


def make_bound(with_union: bool):
    table = bound_table(with_union)
    words = bytearray(table.tree_words_required * 8)
    dec = Decoder(binding=table, words=words, objects=[], **CAPS)

    def body(msg: bytes) -> int:
        dec.reset()
        dec.feed(msg)
        return 0

    return body


WORKLOADS = {
    "plain_bound": lambda: make_bound(False),
    "union_bound": lambda: make_bound(True),
    "union_visitor": lambda: make_visitor(UnionHandler),
    "union_visitor_field": lambda: make_visitor(UnionHandlerField),
}


def selftest() -> int:
    """The two readings of one message agree about the held option."""
    msg = build_msg(True)

    table = bound_table(True)
    words = bytearray(table.tree_words_required * 8)
    view = memoryview(words)
    dec = Decoder(binding=table, words=words, objects=[], **CAPS)
    assert dec.feed(msg) is Status.COMPLETE
    u, d = view.cast("Q"), view.cast("d")
    assert [u[i] for i in range(N_PLAIN)] == [1000 + i for i in range(N_PLAIN)]
    assert u[W_WHICH] == OPT_F32, u[W_WHICH]
    assert d[W_F32] == 2.5, d[W_F32]

    htable = open_table()
    handler = UnionHandler(htable, bytearray(htable.tree_words_required * 8))
    dec = Decoder(visitor=handler, **CAPS)
    assert dec.feed(msg) is Status.COMPLETE
    assert handler.held == OPT_F32 and handler.value == 2.5

    print(
        f"selftest ok (engine={IMPL}, {len(msg)} bytes, "
        f"{len(msg) - len(build_msg(False))} of them the union)"
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "selftest":
        return selftest()
    name = argv[1]
    if name not in WORKLOADS:
        print(f"unknown driver: {name}; known: {' '.join(WORKLOADS)}", file=sys.stderr)
        return 2
    reps = int(argv[2]) if len(argv) > 2 else 100
    msg = build_msg(name != "plain_bound")  # setup -- cancelled by the subtraction
    body = WORKLOADS[name]()                # ditto
    sink = 0
    for _ in range(reps):
        sink += body(msg)
    print(f"sink={sink} bytes={len(msg)} reps={reps} engine={IMPL}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
