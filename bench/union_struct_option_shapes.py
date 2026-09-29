#!/usr/bin/env python3
"""What a struct union option costs: the destination table vs. the visitor (#169).

The sister row to ``bench/union_shapes.py``, which priced a union whose options
are **scalars**. This one prices the option kind that made the generator keep a
union off the table at all: a ``struct`` option whose members carry declared
defaults, including a ``string`` and an ``array``.

MESSAGE_SPEC §7.4.1: a re-selected option "starts from its own default". For a
struct option that default is its rows', so the table has to restore each member
the arriving frame leaves out. Until #169 the reset wrote the *empty* value over
a ``string``/``array`` member's declared default, so
``generators/python/binding.go`` (``resetLosesDefault``) sent the whole union to
the visitor instead -- correct, but four Python calls per message plus the
handler's own bookkeeping.

  ``plain_bound``            the same message WITHOUT the union. The control.
  ``struct_bound``           the union on the table, no member default declared:
                             what #167 already did.
  ``struct_bound_defaults``  the union on the table WITH the member defaults --
                             what #169 adds. The reset list is longer by the
                             string slot and the array's elements, and the reset
                             runs on every message whose held option is not
                             ``default_id``, so this is the row that says what
                             the fix costs.
  ``struct_bound_into``      the same, with the ``string`` member bound through
                             ``string_into``: the one reset payload that copies
                             rather than stores, into a buffer the caller owns.
  ``struct_visitor``         the union off the table: a handler taking
                             ``on_sequence_begin`` / ``on_string`` /
                             ``on_unsigned`` / ``on_sequence_end`` and applying
                             §7.4.1 itself, defaults and all -- the workaround
                             this change is meant to retire.
  ``struct_visitor_field``   the same handler, also overriding ``on_field``.

Usage:
    python bench/union_struct_option_shapes.py <driver> [reps]
    python bench/union_struct_option_shapes.py selftest
    python bench/union_struct_option_shapes.py time [reps]
"""

from __future__ import annotations

import os
import sys
import time

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

CAPS = dict(
    max_dyn_array_count=ARRAY_MAX,
    max_dyn_string_len=FIXLEN_MAX,
    max_dyn_blob_len=FIXLEN_MAX,
    reassembly=1 << 21,
)

N_PLAIN = 8            # ordinary bound fields, ids 0..7
UNION_ID = 21          # the union field in the root scope
OPT_TIME, OPT_META = 0, 1          # the union's options; default_id = OPT_TIME
M_ISO, M_DATA, M_LVL = 0, 1, 2     # the struct option's members

# words: 0..7 plain, 8 which, 9 time, 10 lvl, 11 data count, 12..15 data
W_WHICH = N_PLAIN
W_TIME, W_LVL, W_DCOUNT, W_DATA = N_PLAIN + 1, N_PLAIN + 2, N_PLAIN + 3, N_PLAIN + 4
W_ILEN = N_PLAIN + 4 + 4        # the string_into row's byte-length slot
O_ISO = 0              # objects

DATA_CAP = 4
ISO_DEFAULT = "1970-01-01"
DATA_DEFAULT = (0, 0, 0, 0)
ISO_SENT, LVL_SENT = "2026-09-29", 7


def build_msg(with_union: bool) -> bytes:
    """The plain fields, then -- for the union rows -- a ``meta`` option that
    carries only ``lvl``. Held option != ``default_id``, so the decode runs the
    reset once, which is the realistic shape: one frame per message (§4.2 has a
    conformant encoder always write a non-default option's id)."""
    enc = Encoder()
    for i in range(N_PLAIN):
        enc.write_unsigned(i, 1000 + i)
    if with_union:
        enc.write_sequence_begin_lazy(UNION_ID)
        enc.write_sequence_begin_lazy(OPT_META)
        enc.write_unsigned(M_LVL, LVL_SENT)
        enc.write_sequence_end()
        enc.write_sequence_end()
    enc.flush()
    return enc.getvalue()


def _plain_rows(b: Binding) -> Binding:
    for i in range(N_PLAIN):
        b.unsigned(i, at=i)
    return b


def _member_table(defaults: bool, into: bool = False) -> Binding:
    meta = Binding(closed=True)
    if into:
        # The one reset payload that COPIES: the slot holds the caller's buffer,
        # so the default's bytes go into it and the length slot says how many.
        meta.string_into(M_ISO, at=O_ISO, maxlen=32, count_at=W_ILEN,
                         default=ISO_DEFAULT)
        meta.unsigned_array(M_DATA, at=W_DATA, cap=DATA_CAP, count_at=W_DCOUNT,
                            elem_max=0xFF, default=DATA_DEFAULT)
        meta.unsigned(M_LVL, at=W_LVL, max_value=0xFF)
        return meta
    if defaults:
        meta.string(M_ISO, at=O_ISO, maxlen=32, default=ISO_DEFAULT)
        meta.unsigned_array(M_DATA, at=W_DATA, cap=DATA_CAP, count_at=W_DCOUNT,
                            elem_max=0xFF, default=DATA_DEFAULT)
    else:
        meta.string(M_ISO, at=O_ISO, maxlen=32)
        meta.unsigned_array(M_DATA, at=W_DATA, cap=DATA_CAP, count_at=W_DCOUNT,
                            elem_max=0xFF)
    meta.unsigned(M_LVL, at=W_LVL, max_value=0xFF)
    return meta


def bound_table(with_union: bool, defaults: bool = False,
                into: bool = False) -> Binding:
    root = _plain_rows(Binding(closed=True))
    if with_union:
        option = Binding(closed=True, which_at=W_WHICH, default_id=OPT_TIME)
        option.unsigned(OPT_TIME, at=W_TIME, max_value=0xFFFFFFFF)
        option.sequence(OPT_META, _member_table(defaults, into))
        root.sequence(UNION_ID, option)
    return root


def open_table() -> Binding:
    # The union's id is not named and the table is OPEN, so the union reaches the
    # visitor -- what the Python backend emits while `resetLosesDefault` holds.
    return _plain_rows(Binding())


class StructHandler(Visitor):
    """The generated-handler shape the workaround produces: the scalars stay on
    the table, the union goes through hooks, and the handler applies §7.4.1 --
    including putting every member back at its declared default."""

    def __init__(self, table: Binding, words: bytearray) -> None:
        self._table = table
        self._words = words
        self._depth = 0
        self.held = OPT_TIME
        self.iso = ISO_DEFAULT
        self.data = DATA_DEFAULT
        self.lvl = 0
        self.time = 0

    def destinations(self):
        return (self._table, self._words, None)

    def on_sequence_begin(self, field_id):
        self._depth += 1
        if self._depth == 2:
            # The option's id IS the selection (§4.2). A different one discards
            # the held option and starts the new one from its own default.
            if self.held != field_id:
                self.held = field_id
                self.iso = ISO_DEFAULT
                self.data = DATA_DEFAULT
                self.lvl = 0
        return None

    def on_sequence_end(self):
        self._depth -= 1

    def on_unsigned(self, field_id, value):
        if self._depth == 2:
            if field_id == M_LVL:
                self.lvl = value
        elif self._depth == 1:
            if self.held != field_id:
                self.held = field_id
                self.time = 0
            self.time = value

    def on_string(self, field_id, value):
        self.iso = value


class StructHandlerField(StructHandler):
    def on_field(self, field):
        return None


def make_visitor(handler_cls):
    table = open_table()
    words = bytearray(table.tree_words_required * 8)
    handler = handler_cls(table, words)
    dec = Decoder(visitor=handler, **CAPS)

    def body(msg: bytes) -> int:
        # Each op is a FRESH message: the held option goes back to `default_id`,
        # exactly as the bound rows put it back in `words`. Without this the
        # option only switches on the first op of the run and the reset -- the
        # thing being priced -- is measured once instead of every time.
        handler.held = OPT_TIME
        dec.reset()
        dec.feed(msg)
        return 0

    return body


def make_bound(with_union: bool, defaults: bool = False, switch: bool = True,
               into: bool = False):
    table = bound_table(with_union, defaults, into)
    words = bytearray(table.tree_words_required * 8)
    objects: list[object] = [None] * max(1, table.tree_objects_required)
    if into:
        objects[O_ISO] = bytearray(32)   # the buffer the caller owns
    view = memoryview(words).cast("Q")
    dec = Decoder(binding=table, words=words, objects=objects, **CAPS)
    which = W_WHICH if with_union else 0
    prepared = OPT_TIME if switch else OPT_META

    def body(msg: bytes) -> int:
        # §4.2: the caller prepares `which_at` with the schema's `default_id`.
        # One store, and the same store the visitor row pays -- see above.
        # `switch=False` prepares it with the option the message actually holds
        # instead, so the reset never fires: the difference between the two rows
        # is what the reset costs, which is the only thing #169 makes bigger.
        view[which] = prepared
        dec.reset()
        dec.feed(msg)
        return 0

    return body


WORKLOADS = {
    "plain_bound": lambda: make_bound(False),
    "struct_bound": lambda: make_bound(True, False),
    "struct_bound_defaults": lambda: make_bound(True, True),
    "struct_bound_no_reset": lambda: make_bound(True, True, switch=False),
    "struct_bound_into": lambda: make_bound(True, True, into=True),
    "struct_visitor": lambda: make_visitor(StructHandler),
    "struct_visitor_field": lambda: make_visitor(StructHandlerField),
}


def selftest() -> int:
    """Every reading of one message agrees -- and only the defaults row is
    §7.4.1-correct, which is the point of the change."""
    msg = build_msg(True)

    table = bound_table(True, defaults=True)
    words = bytearray(table.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    objects: list[object] = [None] * table.tree_objects_required
    dec = Decoder(binding=table, words=words, objects=objects, **CAPS)
    assert dec.feed(msg) is Status.COMPLETE
    assert [view[i] for i in range(N_PLAIN)] == [1000 + i for i in range(N_PLAIN)]
    assert view[W_WHICH] == OPT_META, view[W_WHICH]
    assert view[W_LVL] == LVL_SENT, view[W_LVL]
    assert objects[O_ISO] == ISO_DEFAULT, objects[O_ISO]
    got = tuple(view[W_DATA + i] for i in range(view[W_DCOUNT]))
    assert got == DATA_DEFAULT, got

    # Without the declared defaults the same table loses them -- #169's symptom,
    # kept here so the two rows are known to differ in exactly that.
    table = bound_table(True, defaults=False)
    words = bytearray(table.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    objects = [None] * table.tree_objects_required
    objects[O_ISO] = ISO_DEFAULT
    dec = Decoder(binding=table, words=words, objects=objects, **CAPS)
    assert dec.feed(msg) is Status.COMPLETE
    assert objects[O_ISO] == "", objects[O_ISO]

    htable = open_table()
    handler = StructHandler(htable, bytearray(htable.tree_words_required * 8))
    dec = Decoder(visitor=handler, **CAPS)
    assert dec.feed(msg) is Status.COMPLETE
    assert handler.held == OPT_META and handler.lvl == LVL_SENT
    assert handler.iso == ISO_DEFAULT and handler.data == DATA_DEFAULT

    print(
        f"selftest ok (engine={IMPL}, {len(msg)} bytes, "
        f"{len(msg) - len(build_msg(False))} of them the union)"
    )
    return 0


def timings(reps: int) -> int:
    """Wall-clock per message per workload, cheapest of five runs."""
    print(f"engine={IMPL}  reps={reps}  (best of 5)")
    base = None
    for name, factory in WORKLOADS.items():
        msg = build_msg(name != "plain_bound")
        body = factory()
        best = min(_one_run(body, msg, reps) for _ in range(5))
        per = best / reps * 1e6
        if base is None:
            base = per
        print(f"  {name:24} {per:8.3f} us/msg   {per / base:5.2f}x plain_bound")
    return 0


def _one_run(body, msg: bytes, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        body(msg)
    return time.perf_counter() - t0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "selftest":
        return selftest()
    if argv[1] == "time":
        return timings(int(argv[2]) if len(argv) > 2 else 20000)
    name = argv[1]
    if name not in WORKLOADS:
        print(f"unknown driver: {name}; known: {' '.join(WORKLOADS)}", file=sys.stderr)
        return 2
    reps = int(argv[2]) if len(argv) > 2 else 100
    msg = build_msg(name != "plain_bound")
    body = WORKLOADS[name]()
    sink = 0
    for _ in range(reps):
        sink += body(msg)
    print(f"sink={sink} bytes={len(msg)} reps={reps} engine={IMPL}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
