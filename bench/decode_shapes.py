#!/usr/bin/env python3
"""What each decode shape costs, on one message.

A **supplementary, language-native view** (BENCH_SPEC): not one of the shared
comparison rows, because the shapes it compares are Python's. It exists because
corelib-py has three ways to decode the same bytes and the difference between
them is large enough to be an architecture decision rather than a preference:

  ``pull``          the caller drives — ``while d.next()`` plus an ``if
                    fld.id ==`` chain, one typed read per field. What generated
                    ``message.py`` does today.
  ``drive``         the corelib drives, the caller answers — ``Decoder.drive``
                    over a ``Visitor``, still a ``Field`` object and two Python
                    calls per field.
  ``feed_visitor``  push mode with a visitor: ``feed(chunk)`` parses, the
                    visitor is called per field.
  ``feed_bound``    push mode with a :class:`sofab.Binding`: the decoder writes
                    each value into caller-owned storage itself. Nothing crosses
                    into Python during the decode.
  ``feed_bound_read``  ``feed_bound`` plus the caller reading all 36 values back
                    out, one slot at a time — what a generated object with 36
                    typed attributes costs, as opposed to a consumer that reads
                    a few.
  ``feed_bound_bulk``  the same read-back, but the contiguous scalar slots come
                    across in one ``memoryview.tolist()`` instead of 24
                    indexing operations.
  ``feed_bound_fresh``  ``feed_bound`` with a new ``Decoder`` per message, so
                    the binding is compiled every time. Read against
                    ``feed_bound`` it prices ``Decoder.reset()``.

The message is shaped after the ``vehicle_telemetry.yaml`` of issue #109 —
**36 fields, 12 arrays, 51 elements** — and every driver decodes the same bytes,
so the rows are directly comparable to each other and to that issue's report.

Usage:
    python bench/decode_shapes.py <driver> [reps]     # for run_shapes_callgrind.sh
    python bench/decode_shapes.py selftest            # all drivers agree
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sofab import IMPL, Binding, Decoder, Encoder, Status, Visitor  # noqa: E402

GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1

# 36 fields, ids 1..36. Every third id is an unsigned array (12 of them), the
# rest unsigned scalars (24). Array lengths sum to 51, spread unevenly as a
# telemetry schema's would be.
NFIELDS = 36
ARRAY_IDS = tuple(range(3, NFIELDS + 1, 3))
ARRAY_LENS = {fid: n for fid, n in zip(ARRAY_IDS, (3, 4, 5, 4, 4, 4, 4, 5, 4, 4, 4, 6))}
SCALAR_IDS = tuple(fid for fid in range(1, NFIELDS + 1) if fid not in ARRAY_LENS)
assert len(ARRAY_IDS) == 12 and sum(ARRAY_LENS.values()) == 51

#: Every destination is sized by the caller from the schema, never from the wire
#: (§6.6): the longest array a conforming message may carry.
ARRAY_CAP = max(ARRAY_LENS.values())


def _scalar(fid: int) -> int:
    """A spread of unsigned values exercising 1..5-byte varints."""
    return ((fid * GOLDEN) & MASK64) >> (8 * (fid % 5))


def _elem(fid: int, i: int) -> int:
    return (((fid * 31 + i) * GOLDEN) & MASK64) >> 40


def build_msg() -> bytes:
    enc = Encoder()
    for fid in range(1, NFIELDS + 1):
        if fid in ARRAY_LENS:
            enc.write_unsigned_array(fid, [_elem(fid, i) for i in range(ARRAY_LENS[fid])])
        else:
            enc.write_unsigned(fid, _scalar(fid))
    enc.flush()
    return enc.getvalue()


def expected_sum() -> int:
    """Every scalar plus the first element of each array. Deliberately O(1) per
    array: generated code stores an array, it does not fold over it, and folding
    would put list iteration into the measurement of the drivers that hand one
    back."""
    return sum(
        _elem(fid, 0) if fid in ARRAY_LENS else _scalar(fid)
        for fid in range(1, NFIELDS + 1)
    )


# --- the binding every push driver shares ------------------------------------
#
# Scalars take one slot each; arrays take ARRAY_CAP consecutive slots; every
# field's arrival lands in a count slot. Exactly the table a generator would
# emit from the schema.

SCALAR_AT = {fid: i for i, fid in enumerate(SCALAR_IDS)}
COUNT_AT = {fid: len(SCALAR_IDS) + i for i, fid in enumerate(range(1, NFIELDS + 1))}
ARRAY_AT = {
    fid: len(SCALAR_IDS) + NFIELDS + i * ARRAY_CAP for i, fid in enumerate(ARRAY_IDS)
}


def make_binding() -> Binding:
    b = Binding()
    for fid in range(1, NFIELDS + 1):
        if fid in ARRAY_LENS:
            b.unsigned_array(fid, at=ARRAY_AT[fid], cap=ARRAY_CAP, count_at=COUNT_AT[fid])
        else:
            b.unsigned(fid, at=SCALAR_AT[fid], count_at=COUNT_AT[fid])
    return b


BINDING = make_binding()


def new_storage() -> tuple[bytearray, memoryview]:
    words = bytearray(BINDING.tree_words_required * 8)
    return words, memoryview(words).cast("Q")


# --- driver 1: the pull loop generated code runs today -----------------------
#
# Built as source and exec'd because that is what it is: generated code. The
# ``if fld.id ==`` chain is written out in id order, so a field at position k
# costs k comparisons and the average over the message is ~18.
def _make_pull_decoder():
    lines = [
        "def decode_pull(data, Decoder=Decoder, BytesIO=io.BytesIO):",
        "    d = Decoder(BytesIO(data))",
        "    acc = 0",
        "    while True:",
        "        fld = d.next()",
        "        if fld is None:",
        "            break",
    ]
    for k, fid in enumerate(range(1, NFIELDS + 1)):
        head = "if" if k == 0 else "elif"
        lines.append(f"        {head} fld.id == {fid}:")
        if fid in ARRAY_LENS:
            lines.append(f"            a{fid} = d.read_unsigned_array()")
            lines.append(f"            acc += a{fid}[0]")
        else:
            lines.append("            acc += d.unsigned()")
    lines += ["        else:", "            d.skip()", "    return acc"]
    ns: dict = {"Decoder": Decoder, "io": io}
    exec(compile("\n".join(lines), "<generated>", "exec"), ns)
    return ns["decode_pull"]


decode_pull = _make_pull_decoder()


# --- driver 2/3: the visitor, pulled and pushed ------------------------------


class _SumVisitor(Visitor):
    __slots__ = ("acc",)

    def __init__(self) -> None:
        self.acc = 0

    def on_unsigned(self, field_id: int, value: int) -> None:
        self.acc += value

    def on_unsigned_array(self, field_id: int, values: list) -> None:
        self.acc += values[0]


def decode_drive(data: bytes) -> int:
    v = _SumVisitor()
    Decoder(io.BytesIO(data)).drive(v)
    return v.acc


def make_feed_visitor():
    def body(data: bytes) -> int:
        v = _SumVisitor()
        Decoder(visitor=v).feed(data)
        return v.acc

    return body


# --- driver 4..6: the binding ------------------------------------------------


#: The slots to read back, as literal constants — which is what a generator
#: emits. Looking them up in a dict per field would measure the harness.
READ_SLOTS = tuple(SCALAR_AT[f] for f in SCALAR_IDS) + tuple(
    ARRAY_AT[f] for f in ARRAY_IDS
)


def _read_back(u: memoryview) -> int:
    """What a generated object with 36 typed attributes has to do: bring every
    decoded value across into Python. Measured separately because it is the
    caller's cost, not the decode's — and on this schema it is the larger of the
    two."""
    acc = 0
    for at in READ_SLOTS:
        acc += u[at]
    return acc


#: Scalars occupy slots 0..len(SCALAR_IDS)-1, so one slice materialises all of
#: them in a single C-level call instead of one indexing operation each.
def _read_back_bulk(u: memoryview) -> int:
    acc = 0
    for v in u[: len(SCALAR_IDS)].tolist():
        acc += v
    for at in READ_SLOTS[len(SCALAR_IDS):]:
        acc += u[at]
    return acc


def make_feed_bound():
    words, u = new_storage()
    dec = Decoder(binding=BINDING, words=words)

    def body(data: bytes) -> int:
        dec.reset()
        dec.feed(data)
        return 0

    return body


def make_feed_bound_read():
    words, u = new_storage()
    dec = Decoder(binding=BINDING, words=words)

    def body(data: bytes) -> int:
        dec.reset()
        dec.feed(data)
        return _read_back(u)

    return body


def make_feed_bound_bulk():
    words, u = new_storage()
    dec = Decoder(binding=BINDING, words=words)

    def body(data: bytes) -> int:
        dec.reset()
        dec.feed(data)
        return _read_back_bulk(u)

    return body


def make_feed_bound_fresh():
    def body(data: bytes) -> int:
        words, u = new_storage()
        Decoder(binding=BINDING, words=words).feed(data)
        return _read_back(u)

    return body


# --- workload plumbing (same protocol as bench/perfbench.py) -----------------

WORKLOADS = {
    "pull": decode_pull,
    "drive": decode_drive,
    "feed_visitor": make_feed_visitor,
    "feed_bound": make_feed_bound,
    "feed_bound_read": make_feed_bound_read,
    "feed_bound_bulk": make_feed_bound_bulk,
    "feed_bound_fresh": make_feed_bound_fresh,
}

#: Entries that are a factory to call once, during the excluded setup.
FACTORIES = frozenset(
    {
        "feed_visitor",
        "feed_bound",
        "feed_bound_read",
        "feed_bound_bulk",
        "feed_bound_fresh",
    }
)
#: Drivers that decode but hand nothing back, so they have no checksum.
NO_CHECKSUM = frozenset({"feed_bound"})


def resolve(name: str):
    fn = WORKLOADS[name]
    return fn() if name in FACTORIES else fn


def selftest() -> int:
    msg = build_msg()
    want = expected_sum()
    ok = True
    print(
        f"engine={IMPL} bytes={len(msg)} fields={NFIELDS} arrays={len(ARRAY_IDS)} "
        f"elements={sum(ARRAY_LENS.values())} words={BINDING.tree_words_required}"
    )
    for name in WORKLOADS:
        got = resolve(name)(msg)
        if name in NO_CHECKSUM:
            print(f"  {name:<18} ran (no checksum)")
            continue
        good = got == want
        ok &= good
        print(f"  {name:<18} {'ok' if good else f'MISMATCH got={got} want={want}'}")

    # Chunked: the same bytes one at a time must land the same values, and the
    # final status must be COMPLETE (§5.2 / §7.2 item 4).
    words, u = new_storage()
    dec = Decoder(binding=BINDING, words=words)
    st = Status.COMPLETE
    for i in range(len(msg)):
        st = dec.feed(msg[i : i + 1])
    good = st is Status.COMPLETE and _read_back(u) == want
    ok &= good
    print(f"  {'chunked@1':<18} {'ok' if good else 'MISMATCH'}")
    return 0 if ok else 1


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
    msg = build_msg()  # setup — excluded by the two-rep-count subtraction
    body = resolve(name)  # ditto
    sink = 0
    for _ in range(reps):
        sink += body(msg)
    print(f"sink={sink} bytes={len(msg)} reps={reps} engine={IMPL}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
