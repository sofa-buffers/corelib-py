#!/usr/bin/env python3
"""SofaBuffers Python — performance tools.

Implements BENCH_SPEC (the cross-language benchmark specification): the same
workloads on the same data, measured the same way and printed in the same
grammar as `corelib-rs/benches/bench.rs`, `bench/c/bench.c` and
`corelib-go/cmd/perfbench`, so the numbers line up across languages. Three
views, one per tool BENCH_SPEC requires:

  * ``bench`` (alias ``time``) — throughput in **MB/s** on *this* machine. A
    "speedtest" for the library on the current host, measured against process
    CPU time over a ~1s loop per workload (MB = 1e6 bytes).

  * ``perf`` — per-op cost (CPU time/op in ns + MB/s) for the shared 12-field
    "perf" message, in the same format as the C/C++/Rust/… per-op tools. CPython
    exposes no portable hardware cycle counter, so cycles/op is reported
    unavailable and CPU time/op (process CPU time) is the comparable metric.

  * ``<workload> [reps]`` — runs one workload ``reps`` times after an excluded
    one-time setup, then prints ``sink``/``bytes`` to stderr. This is the mode
    driven by ``run_callgrind.sh`` to obtain **instructions/op**, a cost metric
    that is independent of the CPU clock speed and OS scheduler (see that
    script for how the fixed startup cost is cancelled out).

Workloads (the keys of :data:`WORKLOADS`, in the order ``bench`` prints them)::

    encode_u64_array   encode_typical   encode_blob_oneshot
    encode_blob_stream encode_composite
    decode_u64_array   decode_typical   decode_blob
    decode_composite   decode_composite_skip

Usage:
    python bench/perfbench.py bench
    python bench/perfbench.py perf
    python bench/perfbench.py encode_typical 1000

``SOFAB_BENCH_SECONDS`` shortens the per-workload measurement loop (default
``1.0``); it exists for the format test, not for reportable numbers.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Callable

from sofab import IMPL, Decoder, Encoder, WireType

N = 1000
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
ARR16 = [10, 20, 30, 40]

#: Payload size of the ``blob 1MB`` message, and its encoded size: a 1-byte
#: field header, a 4-byte fixlen word and the payload (BENCH_SPEC). Both are
#: parity checks across ports.
BLOB_N = 1_000_000
BLOB_ENCODED = 1_000_005

#: The fixed buffer/chunk size the streaming blob rows are driven with — the
#: same 4096 on every port, so the rows stay comparable (BENCH_SPEC). It is not
#: this port's own buffer size and has nothing to do with MIN_OUTPUT_BUFFER,
#: which is at most 20 and therefore always satisfied here.
STREAM_BUFFER = 4096


def make_src() -> list[int]:
    """A spread of unsigned values exercising 1..10-byte varints."""
    return [(i * GOLDEN) & MASK64 for i in range(N)]


# ---- message builders (identical ids/values to the C/Rust/Go tools) ---------


def encode_u64_array(src: list[int]) -> bytes:
    enc = Encoder()
    enc.write_unsigned_array(1, src)
    enc.flush()
    return enc.getvalue()


def encode_typical(enc: Encoder) -> None:
    enc.write_unsigned(1, 0xDEADBEEF)
    enc.write_signed(2, -12345)
    enc.write_bool(3, True)
    enc.write_float32(4, 3.14159)
    enc.write_string(5, "sofab")
    enc.write_unsigned_array(6, ARR16)
    enc.write_sequence_begin_lazy(7)
    enc.write_unsigned(1, 99)
    enc.write_signed(2, -7)
    enc.write_sequence_end()


def encode_typical_msg() -> bytes:
    enc = Encoder()
    encode_typical(enc)
    enc.flush()
    return enc.getvalue()


# ---- decode workloads (fold values into a checksum so nothing is elided) -----


def decode_u64_array(data: bytes) -> int:
    dec = Decoder(io.BytesIO(data))
    acc = 0
    while (f := dec.next()) is not None:
        if f.type == WireType.ARRAY_UNSIGNED:
            a = dec.read_unsigned_array()
            acc += a[0] + a[-1]
        else:
            dec.skip()
    return acc


def decode_typical(data: bytes) -> int:
    dec = Decoder(io.BytesIO(data))
    acc = 0
    while (f := dec.next()) is not None:
        if f.id == 1 and f.type == WireType.UNSIGNED:
            acc += dec.unsigned()
        elif f.id == 2 and f.type == WireType.SIGNED:
            acc += dec.signed() & MASK64
        elif f.id == 3:
            acc += 1 if dec.bool() else 0
        elif f.id == 4:
            acc += int(dec.float32())
        elif f.id == 5:
            acc += len(dec.string())
        elif f.id == 6:
            acc += dec.read_unsigned_array()[0]
        elif f.type == WireType.SEQUENCE_START:
            while (g := dec.next()) is not None and g.type != WireType.SEQUENCE_END:
                if g.id == 1:
                    acc += dec.unsigned()
                elif g.id == 2:
                    acc += dec.signed() & MASK64
                else:
                    dec.skip()
        else:
            dec.skip()
    return acc


# ---- blob 1MB (BENCH_SPEC "blob 1MB") ---------------------------------------
#
# One field, id 1, type blob, declared *without* maxlen — the point being that
# the message is larger than any buffer a caller could pre-size from the schema.
# blob rather than string on purpose: a megabyte of UTF-8 would put the §6.4
# validator in the measurement and dominate it. This workload measures buffer
# handling.


def make_blob() -> bytes:
    """``b[i] = (i * GOLDEN) & 0xFF`` for i in 0..999_999.

    Built from one 256-byte period rather than a million-iteration loop: the low
    byte of ``i * GOLDEN`` depends only on ``i & 0xFF`` (arithmetic mod 256), so
    the sequence repeats every 256 indices. Identical bytes, and cheap enough
    that the setup does not dominate a Callgrind run. ``tests/test_bench_spec``
    checks it against the literal formula over the whole million.
    """
    period = bytes(((i * GOLDEN) & 0xFF) for i in range(256))
    return (period * (BLOB_N // 256 + 1))[:BLOB_N]


class _DiscardSink:
    """A flush sink that consumes and discards (BENCH_SPEC).

    It must not accumulate the bytes and must not do I/O: an accumulator would
    add to the streaming row a copy the one-shot row never pays, and I/O is not
    deterministic under Callgrind. XOR-ing one byte per call is the minimum that
    keeps the call from being elided.

    It returns without installing a buffer, which is §5.1's "the sink copied":
    the active buffer stays active and the encoder resumes at 0 — i.e. this is
    the copy path, with pass-through *not* granted (this port implements no
    pass-through at all, so BENCH_SPEC's optional third blob row is omitted
    rather than printed as a placeholder).
    """

    __slots__ = ("xor",)

    def __init__(self) -> None:
        self.xor = 0

    def __call__(self, chunk: bytes) -> None:
        self.xor ^= chunk[0]


class _ChunkReader:
    """Feeds a decoder in fixed-size chunks, however much it asks for.

    The cap is the point: :class:`~sofab.Decoder` requests as much as the
    construct in flight still needs, so without a cap here a "streaming" decode
    of a megabyte would be handed the megabyte in one read and measure nothing
    of the refill path.
    """

    __slots__ = ("_data", "_pos", "_chunk")

    def __init__(self, data: bytes, chunk: int) -> None:
        self._data = data
        self._pos = 0
        self._chunk = chunk

    def read(self, n: int) -> bytes:
        take = n if n < self._chunk else self._chunk
        end = self._pos + take
        out = self._data[self._pos : end]
        self._pos += len(out)
        return out


def encode_blob_oneshot(enc: Encoder, blob: bytes) -> int:
    """The floor: one contiguous write into a caller buffer, no sink, no flush."""
    enc.write_bytes(1, blob)
    return enc.bytes_used()


def encode_blob_stream(enc: Encoder, blob: bytes) -> int:
    """The same bytes through ~245 flushes of a 4096-byte caller buffer."""
    enc.write_bytes(1, blob)
    return enc.flush()


def decode_blob(data: bytes) -> int:
    dec = Decoder(_ChunkReader(data, STREAM_BUFFER), chunk_size=STREAM_BUFFER)
    acc = 0
    while (f := dec.next()) is not None:
        if f.type == WireType.FIXLEN:
            payload = dec.bytes()
            acc += len(payload) + payload[0]
        else:
            dec.skip()
    return acc


# ---- composite (BENCH_SPEC "composite") -------------------------------------
#
# The other datasets are flat: every field present, every array a compact scalar
# array, one sequence one level deep, every id a one-byte header. This one
# exercises the paths that leaves unrun — a wrapper array (a header per element,
# MESSAGE_SPEC §5.1, with element ids straddling the one-byte header boundary),
# a non-ASCII string through the §6.4 validator, nesting at depth 3, a field the
# encoder must *omit* (the hold-back's discard path), and a two-byte field
# header.

COMPOSITE_ITEMS = [f"item-{i}" for i in range(64)]
COMPOSITE_TEXT = "aä€𝄞" * 32  # 320 UTF-8 bytes: 1-, 2-, 3- and 4-byte sequences

#: Encoded size of the composite message — the parity check across ports, as
#: the perf message's 170 is.
COMPOSITE_ENCODED = 956


def encode_composite(enc: Encoder) -> None:
    # 1: string array in wrapper form — one field header per element, element id
    #    = 0-based array index (§5.1). Ids 0..15 are one-byte headers, 16..63
    #    two-byte ones.
    enc.write_sequence_begin_lazy(1)
    for i, item in enumerate(COMPOSITE_ITEMS):
        enc.write_string(i, item)
    enc.write_sequence_end()
    # 2: non-ASCII string, through the UTF-8 validator.
    enc.write_string(2, COMPOSITE_TEXT)
    # 3: { 1: { 1: { 1: unsigned 7 } }, 2: signed -1 } — depth 3, so the lazy
    #    hold-back run grows past the single level typical/perf reach.
    enc.write_sequence_begin_lazy(3)
    enc.write_sequence_begin_lazy(1)
    enc.write_sequence_begin_lazy(1)
    enc.write_unsigned(1, 7)
    enc.write_sequence_end()
    enc.write_sequence_end()
    enc.write_signed(2, -1)
    enc.write_sequence_end()
    # 4: a struct equal to its declared default. Every child is then equal to its
    #    own default and is omitted, so the sequence never receives content and
    #    write_sequence_end discards the held-back frame (MESSAGE_SPEC §2) — the
    #    one field in the suite the encoder is required to *not* write.
    enc.write_sequence_begin_lazy(4)
    enc.write_sequence_end()
    # 130: the only two-byte field header in the suite — (130 << 3) | 0.
    enc.write_unsigned(130, 0xDEADBEEF)


def encode_composite_msg() -> bytes:
    enc = Encoder()
    encode_composite(enc)
    enc.flush()
    return enc.getvalue()


def _decode_seq(dec: Decoder) -> int:
    """Read one open sequence to its end, recursing into nested ones."""
    acc = 0
    while (g := dec.next()) is not None and g.type != WireType.SEQUENCE_END:
        if g.type == WireType.SEQUENCE_START:
            acc += _decode_seq(dec)
        elif g.type == WireType.UNSIGNED:
            acc += dec.unsigned()
        elif g.type == WireType.SIGNED:
            acc += dec.signed() & MASK64
        elif g.type == WireType.FIXLEN:
            acc += len(dec.string())
        else:
            dec.skip()
    return acc


def decode_composite(data: bytes) -> int:
    """Whole message, all fields read."""
    dec = Decoder(io.BytesIO(data))
    acc = 0
    while (f := dec.next()) is not None:
        if f.type == WireType.SEQUENCE_START:
            acc += _decode_seq(dec)
        elif f.type == WireType.UNSIGNED:
            acc += dec.unsigned()
        elif f.type == WireType.FIXLEN:
            acc += len(dec.string())
        else:
            dec.skip()
    return acc


def decode_composite_skip(data: bytes) -> int:
    """Same bytes, every field and sub-sequence skipped — the path a router or a
    filter runs: walk the message, materialize nothing. ``skip()`` on a sequence
    start consumes the whole sub-tree."""
    dec = Decoder(io.BytesIO(data))
    n = 0
    while dec.next() is not None:
        dec.skip()
        n += 1
    return n


# ---- workload registry ------------------------------------------------------
#
# One definition of each workload, shared by the throughput loop and the
# Callgrind rep runner, so the two can never drift apart. A builder does all the
# setup — message building, buffer allocation — and returns the body to measure
# plus the encoded size the MB/s figure is computed from. Setup is excluded from
# the timed loop, and is identical at both Callgrind rep counts, so the
# subtraction cancels it exactly.

Body = Callable[[], int]
Workload = Callable[[], "tuple[Body, int]"]


def _w_encode_u64_array() -> tuple[Callable[[], int], int]:
    src = make_src()
    nbytes = len(encode_u64_array(src))
    return (lambda: len(encode_u64_array(src))), nbytes


def _w_encode_typical() -> tuple[Callable[[], int], int]:
    nbytes = len(encode_typical_msg())
    return (lambda: len(encode_typical_msg())), nbytes


def _w_encode_blob_oneshot() -> tuple[Callable[[], int], int]:
    blob = make_blob()
    # Sized by hand, not from a generated MAX_SIZE: this schema is unbounded, so
    # its MAX_SIZE would be the configured ceiling rather than a size the message
    # cannot exceed. No sink, so no minimum applies to the buffer and the
    # message has to fit exactly.
    buf = bytearray(BLOB_ENCODED)
    return (lambda: encode_blob_oneshot(Encoder.over_buffer(buf), blob)), BLOB_ENCODED


def _w_encode_blob_stream() -> tuple[Callable[[], int], int]:
    blob = make_blob()
    buf = bytearray(STREAM_BUFFER)
    sink = _DiscardSink()
    return (lambda: encode_blob_stream(Encoder.over_buffer(buf, 0, sink), blob)), BLOB_ENCODED


def _w_encode_composite() -> tuple[Callable[[], int], int]:
    nbytes = len(encode_composite_msg())
    buf = bytearray(nbytes)

    def body() -> int:
        enc = Encoder.over_buffer(buf)
        encode_composite(enc)
        return enc.bytes_used()

    return body, nbytes


def _w_decode_u64_array() -> tuple[Callable[[], int], int]:
    data = encode_u64_array(make_src())
    return (lambda: decode_u64_array(data)), len(data)


def _w_decode_typical() -> tuple[Callable[[], int], int]:
    data = encode_typical_msg()
    return (lambda: decode_typical(data)), len(data)


def _w_decode_blob() -> tuple[Callable[[], int], int]:
    enc = Encoder()
    enc.write_bytes(1, make_blob())
    enc.flush()
    data = enc.getvalue()
    return (lambda: decode_blob(data)), len(data)


def _w_decode_composite() -> tuple[Callable[[], int], int]:
    data = encode_composite_msg()
    return (lambda: decode_composite(data)), len(data)


def _w_decode_composite_skip() -> tuple[Callable[[], int], int]:
    data = encode_composite_msg()
    return (lambda: decode_composite_skip(data)), len(data)


#: ``key -> (BENCH_SPEC row label, builder)``, in the order the rows print.
WORKLOADS: dict[str, tuple[str, Workload]] = {
    "encode_u64_array": ("encode: u64 array (1000)", _w_encode_u64_array),
    "encode_typical": ("encode: typical message", _w_encode_typical),
    "encode_blob_oneshot": ("encode: blob 1MB one-shot", _w_encode_blob_oneshot),
    "encode_blob_stream": ("encode: blob 1MB streaming", _w_encode_blob_stream),
    "encode_composite": ("encode: composite", _w_encode_composite),
    "decode_u64_array": ("decode: u64 array (1000)", _w_decode_u64_array),
    "decode_typical": ("decode: typical message", _w_decode_typical),
    "decode_blob": ("decode: blob 1MB", _w_decode_blob),
    "decode_composite": ("decode: composite", _w_decode_composite),
    "decode_composite_skip": ("decode: composite skip-all", _w_decode_composite_skip),
}


# ---- throughput (MB/s) ------------------------------------------------------
#
# The clock is read once per *batch*, not once per operation: a
# ``time.process_time()`` call costs about a microsecond, which is a fixed cost
# per operation rather than a scaling factor — barely visible on a 1000-element
# array, dominant on a 37-byte message. Each batch is grown until it spans
# BATCH_SECONDS, so one clock reading is a rounding error against what it
# measures. Calibration doubles as extra warmup.

BATCH_SECONDS = 0.01  # clock cost lands under ~0.01% of a batch

#: Length of the measurement loop, in CPU seconds. BENCH_SPEC's ~1s; the
#: environment override exists so the format test can run the whole table in a
#: fraction of a second, and never for a number anyone reports.
LOOP_SECONDS = float(os.environ.get("SOFAB_BENCH_SECONDS", "1.0"))


def _calibrate_batch(body) -> int:
    """Grow a batch until it spans BATCH_SECONDS."""
    batch = 1
    while True:
        t0 = time.process_time()
        for _ in range(batch):
            body()
        if time.process_time() - t0 >= BATCH_SECONDS:
            return batch
        batch *= 2


def _loop(body) -> tuple[int, float]:
    """Run ``body`` (after a warmup) for ~LOOP_SECONDS of CPU time →
    (iterations, elapsed CPU seconds)."""
    body()  # warmup
    batch = _calibrate_batch(body)
    t0 = time.process_time()
    iters = 0
    el = 0.0
    while el < LOOP_SECONDS:
        for _ in range(batch):
            body()
        iters += batch
        el = time.process_time() - t0
    return iters, el


def measure(body, msg_bytes: int) -> float:
    """Run ``body`` for ~1s of CPU time (after a warmup) → MB/s (MB = 1e6)."""
    iters, el = _loop(body)
    return msg_bytes * iters / el / 1e6


def run_timed() -> None:
    print(f"=== SofaBuffers Python throughput (CPU time, MB/s) [engine: {IMPL}] ===")
    print(f"{'Workload':<26} {'MB/s':>12}")
    print(f"{'--------':<26} {'----':>12}")
    for label, build in WORKLOADS.values():
        body, nbytes = build()
        print(f"{label:<26} {measure(body, nbytes):>12.2f}")
    print("\nMB = 1e6 bytes. ~1s CPU-time loop per workload.")


# ---- per-op cost (perf) -----------------------------------------------------
#
# The 12-field "perf" message — identical ids/types/values to perf.c, perf.cpp
# and corelib-rs/benches/perf.rs — measured over a ~1s process-CPU-time loop and
# printed in the shared per-op format.

PERF_STRING = "perf-benchmark-message"
PERF_SAMPLES = [1_000_000, 2_000_000, 3_000_000, 4_000_000,
                5_000_000, 6_000_000, 7_000_000, 8_000_000]
PERF_DELTAS = [-100_000, -200_000, -300_000, -400_000, -500_000, -600_000, -700_000, -800_000]
PERF_FP64 = [3.14159265, 6.28318530, 9.42477795, 12.56637060]


def encode_perf(enc: Encoder) -> None:
    enc.write_unsigned(1, 0xDEADBEEF)
    enc.write_signed(2, -12345)
    enc.write_unsigned(3, 0x0123456789ABCDEF)
    enc.write_signed(4, -5_000_000_000_000)
    enc.write_bool(5, True)
    enc.write_float32(6, 3.14159)
    enc.write_float64(7, 2.718281828459045)
    enc.write_string(8, PERF_STRING)
    enc.write_unsigned_array(9, PERF_SAMPLES)
    enc.write_signed_array(10, PERF_DELTAS)
    enc.write_float64_array(11, PERF_FP64)
    enc.write_sequence_begin_lazy(12)
    enc.write_unsigned(1, 99)
    enc.write_signed(2, -7)
    enc.write_sequence_end()


def encode_perf_msg() -> bytes:
    enc = Encoder()
    encode_perf(enc)
    enc.flush()
    return enc.getvalue()


def decode_perf(data: bytes) -> int:
    """Decode the perf message, folding every value into a checksum."""
    dec = Decoder(io.BytesIO(data))
    acc = 0
    while (f := dec.next()) is not None:
        if f.type == WireType.SEQUENCE_START:
            while (g := dec.next()) is not None and g.type != WireType.SEQUENCE_END:
                if g.id == 1:
                    acc += dec.unsigned()
                elif g.id == 2:
                    acc += dec.signed() & MASK64
                else:
                    dec.skip()
        elif f.id in (1, 3):
            acc += dec.unsigned()
        elif f.id in (2, 4):
            acc += dec.signed() & MASK64
        elif f.id == 5:
            acc += 1 if dec.bool() else 0
        elif f.id == 6:
            acc += int(dec.float32())
        elif f.id == 7:
            acc += int(dec.float64())
        elif f.id == 8:
            acc += len(dec.string())
        elif f.id == 9:
            a = dec.read_unsigned_array()
            acc += a[0] + a[-1]
        elif f.id == 10:
            a = dec.read_signed_array()
            acc += (a[0] + a[-1]) & MASK64
        elif f.id == 11:
            a = dec.read_float64_array()
            acc += int(a[0])
        else:
            dec.skip()
    return acc


def measure_perop(body, msg_bytes: int) -> tuple[int, float, float]:
    """Run ``body`` for ~1s CPU time → (iterations, ns/op, MB/s)."""
    iters, el = _loop(body)
    return iters, el / iters * 1e9, msg_bytes * iters / el / 1e6


def perf_report(what: str, iters: int, ns_op: float, mb_s: float, msg_bytes: int) -> None:
    print(f"\n--- perf: {what} ---")
    print(f"  iterations    : {iters}")
    print(f"  message size  : {msg_bytes} bytes")
    print("  cycles/op     : (cycle counter unavailable on CPython)")
    print(f"  CPU time/op   : {ns_op:.1f} ns  (process CPU time, not wall-clock)")
    print(f"  throughput    : {mb_s:.1f} MB/s  (speedtest, MB = 1e6 bytes)")


def run_perf() -> None:
    msg = encode_perf_msg()
    nbytes = len(msg)

    print(f"=== SofaBuffers Python per-op cost (cycles/op + throughput MB/s) [engine: {IMPL}] ===")

    it, ns, mb = measure_perop(encode_perf_msg, nbytes)
    perf_report("serialize (stream API)", it, ns, mb, nbytes)

    it, ns, mb = measure_perop(lambda: decode_perf(msg), nbytes)
    perf_report("deserialize (stream API)", it, ns, mb, nbytes)

    print("\ncycles/op tracks code cost; MB/s is this machine's throughput.")


# ---- single workload, N reps (for Callgrind instructions/op) ----------------


def run_workload(name: str, reps: int) -> None:
    entry = WORKLOADS.get(name)
    if entry is None:
        print(f"unknown workload: {name}", file=sys.stderr)
        print(f"known: {' '.join(WORKLOADS)}", file=sys.stderr)
        raise SystemExit(2)
    body, nbytes = entry[1]()  # setup — excluded, and identical at every rep count
    sink = 0
    for _ in range(reps):
        sink += body()
    # to stderr so it doesn't pollute Callgrind's stdout capture
    print(f"sink={sink} bytes={nbytes} reps={reps}", file=sys.stderr)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] in ("bench", "time"):  # "bench" is BENCH_SPEC's name for the tool
        run_timed()
        return 0
    if argv[1] == "perf":
        run_perf()
        return 0
    reps = int(argv[2]) if len(argv) > 2 else 1000
    run_workload(argv[1], reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
