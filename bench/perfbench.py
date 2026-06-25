#!/usr/bin/env python3
"""SofaBuffers Python — performance tools.

Mirrors `bench/c/bench.c`, `corelib-rs/benches/bench.rs` and
`corelib-go/cmd/perfbench`: the same four workloads, with identical field ids,
types and values, so the numbers line up across languages. Two complementary
views:

  * ``time`` — throughput in **MB/s** on *this* machine. A "speedtest" for the
    library on the current host, measured against process CPU time over a ~1s
    loop per workload (MB = 1e6 bytes).

  * ``<workload> [reps]`` — runs one workload ``reps`` times after an excluded
    one-time setup, then prints ``sink``/``bytes`` to stderr. This is the mode
    driven by ``run_callgrind.sh`` to obtain **instructions/op**, a cost metric
    that is independent of the CPU clock speed and OS scheduler (see that
    script for how the fixed startup cost is cancelled out).

Workloads: ``encode_u64_array``, ``encode_typical``, ``decode_u64_array``,
``decode_typical``.

Usage:
    python bench/perfbench.py time
    python bench/perfbench.py encode_typical 1000
"""

from __future__ import annotations

import io
import sys
import time

from sofab import Decoder, Encoder, WireType

N = 1000
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
ARR16 = [10, 20, 30, 40]


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
    enc.write_sequence_begin(7)
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


# ---- throughput (MB/s) ------------------------------------------------------


def measure(body, msg_bytes: int) -> float:
    """Run ``body`` for ~1s of CPU time (after a warmup) → MB/s (MB = 1e6)."""
    body()  # warmup
    t0 = time.process_time()
    iters = 0
    el = 0.0
    while True:
        body()
        iters += 1
        el = time.process_time() - t0
        if el >= 1.0:
            break
    return msg_bytes * iters / el / 1e6


def run_timed() -> None:
    src = make_src()
    u64 = encode_u64_array(src)
    typ = encode_typical_msg()
    ba, bt = len(u64), len(typ)

    enc_u64 = measure(lambda: encode_u64_array(src), ba)
    enc_typ = measure(encode_typical_msg, bt)
    dec_u64 = measure(lambda: decode_u64_array(u64), ba)
    dec_typ = measure(lambda: decode_typical(typ), bt)

    print("=== SofaBuffers Python throughput (CPU time, MB/s) ===")
    print(f"{'Workload':<26} {'MB/s':>12}")
    print(f"{'--------':<26} {'----':>12}")
    print(f"{'encode: u64 array (1000)':<26} {enc_u64:>12.2f}")
    print(f"{'encode: typical message':<26} {enc_typ:>12.2f}")
    print(f"{'decode: u64 array (1000)':<26} {dec_u64:>12.2f}")
    print(f"{'decode: typical message':<26} {dec_typ:>12.2f}")
    print("\nMB = 1e6 bytes. ~1s CPU-time loop per workload.")


# ---- single workload, N reps (for Callgrind instructions/op) ----------------


def run_workload(name: str, reps: int) -> None:
    src = make_src()
    sink = 0
    nbytes = 0

    if name == "encode_u64_array":
        nbytes = len(encode_u64_array(src))  # setup: learn size (cancels out)
        out = b""
        for _ in range(reps):
            out = encode_u64_array(src)
        sink = len(out)
    elif name == "encode_typical":
        nbytes = len(encode_typical_msg())
        out = b""
        for _ in range(reps):
            out = encode_typical_msg()
        sink = len(out)
    elif name == "decode_u64_array":
        data = encode_u64_array(src)
        nbytes = len(data)
        for _ in range(reps):
            sink += decode_u64_array(data)
    elif name == "decode_typical":
        data = encode_typical_msg()
        nbytes = len(data)
        for _ in range(reps):
            sink += decode_typical(data)
    else:
        print(f"unknown workload: {name}", file=sys.stderr)
        raise SystemExit(2)

    # to stderr so it doesn't pollute Callgrind's stdout capture
    print(f"sink={sink} bytes={nbytes} reps={reps}", file=sys.stderr)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "time":
        run_timed()
        return 0
    reps = int(argv[2]) if len(argv) > 2 else 1000
    run_workload(argv[1], reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
