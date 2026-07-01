#!/usr/bin/env python3
"""Head-to-head: SofaBuffers (native + pure Python) vs protobuf-python.

Runs the four shared workloads (``encode``/``decode`` × ``u64 array``/``typical``)
against three back-ends and prints a best-of-N MB/s table:

  * ``sofab-native`` — the compiled Cython accelerator (``sofab._speedups``).
  * ``sofab-pure``   — the pure-Python fallback (``SOFAB_PUREPYTHON``).
  * ``protobuf``     — ``protobuf``'s Python API on an equivalent message,
                       parsing/serializing with **full materialization** so the
                       comparison is apples-to-apples with the SofaBuffers pull
                       API (protobuf's repeated-scalar container otherwise boxes
                       integers lazily, which would only touch the two elements
                       the checksum reads rather than all 1000).

protobuf's descriptors are built at run time, so no ``protoc`` is required; if
``protobuf`` is not installed that column is simply omitted.

Usage::

    python bench/compare_protobuf.py [best_of]   # default best_of=5
"""

from __future__ import annotations

import importlib
import io
import os
import sys
import time

GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
N = 1000
SRC = [(i * GOLDEN) & MASK64 for i in range(N)]
ARR16 = [10, 20, 30, 40]


def _measure(body, msg_bytes: int, best_of: int) -> float:
    """Best-of-N MB/s over a ~0.4s CPU-time loop each (MB = 1e6 bytes)."""
    best = 0.0
    for _ in range(best_of):
        body()  # warmup
        t0 = time.process_time()
        iters = 0
        el = 0.0
        while el < 0.4:
            body()
            iters += 1
            el = time.process_time() - t0
        best = max(best, msg_bytes * iters / el / 1e6)
    return best


# --- SofaBuffers back-end (native or pure, selected by env) ------------------


def _load_sofab(pure: bool):
    os.environ["SOFAB_PUREPYTHON"] = "1" if pure else "0"
    for m in list(sys.modules):
        if m.startswith("sofab"):
            del sys.modules[m]
    sofab = importlib.import_module("sofab")
    want = "python" if pure else "native"
    if sofab.IMPL != want:
        return None  # native not built → skip that column
    return sofab


def _sofab_bodies(sofab):
    Encoder, Decoder, WireType = sofab.Encoder, sofab.Decoder, sofab.WireType

    def enc_arr():
        e = Encoder(); e.write_unsigned_array(1, SRC); e.flush(); return e.getvalue()

    def enc_typ():
        e = Encoder()
        e.write_unsigned(1, 0xDEADBEEF); e.write_signed(2, -12345); e.write_bool(3, True)
        e.write_float32(4, 3.14159); e.write_string(5, "sofab"); e.write_unsigned_array(6, ARR16)
        e.write_sequence_begin(7); e.write_unsigned(1, 99); e.write_signed(2, -7); e.write_sequence_end()
        e.flush(); return e.getvalue()

    arr_bytes = enc_arr()
    typ_bytes = enc_typ()

    def dec_arr():
        d = Decoder(io.BytesIO(arr_bytes)); acc = 0
        while (f := d.next()) is not None:
            if f.type == WireType.ARRAY_UNSIGNED:
                a = d.read_unsigned_array(); acc += a[0] + a[-1]
            else:
                d.skip()
        return acc

    def dec_typ():
        d = Decoder(io.BytesIO(typ_bytes)); acc = 0
        while (f := d.next()) is not None:
            if f.id == 1 and f.type == WireType.UNSIGNED:
                acc += d.unsigned()
            elif f.id == 2 and f.type == WireType.SIGNED:
                acc += d.signed() & MASK64
            elif f.id == 3:
                acc += 1 if d.bool() else 0
            elif f.id == 4:
                acc += int(d.float32())
            elif f.id == 5:
                acc += len(d.string())
            elif f.id == 6:
                acc += d.read_unsigned_array()[0]
            elif f.type == WireType.SEQUENCE_START:
                while (g := d.next()) is not None and g.type != WireType.SEQUENCE_END:
                    if g.id == 1:
                        acc += d.unsigned()
                    elif g.id == 2:
                        acc += d.signed() & MASK64
                    else:
                        d.skip()
            else:
                d.skip()
        return acc

    return {
        "encode: u64 array (1000)": (enc_arr, len(arr_bytes)),
        "encode: typical message": (enc_typ, len(typ_bytes)),
        "decode: u64 array (1000)": (dec_arr, len(arr_bytes)),
        "decode: typical message": (dec_typ, len(typ_bytes)),
    }


# --- protobuf back-end (dynamic descriptors, full materialization) -----------


def _protobuf_bodies():
    try:
        from google.protobuf import descriptor_pb2 as dpb
        from google.protobuf import descriptor_pool, message_factory
    except ImportError:
        return None

    pool = descriptor_pool.DescriptorPool()
    fdp = dpb.FileDescriptorProto(name="cmp.proto", syntax="proto3")
    F = dpb.FieldDescriptorProto

    nested = fdp.message_type.add(name="Nested")
    nested.field.add(name="a", number=1, label=F.LABEL_OPTIONAL, type=F.TYPE_UINT64)
    nested.field.add(name="b", number=2, label=F.LABEL_OPTIONAL, type=F.TYPE_SINT64)

    typ = fdp.message_type.add(name="Typical")
    typ.field.add(name="f1", number=1, label=F.LABEL_OPTIONAL, type=F.TYPE_UINT64)
    typ.field.add(name="f2", number=2, label=F.LABEL_OPTIONAL, type=F.TYPE_SINT64)
    typ.field.add(name="f3", number=3, label=F.LABEL_OPTIONAL, type=F.TYPE_BOOL)
    typ.field.add(name="f4", number=4, label=F.LABEL_OPTIONAL, type=F.TYPE_FLOAT)
    typ.field.add(name="f5", number=5, label=F.LABEL_OPTIONAL, type=F.TYPE_STRING)
    typ.field.add(name="f6", number=6, label=F.LABEL_REPEATED, type=F.TYPE_UINT64)
    typ.field.add(name="f7", number=7, label=F.LABEL_OPTIONAL, type=F.TYPE_MESSAGE, type_name=".Nested")

    arr = fdp.message_type.add(name="U64Array")
    arr.field.add(name="vals", number=1, label=F.LABEL_REPEATED, type=F.TYPE_UINT64)

    pool.Add(fdp)
    get = (message_factory.GetMessageClass if hasattr(message_factory, "GetMessageClass")
           else message_factory.MessageFactory().GetPrototype)
    Typical = get(pool.FindMessageTypeByName("Typical"))
    U64Array = get(pool.FindMessageTypeByName("U64Array"))

    def enc_arr():
        m = U64Array(); m.vals.extend(SRC); return m.SerializeToString()

    def enc_typ():
        m = Typical(); m.f1 = 0xDEADBEEF; m.f2 = -12345; m.f3 = True; m.f4 = 3.14159
        m.f5 = "sofab"; m.f6.extend(ARR16); m.f7.a = 99; m.f7.b = -7
        return m.SerializeToString()

    arr_bytes = enc_arr()
    typ_bytes = enc_typ()

    def dec_arr():
        m = U64Array(); m.ParseFromString(arr_bytes)
        a = list(m.vals)  # full materialization, like read_unsigned_array()
        return a[0] + a[-1]

    def dec_typ():
        m = Typical(); m.ParseFromString(typ_bytes)
        return (m.f1 + (m.f2 & MASK64) + (1 if m.f3 else 0) + int(m.f4)
                + len(m.f5) + list(m.f6)[0] + m.f7.a + (m.f7.b & MASK64))

    return {
        "encode: u64 array (1000)": (enc_arr, len(arr_bytes)),
        "encode: typical message": (enc_typ, len(typ_bytes)),
        "decode: u64 array (1000)": (dec_arr, len(arr_bytes)),
        "decode: typical message": (dec_typ, len(typ_bytes)),
    }


WORKLOADS = [
    "encode: u64 array (1000)",
    "encode: typical message",
    "decode: u64 array (1000)",
    "decode: typical message",
]


def main(argv: list[str]) -> int:
    best_of = int(argv[1]) if len(argv) > 1 else 5

    columns: list[tuple[str, dict]] = []
    native = _load_sofab(pure=False)
    if native is not None:
        columns.append(("sofab-native", _sofab_bodies(native)))
    pure = _load_sofab(pure=True)
    if pure is not None:
        columns.append(("sofab-pure", _sofab_bodies(pure)))
    pb = _protobuf_bodies()
    if pb is not None:
        columns.append(("protobuf", pb))

    results: dict[str, dict[str, float]] = {name: {} for name, _ in columns}
    for name, bodies in columns:
        for wl in WORKLOADS:
            body, nbytes = bodies[wl]
            results[name][wl] = _measure(body, nbytes, best_of)

    head = f"{'Workload':<26}" + "".join(f"{name:>16}" for name, _ in columns)
    print("=== throughput MB/s (best of %d, MB = 1e6 bytes) ===" % best_of)
    print(head)
    print("-" * len(head))
    for wl in WORKLOADS:
        row = f"{wl:<26}"
        for name, _ in columns:
            row += f"{results[name][wl]:>16.2f}"
        print(row)
    if any(n == "sofab-native" for n, _ in columns) and any(n == "protobuf" for n, _ in columns):
        print("\nspeedup (sofab-native / protobuf):")
        for wl in WORKLOADS:
            r = results["sofab-native"][wl] / results["protobuf"][wl]
            print(f"  {wl:<26} {r:>5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
