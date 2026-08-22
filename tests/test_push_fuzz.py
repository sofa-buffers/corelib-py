"""Differential fuzz: the push paths must decode what the pull path decodes.

The pull decoder is the one already validated against the shared conformance
vectors, so the strongest statement available about the push paths is that they
agree with it — on randomly generated messages, at several chunkings, on both
engines. CORELIB_PLAN §7.2 item 4 is the rule being enforced: where the chunk
boundaries fall must never change the outcome.

Seeds are fixed, so a failure is reproducible from its number alone.
"""

from __future__ import annotations

import io
import random

import pytest
from test_push_feed import Collect
from vectors import ENGINE_PAIRS

from sofab import Binding, FixlenSubtype, Status, WireType

#: Kept small enough that the suite stays fast; the same generator with a wider
#: range is what a longer soak would use.
SEEDS = 120

CAP = 8
MAXLEN = 40
SEQ_ID = 20
#: One fixed wire type per field id, the way a schema fixes them. The binding
#: below is built from exactly this table.
SCHEMA = {
    1: "u", 2: "s", 3: "f32", 4: "f64", 5: "str",
    6: "blob", 7: "ua", 8: "sa", 9: "f32a", 10: "f64a",
}


def _emit(rng: random.Random, enc, allow_seq: bool = True) -> None:
    for _ in range(rng.randint(0, 10)):
        if allow_seq and rng.random() < 0.15:
            enc.write_sequence_begin_lazy(SEQ_ID)
            _emit(rng, enc, allow_seq=False)
            enc.write_sequence_end_keep()
            continue
        fid = rng.choice(list(SCHEMA))
        kind = SCHEMA[fid]
        n = rng.randint(0, CAP)
        if kind == "u":
            enc.write_unsigned(fid, rng.getrandbits(rng.choice([1, 7, 64])))
        elif kind == "s":
            enc.write_signed(fid, rng.randint(-(1 << 62), 1 << 62))
        elif kind == "f32":
            enc.write_float32(fid, rng.uniform(-1e6, 1e6))
        elif kind == "f64":
            enc.write_float64(fid, rng.uniform(-1e18, 1e18))
        elif kind == "str":
            enc.write_string(fid, "".join(rng.choice("aä€𝄞") for _ in range(rng.randint(0, 9))))
        elif kind == "blob":
            enc.write_bytes(fid, bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 20))))
        elif kind == "ua":
            enc.write_unsigned_array(fid, [rng.getrandbits(40) for _ in range(n)])
        elif kind == "sa":
            enc.write_signed_array(fid, [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)])
        elif kind == "f32a":
            enc.write_float32_array(fid, [rng.uniform(-1e6, 1e6) for _ in range(n)])
        else:
            enc.write_float64_array(fid, [rng.uniform(-1e9, 1e9) for _ in range(n)])


def _message(enc_cls, seed: int) -> bytes:
    enc = enc_cls()
    _emit(random.Random(seed), enc)
    enc.flush()
    return enc.getvalue()


# --- the visitor path --------------------------------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 5, 4096])
def test_push_visitor_matches_the_pull_driver(enc_cls, dec_cls, chunk):
    for seed in range(SEEDS):
        msg = _message(enc_cls, seed)
        want = Collect()
        dec_cls(io.BytesIO(msg)).drive(want)

        got = Collect()
        dec = dec_cls(visitor=got)
        st = Status.COMPLETE
        for off in range(0, len(msg), chunk):
            st = dec.feed(msg[off : off + chunk])
        assert st is Status.COMPLETE, seed
        assert got.events == want.events, seed


# --- the bound path ----------------------------------------------------------
#
# Slot layout: the outer scope's numeric fields from slot 0, the sequence's from
# slot 200, each scope's arrival flags from its own base + 100 + id, and one
# string plus one blob slot per scope in ``objects``. Disjoint on purpose — a
# child binding shares the parent's storage, so overlapping them would be the
# caller's bug, not the decoder's.


def _binding() -> Binding:
    outer, inner = Binding(), Binding()
    for table, base_w, base_o in ((outer, 0, 0), (inner, 200, 8)):
        at = base_w
        for fid, kind in SCHEMA.items():
            count_at = base_w + 100 + fid
            if kind == "u":
                table.unsigned(fid, at=at, count_at=count_at)
            elif kind == "s":
                table.signed(fid, at=at, count_at=count_at)
            elif kind == "f32":
                table.float32(fid, at=at, count_at=count_at)
            elif kind == "f64":
                table.float64(fid, at=at, count_at=count_at)
            elif kind == "str":
                table.string(fid, at=base_o, maxlen=MAXLEN, count_at=count_at)
                continue
            elif kind == "blob":
                table.bytes(fid, at=base_o + 1, maxlen=MAXLEN, count_at=count_at)
                continue
            elif kind == "ua":
                table.unsigned_array(fid, at=at, cap=CAP, count_at=count_at)
            elif kind == "sa":
                table.signed_array(fid, at=at, cap=CAP, count_at=count_at)
            elif kind == "f32a":
                table.float32_array(fid, at=at, cap=CAP, count_at=count_at)
            else:
                table.float64_array(fid, at=at, cap=CAP, count_at=count_at)
            at += 1 if kind in ("u", "s", "f32", "f64") else CAP
    return outer.sequence(SEQ_ID, inner, count_at=99)


def _pull_last(dec_cls, msg: bytes) -> dict:
    """The last value seen per ``(scope, id)`` — which is what a binding holds
    when a field repeats, so it is what the comparison has to be against."""
    dec = dec_cls(io.BytesIO(msg))
    out: dict = {}
    depth = 0
    while (f := dec.next()) is not None:
        t = f.type
        if t is WireType.SEQUENCE_START:
            depth += 1
            continue
        if t is WireType.SEQUENCE_END:
            depth -= 1
            continue
        key = (min(depth, 1), f.id)
        if t is WireType.UNSIGNED:
            out[key] = dec.unsigned()
        elif t is WireType.SIGNED:
            out[key] = dec.signed()
        elif t is WireType.FIXLEN:
            st = f.subtype
            out[key] = (
                dec.float32() if st is FixlenSubtype.FP32
                else dec.float64() if st is FixlenSubtype.FP64
                else dec.string() if st is FixlenSubtype.STRING
                else dec.bytes()
            )
        elif t is WireType.ARRAY_UNSIGNED:
            out[key] = tuple(dec.read_unsigned_array())
        elif t is WireType.ARRAY_SIGNED:
            out[key] = tuple(dec.read_signed_array())
        else:
            out[key] = tuple(
                dec.read_float32_array() if f.subtype is FixlenSubtype.FP32
                else dec.read_float64_array()
            )
    return out


def _read_slots(binding: Binding, words: bytearray, objects: list) -> dict:
    mv = memoryview(words)
    u, q, d = mv.cast("Q"), mv.cast("q"), mv.cast("d")
    out: dict = {}
    for scope, base_w, base_o in ((0, 0, 0), (1, 200, 8)):
        at = base_w
        for fid, kind in SCHEMA.items():
            arrived = u[base_w + 100 + fid]
            if kind in ("str", "blob"):
                if arrived:
                    out[(scope, fid)] = objects[base_o + (0 if kind == "str" else 1)]
                continue
            if kind in ("u", "s", "f32", "f64"):
                if arrived:
                    view = u if kind == "u" else q if kind == "s" else d
                    out[(scope, fid)] = view[at]
                at += 1
                continue
            view = u if kind == "ua" else q if kind == "sa" else d
            out[(scope, fid)] = tuple(view[at : at + arrived])
            at += CAP
    return out


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 5, 4096])
def test_bound_destinations_match_the_pull_reads(enc_cls, dec_cls, chunk):
    binding = _binding()
    for seed in range(SEEDS):
        msg = _message(enc_cls, seed)
        want = _pull_last(dec_cls, msg)

        words = bytearray(binding.tree_words_required * 8)
        objects: list = [None] * binding.tree_objects_required
        dec = dec_cls(binding=binding, words=words, objects=objects)
        st = Status.COMPLETE
        for off in range(0, len(msg), chunk):
            st = dec.feed(msg[off : off + chunk])
        assert st is Status.COMPLETE, seed

        got = _read_slots(binding, words, objects)
        for key, value in want.items():
            assert key in got, (seed, key)
            assert got[key] == value, (seed, key)
