"""Differential fuzz: the two push paths must agree, at every chunking.

A binding writes into raw memory through a compiled table and a visitor hands
values to Python; they are separate code paths over the same parser, so making
them agree on randomly generated messages is the strongest statement available
about either. CORELIB_PLAN §7.2 item 4 is the other rule enforced here: where
the chunk boundaries fall must never change the outcome.

Seeds are fixed, so a failure is reproducible from its number alone.
"""

from __future__ import annotations

import random

import pytest
from test_push_feed import Collect
from vectors import ENGINE_PAIRS, NO_CAPS

from sofab import Binding, Status, Visitor

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
def test_push_visitor_is_chunking_independent(enc_cls, dec_cls, chunk):
    for seed in range(SEEDS):
        msg = _message(enc_cls, seed)
        want = Collect()
        assert dec_cls(**NO_CAPS, visitor=want).feed(msg) is Status.COMPLETE

        got = Collect()
        dec = dec_cls(**NO_CAPS, visitor=got)
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


class _LastPerScope(Visitor):
    """The last value seen per ``(scope, id)``.

    A binding holds the last value when a field repeats, so that is what the
    comparison has to be against. ``scope`` is 0 at the top level and 1 inside
    any sequence — the fuzz nests one deep.
    """

    def __init__(self) -> None:
        self.out: dict = {}
        self.depth = 0

    def on_sequence_begin(self, field_id):
        self.depth += 1
        return None

    def on_sequence_end(self):
        self.depth -= 1

    def _put(self, field_id, value):
        self.out[(min(self.depth, 1), field_id)] = value

    def on_unsigned(self, field_id, value):
        self._put(field_id, value)

    def on_signed(self, field_id, value):
        self._put(field_id, value)

    def on_float32(self, field_id, value):
        self._put(field_id, value)

    def on_float64(self, field_id, value):
        self._put(field_id, value)

    def on_string(self, field_id, value):
        self._put(field_id, value)

    def on_bytes(self, field_id, value):
        self._put(field_id, value)

    def on_unsigned_array(self, field_id, values):
        self._put(field_id, tuple(values))

    def on_signed_array(self, field_id, values):
        self._put(field_id, tuple(values))

    def on_float32_array(self, field_id, values):
        self._put(field_id, tuple(values))

    def on_float64_array(self, field_id, values):
        self._put(field_id, tuple(values))


def _visitor_last(dec_cls, msg: bytes) -> dict:
    v = _LastPerScope()
    assert dec_cls(**NO_CAPS, visitor=v).feed(msg) is Status.COMPLETE
    return v.out


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
def test_bound_destinations_match_the_visitor(enc_cls, dec_cls, chunk):
    binding = _binding()
    for seed in range(SEEDS):
        msg = _message(enc_cls, seed)
        want = _visitor_last(dec_cls, msg)

        words = bytearray(binding.tree_words_required * 8)
        objects: list = [None] * binding.tree_objects_required
        dec = dec_cls(**NO_CAPS, binding=binding, words=words, objects=objects)
        st = Status.COMPLETE
        for off in range(0, len(msg), chunk):
            st = dec.feed(msg[off : off + chunk])
        assert st is Status.COMPLETE, seed

        got = _read_slots(binding, words, objects)
        for key, value in want.items():
            assert key in got, (seed, key)
            assert got[key] == value, (seed, key)
