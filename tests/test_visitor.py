"""Visitor-driver tests.

A recording visitor must reproduce exactly what the equivalent pull-decode loop
produces (it's the same hot path, just dispatched), and the ``on_field`` /
``on_sequence_begin`` skip hooks must drop fields and whole sub-trees the same
way the harness' ``skip_ids`` scenario does.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from sofab import Decoder, FixlenSubtype, Visitor, WireType

VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "assets" / "test_vectors.json").read_text()
)["vectors"]
_IDS = [v["name"] for v in VECTORS]


class Recorder(Visitor):
    """Records every hook into the same ``(tag, ...)`` tuples the conformance
    harness' ``_decode_stream`` emits, so the two can be compared directly."""

    def __init__(self, skip_ids=()):
        self.events = []
        self._skip = frozenset(skip_ids)

    def on_field(self, field):
        if field.id in self._skip:
            return False
        return None

    def on_sequence_begin(self, field_id):
        if field_id in self._skip:
            return False
        self.events.append(("seq", field_id))
        return None

    def on_sequence_end(self):
        self.events.append(("end",))

    def on_unsigned(self, fid, v):
        self.events.append(("u", fid, v))

    def on_signed(self, fid, v):
        self.events.append(("s", fid, v))

    def on_float32(self, fid, v):
        self.events.append(("f32", fid, v))

    def on_float64(self, fid, v):
        self.events.append(("f64", fid, v))

    def on_string(self, fid, v):
        self.events.append(("str", fid, v))

    def on_bytes(self, fid, v):
        self.events.append(("blob", fid, v))

    def on_unsigned_array(self, fid, v):
        self.events.append(("ua", fid, v))

    def on_signed_array(self, fid, v):
        self.events.append(("sa", fid, v))

    def on_float32_array(self, fid, v):
        self.events.append(("f32a", fid, v))

    def on_float64_array(self, fid, v):
        self.events.append(("f64a", fid, v))


def _pull_decode(data, skip_ids=()):
    """Reference: the plain pull loop, mirroring the visitor's recording."""
    skip = frozenset(skip_ids)
    dec = Decoder(io.BytesIO(data))
    out = []
    while (f := dec.next()) is not None:
        t = f.type
        if t == WireType.SEQUENCE_END:
            out.append(("end",))
            continue
        if f.id in skip:
            dec.skip()
            continue
        if t == WireType.UNSIGNED:
            out.append(("u", f.id, dec.unsigned()))
        elif t == WireType.SIGNED:
            out.append(("s", f.id, dec.signed()))
        elif t == WireType.FIXLEN:
            st = f.subtype
            if st == FixlenSubtype.FP32:
                out.append(("f32", f.id, dec.float32()))
            elif st == FixlenSubtype.FP64:
                out.append(("f64", f.id, dec.float64()))
            elif st == FixlenSubtype.STRING:
                out.append(("str", f.id, dec.string()))
            else:
                out.append(("blob", f.id, dec.bytes()))
        elif t == WireType.ARRAY_UNSIGNED:
            out.append(("ua", f.id, dec.read_unsigned_array()))
        elif t == WireType.ARRAY_SIGNED:
            out.append(("sa", f.id, dec.read_signed_array()))
        elif t == WireType.ARRAY_FIXLEN:
            if f.subtype == FixlenSubtype.FP32:
                out.append(("f32a", f.id, dec.read_float32_array()))
            else:
                out.append(("f64a", f.id, dec.read_float64_array()))
        elif t == WireType.SEQUENCE_START:
            out.append(("seq", f.id))
    return out


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_visitor_matches_pull(vec):
    data = bytes.fromhex(vec["serialized"]["hex"])
    rec = Recorder()
    Decoder(io.BytesIO(data)).drive(rec)
    assert rec.events == _pull_decode(data)


@pytest.mark.parametrize("vec", [v for v in VECTORS if v.get("skip_ids")],
                         ids=[v["name"] for v in VECTORS if v.get("skip_ids")])
def test_visitor_skip_hooks(vec):
    data = bytes.fromhex(vec["serialized"]["hex"])
    skip = vec["skip_ids"]
    rec = Recorder(skip_ids=skip)
    Decoder(io.BytesIO(data)).drive(rec)
    assert rec.events == _pull_decode(data, skip_ids=skip)


def test_default_visitor_consumes_everything():
    """An unmodified Visitor (all no-ops) must still walk a message cleanly to
    EOF — unknown fields are consumed, not left dangling."""
    data = bytes.fromhex(next(v for v in VECTORS if v["name"] == "full_scale_example")["serialized"]["hex"])
    Decoder(io.BytesIO(data)).drive(Visitor())  # must not raise
