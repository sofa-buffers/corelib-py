"""The Visitor contract.

What a handler is promised: one typed hook per wire type, ``on_field`` and
``on_sequence_begin`` able to decline before the value is decoded, an unmodified
:class:`sofab.Visitor` still consuming a whole message, and none of it depending
on where the chunk boundaries fall (§7.2 item 4).

The *values* a visitor receives are checked against the shared vectors'
declared fields in ``test_conformance_vectors``; this suite is about the hooks.
"""

from __future__ import annotations

import pytest
from vectors import VECTORS

from sofab import Decoder, Field, Status, Visitor, WireType

_IDS = [v["name"] for v in VECTORS]

# Decode-side driver: it reads each vector's `serialized` hex, the dense
# primitive-layer ground truth. The `serialized_sparse` column (MESSAGE_SPEC §2
# omission of an all-default sequence *field*) is not read here or anywhere else in
# this repo — see the note in tests/test_conformance_vectors.py for why a corelib
# cannot produce it and where it is actually exercised (the generator's
# conformance drivers).


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


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
@pytest.mark.parametrize("chunk", [1, 4096])
def test_a_vector_decodes_the_same_however_it_is_chunked(vec, chunk):
    data = bytes.fromhex(vec["serialized"]["hex"])
    whole = Recorder()
    if Decoder(visitor=whole).feed(data) is not Status.COMPLETE:
        pytest.skip("vector is not a complete message on its own")

    piecewise = Recorder()
    dec = Decoder(visitor=piecewise)
    status = Status.COMPLETE
    for off in range(0, len(data), chunk):
        status = dec.feed(data[off : off + chunk])
    assert status is Status.COMPLETE
    assert piecewise.events == whole.events


@pytest.mark.parametrize("vec", [v for v in VECTORS if v.get("skip_ids")],
                         ids=[v["name"] for v in VECTORS if v.get("skip_ids")])
def test_declined_fields_leave_the_rest_intact(vec):
    """``on_field`` / ``on_sequence_begin`` returning ``False`` drops that field
    — a whole sub-tree, for a sequence — and nothing else."""
    data = bytes.fromhex(vec["serialized"]["hex"])
    skip = frozenset(vec["skip_ids"])

    everything = Recorder()
    if Decoder(visitor=everything).feed(data) is not Status.COMPLETE:
        pytest.skip("vector is not a complete message on its own")

    kept = Recorder(skip_ids=skip)
    assert Decoder(visitor=kept).feed(data) is Status.COMPLETE

    # Everything the declining run reported was also reported by the full run,
    # in the same order, and every declined id is gone from it.
    assert all(e in everything.events for e in kept.events)
    assert not [e for e in kept.events if len(e) > 1 and e[1] in skip]
    assert len(kept.events) < len(everything.events)


def test_every_typed_hook_fires_for_its_wire_type(enc_cls=None):
    """One hook per wire type, and the value arrives decoded."""
    from sofab import Encoder

    enc = Encoder()
    enc.write_unsigned(1, 300)
    enc.write_signed(2, -7)
    enc.write_float32(3, 1.5)
    enc.write_float64(4, -2.25)
    enc.write_string(5, "hi")
    enc.write_bytes(6, b"\x00\xff")
    enc.write_unsigned_array(7, [1, 2])
    enc.write_signed_array(8, [-1, 2])
    enc.write_float32_array(9, [1.5])
    enc.write_float64_array(10, [2.5])
    enc.write_sequence_begin_lazy(11)
    enc.write_unsigned(1, 9)
    enc.write_sequence_end()
    enc.flush()

    rec = Recorder()
    assert Decoder(visitor=rec).feed(enc.getvalue()) is Status.COMPLETE
    assert rec.events == [
        ("u", 1, 300), ("s", 2, -7), ("f32", 3, 1.5), ("f64", 4, -2.25),
        ("str", 5, "hi"), ("blob", 6, b"\x00\xff"),
        ("ua", 7, [1, 2]), ("sa", 8, [-1, 2]),
        ("f32a", 9, [1.5]), ("f64a", 10, [2.5]),
        ("seq", 11), ("u", 1, 9), ("end",),
    ]


def test_default_visitor_consumes_everything():
    """An unmodified Visitor (all no-ops) must still walk a message cleanly to
    EOF — unknown fields are consumed, not left dangling."""
    data = bytes.fromhex(
        next(v for v in VECTORS if v["name"] == "full_scale_example")["serialized"]["hex"]
    )
    assert Decoder(visitor=Visitor()).feed(data) is Status.COMPLETE


# --- the elision contract both engines rely on ------------------------------


def test_the_base_control_hooks_do_not_decline():
    """``on_field`` and ``on_sequence_begin`` default to "proceed".

    This is what lets the driver skip calling them at all when a visitor leaves
    them alone — and, for ``on_field``, skip building the Field they are the only
    consumer of. The saving is only sound because the calls it removes could not
    have changed the outcome, so the base returns are pinned here: they are no
    longer reached through a decode.
    """
    base = Visitor()
    assert base.on_field(Field(1, WireType.UNSIGNED)) is not False
    assert base.on_sequence_begin(1) is not False
    # on_array_begin's default is the same promise in its own shape: no
    # destination, no declared width, so the array takes the list route.
    assert base.on_array_begin(1, WireType.ARRAY_UNSIGNED, 3) is None


class _Delegating(Visitor):
    """Overrides the control hooks but defers to the base for the verdict — the
    case the ``is not Visitor.on_field`` test must treat as an override."""

    def __init__(self):
        self.fields = []
        self.seqs = []

    def on_field(self, field):
        self.fields.append(field.id)
        return super().on_field(field)

    def on_sequence_begin(self, field_id):
        self.seqs.append(field_id)
        return super().on_sequence_begin(field_id)


def test_a_hook_that_delegates_to_the_base_still_receives_every_call():
    from sofab import Encoder

    enc = Encoder()
    enc.write_unsigned(1, 7)
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(3, 8)
    enc.write_sequence_end()
    enc.flush()

    v = _Delegating()
    assert Decoder(visitor=v).feed(enc.getvalue()) is Status.COMPLETE
    assert v.fields == [1, 3]
    assert v.seqs == [2]
