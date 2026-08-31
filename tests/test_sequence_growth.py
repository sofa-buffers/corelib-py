"""The shared ``sequence_growth`` cases (CORELIB_PLAN §7.2 item 8).

A wrapper array carries no length: MESSAGE_SPEC §5.1 makes it *highest present
id + 1*, so its container grows as elements arrive. The growth itself belongs to
the static helper / generated layer (§6.6.1) and never to the codec — this port
ships that layer as ``sofab.collectors``, so the container below is the
library's own, driven through the public visitor API exactly as generated code
drives it ("the port builds the message from `deliver` and asserts `expect`").
Its growth **geometry** is measured in ``tests/test_collectors.py``.

What the codec owes, and what these cases pin, is §6.2.1's other half:

    for a **sequence array** it surfaces the **index** of the element in hand —
    a wrapper array's length is *highest present id + 1*, so the index is what
    has to be checked, there being no count header to check; the visitor decides.

So the decoder must hand over each element's id, unrenumbered and uncompacted,
**before** the container it indexes into is extended; and a rejection at that
index must be terminal (§6.3), which is what stops a later, lower id from
landing behind it.

Cases are keyed by a delivery sequence rather than by bytes, because two ports
that grow differently emit identical bytes. Indices are cap-relative: the case's
``id_from_cap`` is an offset onto *this* port's ``max_dyn_array_count``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import ENCODER_ENGINES as ENCODERS
from vectors import NO_CAPS

from sofab import NestedSeq, SofaLimitError, Status, StringSeq, Visitor

_VECTORS = json.loads(
    (Path(__file__).resolve().parent.parent / "assets" / "test_vectors.json").read_text()
)
CASES = _VECTORS.get("sequence_growth", [])
IDS = [c["name"] for c in CASES]

#: This port's configured element cap for the block. The vectors require at
#: least 4 and resolve every ``*_from_cap`` against whatever the port picks.
CAP = 8


class Element(Visitor):
    """One struct element: an ``unsigned`` at id 0, per the block's note."""

    def __init__(self) -> None:
        self.fields: dict[int, int] = {}

    def on_unsigned(self, field_id, value):
        self.fields[field_id] = value


class Root(Visitor):
    """The object that holds the array field, handing its scope to a collector.

    This is the shape generated code takes: a wrapper array's scope is delegated
    to the collector for that field, so the ids the collector sees are that
    array's indices.
    """

    def __init__(self, case: dict, out: list) -> None:
        self._case = case
        self._out = out

    def on_sequence_begin(self, field_id):
        if field_id != self._case["field_id"]:
            return None
        if self._case.get("element_type") == "struct":
            return NestedSeq(self._out, factory=Element, max_dyn_array_count=CAP)
        return StringSeq(self._out, max_dyn_array_count=CAP)


def collector_for(case: dict, out: list):
    """The library's own collector for this case's element type.

    This is the point of the block for a codec-only port: the container is
    ``sofab.collectors``' — the static helper layer §6.6.1 puts beside the codec
    for exactly this — driven through the public visitor API the generated layer
    uses. Nothing here is written for the test.
    """
    return Root(case, out)


def _element_default(case: dict):
    """What an omitted interior element leaves behind (MESSAGE_SPEC §2)."""
    return None if case.get("element_type") == "struct" else ""


def _resolve(entry: dict, key: str) -> int:
    """Cap-relative or absolute, whichever the case states."""
    if key in entry:
        return entry[key]
    return CAP + entry[key + "_from_cap"]


def _build(enc_cls, case: dict) -> bytes:
    """The message the delivery sequence describes."""
    enc = enc_cls()
    enc.write_sequence_begin_lazy(case["field_id"])
    for entry in case["deliver"]:
        index = _resolve(entry, "id")
        if case.get("element_type") == "struct":
            enc.write_sequence_begin_lazy(index)
            enc.write_unsigned(0, entry["value"])
            enc.write_sequence_end_keep()
        else:
            enc.write_string(index, entry["value"])
    enc.write_sequence_end_keep()
    enc.flush()
    return enc.getvalue()


@pytest.mark.skipif(not CASES, reason="vectors carry no sequence_growth block")
@pytest.mark.parametrize("dec_cls", ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_sequence_growth_case(case, enc_cls, dec_cls):
    assert "dynamic_arrays" in case["requires"], case["name"]
    wire = _build(enc_cls, case)
    out: list = []
    dec = dec_cls(**NO_CAPS, visitor=collector_for(case, out))
    expect = case["expect"]

    if expect["outcome"] == "limit_exceeded":
        with pytest.raises(SofaLimitError):
            dec.feed(wire)
        assert len(out) <= expect["max_length"], (
            "the container was extended past the rejected index"
        )
        if expect.get("terminal"):
            # §6.3: a terminal policy rejection. Nothing delivered after it --
            # which is what keeps the lower id in `growth_no_partial_extension`
            # from landing behind the one that was refused.
            with pytest.raises(SofaLimitError):
                dec.feed(b"")
            assert len(out) <= expect["max_length"]
        return

    assert dec.feed(wire) is Status.COMPLETE
    assert len(out) == _resolve(expect, "length")
    for gap in expect.get("default_ids", []):
        assert out[gap] == _element_default(case), (
            f"id {gap} should have kept the element default"
        )


@pytest.mark.skipif(not CASES, reason="vectors carry no sequence_growth block")
def test_every_case_is_gated_on_dynamic_arrays():
    """The block's own gating: a port that never grows is excluded by it. This
    one does grow -- in the layer above the codec -- so it runs every case."""
    assert CASES, "the block must not be silently empty"
    assert all("dynamic_arrays" in c["requires"] for c in CASES)
