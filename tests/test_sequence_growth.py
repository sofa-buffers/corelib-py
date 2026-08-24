"""The shared ``sequence_growth`` cases (CORELIB_PLAN §7.2 item 8).

A wrapper array carries no length: MESSAGE_SPEC §5.1 makes it *highest present
id + 1*, so its container grows as elements arrive. The growth itself belongs to
the static helper / generated layer (§6.6.1) and never to the codec — this port
ships no such layer, so the container below is the test's, standing in for
generated code exactly as the vectors' own note describes ("the port builds the
message from `deliver` and asserts `expect`").

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

from sofab import SofaLimitError, Status, Visitor

_VECTORS = json.loads(
    (Path(__file__).resolve().parent.parent / "assets" / "test_vectors.json").read_text()
)
CASES = _VECTORS.get("sequence_growth", [])
IDS = [c["name"] for c in CASES]

#: This port's configured element cap for the block. The vectors require at
#: least 4 and resolve every ``*_from_cap`` against whatever the port picks.
CAP = 8


class Collector(Visitor):
    """The wrapper array's container — the generated layer's job, not the codec's.

    It extends to ``id + 1`` as elements arrive, fills a gap with the element
    default, and refuses an index at or past the cap. The refusal is §6.2.1's:
    the codec surfaced the index, and this is the visitor deciding on it, before
    the container is extended.
    """

    def __init__(self, cap: int = CAP) -> None:
        self.cap = cap
        self.items: list = []
        self._depth = 0
        self._element = None

    # --- the container ---
    def _place(self, index: int, value) -> None:
        if index >= self.cap:
            raise SofaLimitError(
                f"element index {index} exceeds max_dyn_array_count {self.cap}"
            )
        while len(self.items) <= index:
            self.items.append(None)  # the element default
        self.items[index] = value

    # --- the wire ---
    def on_sequence_begin(self, field_id):
        self._depth += 1
        if self._depth == 2:  # a framed (struct) element of the wrapper
            self._place(field_id, {})
            self._element = field_id
        return None

    def on_sequence_end(self):
        if self._depth == 2:
            self._element = None
        self._depth -= 1

    def on_string(self, field_id, value):
        if self._depth == 1:  # a leaf element of the wrapper
            self._place(field_id, value)

    def on_unsigned(self, field_id, value):
        if self._depth == 2 and self._element is not None:
            self.items[self._element][field_id] = value


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
    coll = Collector()
    dec = dec_cls(visitor=coll)
    expect = case["expect"]

    if expect["outcome"] == "limit_exceeded":
        with pytest.raises(SofaLimitError):
            dec.feed(wire)
        assert len(coll.items) <= expect["max_length"], (
            "the container was extended past the rejected index"
        )
        if expect.get("terminal"):
            # §6.3: a terminal policy rejection. Nothing delivered after it --
            # which is what keeps the lower id in `growth_no_partial_extension`
            # from landing behind the one that was refused.
            with pytest.raises(SofaLimitError):
                dec.feed(b"")
            assert len(coll.items) <= expect["max_length"]
        return

    assert dec.feed(wire) is Status.COMPLETE
    assert len(coll.items) == _resolve(expect, "length")
    for gap in expect.get("default_ids", []):
        assert coll.items[gap] is None, f"id {gap} should have kept the default"


@pytest.mark.skipif(not CASES, reason="vectors carry no sequence_growth block")
def test_every_case_is_gated_on_dynamic_arrays():
    """The block's own gating: a port that never grows is excluded by it. This
    one does grow -- in the layer above the codec -- so it runs every case."""
    assert CASES, "the block must not be silently empty"
    assert all("dynamic_arrays" in c["requires"] for c in CASES)
