"""The shared ``boolean_tolerant`` cases: canonical on encode, tolerant on decode.

CORELIB_PLAN §4.4 is normative and has two halves, one for each direction:

    **Canonical on encode, tolerant on decode.** An encoder **MUST** write
    ``true`` as ``1``. A decoder **MUST** read **every value other than ``0``**
    as ``true``: such a value is **not** ``INVALID`` (§5.2), it is normalized
    away, and a re-encode emits ``1``.

A boolean has no wire type of its own — it is an unsigned integer (``0b000``),
and a boolean array rides the unsigned-varint array type (``0b011``). So every
byte string in this block is ordinary, well-formed wire. What is under test is
only how the **boolean read surface** interprets and stores it, and what the
**boolean write surface** emits afterwards.

The positive ``vectors`` array cannot reach this half of §4.4: its bytes come
from replaying ``fields`` through a conforming encoder, and a conforming encoder
never writes a non-canonical boolean. Bytes carrying ``2``, ``256`` or
``2**64-1`` at a boolean position only ever arrive from *someone else's*
encoder — hence a separate, hand-authored block.

**Three defects, three assertions.** The block is built so that each lands in a
different one, and dropping any of the three certifies a decoder that violates
§4.4:

* answering ``INVALID`` for ``256`` (a boolean read as a bounded 1-byte type) —
  caught by the **outcome** assertion;
* masking the accumulated varint down to the destination width before the
  zero-test, so ``256`` silently becomes ``false`` — caught by the **value**
  assertion, and by nothing else: the outcome is ``COMPLETE`` and looks perfect;
* storing the raw ``2`` without normalizing — caught by the **re-encode**
  assertion alone, since ``2`` is true under every truthiness test there is.

The value assertion and the re-encode assertion are not redundant in the other
direction either: a port that *stores* ``2`` but whose encoder maps any non-zero
to ``1`` re-encodes correctly while holding a value its own type system says is
impossible. That is this port today — see the two findings below.

**What this port offers as a boolean surface.**

* Write: :meth:`sofab.Encoder.write_bool`, which maps to ``1``/``0`` (§4.4's
  encode half, and it is correct).
* Read: :meth:`sofab.Binding.boolean` for one field and
  :meth:`sofab.Binding.boolean_array` for the element half, both of which
  normalize on store (``K_BOOLEAN`` / ``K_ARRAY_BOOLEAN``). The unsigned array
  reader could not stand in for the latter: ``256`` is a perfectly ordinary
  unsigned element, and normalization is precisely the behaviour the boolean
  surface adds, so reading through it would test the unsigned path and hide both
  truncation and missing normalization.

When this runner was first written neither surface normalized: ``Binding.boolean``
was the plain unsigned it is on the wire, and ``Binding.boolean_array`` did not
exist at all, so six of the eight cases failed and the array pair had no
destination to decode into. Those were conformance gaps in this port rather than
defects in the block, so the assertions were left stating §4.4 rather than the
observed behaviour — and #156 then closed both. The assertions are unchanged;
only this note and the surfaces beneath them moved. (sofa-buffers/crucible#189
tracks the family-wide rollout.)

**``requires`` here means REJECT, not skip** — the block narrows the corpus rule
and the narrowing is normative for it. §4.4 lifts the width bound the *type*
carries, never the one a particular *build* has: under a 32-bit accumulator a
boolean carrying ``2**64-1`` overflows before any boolean rule can apply, and
§6.2.2 makes rejecting it the conformant answer. Skipping such a case asserts
nothing and leaves that truncation untested in exactly the build most likely to
have it. This port has no compile-time feature switches, so its capability set
is complete, every case runs positively and the reject path below is
unreachable — it is implemented anyway so a reduced profile would not have to
grow one.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import ENGINE_PAIRS, NO_CAPS, VECTOR_DOC, Slots

from sofab import Binding, Status

CASES = VECTOR_DOC.get("boolean_tolerant", [])
IDS = [c["name"] for c in CASES]

#: A floor, never an equality: the block may grow upstream and a port must not
#: have to be edited for it to. Eight is what the corpus carries today.
FLOOR = 8

#: Every capability tag this block's gate knows, and every one this port HAS.
#: The two sets are equal: this port compiles nothing out — there is one build,
#: it has 64-bit values and arrays, and a reduced profile does not exist. A tag
#: the corpus adds later and this set does not know is ignored, not treated as
#: missing (forward compatibility), so all ports agree on the same behaviour.
KNOWN = frozenset({"array", "int64", "fixlen", "sequence", "fp64"})
SUPPORTED = KNOWN

#: Written into every destination slot before the feed, so a decoder that never
#: writes cannot pass the ``[false]`` case against a zero-initialized buffer.
#: As a ``words`` byte it is neither ``0`` nor ``1`` at any byte of any slot.
POISON = 0xAA

#: One byte offered after a rejection: §5.2.3 says the verdict is terminal, so a
#: further feed must re-issue it rather than lift it.
MORE = b"\x00"

#: What the run did, for the §12 report ``tests/conftest.py`` prints. ``found``
#: is the file's; the rest are filled in as cases execute, across every engine.
TALLY = {"found": len(CASES), "decoded": 0, "rejected": 0, "checks": 0}


def _needed(case: dict) -> frozenset:
    """The capability tags this case needs, unknown ones dropped (§7)."""
    return frozenset(case.get("requires", ())) & KNOWN


def _storage(binding: Binding) -> tuple[bytearray, list, Slots]:
    """A poisoned destination sized from the binding, and a view of it."""
    words = bytearray([POISON]) * (binding.tree_words_required * 8)
    objects: list = [None] * binding.tree_objects_required
    return words, objects, Slots(words, objects)


def _bind(case: dict) -> Binding:
    """This port's **boolean** destination for the case's arity.

    Scalar and array are different surfaces, chosen by ``len(expect.values)`` as
    the block specifies. The array one does not exist here; say which name is
    missing and why the unsigned reader is not a substitute, rather than quietly
    testing the wrong path.
    """
    field_id, count = case["id"], len(case["expect"]["values"])
    if count == 1:
        return Binding().boolean(field_id, at=0)
    binder = getattr(Binding, "boolean_array", None)
    if binder is None:
        pytest.fail(
            f"{case['name']}: this port exposes no boolean-typed ARRAY destination "
            "(`Binding.boolean_array`, the counterpart of `Binding.unsigned_array` "
            "and of the reference's sofab_istream_read_array_of_bool). The unsigned "
            "array reader cannot stand in: normalization is exactly the behaviour "
            "the boolean surface adds, so reading 256 back as 256 would be correct "
            "there and the case would assert nothing (CORELIB_PLAN §4.4)."
        )
    return binder(Binding(), field_id, at=0, cap=count, count_at=count)


def _stored(case: dict, slots: Slots) -> list[int]:
    """The destination read back, one entry per expected element.

    This port's boolean destination is a ``words`` slot, i.e. byte-shaped
    storage the test owns, so the raw stored representation is compared — the
    strictly stronger check of the two the block allows. A conformant slot holds
    exactly ``0`` or ``1``; nothing is coerced to a ``bool`` on the way, because
    that would map a stored ``2`` onto ``True`` and destroy the evidence.
    """
    count = len(case["expect"]["values"])
    return slots.arr_u(0, count)


def _reencode(enc_cls, case: dict, stored: list[int]) -> bytes:
    """The decoded destination written back out at the case's own field id.

    The values are the ones the *decoder* produced, never the JSON's — feeding
    ``expect.values`` in would make every re-encode match trivially and leave the
    decode half unverified.
    """
    enc = enc_cls()
    if len(stored) == 1:
        enc.write_bool(case["id"], stored[0])
    else:
        # §9: through the boolean array writer where one exists, otherwise the
        # unsigned array writer — an integer array's element width is an API
        # concern and never reaches the wire (§4.7).
        enc.write_unsigned_array(case["id"], stored)
    enc.flush()
    assert enc.error is None, f"{case['name']}: the encoder refused the re-encode"
    return enc.getvalue()


def _expect_rejected(dec_cls, case: dict, msg: bytes) -> None:
    """The reduced-build path: this message must be REJECTED, and terminally.

    Unreachable in this port (its capability set is complete); kept so that a
    profile-reduced build would not have to grow one, and so the rule this block
    narrows is written down where it applies.
    """
    binding = Binding()  # binds nothing: what is asserted is the verdict
    words, objects, _slots = _storage(binding)
    dec = dec_cls(binding=binding, words=words, objects=objects, **NO_CAPS)
    assert dec.feed(msg) is Status.INVALID, (
        f"{case['name']}: a value this build cannot represent is INVALID (§5.2.2), "
        "never read as true by truncation"
    )
    was = dec.error
    assert dec.feed(MORE) is Status.INVALID, f"{case['name']}: the verdict was lifted"
    assert dec.error is was


@pytest.mark.skipif(not CASES, reason="vectors carry no boolean_tolerant block")
@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_boolean_tolerant_case(case, enc_cls, dec_cls):
    msg = bytes.fromhex(case["serialized_hex"])
    expect = case["expect"]

    if _needed(case) - SUPPORTED:
        TALLY["rejected"] += 1
        TALLY["checks"] += 1
        _expect_rejected(dec_cls, case, msg)
        return

    TALLY["decoded"] += 1
    assert expect["outcome"] == "complete", (
        f"{case['name']}: a tolerated value is not a rejected one; this reader "
        f"knows no outcome {expect['outcome']!r}"
    )

    # --- decode: the value is read, not rejected, and not truncated ----------
    TALLY["checks"] += 1
    binding = _bind(case)
    words, objects, slots = _storage(binding)
    dec = dec_cls(binding=binding, words=words, objects=objects, **NO_CAPS)
    assert dec.feed(msg) is Status.COMPLETE, f"error: {dec.error!r}"

    want = [1 if v else 0 for v in expect["values"]]
    stored = _stored(case, slots)
    assert stored == want, (
        f"{case['name']}: §4.4 — every value other than 0 reads as true and is "
        f"normalized away; the destination holds {stored}"
    )
    if len(want) > 1:
        # §13: the wire's own element count, where the surface exposes it.
        assert slots.u[len(want)] == len(want), f"{case['name']}: element count"

    # --- re-encode: the normalized value goes back out canonical -------------
    TALLY["checks"] += 1
    assert _reencode(enc_cls, case, stored).hex() == expect["reencoded_hex"], (
        f"{case['name']}: §4.4 — an encoder MUST write true as 1"
    )


@pytest.mark.skipif(not CASES, reason="vectors carry no boolean_tolerant block")
@pytest.mark.parametrize("dec_cls", ENGINES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_boolean_tolerant_case_chunked(case, dec_cls):
    """§13: the same values and the same verdict, fed one byte at a time.

    The ``u64_max`` cases carry a ten-byte varint, so this is where an
    accumulator that spans feed boundaries is exercised at a boolean position.
    """
    msg = bytes.fromhex(case["serialized_hex"])
    if _needed(case) - SUPPORTED:
        pytest.skip("gated out; the reject path is asserted by the case above")

    binding = _bind(case)
    words, objects, slots = _storage(binding)
    dec = dec_cls(binding=binding, words=words, objects=objects, **NO_CAPS)
    status = Status.COMPLETE
    for off in range(len(msg)):
        status = dec.feed(msg[off : off + 1])
    assert status is Status.COMPLETE, f"error: {dec.error!r}"
    assert _stored(case, slots) == [1 if v else 0 for v in case["expect"]["values"]]


@pytest.mark.skipif(not CASES, reason="vectors carry no boolean_tolerant block")
def test_the_block_is_present_in_full():
    """A floor, and the shape every case must have.

    The failure this guards is the quiet one: a port whose ``test_vectors.json``
    predates the block looks up an absent key, iterates nothing and passes.
    """
    assert len(CASES) >= FLOOR, f"the block carries {len(CASES)} cases, floor {FLOOR}"
    for case in CASES:
        assert case["group"] == "boolean/tolerant", case["name"]
        assert case["expect"]["outcome"] == "complete", case["name"]
        assert case["expect"]["values"], case["name"]
        assert len(case["serialized_hex"]) % 2 == 0, case["name"]
        assert case["expect"]["reencoded_hex"], case["name"]
        assert _needed(case) == frozenset(case.get("requires", ())), (
            f"{case['name']}: unknown capability tags "
            f"{sorted(set(case.get('requires', ())) - KNOWN)}"
        )


@pytest.mark.skipif(not CASES, reason="vectors carry no boolean_tolerant block")
def test_every_case_is_either_decoded_or_rejected():
    """§12: ``found`` equals ``decoded + rejected``; nothing is inapplicable.

    A gated-out case is a negative case for this build, never one to drop, so
    the two paths must account for the whole block between them.
    """
    decoded = [c for c in CASES if not _needed(c) - SUPPORTED]
    rejected = [c for c in CASES if _needed(c) - SUPPORTED]
    assert len(decoded) + len(rejected) == len(CASES)
    # This port compiles nothing out, so the whole block runs positively.
    assert not rejected, [c["name"] for c in rejected]


@pytest.mark.skipif(not CASES, reason="vectors carry no boolean_tolerant block")
def test_the_block_carries_both_halves_of_the_rule():
    """The cases that make the block worth running, by name.

    ``255``/``256`` is the pair that separates "reads wide varints" from
    "truncates to the destination width", and the array cases are the only place
    the rule is asserted per element. A runner that lost either — a stale copy, a
    ``len(values) == 1`` filter — would still be green without this.
    """
    by_name = {c["name"]: c for c in CASES}
    assert {"boolean_tolerant_255", "boolean_tolerant_256"} <= set(by_name)
    assert by_name["boolean_tolerant_256"]["expect"]["values"] == [True]
    assert by_name["boolean_tolerant_256"]["expect"]["reencoded_hex"] == "0001"
    arrays = [c for c in CASES if len(c["expect"]["values"]) > 1]
    assert len(arrays) >= 2, "the element-level half of §4.4 is not covered"
    # A non-canonical element among canonical ones: the array re-encode is what
    # makes a missing normalization observable.
    assert any(
        c["expect"]["reencoded_hex"] != c["serialized_hex"] for c in arrays
    ), "no array case forces a re-encode that differs from its input"
