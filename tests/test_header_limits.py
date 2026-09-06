"""The shared ``header_limits`` cases: the ceiling answers at the length word.

The block carries the **truncated over-ceiling header** — bytes that *declare* a
length or a count and then end, with not one payload byte behind them::

    02 a2 06   then EOF
    ^^ id 0, wire type 2 (fixlen)
       ^^^^^ length word (100 << 3) | 2  ->  a 100-byte STRING is declared

A conformant decoder answers **at that word**, before the payload is asked for,
so the answer is the ceiling's and it is **terminal**. ``INCOMPLETE`` would be
the outcome CORELIB_PLAN §6.2.1's enforcement point exists to prevent: §5.2.1
defines it as the outcome more bytes *can* change and §5.2.4 has a streaming
caller read it as "feed me the next chunk", and after a ceiling has fired that is
a false statement about the state (§6.3 calls the rejection terminal).

**Which ceiling speaks is the subject**, and the two give opposite answers on the
same word:

* a case stating ``schema: {maxlen: N}`` — the bound is the schema's, so a
  breach is ``INVALID`` (MESSAGE_SPEC §7.1). Here that bound is declared where
  generated code declares it, on a :class:`sofab.Binding`.
* a case stating ``limits: {max_dyn_*: N}`` — the bound is the receiver's, so
  the bytes are well-formed and the breach is ``LimitExceeded``
  (:class:`sofab.SofaLimitError`, §6.2.1/§6.3).

``header_string_schema_bounded`` and ``header_string_over_cap`` carry the
**identical bytes** and differ only in which ceiling the case configures; a port
that routes both to one category passes every other case here and fails that
pair. This port keeps them apart — ``tests/test_schema_bounded.py`` is where that
split is specified in full; this file replays the family's shared statement of it.

**Every rejection is paired with its in-cap control**: the same shape at a length
the ceiling admits, which must still answer ``INCOMPLETE``. They are not filler —
a port that rejects every short read passes all six rejection cases and is badly
broken. Each control is therefore also fed the payload it is still waiting for,
which must complete it: that is what ``INCOMPLETE`` claims, and the one property
that tells it apart from a terminal verdict.

**An unsatisfied ``requires`` tag means SKIP here, for every tag** — not the
reduced-build rejection a *vector* gets. These cases already assert a rejection
with a specific category, so a build that cannot represent the construct would
reject it for an unrelated reason and appear to pass while testing nothing.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import NO_CAPS, VECTOR_DOC, Recorder, capped

from sofab import (
    Binding,
    FixlenSubtype,
    SofaDecodeError,
    SofaLimitError,
    Status,
    WireType,
)

CASES = VECTOR_DOC.get("header_limits", [])
IDS = [c["name"] for c in CASES]

#: Capabilities this port has, for this block's gating. The wire-construct tags
#: are the vectors' own; ``receiver_caps`` is a **profile** capability — a port
#: declares it when its generated code carries §6.2.1 receiver caps *distinct
#: from* schema bounds. This one does: the three ``max_dyn_*`` arguments are the
#: caller's caps, and a :class:`Binding`'s ``maxlen``/``cap`` is the schema bound,
#: which takes the field out of the cap's reach (``tests/test_schema_bounded.py``).
SUPPORTED = frozenset({"fixlen", "array", "int64", "receiver_caps"})

#: The receiver cap that governs each construct (§6.2.1). Blob and string are
#: deliberately separate caps, so the enforcement point has to be asserted on
#: both — a port can wire one and miss the other.
CAP_FOR = {
    "string": "max_dyn_string_len",
    "blob": "max_dyn_blob_len",
    "array": "max_dyn_array_count",
}

#: Bytes offered *after* a rejection, standing in for the payload that would have
#: followed. Zeroes are valid content for all three constructs, so a decoder that
#: resumed instead of re-raising would consume them happily.
MORE = bytes(8)


def _read_uvarint(raw: bytes, pos: int) -> tuple[int, int]:
    """One base-128 little-endian varint (§4.1); returns ``(value, next_pos)``."""
    value = shift = 0
    while True:
        byte = raw[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _header(case: dict) -> tuple[int, str, int]:
    """``(field_id, construct, declared)`` read off the case's own bytes.

    The case *states* ``field_id`` and ``declared``; reading them back off the
    wire instead and asserting the two agree is what keeps this reader honest —
    a reader that configured a ceiling for a construct the bytes do not carry
    would otherwise still see the expected verdict for the wrong reason.
    """
    raw = bytes.fromhex(case["serialized"])
    tag, pos = _read_uvarint(raw, 0)
    field_id, wire = tag >> 3, WireType(tag & 0x7)
    word, _ = _read_uvarint(raw, pos)
    if wire is WireType.FIXLEN:
        subtype = FixlenSubtype(word & 0x7)
        assert subtype in (FixlenSubtype.STRING, FixlenSubtype.BLOB), (
            f"{case['name']}: {subtype!r} declares a fixed width, not a length"
        )
        construct = "string" if subtype is FixlenSubtype.STRING else "blob"
        return field_id, construct, word >> 3
    assert wire in (
        WireType.ARRAY_UNSIGNED,
        WireType.ARRAY_SIGNED,
        WireType.ARRAY_FIXLEN,
    ), f"{case['name']}: {wire!r} carries no header word to bind"
    return field_id, "array", word


def _ceiling(case: dict) -> tuple[str, int]:
    """``("schema"|"limits", N)`` — the one ceiling the case configures.

    Never both: §6.2.1 forbids applying a receiver cap to a field the schema
    already bounds, so the case says which of the two is in play and that is the
    whole subject of the block.
    """
    assert ("schema" in case) != ("limits" in case), (
        f"{case['name']}: a case configures exactly one ceiling"
    )
    if "schema" in case:
        return "schema", case["schema"]["maxlen"]
    (value,) = case["limits"].values()
    return "limits", value


def _decoder(dec_cls, case: dict):
    """A decoder carrying this case's ceiling, and a view of what it delivered.

    The two ceilings live in different places, which is the point:

    * a receiver cap is a decoder argument, so the case's ``limits`` go straight
      in and the other two caps stay at their format ceilings, where they cannot
      fire — only the stated one can speak.
    * a schema bound is declared by the caller *for a field*, and a
      :class:`Binding` entry's ``maxlen``/``cap`` is where this port takes that
      declaration. Every receiver cap is left at its ceiling there, so an
      ``INVALID`` can only have come from the declared bound.
    """
    field_id, construct, _declared = _header(case)
    kind, bound = _ceiling(case)

    if kind == "limits":
        assert set(case["limits"]) == {CAP_FOR[construct]}, (
            f"{case['name']}: the cap named does not govern the construct on the wire"
        )
        rec = Recorder()
        return dec_cls(visitor=rec, **capped(**case["limits"])), (lambda: rec.events)

    binding = Binding()
    if construct == "string":
        binding.string(field_id, at=0, maxlen=bound)
    elif construct == "blob":
        binding.bytes(field_id, at=0, maxlen=bound)
    else:
        # No such case exists yet; an array's count lands in `words`, not in
        # `objects`, so it needs its own destination and its own view of what
        # was delivered. Say so rather than quietly checking the wrong thing.
        pytest.fail(f"{case['name']}: this reader has no schema-bounded array path")
    objects: list = [None] * binding.tree_objects_required
    dec = dec_cls(
        binding=binding,
        words=bytearray(binding.tree_words_required * 8),
        objects=objects,
        **NO_CAPS,
    )
    return dec, (lambda: [o for o in objects if o is not None])


def _feed(dec, chunks) -> str:
    """Feed the case's bytes and name the outcome in the block's vocabulary.

    A receiver-cap rejection is *raised* (§6.3: the message is well-formed and
    the receiver declined it, so it is not one of the three statuses); the other
    two outcomes come back as a :class:`Status`.
    """
    status = Status.COMPLETE
    try:
        for chunk in chunks:
            status = dec.feed(chunk)
    except SofaLimitError:
        return "limit_exceeded"
    return {
        Status.COMPLETE: "complete",
        Status.INCOMPLETE: "incomplete",
        Status.INVALID: "invalid",
    }[status]


def _chunks(case: dict) -> list:
    """The case's delivery: its ``chunks`` where it states them, else one feed.

    ``header_string_over_cap_split`` divides the length varint itself, so the
    ceiling fires on a word no single feed delivered whole — the verdict is a
    property of the bytes, not of the chunking (§7.2 item 4).
    """
    hexes = case.get("chunks") or [case["serialized"]]
    assert "".join(hexes) == case["serialized"], f"{case['name']}: chunks are not the bytes"
    return [bytes.fromhex(h) for h in hexes]


def _check_requires(case: dict) -> None:
    """In this block an unsatisfied tag is a SKIP, for every tag (see the module
    docstring). Nothing is skipped here — this port supports all four — but the
    gate keeps the harness honest for a footprint-reduced one."""
    missing = set(case.get("requires", ())) - SUPPORTED
    if missing:
        pytest.skip(f"requires unsupported capabilities: {sorted(missing)}")


@pytest.mark.skipif(not CASES, reason="vectors carry no header_limits block")
@pytest.mark.parametrize("dec_cls", ENGINES)
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_header_limits_case(case, dec_cls):
    _check_requires(case)
    field_id, construct, declared = _header(case)
    assert (field_id, declared) == (case["field_id"], case["declared"]), (
        "the case's stated header does not match its bytes"
    )

    dec, delivered = _decoder(dec_cls, case)
    outcome = _feed(dec, _chunks(case))
    assert outcome == case["expect"]["outcome"], f"error: {dec.error!r}"

    if outcome == "incomplete":
        # THE CONTROL. The ceiling admits this length, so the short read is the
        # state more bytes can lift (§5.2.1) -- and lift it they do: the payload
        # it is still waiting for completes the message. Without that second
        # half, a decoder that answered INCOMPLETE and then wedged would pass.
        assert dec.error is None
        assert not delivered()
        filler = bytes([1]) * declared if construct == "array" else b"x" * declared
        assert dec.feed(filler) is Status.COMPLETE
        assert len(delivered()) == 1
        return

    # A rejection, and it is decided at the header: not one payload byte was
    # fed, so nothing but the length/count word can have produced this verdict.
    assert not delivered(), "the payload was materialized behind the rejection"
    if outcome == "limit_exceeded":
        assert isinstance(dec.error, SofaLimitError)
    else:
        assert isinstance(dec.error, SofaDecodeError)
        assert not isinstance(dec.error, SofaLimitError), (
            "a schema bound is a statement about validity, not about capacity"
        )

    if not case["expect"].get("terminal"):
        return
    # §6.3: terminal. A further feed re-issues the verdict rather than consuming
    # -- INVALID comes back from every later feed with the reason still on
    # `error` (§5.2.3), a cap rejection re-raises.
    was = dec.error
    assert _feed(dec, [MORE]) == outcome
    assert dec.error is was
    assert not delivered()


@pytest.mark.skipif(not CASES, reason="vectors carry no header_limits block")
def test_every_rejection_is_paired_with_an_in_cap_control():
    """A missing control is a bug in the block, not an omission: it is the only
    thing standing between a conformant port and one that rejects every short
    read."""
    controls = {
        (_header(c)[1], *_ceiling(c))
        for c in CASES
        if c["expect"]["outcome"] == "incomplete"
    }
    for case in CASES:
        if case["expect"]["outcome"] == "incomplete":
            continue
        key = (_header(case)[1], *_ceiling(case))
        assert key in controls, f"{case['name']} has no in-cap control"


@pytest.mark.skipif(not CASES, reason="vectors carry no header_limits block")
def test_the_block_pins_both_ceilings_on_the_same_word():
    """The pair that keeps the two categories apart: identical bytes, opposite
    verdicts, because the case configures a different ceiling."""
    by_name = {c["name"]: c for c in CASES}
    cap, schema = by_name["header_string_over_cap"], by_name["header_string_schema_bounded"]
    assert cap["serialized"] == schema["serialized"]
    assert (cap["expect"]["outcome"], schema["expect"]["outcome"]) == (
        "limit_exceeded",
        "invalid",
    )


@pytest.mark.skipif(not CASES, reason="vectors carry no header_limits block")
def test_every_case_declares_a_known_capability():
    declared = {tag for c in CASES for tag in c.get("requires", ())}
    assert declared, "the block must not be silently ungated"
    assert declared <= SUPPORTED, f"unknown capability tags: {declared - SUPPORTED}"
    # `terminal` describes a rejection; INCOMPLETE is precisely the state more
    # bytes can lift, so the block never marks one terminal.
    for case in CASES:
        if case["expect"]["outcome"] == "incomplete":
            assert "terminal" not in case["expect"], case["name"]
        else:
            assert case["expect"]["terminal"] is True, case["name"]
