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

``header_limits_nested`` — the same assertion one or two frames deeper
-----------------------------------------------------------------------

Every case in the flat block puts its field at ``field_id 0`` in the top-level
scope, so one axis stays untested: the identical over-ceiling header delivered
**inside an open sequence**. ``header_limits_nested`` is that axis and only that
axis — same keys, same outcome vocabulary, same terminality rule, same pairing
with an in-ceiling control — plus one new key, ``frames``: the chain of sequence
field ids the target field is nested in, outermost first. It is a **separate
top-level block** because its bytes begin with a sequence header, so a runner
that ignored ``frames`` would bind its ceiling at the top level and answer
``INCOMPLETE`` where the case demands a rejection.

Depth is its own axis because the nested cases end with their frames **still
open**, which hands a decoder a *second, independent* reason to answer
``INCOMPLETE``. Two things follow, and both are load-bearing here:

* the leaf is **shared** with the flat block — :func:`_decoder`, :func:`_feed`
  and :func:`_header` take ``frames`` and are otherwise the same code, so the
  nested cases cannot pass by a different mechanism than the flat ones;
* the block carries a **negative control**
  (:func:`test_the_nested_control_shows_the_ceiling_caused_every_rejection`),
  which re-runs every rejection with the ceiling lifted far above ``declared``
  and asserts the answer *changed*. Without it a port that rejected these bytes
  for an unrelated reason — a depth guard, a refusal of unclosed frames — would
  look green while never having consulted the ceiling under test. The control
  asserts how many cases it checked, so it cannot quietly degenerate into a loop
  that examines nothing.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

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

#: The nested block: the same cases one or two sequence frames deeper.
NESTED = VECTOR_DOC.get("header_limits_nested", [])
NESTED_IDS = [c["name"] for c in NESTED]

#: Capabilities this port has, for this block's gating. The wire-construct tags
#: are the vectors' own; ``receiver_caps`` is a **profile** capability — a port
#: declares it when its generated code carries §6.2.1 receiver caps *distinct
#: from* schema bounds. This one does: the three ``max_dyn_*`` arguments are the
#: caller's caps, and a :class:`Binding`'s ``maxlen``/``cap`` is the schema bound,
#: which takes the field out of the cap's reach (``tests/test_schema_bounded.py``).
#:
#: ``sequence`` is here because nothing in this port can be compiled out: there
#: is no ``SOFAB_DISABLE_*`` switch, so every wire-construct tag is satisfied
#: unconditionally and both blocks run whole.
SUPPORTED = frozenset({"sequence", "fixlen", "array", "int64", "receiver_caps"})

#: The ceiling the negative control raises every rejection to (§6 of the block's
#: spec): far above every ``declared`` in either block, and small enough that
#: lifting it cannot provoke an allocation worth worrying about. A rejection that
#: survives this was not the ceiling's.
LIFTED = 65536

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


def _frames(case: dict) -> tuple[int, ...]:
    """The sequence field ids the target is nested in, outermost first.

    Absent in the flat block — the field is at the top level — and **non-empty**
    wherever it appears, since a nested case with no frame is a flat case.
    """
    if "frames" not in case:
        return ()
    frames = tuple(case["frames"])
    assert frames, f"{case['name']}: `frames` is the block's subject and cannot be empty"
    return frames


def _header(case: dict) -> tuple[int, str, int]:
    """``(field_id, construct, declared)`` read off the case's own bytes.

    The case *states* ``field_id`` and ``declared``; reading them back off the
    wire instead and asserting the two agree is what keeps this reader honest —
    a reader that configured a ceiling for a construct the bytes do not carry
    would otherwise still see the expected verdict for the wrong reason. The
    same goes for ``frames``: the sequence opens are walked here and each one is
    checked against the id the case names, so a chain built to the wrong depth
    is caught before any decoder sees the bytes.
    """
    raw = bytes.fromhex(case["serialized"])
    pos = 0
    for depth, frame_id in enumerate(_frames(case)):
        tag, pos = _read_uvarint(raw, pos)
        assert (tag >> 3, WireType(tag & 0x7)) == (frame_id, WireType.SEQUENCE_START), (
            f"{case['name']}: frame {depth} on the wire is not the sequence at id {frame_id}"
        )
    tag, pos = _read_uvarint(raw, pos)
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


class Receiver(NamedTuple):
    """What :func:`_decoder` hands back: the decoder, and two views of the walk.

    ``delivered()`` is what the decode materialized — it must stay empty behind a
    rejection (§6.2.1 is "rejected, never clamped"). ``frames_seen()`` is the
    chain of sequence ids the decoder actually descended into, which is what
    tells a ceiling that fired at the nested field apart from one that fired at
    the top level. Both are empty for a flat case, which has neither.
    """

    dec: object
    delivered: Callable[[], list]
    frames_seen: Callable[[], list]


def _decoder(dec_cls, case: dict, ceiling: int | None = None) -> Receiver:
    """A decoder carrying this case's ceiling, and a view of what it delivered.

    The two ceilings live in different places, which is the point:

    * a receiver cap is a decoder argument — §6.2.1's "as an ARGUMENT" shape —
      so the case's ``limits`` go straight in and the other two caps stay at
      their format ceilings, where they cannot fire: only the stated one can
      speak. An argument governs the whole walk, at every depth, so what a
      nested case has to establish is that the walk *reached* the nested field;
      ``frames_seen()`` is that evidence.
    * a schema bound is declared by the caller *for a field*, and a
      :class:`Binding` entry's ``maxlen``/``cap`` is where this port takes that
      declaration. For a nested case the chain is built outermost-first out of
      child tables, so the bound exists **only** inside the innermost one — a
      decoder that failed to descend would find no bound to breach. Every
      receiver cap is left at its ceiling there, so an ``INVALID`` can only have
      come from the declared bound.

    ``ceiling`` overrides the case's own number, which is what the negative
    control needs: the same receiver and the same bytes, with the one ceiling
    under test lifted out of reach.
    """
    field_id, construct, _declared = _header(case)
    frames = _frames(case)
    kind, bound = _ceiling(case)
    if ceiling is not None:
        bound = ceiling

    if kind == "limits":
        assert set(case["limits"]) == {CAP_FOR[construct]}, (
            f"{case['name']}: the cap named does not govern the construct on the wire"
        )
        rec = Recorder()
        dec = dec_cls(visitor=rec, **capped(**{CAP_FOR[construct]: bound}))
        return Receiver(
            dec,
            lambda: [e for e in rec.events if e[0] not in ("seq{", "seq}")],
            lambda: [e[1] for e in rec.events if e[0] == "seq{"],
        )

    # The leaf table, then one enclosing table per frame, outermost last so the
    # chain is built inside-out and `binding` ends up being the top-level scope.
    # The children are closed: there is no visitor here, and a child that handed
    # an unnamed id over would hand it over under the parent's identity.
    binding = Binding(closed=True)
    if construct == "string":
        binding.string(field_id, at=0, maxlen=bound)
    elif construct == "blob":
        binding.bytes(field_id, at=0, maxlen=bound)
    else:
        # No such case exists yet; an array's count lands in `words`, not in
        # `objects`, so it needs its own destination and its own view of what
        # was delivered. Say so rather than quietly checking the wrong thing.
        pytest.fail(f"{case['name']}: this reader has no schema-bounded array path")
    for depth, frame_id in reversed(list(enumerate(frames))):
        # `count_at` counts the sequence's occurrences and is written when the
        # frame OPENS, so it reads back as "the decoder descended this far" even
        # on a message that never closes the frame -- which every case here is.
        parent = Binding(closed=True)
        parent.sequence(frame_id, binding, count_at=depth)
        binding = parent

    words = bytearray(max(binding.tree_words_required, len(frames)) * 8)
    objects: list = [None] * binding.tree_objects_required
    dec = dec_cls(binding=binding, words=words, objects=objects, **NO_CAPS)
    entered = memoryview(words).cast("Q")
    return Receiver(
        dec,
        lambda: [o for o in objects if o is not None],
        lambda: [fid for depth, fid in enumerate(frames) if entered[depth]],
    )


def _feed(dec, chunks) -> str:
    """Feed the case's bytes and name the outcome in the block's vocabulary.

    A receiver-cap rejection is *raised* (§6.3: the message is well-formed and
    the receiver declined it, so it is not one of the three statuses); the other
    two outcomes come back as a :class:`Status`.
    """
    status = Status.COMPLETE
    try:
        for n, chunk in enumerate(chunks):
            status = dec.feed(chunk)
            if n < len(chunks) - 1:
                # A verdict before the last chunk would be a verdict on bytes
                # the decoder had not yet seen; only the final feed answers.
                assert status is Status.INCOMPLETE, f"chunk {n} answered {status!r}"
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

    dec, delivered, _frames_seen = _decoder(dec_cls, case)
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


#: The two blocks, for the shape assertions that hold of both. A block missing
#: from the vector file is not skipped here -- it is the file that is wrong.
BLOCKS = [
    pytest.param(CASES, id="header_limits"),
    pytest.param(NESTED, id="header_limits_nested"),
]


@pytest.mark.parametrize("block", BLOCKS)
def test_every_rejection_is_paired_with_an_in_cap_control(block):
    """A missing control is a bug in the block, not an omission: it is the only
    thing standing between a conformant port and one that rejects every short
    read."""
    assert block, "the vector file carries the block, and it is not empty"
    controls = {
        (_frames(c), _header(c)[1], *_ceiling(c))
        for c in block
        if c["expect"]["outcome"] == "incomplete"
    }
    for case in block:
        if case["expect"]["outcome"] == "incomplete":
            continue
        key = (_frames(case), _header(case)[1], *_ceiling(case))
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


@pytest.mark.parametrize("block", BLOCKS)
def test_every_case_declares_a_known_capability(block):
    assert block, "the vector file carries the block, and it is not empty"
    declared = {tag for c in block for tag in c.get("requires", ())}
    assert declared, "the block must not be silently ungated"
    assert declared <= SUPPORTED, f"unknown capability tags: {declared - SUPPORTED}"
    # `terminal` describes a rejection; INCOMPLETE is precisely the state more
    # bytes can lift, so the block never marks one terminal.
    for case in block:
        if case["expect"]["outcome"] == "incomplete":
            assert "terminal" not in case["expect"], case["name"]
        else:
            assert case["expect"]["terminal"] is True, case["name"]


# --- header_limits_nested: the same assertion one or two frames deeper --------


def _nested_lift(case: dict) -> bytes:
    """The bytes that would lift an ``INCOMPLETE`` nested case to ``COMPLETE``.

    The payload the header promised, then one sequence-end marker per open
    frame. Only the in-ceiling controls are fed this: ``INCOMPLETE`` claims that
    more bytes can change the verdict, and this is the shortest continuation
    that proves the claim rather than taking it on trust.
    """
    _field_id, construct, declared = _header(case)
    payload = bytes([1]) * declared if construct == "array" else b"x" * declared
    return payload + bytes([WireType.SEQUENCE_END]) * len(_frames(case))


@pytest.mark.parametrize("dec_cls", ENGINES)
@pytest.mark.parametrize("case", NESTED, ids=NESTED_IDS)
def test_header_limits_nested_case(case, dec_cls):
    """The flat block's assertion, with the field inside an open sequence chain.

    Same leaf, same ceilings, same vocabulary; the one new thing is ``frames``,
    and the one new hazard is that these bytes end with a frame still open, so
    ``INCOMPLETE`` now has a second, unrelated reason to be true. That is why the
    frame chain the decoder actually walked is asserted here, and why the
    negative control below is not optional.
    """
    _check_requires(case)
    field_id, construct, declared = _header(case)
    frames = _frames(case)
    assert (field_id, declared) == (case["field_id"], case["declared"]), (
        "the case's stated header does not match its bytes"
    )

    dec, delivered, frames_seen = _decoder(dec_cls, case)
    outcome = _feed(dec, _chunks(case))
    assert outcome == case["expect"]["outcome"], f"{case['description']} (error: {dec.error!r})"
    # The verdict was reached at the nested field, not at the top level: the
    # decoder descended every frame the case names, in order, before answering.
    assert frames_seen() == list(frames), f"{case['name']}: the chain walked is not `frames`"

    if outcome == "incomplete":
        # THE CONTROL. The ceiling admits this size, so the short read is the
        # state more bytes can lift (§5.2.1) -- and lift it they do: the payload
        # plus one end marker per open frame completes the message.
        assert dec.error is None
        assert not delivered()
        assert dec.feed(_nested_lift(case)) is Status.COMPLETE
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
    # §6.3: terminal, and the open frames change nothing about that. A further
    # feed of would-be payload re-issues the verdict rather than consuming it,
    # and nothing is materialized behind it even late.
    was = dec.error
    assert _feed(dec, [MORE]) == outcome
    assert dec.error is was
    assert not delivered()


@pytest.mark.parametrize("dec_cls", ENGINES)
def test_the_nested_control_shows_the_ceiling_caused_every_rejection(dec_cls):
    """THE NEGATIVE CONTROL, and the reason this block can be believed at all.

    Every case here ends with its frames still open, so a decoder has a second,
    fully independent reason to answer ``INCOMPLETE`` — and, the other way
    round, a port that rejected these bytes for some unrelated reason (a depth
    guard, a frame-count guard, a blanket refusal of an unclosed frame) would
    pass the forward pass above while never having consulted the ceiling under
    test. Only this tells the two apart.

    So: the same receiver, the same bytes, and the **same kind** of ceiling the
    case names — a ``limits`` case gets a lifted receiver cap, a ``schema`` case
    a lifted schema bound, never the other one and never both — raised to
    :data:`LIFTED`, far above every ``declared`` in the block. The rejection must
    then be **gone**. What replaces it is not asserted: that would be a claim
    about the alternative answer, where the point is only that the ceiling is
    what caused the rejection.

    The number of cases checked is asserted too, because a control loop that
    quietly ``continue``s through every iteration is green and proves nothing.
    """
    checked = []
    for case in NESTED:
        if case["expect"]["outcome"] not in ("limit_exceeded", "invalid"):
            continue  # only a rejection can be shown to depend on its ceiling
        if set(case.get("requires", ())) - SUPPORTED:
            continue  # gated in the forward pass, so not checkable here either
        assert case["declared"] < LIFTED, (
            f"{case['name']} declares more than the control lifts to; skip it explicitly"
        )
        dec, delivered, frames_seen = _decoder(dec_cls, case, ceiling=LIFTED)
        outcome = _feed(dec, _chunks(case))
        assert outcome != case["expect"]["outcome"], (
            f"{case['name']}: lifting the ceiling changed nothing, so the rejection was "
            f"never the ceiling's -- error: {dec.error!r}"
        )
        # ... and the walk still went as deep, so the ceiling was lifted on the
        # field under test rather than the message being refused earlier.
        assert frames_seen() == list(_frames(case))
        assert not delivered()
        checked.append(case["name"])

    # Four rejections, all four controllable: this block has no amplification
    # case the control cannot lift past (the flat block's 1 GiB one is the
    # exemption, and it is not here). A port without receiver caps would check
    # only the one `schema` rejection.
    expected = sum(
        1
        for c in NESTED
        if c["expect"]["outcome"] in ("limit_exceeded", "invalid")
        and not set(c.get("requires", ())) - SUPPORTED
    )
    assert len(checked) == expected == 4, checked


def test_the_nested_block_reports_what_it_ran():
    """``ran + gated == total``, with the gating tag named per gated case.

    Mandatory, not cosmetic: a mis-spelled capability name or a probe that
    answered "unsupported" by accident turns this runner into a no-op that
    reports green, and the counts are the cheap way to see it. A port that
    legitimately gates (no receiver caps runs 4 of 8) must be distinguishable in
    CI output from one that is broken.
    """
    assert NESTED, "the vector file carries no header_limits_nested block"
    ran, gated = [], {}
    for case in NESTED:
        missing = sorted(set(case.get("requires", ())) - SUPPORTED)
        if missing:
            gated[case["name"]] = missing
        else:
            ran.append(case["name"])

    print(f"header_limits_nested: ran = {len(ran)}, gated = {len(gated)} of {len(NESTED)}")
    for name, tags in gated.items():
        print(f"  gated {name}: {tags}")
    assert len(ran) + len(gated) == len(NESTED)

    # This port compiles nothing out and carries §6.2.1 receiver caps, so it
    # runs the block whole; a reduced build would say so here instead.
    assert not gated, f"unexpectedly gated: {gated}"
    # Depth 2 is its own trap -- a chain builder off by one handles `[7]` and
    # mishandles `[7, 3]` -- so assert a two-frame case is among the ones that
    # ran rather than trusting the total.
    assert any(len(_frames(c)) == 2 for c in NESTED if c["name"] in ran)


def test_the_nested_block_pins_both_ceilings_on_the_same_nested_word():
    """The highest-value pair in the block: identical bytes at identical depth,
    opposite verdicts, because the case configures a different ceiling. A port
    that collapses ``limit_exceeded`` and ``invalid`` into one category passes
    seven of the eight cases and fails this one."""
    by_name = {c["name"]: c for c in NESTED}
    cap = by_name["nested_string_over_cap"]
    schema = by_name["nested_string_schema_bounded"]
    assert cap["serialized"] == schema["serialized"]
    assert _frames(cap) == _frames(schema)
    assert (cap["expect"]["outcome"], schema["expect"]["outcome"]) == (
        "limit_exceeded",
        "invalid",
    )


def test_the_nested_block_is_the_flat_block_one_frame_deeper():
    """Depth is the **only** axis this block adds.

    Every case carries a non-empty ``frames`` (that is what makes it nested and
    what makes a separate top-level block necessary: a runner blind to ``frames``
    would bind its ceiling at the top level), and the key set is otherwise the
    flat block's, so the shared leaf above can run both.
    """
    flat_keys = {k for c in CASES for k in c}
    for case in NESTED:
        assert _frames(case), case["name"]
        assert set(case) - {"frames"} <= flat_keys, (
            f"{case['name']} carries keys the flat block's leaf does not know: "
            f"{sorted(set(case) - {'frames'} - flat_keys)}"
        )
