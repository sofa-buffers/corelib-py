"""Booleans: canonical on encode, tolerant on decode (CORELIB_PLAN §4.4).

§4.4 is two rules pointing in opposite directions, and they meet in the corelib
rather than above it:

* an encoder **MUST** write ``true`` as ``1``;
* a decoder **MUST** read *every* value other than ``0`` as ``true`` — such a
  value is **not** INVALID, it is normalized away, and a re-encode emits ``1``.

The second half is what this suite is about. A boolean has no wire type of its
own, so it arrives under the unsigned tag and nothing in the *header* tells the
two apart — only the schema does, which is why the rule lives on
:meth:`sofab.Binding.boolean` and :meth:`sofab.Binding.boolean_array` and not on
the visitor: without a table there is no boolean, only an unsigned integer.

The distinction §4.4 draws explicitly, and the one these tests pin hardest: a
boolean carries **no** declared width, unlike an ``enum`` or a ``bitfield``. A
value outside ``0..1`` is normalized, never rejected.
"""

from __future__ import annotations

import pytest
from vectors import (
    DECODER_ENGINES,
    ENCODER_ENGINES,
    ENGINE_PAIRS,
    Recorder,
    Status,
    bound,
    walk,
)

from sofab import Binding, SofaArgumentError, SofaBufferError

#: Values a conforming encoder never writes, which a decoder must still read as
#: true. ``1 << 63`` and ``(1 << 64) - 1`` are there because the slot is 64 bits
#: wide and the top bit is the one a sloppy normalization drops.
NON_CANONICAL = [2, 3, 42, 128, 255, 256, 1 << 32, 1 << 63, (1 << 64) - 1]


def _msg(enc_cls, fn) -> bytes:
    enc = enc_cls()
    fn(enc)
    enc.flush()
    return enc.getvalue()


# --- the scalar ---------------------------------------------------------------


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("raw", NON_CANONICAL)
def test_a_non_canonical_boolean_is_normalized_to_one(enc_cls, dec_cls, raw):
    """The clause itself: not INVALID, normalized away. The slot holds ``1``,
    never the value the sender happened to write."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned(1, raw))
    b = Binding().boolean(1, at=0, count_at=1)
    status, dec, s = bound(dec_cls, msg, b)
    assert status is Status.COMPLETE, dec.error
    assert s.u[0] == 1
    assert s.u[1] == 1, "arrival is still recorded in count_at"


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_zero_is_the_only_false(enc_cls, dec_cls):
    msg = _msg(enc_cls, lambda e: e.write_unsigned(1, 0))
    status, dec, s = bound(dec_cls, msg, Binding().boolean(1, at=0, count_at=1))
    assert status is Status.COMPLETE, dec.error
    assert s.u[0] == 0
    assert s.u[1] == 1, "false arrived; absence is a different thing"


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_absent_boolean_leaves_its_slot_alone(enc_cls, dec_cls):
    """Absence is reported by count_at staying untouched, not by a false in the
    value slot — the same contract every other bound kind has.

    The message carries an *unrelated* id rather than being empty: the walk has
    to actually run past the bound field, or the assertion holds no matter what
    the boolean path does."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned(7, 1))
    b = Binding().boolean(1, at=0, count_at=1)
    words = bytearray(b"\xAA" * (b.tree_words_required * 8))
    dec = dec_cls(
        binding=b,
        words=words,
        objects=[],
        max_dyn_array_count=1,
        max_dyn_string_len=1,
        max_dyn_blob_len=1,
        reassembly=4096,
    )
    assert dec.feed(msg) is Status.COMPLETE
    assert list(memoryview(words).cast("Q")) == [0xAAAAAAAAAAAAAAAA] * 2


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_boolean_carries_no_declared_width(enc_cls, dec_cls):
    """§4.4's own contrast. An ``enum``/``bitfield`` bound to a width rejects a
    value outside it (§1, §7.1); the widest possible value read as a boolean is
    merely true. Both bindings, same bytes, deliberately different verdicts."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned(1, (1 << 64) - 1))

    status, dec, s = bound(dec_cls, msg, Binding().boolean(1, at=0))
    assert status is Status.COMPLETE, dec.error
    assert s.u[0] == 1

    status, _dec, _s = bound(dec_cls, msg, Binding().unsigned(1, at=0, max_value=1))
    assert status is Status.INVALID, "a declared width is what makes it INVALID"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_tolerated_boolean_re_encodes_canonically(enc_cls, dec_cls):
    """"...and a re-encode emits ``1``". True end to end only because the slot
    was normalized: a raw 256 written back out would still be 256."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned(1, 256))
    status, dec, s = bound(dec_cls, msg, Binding().boolean(1, at=0))
    assert status is Status.COMPLETE, dec.error

    assert s.u[0] == 1, "the slot itself is what §4.4 normalizes"
    # Re-encoded from the slot as the raw number it holds — NOT through
    # ``bool()``, which would collapse a still-raw 256 to True and make this
    # pass against exactly the behaviour it is here to rule out.
    again = _msg(enc_cls, lambda e: e.write_unsigned(1, s.u[0]))
    assert again == _msg(enc_cls, lambda e: e.write_bool(1, True))
    assert again.endswith(b"\x01")


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_contradicting_tag_is_still_skipped(enc_cls, dec_cls):
    """Tolerance is about the *value*, not the tag. A boolean's tag is the
    unsigned one, so a signed field at the same id is the §7.3 mismatch: skipped
    like an unknown id, decode stays COMPLETE, and the slot is never written."""
    msg = _msg(enc_cls, lambda e: e.write_signed(1, -5))
    b = Binding().boolean(1, at=0, count_at=1)
    words = bytearray(b"\xAA" * (b.tree_words_required * 8))
    dec = dec_cls(
        binding=b,
        words=words,
        objects=[],
        max_dyn_array_count=1,
        max_dyn_string_len=1,
        max_dyn_blob_len=1,
        reassembly=4096,
    )
    assert dec.feed(msg) is Status.COMPLETE
    assert list(memoryview(words).cast("Q")) == [0xAAAAAAAAAAAAAAAA] * 2


# --- the array ----------------------------------------------------------------


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_array_of_boolean_normalizes_every_element(enc_cls, dec_cls):
    """§4.4 applies per element — the array half of the same clause, and the
    reason ``boolean_array`` exists rather than callers reaching for
    ``unsigned_array``."""
    raw = [0, 1, 2, 255, 0, 1 << 63, (1 << 64) - 1, 0]
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(3, raw))
    b = Binding().boolean_array(3, at=0, cap=8, count_at=8)
    status, dec, s = bound(dec_cls, msg, b)
    assert status is Status.COMPLETE, dec.error
    assert s.u[8] == len(raw)
    assert s.arr_u(0, len(raw)) == [0, 1, 1, 1, 0, 1, 1, 0]


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_array_of_boolean_writes_no_slot_past_its_count(enc_cls, dec_cls):
    """The normalization pass runs over the elements that arrived, not over the
    capacity the table declared: the slots behind them are the caller's."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(3, [7, 0]))
    b = Binding().boolean_array(3, at=0, cap=6, count_at=6)
    words = bytearray(b"\xAA" * (b.tree_words_required * 8))
    dec = dec_cls(
        binding=b,
        words=words,
        objects=[],
        max_dyn_array_count=64,
        max_dyn_string_len=1,
        max_dyn_blob_len=1,
        reassembly=4096,
    )
    assert dec.feed(msg) is Status.COMPLETE
    u = memoryview(words).cast("Q")
    assert list(u[0:2]) == [1, 0]
    assert list(u[2:6]) == [0xAAAAAAAAAAAAAAAA] * 4
    assert u[6] == 2


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_empty_array_of_boolean_is_an_array(enc_cls, dec_cls):
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(3, []))
    b = Binding().boolean_array(3, at=0, cap=4, count_at=4)
    status, dec, s = bound(dec_cls, msg, b)
    assert status is Status.COMPLETE, dec.error
    assert s.u[4] == 0


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_array_of_boolean_is_bounded_by_the_schema_not_the_wire(enc_cls, dec_cls):
    """``cap`` is the schema's declared element count and still binds: a wire
    count past it is INVALID (§7.1). Tolerance is about element *values*."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(3, [1] * 5))
    status, _dec, _s = bound(dec_cls, msg, Binding().boolean_array(3, at=0, cap=4))
    assert status is Status.INVALID


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_both_boolean_kinds_work_inside_a_closed_sequence(enc_cls, dec_cls):
    """A nested scope is where a new kind is most likely to fall out: the child
    table is reached through a different path. The child is *closed* and the
    message carries an id it does not name, so the §4.9 skip is exercised too —
    the visitor must not see that id under the parent's identity (#150)."""

    def write(e):
        e.write_sequence_begin_lazy(9)
        e.write_unsigned(1, 77)
        e.write_unsigned_array(2, [5, 0, 9])
        e.write_unsigned(5, 3)  # an id the closed child does not name
        e.write_sequence_end()

    child = Binding(closed=True).boolean(1, at=0).boolean_array(2, at=1, cap=4, count_at=5)
    root = Binding().sequence(9, child)

    class Handler(Recorder):
        words = bytearray(root.tree_words_required * 8)

        def destinations(self):
            return (root, self.words, None)

    rec = Handler()
    status, _rec, dec = walk(dec_cls, _msg(enc_cls, write), recorder=rec)
    assert status is Status.COMPLETE, dec.error
    u = memoryview(Handler.words).cast("Q")
    assert u[0] == 1
    assert list(u[1:4]) == [1, 0, 1]
    assert u[5] == 3
    assert rec.events == [], "a closed child hands nothing to the visitor"


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_declared_boolean_array_spends_the_receiver_cap(enc_cls, dec_cls):
    """``cap`` is the schema's bound, and §6.2.1 forbids applying a receiver cap
    to a field the schema already bounds — the new array kind reaches that rule
    through the same ``_settle_bound`` every other array kind does."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(2, [1] * 10))
    b = Binding().boolean_array(2, at=0, cap=16, count_at=16)
    status, dec, s = bound(dec_cls, msg, b, max_dyn_array_count=2)
    assert status is Status.COMPLETE, dec.error
    assert s.u[16] == 10


# --- chunking: the resume transaction (§5.2) ----------------------------------


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("chunk", [1, 3])
def test_normalization_survives_any_chunking(enc_cls, dec_cls, chunk):
    """A multi-byte non-canonical value straddles chunks, and an array resumes
    from element zero — the normalization must not be applied twice, nor to a
    half-filled array. Same bytes, same slots, whatever the chunking."""
    raw = [0, 300, 1, 1 << 63, 0]
    msg = _msg(
        enc_cls,
        lambda e: (e.write_unsigned(1, 70000), e.write_unsigned_array(3, raw)),
    )
    b = Binding().boolean(1, at=0, count_at=1).boolean_array(3, at=2, cap=8, count_at=10)
    status, dec, s = bound(dec_cls, msg, b, chunk=chunk)
    assert status is Status.COMPLETE, dec.error
    assert s.u[0] == 1
    assert s.u[10] == len(raw)
    assert s.arr_u(2, len(raw)) == [0, 1, 1, 1, 0]


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_half_fed_array_may_still_hold_raw_values(enc_cls, dec_cls):
    """The normalization is one pass after the payload lands, so a decode caught
    INCOMPLETE *inside* the array leaves raw wire values in the slots it already
    wrote. Both engines do, identically, and ``Binding.boolean_array`` documents
    it — pinned here so it reads as a decision rather than an accident, and so a
    future per-element normalization is a deliberate change and not a silent one.

    What the contract actually promises is about a **completed** decode, and the
    second half of this test is that promise: the retry refills from element
    zero and every slot ends up 0/1."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned_array(3, [255] * 4))
    b = Binding().boolean_array(3, at=0, cap=4, count_at=4)
    words = bytearray(b.tree_words_required * 8)
    dec = dec_cls(
        binding=b,
        words=words,
        objects=[],
        max_dyn_array_count=64,
        max_dyn_string_len=1,
        max_dyn_blob_len=1,
        reassembly=4096,
    )
    u = memoryview(words).cast("Q")
    assert dec.feed(msg[:-2]) is Status.INCOMPLETE
    assert list(u[0:2]) == [255, 255], "raw, because the array has not completed"

    assert dec.feed(msg[-2:]) is Status.COMPLETE
    assert list(u[0:4]) == [1, 1, 1, 1]
    assert u[4] == 4


# --- the encode half, and where the rule is NOT ------------------------------


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_write_bool_array_is_canonical(enc_cls):
    """The array half of ``write_bool``: an element is tested for truth and
    ``true`` goes out as ``1``, whatever the element was."""
    truthy = [0, 1, 2, "", "x", None, [], [0], False, True]
    got = _msg(enc_cls, lambda e: e.write_bool_array(3, truthy))
    want = _msg(
        enc_cls,
        lambda e: e.write_unsigned_array(3, [1 if v else 0 for v in truthy]),
    )
    assert got == want
    assert b"\x02" not in got[2:], "no element went out as anything but 0 or 1"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_write_bool_array_drains_a_buffer_it_outgrows(enc_cls):
    """The element loop is inlined over the output buffer, so it has to hand
    over to the drain when the buffer fills — 5000 one-byte elements through a
    1 KiB scratch buffer is several drains, and the bytes must not notice."""
    flags = [i % 3 for i in range(5000)]
    got = _msg(enc_cls, lambda e: e.write_bool_array(3, flags))
    assert got == _msg(
        enc_cls, lambda e: e.write_unsigned_array(3, [1 if v else 0 for v in flags])
    )
    assert got.endswith(bytes(1 if v else 0 for v in flags))


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_raising_element_leaves_the_partial_write(enc_cls):
    """Both engines must reach the *same* state, not merely the same exception.
    The element test runs inside the loop, exactly where the other array writers
    convert theirs, so what was written stays written: header, count, and the
    elements that got through. Feeding the truth test up front would leave an
    empty buffer in one engine and a partial one in the other."""

    class Boom:
        def __bool__(self):
            raise RuntimeError("boom")

    enc = enc_cls()
    with pytest.raises(RuntimeError):
        enc.write_bool_array(3, [True, Boom(), False])
    assert enc.getvalue() == bytes.fromhex("1b0301")

    same = enc_cls()
    with pytest.raises(Exception):  # noqa: B017 - the point is the state, below
        same.write_unsigned_array(3, [1, "x", 0])
    assert same.getvalue() == bytes.fromhex("1b0301"), "the house shape"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_sticky_encoder_latches_over_write_bool_array(enc_cls):
    """Sticky mode turns every write after the first failure into a no-op, and
    the new writer obeys it like every other."""
    enc = enc_cls(sticky=True)
    enc.write_unsigned(1, -1)
    assert isinstance(enc.error, SofaArgumentError)
    enc.write_bool_array(3, [True, False])  # skipped: the failure is latched
    enc.flush()
    assert enc.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_write_bool_array_reports_a_buffer_it_cannot_fill(enc_cls):
    """A caller-owned buffer with no flush sink is the shape generated code
    sizes from a schema's MAX_SIZE: overrunning it is :class:`SofaBufferError`,
    raised from inside the element loop like any other write."""
    enc = enc_cls.over_buffer(bytearray(6), 0)
    with pytest.raises(SofaBufferError):
        enc.write_bool_array(3, [True] * 32)


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_write_bool_array_round_trips_through_the_binding(enc_cls, dec_cls):
    flags = [True, False, True, True]
    msg = _msg(enc_cls, lambda e: e.write_bool_array(3, flags))
    b = Binding().boolean_array(3, at=0, cap=4, count_at=4)
    status, dec, s = bound(dec_cls, msg, b)
    assert status is Status.COMPLETE, dec.error
    assert [bool(v) for v in s.arr_u(0, s.u[4])] == flags


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_visitor_still_sees_an_unsigned(enc_cls, dec_cls):
    """Where the rule is *not*, on purpose. Without a table there is no boolean
    on the wire to recognise — §4.4's mapping needs the schema — so a visitor
    gets the unsigned integer it asked for, raw. That is not the §4.4 gap this
    suite closes: ``on_unsigned`` was never claimed to be a boolean read."""
    msg = _msg(enc_cls, lambda e: e.write_unsigned(1, 256))
    status, rec, _dec = walk(dec_cls, msg)
    assert status is Status.COMPLETE
    assert rec.events == [("u", 1, 256)]
