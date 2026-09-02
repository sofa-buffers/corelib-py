"""Push-feed decoding: ``feed`` and the three-valued outcome (CORELIB_PLAN §5.2).

The other half of §5.2's push-feed / pull-read model. What is under test here is
the *contract*, not the speed: that ``feed`` returns ``COMPLETE`` /
``INCOMPLETE`` / ``INVALID`` and never folds one into another, that there is no
finalize step that could turn a truncation into an error, that a chunk is
borrowed only for the duration of the call, and that the two engines agree on
every one of those. The binding destinations get their own suite
(``test_binding``).
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES, ENGINE_PAIRS, NO_CAPS, ROOMY_REASSEMBLY, VECTORS, capped

from sofab import FIXLEN_MAX, Binding, SofaArgumentError, SofaLimitError, Status, Visitor


class Collect(Visitor):
    """Records every value the decoder hands over, in order."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_unsigned(self, field_id, value):
        self.events.append(("u", field_id, value))

    def on_signed(self, field_id, value):
        self.events.append(("s", field_id, value))

    def on_float32(self, field_id, value):
        self.events.append(("f32", field_id, value))

    def on_float64(self, field_id, value):
        self.events.append(("f64", field_id, value))

    def on_string(self, field_id, value):
        self.events.append(("str", field_id, value))

    def on_bytes(self, field_id, value):
        self.events.append(("blob", field_id, value))

    def on_unsigned_array(self, field_id, values):
        self.events.append(("ua", field_id, tuple(values)))

    def on_signed_array(self, field_id, values):
        self.events.append(("sa", field_id, tuple(values)))

    def on_float32_array(self, field_id, values):
        self.events.append(("f32a", field_id, tuple(values)))

    def on_float64_array(self, field_id, values):
        self.events.append(("f64a", field_id, tuple(values)))

    def on_sequence_begin(self, field_id):
        self.events.append(("seq{", field_id))
        return None

    def on_sequence_end(self):
        self.events.append(("seq}",))


def sample(enc_cls) -> bytes:
    """One message touching every wire type, including a nested sequence."""
    enc = enc_cls()
    enc.write_unsigned(1, 300)
    enc.write_signed(2, -7)
    enc.write_float32(3, 1.5)
    enc.write_float64(4, -2.25)
    enc.write_string(5, "grüß dich")
    enc.write_bytes(6, b"\x00\xff\x10")
    enc.write_unsigned_array(7, [1, 2, 300])
    enc.write_signed_array(8, [-1, 2, -3])
    enc.write_float32_array(9, [1.5, -2.5])
    enc.write_float64_array(10, [1e300, -0.5])
    enc.write_sequence_begin_lazy(11)
    enc.write_unsigned(1, 42)
    enc.write_sequence_end()
    enc.flush()
    return enc.getvalue()


# --- the three outcomes ------------------------------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_whole_message_is_complete(enc_cls, dec_cls):
    v = Collect()
    assert dec_cls(**NO_CAPS, visitor=v).feed(sample(enc_cls)) is Status.COMPLETE


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_no_prefix_of_a_valid_message_is_invalid(enc_cls, dec_cls):
    """§5.2 forbids folding INCOMPLETE into INVALID: a prefix the caller may
    still extend is not malformed. A prefix ending *between* fields is
    legitimately COMPLETE — a valid message may end there — so what every cut
    has in common is only that none of them is INVALID."""
    msg = sample(enc_cls)
    for cut in range(1, len(msg)):
        dec = dec_cls(**NO_CAPS, visitor=Collect())
        assert dec.feed(msg[:cut]) is not Status.INVALID, cut
        assert dec.error is None


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_cut_inside_a_construct_is_incomplete(enc_cls, dec_cls):
    """The other half of the same rule: stopping *inside* a field is
    INCOMPLETE, and nothing of that field has been handed over yet."""
    enc = enc_cls()
    enc.write_string(1, "a long enough payload to cut in half")
    enc.flush()
    msg = enc.getvalue()
    for cut in range(1, len(msg)):
        v = Collect()
        dec = dec_cls(**NO_CAPS, visitor=v)
        assert dec.feed(msg[:cut]) is Status.INCOMPLETE, cut
        assert v.events == []


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_feeding_the_rest_completes_the_message(enc_cls, dec_cls):
    """And the retained tail is not lost: the same values arrive whatever the
    split, which is what makes INCOMPLETE a first-class outcome rather than an
    error (§5.2, §7.2 item 4)."""
    msg = sample(enc_cls)
    whole = Collect()
    dec_cls(**NO_CAPS, visitor=whole).feed(msg)
    for cut in range(1, len(msg)):
        split = Collect()
        dec = dec_cls(**NO_CAPS, visitor=split)
        assert dec.feed(msg[:cut]) is not Status.INVALID
        assert dec.feed(msg[cut:]) is Status.COMPLETE
        assert split.events == whole.events, cut


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_one_byte_at_a_time_decodes_the_same(enc_cls, dec_cls):
    msg = sample(enc_cls)
    whole = Collect()
    dec_cls(**NO_CAPS, visitor=whole).feed(msg)
    drip = Collect()
    dec = dec_cls(**NO_CAPS, visitor=drip)
    st = None
    for i in range(len(msg)):
        st = dec.feed(msg[i : i + 1])
    assert st is Status.COMPLETE
    assert drip.events == whole.events


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize(
    ("data", "what"),
    [
        (b"\x08" + b"\x80" * 10 + b"\x02", "overlong varint"),
        (b"\x3f", "sequence end with nothing open"),
        (b"\x0a\x2c", "reserved fixlen subtype"),
        (b"\x0a\x58", "fp64 fixlen of the wrong width"),
    ],
)
def test_malformed_input_is_invalid(dec_cls, data, what):
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(data) is Status.INVALID, what
    assert dec.error is not None


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_invalid_is_terminal(dec_cls):
    """§5.2: INVALID means no continuation of bytes can make the stream valid,
    so the decoder stays there rather than resynchronising."""
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(b"\x3f") is Status.INVALID
    reason = dec.error
    assert dec.feed(b"\x08\x01") is Status.INVALID
    assert dec.error is reason


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_an_open_sequence_at_end_of_input_is_incomplete(enc_cls, dec_cls):
    """Bytes that end inside a sequence that never closed are truncation, not
    malformation — the caller's framing decides whether that is an error."""
    enc = enc_cls()
    enc.write_sequence_begin_lazy(1)
    enc.write_unsigned(2, 5)
    enc.flush()
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(enc.getvalue()) is Status.INCOMPLETE
    assert dec.error is None


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_a_lone_continuation_byte_is_incomplete(dec_cls):
    """§5.2 names this one explicitly: 0x80 is a well-formed *prefix* of a
    varint, so it is INCOMPLETE, not INVALID."""
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(b"\x80") is Status.INCOMPLETE


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_feed_is_the_only_place_the_outcome_is_reported(dec_cls):
    """§5.2.4: "The status ``feed``/``decode`` returns *is* the answer." Each
    call hands back the outcome for the bytes so far, and nothing on the object
    repeats it -- a second accessor is a second thing to keep in step, and this
    port once had it answer COMPLETE for a message ``feed`` had refused."""
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(b"\x80") is Status.INCOMPLETE  # a varint prefix, mid-header...
    assert dec.feed(b"\x01\x05") is Status.COMPLETE  # ...finishing id 16 = 5
    assert not hasattr(dec, "status")


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_there_is_no_finalize_step(dec_cls):
    """§5.2: a decoder MUST NOT provide a finish/finalize that reclassifies
    INCOMPLETE. Nothing on the object may promote a truncation to an error."""
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    for name in ("finish", "finalize", "end", "close", "eof"):
        assert not hasattr(dec, name), name


# --- chunk lifetime (§6, normative) -----------------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 3, 1000])
def test_a_fed_chunk_may_be_overwritten_afterwards(enc_cls, dec_cls, chunk):
    """"A fed chunk is borrowed only for the duration of the feed call": the
    decoded message must not change when the caller reuses that memory. Feeding
    from a bytearray that is then scribbled over is the direct test."""
    msg = sample(enc_cls)
    whole = Collect()
    dec_cls(**NO_CAPS, visitor=whole).feed(msg)

    got = Collect()
    dec = dec_cls(**capped(reassembly=len(msg) + 16), visitor=got)
    scratch = bytearray(chunk)
    view = memoryview(scratch)
    for off in range(0, len(msg), chunk):
        piece = msg[off : off + chunk]
        scratch[: len(piece)] = piece
        # A *view* of the scratch, not a slice of it. ``scratch[:n]`` copies —
        # the object fed would be a fresh bytearray, the scrub below could not
        # reach it, and the case would pass whatever the decoder did with the
        # bytes. §7.2 item 4 wants the buffer itself fed and then overwritten.
        dec.feed(view[: len(piece)])
        scratch[:] = b"\xa5" * len(scratch)  # the caller reuses its buffer
    view.release()
    assert got.events == whole.events


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_feed_accepts_any_buffer(dec_cls):
    msg = b"\x08\xac\x02"
    for shape in (msg, bytearray(msg), memoryview(msg)):
        v = Collect()
        assert dec_cls(**NO_CAPS, visitor=v).feed(shape) is Status.COMPLETE
        assert v.events == [("u", 1, 300)]


# --- construction rules ------------------------------------------------------


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_a_decoder_needs_a_source_or_a_handler(dec_cls):
    with pytest.raises(SofaArgumentError):
        dec_cls()


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_feed_is_not_re_entrant(dec_cls):
    class Reentrant(Visitor):
        def on_unsigned(self, field_id, value):
            dec.feed(b"\x08\x01")

    dec = dec_cls(**NO_CAPS, visitor=Reentrant())
    with pytest.raises(SofaArgumentError):
        dec.feed(b"\x08\x01")


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_reset_starts_a_new_message(enc_cls, dec_cls):
    msg = sample(enc_cls)
    v = Collect()
    dec = dec_cls(**NO_CAPS, visitor=v)
    assert dec.feed(msg) is Status.COMPLETE
    n = len(v.events)
    dec.reset()
    assert dec.feed(msg) is Status.COMPLETE
    assert len(v.events) == 2 * n


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_reset_clears_a_terminal_invalid(dec_cls):
    dec = dec_cls(**NO_CAPS, visitor=Collect())
    assert dec.feed(b"\x3f") is Status.INVALID
    dec.reset()
    assert dec.error is None
    assert dec.feed(b"\x08\xac\x02") is Status.COMPLETE


# --- receiver caps are not INVALID (§6.2.1 / §6.3) ---------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_receiver_cap_is_raised_not_folded_into_invalid(enc_cls, dec_cls):
    """A capped message is *well-formed* — the same bytes decode under a looser
    limit — so it must not arrive as INVALID. §6.3 leaves the channel open;
    Python is an exceptions language, so it is raised."""
    enc = enc_cls()
    enc.write_unsigned_array(1, [1, 2, 3, 4, 5])
    enc.flush()
    dec = dec_cls(reassembly=ROOMY_REASSEMBLY, max_dyn_blob_len=FIXLEN_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Collect(), max_dyn_array_count=2)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_declared_bound_takes_the_receiver_cap_off_the_field(enc_cls, dec_cls):
    """§6.2.1: a field the schema bounds is not subject to the receiver-side cap.
    A binding that declares the bound *is* that declaration, so the same message
    that a cap rejects decodes when the field is bound — and a message past the
    declared bound is INVALID (§7.1), not a limit rejection."""
    enc = enc_cls()
    enc.write_unsigned_array(1, [1, 2, 3, 4, 5])
    enc.flush()
    msg = enc.getvalue()
    b = Binding().unsigned_array(1, at=0, cap=8, count_at=8)
    words = bytearray(b.tree_words_required * 8)
    assert dec_cls(reassembly=ROOMY_REASSEMBLY, max_dyn_blob_len=FIXLEN_MAX, max_dyn_string_len=FIXLEN_MAX, binding=b, words=words, max_dyn_array_count=2).feed(msg) is Status.COMPLETE

    tight = Binding().unsigned_array(1, at=0, cap=3, count_at=8)
    dec = dec_cls(**NO_CAPS, binding=tight, words=bytearray(tight.tree_words_required * 8))
    assert dec.feed(msg) is Status.INVALID


# --- the whole shared vector set, through feed -------------------------------


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
@pytest.mark.parametrize("chunk", [1, 2, 7, 4096])
def test_every_decode_vector_is_chunking_independent(dec_cls, chunk):
    """§7.2 item 4: the chunk boundaries must never change the outcome. Feeding
    each vector whole and then in slices has to hand over the same fields."""
    for vec in VECTORS:
        if "bytes" not in vec:
            continue
        data = bytes.fromhex(vec["bytes"])
        want = Collect()
        if dec_cls(**NO_CAPS, visitor=want).feed(data) is not Status.COMPLETE:
            continue  # the vector set includes malformed input

        got = Collect()
        dec = dec_cls(**NO_CAPS, visitor=got)
        st = Status.COMPLETE
        for off in range(0, len(data), chunk):
            st = dec.feed(data[off : off + chunk])
        assert st is Status.COMPLETE, vec.get("name")
        assert got.events == want.events, vec.get("name")


# --- state the field walk leaves behind -------------------------------------
#
# Both of these guard invariants an optimisation can quietly break: the decode
# stays correct while the *bookkeeping* around it drifts, so nothing else in the
# suite notices. Both did break while the bound path was being made faster.


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_a_declined_field_that_straddles_a_chunk_does_not_replay(enc_cls, dec_cls, chunk):
    """A field the handler declines is walked by the *next* header parse. That
    walk commits — so if the same call then runs out of bytes (at end of input
    inside an open sequence, say), the retry must not rewind to before it and
    read the skipped value's bytes as a new field.

    Only a push decoder can reach this: a reader-backed one blocks inside the
    refill instead of returning to the caller mid-call. It stalled forever at
    INCOMPLETE before the resume point was re-armed after the skip.
    """
    enc = enc_cls()
    enc.write_unsigned(1, 1)
    enc.write_sequence_begin_lazy(3)
    enc.write_unsigned(5, 5_555_555)  # declined, and four varint bytes wide
    enc.write_unsigned(6, 6)
    enc.write_sequence_end_keep()
    enc.write_unsigned(9, 9)
    enc.flush()
    wire = enc.getvalue()

    rec = _DecliningCollect(5)
    dec = dec_cls(**NO_CAPS, visitor=rec)
    status = Status.COMPLETE
    for off in range(0, len(wire), chunk):
        status = dec.feed(wire[off : off + chunk])
    assert status is Status.COMPLETE
    assert rec.events == [
        ("u", 1, 1),
        ("seq{", 3),
        ("u", 6, 6),
        ("seq}",),
        ("u", 9, 9),
    ]


class _DecliningCollect(Collect):
    """Records like :class:`Collect`, but declines one field id."""

    def __init__(self, decline_id: int) -> None:
        super().__init__()
        self._decline_id = decline_id

    def on_field(self, field):
        return False if field.id == self._decline_id else None


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_truncated_fixlen_payload_is_incomplete(enc_cls, dec_cls):
    """The payload read and the skip over one both run out in their own place;
    neither may report anything but INCOMPLETE."""
    enc = enc_cls()
    enc.write_bytes(1, b"\xa5" * 40)
    enc.flush()
    cut = enc.getvalue()[:-10]

    assert dec_cls(**NO_CAPS, visitor=Collect()).feed(cut) is Status.INCOMPLETE
    assert dec_cls(**NO_CAPS, visitor=_DecliningCollect(1)).feed(cut) is Status.INCOMPLETE
