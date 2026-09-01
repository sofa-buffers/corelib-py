"""A string and a blob into storage the handler declared **before** the decode.

CORELIB_PLAN §6.6.3 gives a conforming port three shapes for delivering an
aggregate. The third is:

    into a destination the caller declared **before the decode began** — a
    field-id → slot table (§5.3.1) — which is the bullet above with the choice
    made once instead of per field. The codec's obligations are identical: the
    announced count or length is settled against what the schema declares first,
    a destination too short is **`InvalidArgument`**, and it is never grown.

A ``Binding`` had that shape for every numeric field and for both array kinds
already: an entry names ``words`` slots the caller sized from the schema. It did
**not** have it for the two aggregates with no fixed-width machine form. A
``string``/``blob`` row named a slot in ``objects`` to *put a value in*, and the
value was one the codec had to build — from the only size it has, the wire's. So
a whole message could decode into caller storage and a 1 MiB string would still
cost a 1 MiB allocation inside the codec, on the very route §6.6.3 names.

:meth:`sofab.Binding.string_into` / :meth:`~sofab.Binding.blob_into` close that:
the slot already holds the buffer, and the payload is copied into it.

What is tested here is that the *rules* did not fork with the route (§5.3.1,
§6.2.1): the same schema bound, the same MESSAGE_SPEC §7.3 tag test, the same
UTF-8 verdict, the same resume transaction, and the same `InvalidArgument` for a
destination too short — all of them reaching the implementation an
``on_string_begin`` destination reaches.
"""

from __future__ import annotations

import array

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import ENCODER_ENGINES as ENCODERS
from vectors import NO_CAPS, capped

from sofab import (
    Binding,
    Encoder,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    SofaLimitError,
    Status,
    Visitor,
)

TEXT = "héllo wörld " * 4  # multi-byte, so bytes != characters
BLOB = bytes(range(256)) * 2


def _wire(enc_cls, *, text=TEXT, blob=BLOB):
    enc = enc_cls()
    enc.write_string(1, text)
    enc.write_bytes(2, blob)
    enc.flush()
    return enc.getvalue()


def _string_field(payload: bytes, field_id: int = 1) -> bytes:
    """A ``string`` field carrying ``payload`` verbatim, valid UTF-8 or not."""
    enc = Encoder()
    enc.write_bytes(field_id, payload)
    enc.flush()
    wire = bytearray(enc.getvalue())
    wire[1] = (len(payload) << 3) | FixlenSubtype.STRING
    return bytes(wire)


def _table(**kw):
    b = Binding().string_into(1, at=0, count_at=0, **kw).blob_into(2, at=1, count_at=1)
    return b, [bytearray(1024), bytearray(1024)], bytearray(b.tree_words_required * 8)


# --- the route itself --------------------------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_both_aggregates_land_in_the_declared_buffers(dec_cls, enc_cls):
    """The payload's own bytes, and the byte length in ``count_at`` — which is
    what tells the caller how much of its buffer the message filled."""
    table, objs, words = _table()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(_wire(enc_cls)) is Status.COMPLETE

    n = memoryview(words).cast("Q")
    utf8 = TEXT.encode()
    assert len(utf8) != len(TEXT), "the case must distinguish bytes from characters"
    assert n[0] == len(utf8)
    assert bytes(objs[0][: n[0]]) == utf8
    assert n[1] == len(BLOB)
    assert bytes(objs[1][: n[1]]) == BLOB
    # The buffers the caller put there are the buffers that were written: the
    # decoder replaced neither, which is what a value-building route would have
    # had to do.
    assert isinstance(objs[0], bytearray) and isinstance(objs[1], bytearray)


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_plain_rows_still_hand_back_a_value(dec_cls, enc_cls):
    """``string``/``bytes`` are untouched: the new rows are an opt-in beside
    them, not a change to them."""
    table = Binding().string(1, at=0).bytes(2, at=1)
    objs = [None, None]
    dec = dec_cls(**NO_CAPS, binding=table, words=bytearray(64), objects=objs)
    assert dec.feed(_wire(enc_cls)) is Status.COMPLETE
    assert objs == [TEXT, BLOB]


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_an_absent_field_leaves_its_buffer_and_its_count_alone(dec_cls, enc_cls):
    """Absence needs no sentinel here either: what the caller prepared stays."""
    enc = enc_cls()
    enc.write_string(1, "ok")
    enc.flush()
    table, objs, words = _table()
    objs[1][:4] = b"keep"
    words[8:16] = (7).to_bytes(8, "little")  # the blob's count_at slot
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert bytes(objs[1][:4]) == b"keep"
    assert memoryview(words).cast("Q")[1] == 7


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_an_empty_payload_writes_a_zero_length(dec_cls, enc_cls):
    table, objs, words = _table()
    enc = enc_cls()
    enc.write_string(1, "")
    enc.write_bytes(2, b"")
    enc.flush()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    n = memoryview(words).cast("Q")
    assert (n[0], n[1]) == (0, 0)


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("dst", [None, "not a buffer", b"read-only", array.array("d", [0.0] * 8)])
def test_a_slot_that_is_not_a_writable_byte_buffer_is_refused(dec_cls, enc_cls, dst):
    """§6.3's ``InvalidArgument`` tier: the message is fine, the storage is not.

    ``None`` is the shape that matters — it is what ``[None] * n`` leaves, and a
    caller that declared ``string_into`` and forgot to put a buffer there must be
    told so rather than have one made for it.
    """
    table, objs, words = _table()
    objs[0] = dst
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    with pytest.raises(SofaArgumentError):
        dec.feed(_wire(enc_cls))


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("which", [0, 1])
def test_a_short_destination_is_an_argument_error_and_is_never_grown(
    dec_cls, enc_cls, which
):
    """§6.6.3: "with the codec refusing a destination too short rather than
    growing it. That refusal is ``InvalidArgument``" — the same verdict
    ``on_string_begin`` reaches for the same reason."""
    table, objs, words = _table()
    objs[which] = bytearray(4)
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    with pytest.raises(SofaArgumentError):
        dec.feed(_wire(enc_cls))
    assert len(objs[which]) == 4, "the destination was grown"


# --- the rules did not fork with the route ----------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_declared_maxlen_is_the_schema_bound_and_a_longer_payload_is_invalid(
    dec_cls, enc_cls
):
    """MESSAGE_SPEC §7.1 through §6.2.1: ``maxlen`` on the row is the *schema's*
    bound, so exceeding it is INVALID — not a policy rejection, and not an
    argument error, even though the buffer would have held it."""
    table, objs, words = _table(maxlen=4)
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(_wire(enc_cls)) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert bytes(objs[0]) == bytes(1024), "INVALID before the destination is touched"


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_receiver_cap_does_not_gate_a_destination_the_caller_declared(
    dec_cls, enc_cls
):
    """§6.2.1's limit exists to stop the *sender* dictating the *receiver's*
    allocation. A caller that put the buffer in the slot sized it itself, so
    there is no allocation of this decoder's left for the cap to prevent — which
    is exactly what ``on_string_begin`` / ``on_blob_begin`` already do with it.
    One rule, one implementation, whichever way the destination was stated.
    """
    table, objs, words = _table()
    dec = dec_cls(
        **capped(max_dyn_string_len=4, max_dyn_blob_len=4),
        binding=table, words=words, objects=objs,
    )
    assert dec.feed(_wire(enc_cls)) is Status.COMPLETE
    assert bytes(objs[0][: memoryview(words).cast("Q")[0]]) == TEXT.encode()


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_value_row_beside_it_is_still_capped(dec_cls, enc_cls):
    """The control for the test above: the same message, the same cap, and the
    row that makes the codec build the value — where the cap still speaks."""
    table = Binding().string(1, at=0)
    dec = dec_cls(
        **capped(max_dyn_string_len=4), binding=table,
        words=bytearray(64), objects=[None],
    )
    with pytest.raises(SofaLimitError):
        dec.feed(_wire(enc_cls))


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("bad", [b"\xff", b"\xc3", b"\x80", b"\xe2\x28\xa1", b"\xed\xa0\x80"])
def test_the_declared_route_still_validates_utf8(dec_cls, bad):
    """§6.7.2: a field the handler **reads** is materialized *and* validated.
    A route that skipped it would be a way around §6.4."""
    table, objs, words = _table()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(_string_field(bad)) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert bytes(objs[0]) == bytes(1024), "no half-written buffer behind a verdict"


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_contradicting_wire_tag_is_skipped_not_rejected(dec_cls, enc_cls):
    """MESSAGE_SPEC §7.3: the row declares a ``string`` for id 1 and the wire
    carries a blob, so the field is skipped like an unknown id — buffer
    untouched, decode COMPLETE."""
    enc = enc_cls()
    enc.write_bytes(1, b"not a string")
    enc.flush()
    table, objs, words = _table()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert bytes(objs[0]) == bytes(1024)
    assert memoryview(words).cast("Q")[0] == 0, "count_at was written for a skip"


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_declared_route_survives_a_chunk_boundary(dec_cls, enc_cls):
    """§5.2's resume transaction is the same one every other read uses: a
    payload split across feeds is written into the destination exactly once,
    complete."""
    wire = _wire(enc_cls)
    table, objs, words = _table()
    dec = dec_cls(**capped(reassembly=bytearray(len(wire) + 64)), binding=table, words=words, objects=objs, )
    for i in range(0, len(wire), 3):
        dec.feed(wire[i : i + 3])
    n = memoryview(words).cast("Q")
    assert bytes(objs[0][: n[0]]) == TEXT.encode()
    assert bytes(objs[1][: n[1]]) == BLOB


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_last_occurrence_wins_and_the_length_follows_it(dec_cls, enc_cls):
    """MESSAGE_SPEC §7.4 applies here like anywhere: the second occurrence
    replaces the first, and ``count_at`` must not be left describing the one that
    lost — otherwise a caller reads a live prefix of a dead payload."""
    enc = enc_cls()
    enc.write_string(1, "a longer first value")
    enc.write_string(1, "short")
    enc.flush()
    table, objs, words = _table()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    n = memoryview(words).cast("Q")
    assert bytes(objs[0][: n[0]]) == b"short"


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_visitor_takes_the_fields_the_table_does_not_name(dec_cls, enc_cls):
    """The table and the visitor compose exactly as they did before."""

    class Rest(Visitor):
        def __init__(self):
            self.seen = []

        def on_unsigned(self, field_id, value):
            self.seen.append((field_id, value))

    enc = enc_cls()
    enc.write_string(1, "bound")
    enc.write_unsigned(9, 42)
    enc.flush()
    table, objs, words = _table()
    rest = Rest()
    dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs, visitor=rest)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert rest.seen == [(9, 42)]
    assert bytes(objs[0][: memoryview(words).cast("Q")[0]]) == b"bound"
