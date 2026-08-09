"""Strict UTF-8 validation is scoped to the string payload — and to nothing else.

CORELIB_PLAN §6.4 makes an invalid-UTF-8 ``string`` that is *read* the INVALID
outcome. Every port reaches that verdict by validating a byte range, and the
shared ``invalid_utf8`` conformance vectors cannot tell whether that range is the
right one: each of them is a whole message of four or five bytes whose string
field sits at the very front, so the payload's offset into the decode buffer is
0 or 1 and its length is the rest. A validator handed a *length* where an
exclusive *end index* was required — or one that starts at the buffer base
instead of the field — passes all eleven of them, and passes them for messages
where the two happen to coincide. That is not a hypothetical: a backend shipped
exactly that bug with a fully green conformance run (generator #166).

So the cases the vectors cannot state are stated here, in both directions:

* an **invalid** payload sitting far enough into the buffer that everything
  before it is valid UTF-8 — a from-the-base validator sees only the prefix and
  wrongly accepts;
* a **valid** payload sitting behind bytes that are *not* UTF-8 at all (a blob
  is opaque, §6.4) — a from-the-base validator wrongly rejects;
* a payload whose invalidity is only visible if validation **stops at the
  payload end**: a 3-byte sequence truncated by the field's own length word,
  where the byte that follows on the wire would complete it. An end-index bug
  reads one byte too far and decodes a perfectly good ``€``.

Each case runs both fully buffered and chunk-fed, including through a reader
that stalls at every byte boundary, because the offsets a chunked decode
presents to the validator are the ones the vectors never produce either: the
verdict must be the same at every split, and a multi-byte sequence merely *split*
across chunks stays INCOMPLETE until its declared payload has arrived (the
vectors README's chunk-boundary-independence rule).
"""

from __future__ import annotations

import io

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import ChunkReader, uvarint

from sofab import (
    Encoder,
    FixlenSubtype,
    SofaDecodeError,
    SofaIncompleteError,
    SofaRangeError,
    WireType,
)

#: Payload bytes that are not valid UTF-8, one per malformed class.
INVALID = {
    "overlong_c0_80": b"\xc0\x80",
    "surrogate_d800": b"\xed\xa0\x80",
    "out_of_range": b"\xf4\x90\x80\x80",
    "bare_continuation": b"\x80",
    "lone_ff": b"\xff",
    "truncated_3byte": b"\xe2\x82",
}

#: A string whose bytes are multi-byte UTF-8 in every plane the format can carry.
VALID_TEXT = "ä€\U0001f600z"


def _fixlen(field_id: int, subtype: FixlenSubtype, payload: bytes) -> bytes:
    """One fixlen field: header ``(id << 3) | FIXLEN`` then ``(len << 3) | subtype``."""
    return (
        uvarint((field_id << 3) | int(WireType.FIXLEN))
        + uvarint((len(payload) << 3) | int(subtype))
        + payload
    )


def _ascii_blob(n: int) -> bytes:
    """A leading field of ``n`` payload bytes that are valid UTF-8 on their own."""
    return _fixlen(1, FixlenSubtype.BLOB, b"a" * n)


class StallingReader:
    """A ``read(n)`` source that hands over ``chunk`` bytes, then stalls once.

    Returning ``b""`` before the message ends is what puts the decoder in §5.2's
    INCOMPLETE state — the shape a non-blocking socket has and the one a
    ``BytesIO`` can never produce. Alternating stall/deliver drives a suspension
    at *every* byte boundary of the message, so a resumed read starts from every
    possible offset into the payload.
    """

    def __init__(self, data: bytes, chunk: int = 1) -> None:
        self._data = bytes(data)
        self._pos = 0
        self._chunk = chunk
        self._stall = True

    def read(self, n: int) -> bytes:
        self._stall = not self._stall
        if self._stall or self._pos >= len(self._data):
            return b""
        end = min(self._pos + min(n, self._chunk), len(self._data))
        out = self._data[self._pos : end]
        self._pos = end
        return out


def _drive_stalling(dec, *, reads: int):
    """Pull ``reads`` fields, retrying every INCOMPLETE, and return the values.

    Every retry re-issues the *same* call: §5.2 says a suspended call consumed
    nothing, so this loop is the whole contract for a stalling source. ``None``
    from :meth:`next` is the between-fields half of the same answer (the bytes
    stopped exactly on a field boundary), so it is retried too. The retry budget
    is finite so a decoder that never makes progress fails the test instead of
    hanging it.
    """
    out = []
    budget = 10_000
    while len(out) < reads:
        try:
            field = dec.next()
        except SofaIncompleteError:
            field = None
        if field is None:
            budget -= 1
            assert budget > 0, "decoder made no progress"
            continue
        while True:
            try:
                if field.subtype == FixlenSubtype.STRING:
                    out.append(dec.string())
                else:
                    out.append(dec.bytes())
                break
            except SofaIncompleteError:
                budget -= 1
                assert budget > 0, "decoder made no progress"
    return out


# --- an invalid payload far into the buffer ----------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
@pytest.mark.parametrize("pad", [0, 7, 64, 300], ids=lambda n: f"pad{n}")
def test_invalid_payload_behind_valid_bytes_is_invalid(engine, payload, pad):
    """The verdict must not depend on how far into the buffer the payload sits.

    With ``pad >= len(payload)`` every byte a from-the-base validator would look
    at is valid UTF-8 (an ASCII blob and its two header bytes), so accepting here
    is exactly the range bug the shared vectors cannot catch.
    """
    data = _ascii_blob(pad) + _fixlen(0, FixlenSubtype.STRING, payload)
    dec = engine(io.BytesIO(data))
    assert dec.next() is not None
    assert dec.bytes() == b"a" * pad  # the padding itself is intact
    assert dec.next() is not None
    with pytest.raises(SofaDecodeError):
        dec.string()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
@pytest.mark.parametrize("chunk", [1, 3, 64], ids=lambda n: f"chunk{n}")
def test_invalid_payload_far_in_is_invalid_when_chunk_fed(engine, payload, chunk):
    """Same message through a chunk-fed reader: refilling moves the payload to a
    different buffer offset on every chunk size, and none of them may change the
    verdict."""
    data = _ascii_blob(300) + _fixlen(0, FixlenSubtype.STRING, payload)
    dec = engine(ChunkReader(data, chunk))
    dec.next(); dec.skip()
    assert dec.next() is not None
    with pytest.raises(SofaDecodeError):
        dec.string()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
def test_invalid_payload_far_in_is_invalid_through_a_stalling_reader(engine, payload):
    """A source that runs dry at every byte boundary suspends the read inside the
    payload and resumes it from its first byte over and over; INVALID must still
    be the verdict, never a truncation-shaped success."""
    data = _ascii_blob(40) + _fixlen(0, FixlenSubtype.STRING, payload)
    dec = engine(StallingReader(data))
    assert _drive_stalling(dec, reads=1) == [b"a" * 40]
    with pytest.raises(SofaDecodeError):
        _drive_stalling(dec, reads=1)


# --- a valid payload behind bytes that are not UTF-8 at all ------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_valid_string_behind_a_non_utf8_blob_still_decodes(engine):
    """A ``blob`` is opaque bytes (§6.4), so nothing before the string field is
    UTF-8 at all. Validating from the buffer base instead of the payload start
    would reject a perfectly valid string — the same range bug seen from the
    other side."""
    junk = bytes(range(0x80, 0x100))  # every byte here is an invalid UTF-8 lead
    data = _fixlen(1, FixlenSubtype.BLOB, junk) + _fixlen(0, FixlenSubtype.STRING,
                                                          VALID_TEXT.encode())
    dec = engine(io.BytesIO(data))
    dec.next()
    assert dec.bytes() == junk
    dec.next()
    assert dec.string() == VALID_TEXT
    assert dec.next() is None


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [1, 5], ids=lambda n: f"chunk{n}")
def test_valid_multibyte_string_split_across_chunks_stays_incomplete(engine, chunk):
    """A multi-byte sequence *split* at a chunk end is INCOMPLETE, not INVALID.

    Every split of ``VALID_TEXT`` cuts at least one sequence in half, and a
    stalling source turns each of those cuts into a suspension. The value must
    come back whole once the declared payload has arrived — a validator that
    judged what it had so far would call a half-delivered ``\U0001f600``
    malformed."""
    data = _fixlen(0, FixlenSubtype.STRING, VALID_TEXT.encode())
    dec = engine(StallingReader(data, chunk))
    assert _drive_stalling(dec, reads=1) == [VALID_TEXT]


# --- validation stops at the payload end -------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_validation_stops_at_the_declared_payload_end(engine):
    """``E2 82`` is a 3-byte sequence the field's own length word cuts short.

    The byte that follows it on the wire is ``AC``, which would complete it as
    ``U+20AC``. It is not part of the string: it opens the next field (a signed
    array, id 21). A validator using an exclusive end one byte too far — or the
    rest of the buffer — decodes ``€`` and accepts a malformed string.
    """
    tail = bytes([0xAC, 0x01, 0x00])  # id 21, ARRAY_SIGNED, count 0
    assert tail[0] == 0xAC  # the byte that would complete E2 82
    data = _fixlen(0, FixlenSubtype.STRING, b"\xe2\x82") + tail

    dec = engine(io.BytesIO(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.string()

    # And the framing itself is sound: skipping the same field walks straight
    # onto the array, so the case above really is about validation range only.
    dec = engine(io.BytesIO(data))
    dec.next(); dec.skip()
    field = dec.next()
    assert field is not None and field.id == 21
    assert dec.read_signed_array() == []
    assert dec.next() is None


@pytest.mark.parametrize("engine", ENGINES)
def test_validation_stops_at_the_payload_end_with_trailing_bytes(engine):
    """The same, with the completing byte outside the message entirely: trailing
    bytes are in the decode buffer but belong to no field this decoder read."""
    data = _fixlen(0, FixlenSubtype.STRING, b"\xe2\x82") + b"\xac"
    dec = engine(io.BytesIO(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.string()


@pytest.mark.parametrize("engine", ENGINES)
def test_payload_ending_exactly_at_the_buffer_end_is_still_validated(engine):
    """The opposite off-by-one: an invalid payload that ends flush with the last
    byte fed. A validator that stopped one byte short would miss the lone ``FF``
    and accept."""
    data = _ascii_blob(64) + _fixlen(0, FixlenSubtype.STRING, b"ok\xff")
    dec = engine(io.BytesIO(data))
    dec.next(); dec.skip()
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.string()


# --- the encode side is offset-independent too --------------------------------


def test_encoder_rejects_a_late_surrogate_in_a_long_string():
    """§6.4's encode half has the same shape: the unencodable code point sits far
    past the start of a long, valid prefix. Refusal must not depend on where in
    the value it is, and nothing may reach the wire for the refused field."""
    enc = Encoder()
    enc.write_string(0, "keep me")
    with pytest.raises(SofaRangeError):
        enc.write_string(1, "a" * 4096 + "\ud800" + "b" * 4096)
    # Only the first field is on the wire; the refused one left nothing behind.
    assert enc.getvalue() == _fixlen(0, FixlenSubtype.STRING, b"keep me")
