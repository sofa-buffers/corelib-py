"""Strict-UTF-8 unit tests (issue #85).

These are plain, self-contained pytest cases that assert the strict-UTF-8
contract directly, without leaning on the shared ``assets/test_vectors.json``
negative group (that group is exercised separately in
``test_conformance_vectors.py``). Keeping a native copy of the behavior here
means the contract is still pinned even if the copied vector file is ever
regenerated or trimmed.

Contract (MESSAGE_SPEC §8, CORELIB_PLAN §6.4): a ``string`` value is UTF-8.
Python ``str`` is a Unicode string type, so it is **always strict** — the
``SOFAB_STRICT_UTF8`` option is a no-op for this port and is omitted entirely
(documented as always-ON). Strictness is symmetric:

* **decode** — an invalid-UTF-8 ``string`` payload **that is read** is the
  INVALID outcome (``SofaDecodeError``). Skipped fields are never validated.
* **encode** — a ``str`` that cannot be encoded as valid UTF-8 (a lone/unpaired
  surrogate) is refused with InvalidArgument (``SofaRangeError``); never
  silently replaced with U+FFFD or dropped.

Embedded U+0000 is valid UTF-8 and round-trips unchanged in both directions.
"""

from __future__ import annotations

import pytest
from vectors import ChunkReader, reader

from sofab import (
    Decoder,
    Encoder,
    FixlenSubtype,
    SofaDecodeError,
    SofaRangeError,
    WireType,
)

# Raw invalid-UTF-8 payloads. These mirror the classes in the shared
# `invalid_utf8` vectors but are spelled out here so the unit tests stand alone.
_INVALID_PAYLOADS = {
    "overlong_nul_c0_80": b"\xc0\x80",
    "overlong_c1_bf": b"\xc1\xbf",
    "overlong_3byte": b"\xe0\x80\x80",
    "overlong_4byte": b"\xf0\x80\x80\x80",
    "surrogate_d800": b"\xed\xa0\x80",
    "surrogate_dfff": b"\xed\xbf\xbf",
    "out_of_range_110000": b"\xf4\x90\x80\x80",
    "bare_continuation": b"\x80",
    "lone_ff": b"\xff",
    "truncated_2byte": b"\xc2",
    "truncated_3byte": b"\xe2\x82",
}


def _string_message(payload: bytes, field_id: int = 0) -> bytes:
    """Hand-frame a single STRING field carrying ``payload`` verbatim.

    Built by hand (not via the encoder, which would reject invalid bytes) so we
    can drive the *decode* side with a wire message the encoder would never
    produce. Header = ``(id << 3) | FIXLEN``; length word = ``(len << 3) |
    STRING``. Both fit one varint byte for these small payloads.
    """
    assert field_id < 16 and len(payload) < 16
    header = (field_id << 3) | int(WireType.FIXLEN)
    length_word = (len(payload) << 3) | int(FixlenSubtype.STRING)
    return bytes([header, length_word]) + payload


def _read_first_string(data: bytes, *, chunk: int | None = None) -> str:
    src = ChunkReader(data, chunk) if chunk is not None else reader(data)
    dec = Decoder(src)
    fld = dec.next()
    assert fld is not None and fld.type == WireType.FIXLEN
    assert fld.subtype == FixlenSubtype.STRING
    return dec.string()


# --- decode: invalid UTF-8 is the INVALID outcome ----------------------------


@pytest.mark.parametrize("payload", _INVALID_PAYLOADS.values(), ids=list(_INVALID_PAYLOADS))
def test_decode_invalid_utf8_is_invalid(payload):
    with pytest.raises(SofaDecodeError):
        _read_first_string(_string_message(payload))


@pytest.mark.parametrize("payload", _INVALID_PAYLOADS.values(), ids=list(_INVALID_PAYLOADS))
def test_decode_invalid_utf8_is_invalid_chunked(payload):
    # INVALID must win over INCOMPLETE even when fed one byte at a time.
    with pytest.raises(SofaDecodeError):
        _read_first_string(_string_message(payload), chunk=1)


@pytest.mark.parametrize("payload", _INVALID_PAYLOADS.values(), ids=list(_INVALID_PAYLOADS))
def test_skipped_invalid_string_is_not_validated(payload):
    # A skipped field is never materialized, so its bytes are never validated.
    data = _string_message(payload)
    dec = Decoder(reader(data))
    fld = dec.next()
    assert fld is not None
    dec.skip()  # must not raise
    assert dec.next() is None


# --- encode: unencodable str is InvalidArgument ------------------------------


@pytest.mark.parametrize(
    "text",
    ["\ud800", "\udfff", "a\ud834b", "\udc00\udc80"],
    ids=["lone_high", "lone_low", "embedded_surrogate", "surrogate_pair_bytes"],
)
def test_encode_lone_surrogate_is_invalid_argument(text):
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_string(0, text)


def test_encode_does_not_replace_or_drop():
    # A strict encoder must never fall back to a lossy encode (errors="replace"
    # would turn the surrogate into U+FFFD; "surrogatepass" would emit raw
    # ED A0 80). Prove neither happened: no bytes were produced at all.
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_string(3, "ok\ud800")
    # Nothing was committed for the rejected field.
    assert enc.getvalue() == b""


def test_encode_surrogate_reconstructed_from_wire_bytes():
    # The surrogateescape trick the conformance runner uses maps invalid wire
    # bytes back to a str; encoding that str must still be refused (it is exactly
    # the value a byte-container port would hand to encode).
    for payload in _INVALID_PAYLOADS.values():
        text = payload.decode("utf-8", "surrogateescape")
        enc = Encoder()
        with pytest.raises(SofaRangeError):
            enc.write_string(0, text)


# --- embedded U+0000 round-trips (valid) -------------------------------------


@pytest.mark.parametrize(
    "text",
    ["\x00", "a\x00b", "\x00\x00", "pre\x00post\U0001f600", ""],
    ids=["nul", "embedded_nul", "double_nul", "nul_with_astral", "empty"],
)
def test_embedded_nul_roundtrips(text):
    enc = Encoder()
    enc.write_string(5, text)
    wire = enc.getvalue()
    assert _read_first_string(wire) == text
    # And byte-at-a-time decode recovers the same value.
    assert _read_first_string(wire, chunk=1) == text


def test_valid_utf8_still_decodes():
    # Sanity: the strict path does not reject legitimate multi-byte UTF-8.
    text = "naïve — 日本語 — \U0001f600"
    enc = Encoder()
    enc.write_string(1, text)
    assert _read_first_string(enc.getvalue()) == text
