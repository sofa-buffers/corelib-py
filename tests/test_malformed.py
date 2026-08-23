"""Malformed-input tests. Byte vectors transcribed from
corelib-c-cpp/test/c/test_istream.c (SOFAB_RET_E_INVALID_MSG cases)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vectors import Recorder, Status, pairs, values, verdict
from vectors import bound as boundfeed

from sofab import (
    Binding,
    Decoder,
    Encoder,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaLimitError,
    SofaRangeError,
    WireType,
)


def _decode_fully(data):
    """Walk the whole message and surface its verdict."""
    verdict(Decoder, bytes(data))


def test_varint_unsigned_overflow():
    data = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_varint_signed_overflow():
    data = [0x01, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_fixlen_length_varint_overflow():
    data = [0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01,
            0x56, 0x0E, 0x49, 0x40]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_fixlen_length_limit_overflow():
    # length header (length << 3 | subtype) whose length exceeds FIXLEN_MAX
    data = [0x02, 0xF8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x03, 0x56, 0x0E, 0x49, 0x40]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def _fixlen_header_over_the_ceiling(subtype):
    """id 1, FIXLEN, a length one past FIXLEN_MAX carrying ``subtype``."""
    out = [0x0A]
    word = ((0x7FFF_FFFF + 1) << 3) | subtype
    while True:
        b = word & 0x7F
        word >>= 7
        out.append(b | (0x80 if word else 0))
        if not word:
            return out


@pytest.mark.parametrize("subtype", [0x2, 0x3], ids=["string", "blob"])
def test_fixlen_length_over_the_ceiling_on_a_variable_length_subtype(subtype):
    """The ceiling is what rejects an oversize string or blob.

    The case above uses an fp subtype, where the exact-width check gets there
    first and FIXLEN_MAX is never consulted -- fp32 and fp64 carry 4 and 8 bytes
    and nothing else. STRING and BLOB have no such width, so §6.2's format-wide
    ceiling is the only thing bounding them, and this is where it fires.
    """
    with pytest.raises(SofaDecodeError):
        _decode_fully(_fixlen_header_over_the_ceiling(subtype))


def test_array_count_varint_overflow():
    data = [0x04, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x53]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_array_count_limit_overflow():
    data = [0x04, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x53]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_array_count_zero_is_valid():
    # §4.7/§4.8: a zero-count array is a valid, fully-specified empty array. The
    # previous behaviour (reject count==0) was a defect under the updated spec.
    # Unsigned array (0x03), signed array (0x04): [header][0x00] then next field —
    # integer arrays never carry a fixlen_word.
    for header in (0x03, 0x04):
        got = pairs(Decoder, bytes([header, 0x00]))
        # No fixlen_word / payload may be consumed; that is the whole message.
        assert len(got) == 1 and got[0][0].count == 0
    # Fixlen array (0x05): [header][0x00][fixlen_word] — the fixlen_word is always
    # present (§4.8), here 0x20 = (4<<3)|fp32, but there is no payload.
    got = pairs(Decoder, bytes([0x05, 0x00, 0x20]))
    assert len(got) == 1
    f, _ev = got[0]
    assert f.count == 0 and f.subtype == FixlenSubtype.FP32


def test_array_fixlen_count_zero_reads_the_fixlen_word():
    # §4.8: an empty fixlen array still carries its fixlen_word, so the bytes
    # after [0x05, 0x00, <fixlen_word>] must be parsed as the NEXT field.
    # 0x20 = (4<<3)|fp32 fixlen_word; 0x50 = (10 << 3) | UNSIGNED, 0x07 = value 7.
    got = pairs(Decoder, bytes([0x05, 0x00, 0x20, 0x50, 0x07]))
    assert len(got) == 2
    (f, ev), (nxt, nxt_ev) = got
    assert f.count == 0 and f.subtype == FixlenSubtype.FP32
    assert ev[2] == ()  # empty fixlen array reads as empty
    assert nxt.id == 10 and nxt_ev == ("u", 10, 7)


def test_string_invalid_utf8_raises_decode_error():
    # fixlen STRING (subtype 0x2) of length 2 with invalid UTF-8 bytes.
    # length_header = (2 << 3) | 0x2 = 0x12; payload 0xFF 0xFE is not valid UTF-8.
    data = [0x02, 0x12, 0xFF, 0xFE]
    with pytest.raises(SofaDecodeError):
        verdict(Decoder, bytes(data))


def test_decode_nesting_beyond_max_depth_rejected():
    # 256 consecutive sequence-start bytes (0x06) must be rejected once depth
    # would exceed MAX_DEPTH (255), with SofaDecodeError.
    data = [0x06] * 256
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_array_fixlen_invalid_subtype():
    # 0x27 => element_size 4, subtype 7 (reserved) in a fixlen array
    data = [0x05, 0x05, 0x27, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40, 0x00,
            0x00, 0x40, 0x40, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0x7F]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_array_fixlen_element_width_mismatch_underflow():
    # Regression (corelib-py#28 / #41): fp32 fixlen array whose fixlen_word
    # declares a 0-byte element width. §4.8/§5.2: fp32 elements are exactly 4
    # bytes, so a 0-width fixlen_word is malformed at header time, before any
    # payload is read (the native engine used to trust the count and read off
    # the end of the buffer — SIGSEGV). Both engines must reject it at next().
    # 0x05 = (0<<3)|ARRAY_FIXLEN, 0x01 = count 1, 0x00 = fixlen_word (0<<3)|fp32.
    with pytest.raises(SofaDecodeError):
        verdict(Decoder, bytes([0x05, 0x01, 0x00]))


def test_array_fixlen_element_width_mismatch_overflow():
    # fp32 array claiming an 8-byte element width (fp64's width): even with the
    # payload present, count*8 != count*4, so the fixlen_word is malformed and
    # rejected eagerly at header time.
    # 0x40 = (8<<3)|fp32; eight payload bytes follow the count-1 element.
    data = [0x05, 0x01, 0x40, 0, 0, 0x80, 0x3F, 0, 0, 0, 0]
    with pytest.raises(SofaDecodeError):
        verdict(Decoder, bytes(data))


def test_array_fixlen_fp64_width_mismatch():
    # Same defect on the fp64 path: subtype fp64 (1) with a 4-byte element width.
    # 0x05 = ARRAY_FIXLEN, 0x01 = count 1, 0x21 = (4<<3)|fp64; four payload bytes.
    data = [0x05, 0x01, 0x21, 0, 0, 0, 0]
    with pytest.raises(SofaDecodeError):
        verdict(Decoder, bytes(data))


def _uvarint(n: int) -> list[int]:
    out = []
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return out


def test_array_fixlen_giant_element_width_rejected_at_header():
    # A fixlen array whose fixlen_word declares a gigantic element width is a
    # wrong-width fp32 (§4.8: fp32 elements are exactly 4 bytes), so §5.2 makes
    # it INVALID at header time — the eager width check rejects it before the
    # count * element_width payload-size arithmetic is ever reached, so it can
    # never wrap to a small/negative size and drive the cursor off the buffer.
    # count = ARRAY_MAX, element width ~2^61 (fixlen_word low 3 bits 0 => fp32).
    count = 0x7FFFFFFF
    elem_word = 0xFFFFFFFFFFFFFFF8  # (elem_size << 3) | fp32, elem_size ~2^61
    data = [0x05] + _uvarint(count) + _uvarint(elem_word)
    with pytest.raises(SofaDecodeError) as exc:
        _decode_fully(data)
    assert not isinstance(exc.value, SofaIncompleteError)


# --- scalar fixlen fp width: INVALID takes precedence over INCOMPLETE (§7) ---
#
# A fixlen fp32/fp64 whose declared length is not the type's fixed width (4/8)
# is malformed regardless of what bytes follow, so INVALID must win over the
# INCOMPLETE a truncated payload would otherwise raise (corelib-py#38). The
# width is validated eagerly at header-decode time, mirroring the fixlen-array
# path (test_array_fixlen_*_width_mismatch above), so the verdict is reached
# before any payload read — hence these expect the raise from ``next()`` itself.
#
# Exercise every available engine so the pure and native decoders stay in
# lockstep (they returned INCOMPLETE together before the fix). ``Decoder``
# imported from ``sofab`` resolves to the native class when it is compiled in,
# so name the pure engine explicitly rather than relying on the public alias.
from sofab import decoder as _pydec_module  # noqa: E402
from sofab.decoder import Decoder as _PyDecoder  # noqa: E402

_DECODERS = [_PyDecoder]
try:  # the native accelerator, when compiled in, must behave identically
    from sofab import _speedups as _sp

    _DECODERS.append(_sp.Decoder)
except ImportError:  # pragma: no cover - pure-Python-only install
    pass


def _decode_one(decoder_cls, data):
    """Decode the single fixlen field, surfacing its verdict."""
    verdict(decoder_cls, bytes(data))


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_fp64_wrong_width_truncated_is_invalid_not_incomplete(decoder_cls):
    # 0x02 = (0<<3)|FIXLEN, 0x59 = (11<<3)|fp64 → length 11 ≠ 8; zero payload
    # bytes present. Wrong-width *and* truncated: INVALID must take precedence.
    with pytest.raises(SofaDecodeError) as exc:
        _decode_one(decoder_cls, [0x02, 0x59])
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_fp32_wrong_width_truncated_is_invalid_not_incomplete(decoder_cls):
    # 0x38 = (7<<3)|fp32 → length 7 ≠ 4; zero payload bytes present.
    with pytest.raises(SofaDecodeError) as exc:
        _decode_one(decoder_cls, [0x02, 0x38])
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_fp64_wrong_width_full_payload_stays_invalid(decoder_cls):
    # Control: wrong width but all 11 declared bytes present → still INVALID.
    with pytest.raises(SofaDecodeError) as exc:
        _decode_one(decoder_cls, [0x02, 0x59] + [0] * 11)
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_fp64_correct_width_truncated_stays_incomplete(decoder_cls):
    # Control: correct width (0x41 = (8<<3)|fp64 → length 8) but only 3 of the 8
    # payload bytes present → genuinely INCOMPLETE, must NOT be reclassified.
    with pytest.raises(SofaIncompleteError):
        _decode_one(decoder_cls, [0x02, 0x41, 0, 0, 0])


# --- fixlen-array fp width: INVALID takes precedence over INCOMPLETE (§7) -----
#
# The array analogue of the scalar checks above (#41 / Crucible F-0014). A
# fixlen-array fixlen_word whose element width is not the subtype's fixed width
# (fp32→4, fp64→8) is malformed regardless of what payload follows, so the
# element width is validated eagerly at header time — the raise comes from
# next() itself, before any payload read.


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_array_fixlen_fp32_zero_width_truncated_is_invalid_not_incomplete(decoder_cls):
    # F-0014 reproducer: 0x75 = field id 14, wtype ARRAY_FIXLEN; 0x60 = count 96;
    # 0x00 = fixlen_word (size 0, fp32) — fp32 must be 4; 0x0d 0x0d = truncated
    # payload. Wrong width *and* truncated: INVALID must win over INCOMPLETE.
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes([0x75, 0x60, 0x00, 0x0D, 0x0D]))
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_array_fixlen_correct_width_truncated_stays_incomplete(decoder_cls):
    # Control: correct fp32 width (0x20 = (4<<3)|fp32) with count 1 but zero
    # payload bytes → genuinely INCOMPLETE, must NOT be reclassified.
    with pytest.raises(SofaIncompleteError):
        verdict(decoder_cls, bytes([0x05, 0x01, 0x20]))


def test_nested_sequence_extra_end():
    data = [0x00, 0x2A, 0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E, 0x00, 0x2A, 0x11, 0x53,
            0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E, 0x00,
            0x2A, 0x11, 0x53, 0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E, 0x00, 0x2A, 0x11,
            0x53, 0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E, 0x00, 0x2A, 0x11, 0x53, 0x0E,
            0x00, 0x2A, 0x11, 0x53, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
            0x07, 0x07, 0x07, 0x11, 0x53]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_truncated_payload():
    # fixlen string claims 12 bytes but only 2 follow — the bytes end inside the
    # field, so this is INCOMPLETE (§7), not malformed.
    data = [0x02, 0x62, 0x48, 0x65]
    with pytest.raises(SofaIncompleteError):
        verdict(Decoder, bytes(data))


# --- three-valued outcome: INCOMPLETE vs INVALID (MESSAGE_SPEC §7) -----------


def test_lone_continuation_byte_is_incomplete_not_malformed():
    # A single dangling 0x80 (continuation bit set, no terminating byte) is the
    # canonical INCOMPLETE case: the bytes end inside a varint, more could follow.
    # It must be neither COMPLETE (next() returning a field / None) nor INVALID.
    with pytest.raises(SofaIncompleteError) as exc:
        verdict(Decoder, bytes([0x80]))
    # INCOMPLETE is a *sibling* of the malformed error, never a subclass, so a
    # caller doing `except SofaDecodeError` does not mistake "need more bytes"
    # for "these bytes are garbage".
    assert not isinstance(exc.value, SofaDecodeError)


def test_varint_over_64_bits_stays_malformed():
    # A varint whose continuation runs past 64 bits is INVALID regardless of what
    # follows — it stays SofaDecodeError, and is NOT reclassified as incomplete.
    # 10 x 0xFF then 0x7F: the shift reaches 64 with the continuation bit set.
    data = [0x00] + [0xFF] * 10 + [0x7F]
    with pytest.raises(SofaDecodeError) as exc:
        _decode_fully(data)
    assert not isinstance(exc.value, SofaIncompleteError)


# --- the 64-bit bound applies to ARRAY ELEMENTS too (§4.1, issue #64) --------
#
# §4.1 puts the bound on the *encoding*, wherever a varint appears — headers,
# fixlen_words, counts, skipped fields and *element values* alike: longer than
# 10 bytes, or a tenth byte whose payload would land at bit >= 64 (any value
# above 0x01), is INVALID, and a decoder MUST NOT silently drop the overflowing
# bits. The array read loops inline the varint codec for speed, so each one
# needs the guard of its own; both engines must reach the same verdict on the
# same bytes (the pure one used to accept these and mask them down to 2^64-1).

# A ten-byte element whose tenth byte is 0x7F carries payload bits 63..69 — those
# high bits are unrepresentable in u64, so masking them away is corruption.
_OVERLONG_ELEM = [0xFF] * 9 + [0x7F]
# Eleven bytes: INVALID by length alone, even though the surplus byte is zero.
_ELEVEN_BYTE_ELEM = [0xFF] * 10 + [0x00]
# The largest *legal* element: 2^64-1, ten bytes with a tenth byte of 0x01.
_MAX_U64_ELEM = [0xFF] * 9 + [0x01]
_OVER_64_ELEMS = [_OVERLONG_ELEM, _ELEVEN_BYTE_ELEM]


@pytest.mark.parametrize("decoder_cls", _DECODERS)
@pytest.mark.parametrize("elem", _OVER_64_ELEMS, ids=["ten-byte", "eleven-byte"])
def test_unsigned_array_element_over_64_bits_is_invalid(decoder_cls, elem):
    # (1<<3)|ARRAY_UNSIGNED, count 1, then one out-of-range element.
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes([(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + elem))
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
@pytest.mark.parametrize("elem", _OVER_64_ELEMS, ids=["ten-byte", "eleven-byte"])
def test_signed_array_element_over_64_bits_is_invalid(decoder_cls, elem):
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes([(1 << 3) | WireType.ARRAY_SIGNED, 0x01] + elem))
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_array_element_over_64_bits_is_invalid_when_chunk_fed(decoder_cls):
    # The same element arriving one byte at a time: the guard has to live in the
    # decode loop itself, not only in a "whole element already buffered" path.
    data = bytes([(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _OVERLONG_ELEM)
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes(data), chunk=1)
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_array_element_over_64_bits_after_valid_elements_is_invalid(decoder_cls):
    # The bad element sits behind two good ones, so the guard must hold on every
    # iteration of the loop and not just the first.
    data = [(1 << 3) | WireType.ARRAY_UNSIGNED, 0x03, 0x01, 0x02] + _OVERLONG_ELEM
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes(data))
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_max_u64_array_element_is_accepted(decoder_cls):
    # Control: the boundary value itself is legal and must survive intact — the
    # guard rejects overflowing payload *bits*, not a full-width element.
    assert values(decoder_cls, bytes([(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _MAX_U64_ELEM))[0][2] == tuple([0xFFFFFFFFFFFFFFFF])


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_min_i64_array_element_is_accepted(decoder_cls):
    # Control for the signed path: the zig-zag encoding of INT64_MIN is 2^64-1.
    assert values(decoder_cls, bytes([(1 << 3) | WireType.ARRAY_SIGNED, 0x01] + _MAX_U64_ELEM))[0][2] == tuple([-(2**63)])


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_skipped_array_with_over_64_bit_element_is_invalid(decoder_cls):
    # Skipping does not lower the bar (§4.1: "and inside skipped fields").
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes([(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _OVERLONG_ELEM))
    assert not isinstance(exc.value, SofaIncompleteError)


# --- decode resource limits (issue #31) -------------------------------------
#
# Part A: unconditional hardening — the untrusted wire count must never drive an
# eager allocation. Part B: opt-in receiver-side limits raising SofaLimitError.


def test_unsigned_array_huge_count_does_not_preallocate():
    # issue #31 Part A: a tiny message claiming a 2^31-element unsigned array but
    # carrying a single payload byte must fail as truncated (INCOMPLETE) promptly
    # — WITHOUT pre-allocating a ~16 GB list from the untrusted count. If the
    # decoder still did `[0] * count` this would OOM/hang instead of raising.
    # 0x03 = (0<<3)|ARRAY_UNSIGNED, then count = 0x7FFFFFFF, then one lone byte.
    data = [0x03] + _uvarint(0x7FFFFFFF) + [0x01]
    with pytest.raises(SofaIncompleteError):
        verdict(Decoder, bytes(data))


def test_signed_array_huge_count_does_not_preallocate():
    # Same hardening on the signed-array path (0x04 = (0<<3)|ARRAY_SIGNED).
    data = [0x04] + _uvarint(0x7FFFFFFF) + [0x01]
    with pytest.raises(SofaIncompleteError):
        verdict(Decoder, bytes(data))


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_array_huge_count_does_not_preallocate(decoder_cls):
    # The same hardening on the fixlen-array path, where the payload size is a
    # *product* (count * element width) rather than a count of varints: 2^31-1
    # fp64 elements claim a 16 GB payload. That size must be reached without
    # materialising anything, and a message carrying none of it must fail
    # promptly as truncated (INCOMPLETE).
    # 0x05 = (0<<3)|ARRAY_FIXLEN, count = 0x7FFFFFFF, fixlen_word = (8<<3)|FP64.
    data = [0x05] + _uvarint(0x7FFFFFFF) + _uvarint((8 << 3) | int(FixlenSubtype.FP64))
    with pytest.raises(SofaIncompleteError):
        verdict(decoder_cls, bytes(data))

    with pytest.raises(SofaIncompleteError):  # and the skip walk is bounded too
        verdict(decoder_cls, bytes(data))


def test_fixlen_array_payload_beyond_the_address_space_is_truncated(monkeypatch):
    # The 32-bit case of the test above, forced on a 64-bit host: there,
    # count * element width can exceed what a Py_ssize_t — and therefore any
    # real buffer — can index. Such a payload can never be satisfied, so it is
    # §5.2's truncated read, not a raw OverflowError escaping from the read
    # underneath (which would leave §6.3's outcome set and diverge from the
    # native engine). Pure engine only: the native decoder does this arithmetic
    # in C against the same ceiling, and its branch is invisible to coverage.py.
    monkeypatch.setattr(_pydec_module, "sys", SimpleNamespace(maxsize=2**31 - 1))
    data = [0x05] + _uvarint(0x7FFFFFFF) + _uvarint((8 << 3) | int(FixlenSubtype.FP64))

    # Both ways in: the driver's readiness check answers first for a field it
    # would read, and the auto-skip reaches the same ceiling for one it walks
    # past.
    with pytest.raises(SofaIncompleteError):
        verdict(_PyDecoder, bytes(data))

    with pytest.raises(SofaIncompleteError):
        verdict(_PyDecoder, bytes(data), recorder=Recorder(decline=lambda f: True))


def test_max_array_count_rejects_oversize_before_alloc():
    # Part B acceptance: with max_array_count=65536 an otherwise-valid message
    # carrying a 65537-element dynamic array is rejected with SofaLimitError on
    # the strength of its count header alone — no element is decoded and no list
    # is allocated. The verdict is reached inside next(); it is RAISED by the
    # call that would consume the field, which is the window §6.2.1 needs for
    # Decoder.schema_bounded (see tests/test_schema_bounded.py). The identical
    # bytes decode unchanged with the limit unset.
    enc = Encoder()
    enc.write_unsigned_array(7, list(range(65537)))
    data = enc.getvalue()

    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(data), max_array_count=65536)

    f, ev = pairs(Decoder, data)[0]
    assert f.count == 65537
    assert ev[2] == tuple(range(65537))


def test_max_array_count_fires_before_any_payload():
    # The cap is enforced on the count varint alone: a header claiming 100
    # elements with NO payload following still raises SofaLimitError (not the
    # truncation that reading the absent elements would give), proving the check
    # runs before allocation/buffering.
    data = [0x03] + _uvarint(100)  # ARRAY_UNSIGNED, count 100, no elements
    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(data), max_array_count=10)


def test_max_array_count_boundary_is_inclusive():
    # count == max_array_count is allowed; count == max + 1 is rejected.
    ok = Encoder()
    ok.write_unsigned_array(0, list(range(8)))
    f, ev = pairs(Decoder, ok.getvalue(), max_array_count=8)[0]
    assert f.count == 8
    assert ev[2] == tuple(range(8))

    over = Encoder()
    over.write_unsigned_array(0, list(range(9)))
    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(over.getvalue()), max_array_count=8)


def test_max_array_count_applies_to_all_array_kinds():
    # The count cap governs every array wire type — signed and fixlen (float)
    # arrays as well as unsigned.
    for write in (
        lambda e: e.write_signed_array(1, list(range(6))),
        lambda e: e.write_float32_array(1, [1.0] * 6),
        lambda e: e.write_float64_array(1, [1.0] * 6),
    ):
        enc = Encoder()
        write(enc)
        # A declined field is a consume too — the payload the walk would buffer
        # is exactly what the cap protects, so it is rejected all the same.
        with pytest.raises(SofaLimitError):
            verdict(
                Decoder, enc.getvalue(), max_array_count=5,
                recorder=Recorder(decline=lambda f: True),
            )


def test_max_string_len_fires_before_payload():
    # A fixlen STRING header claiming length 100 with NO payload bytes is
    # rejected by max_string_len on its length word, before the payload is
    # read/buffered — the read never gets as far as reporting the truncation.
    # 0x02 = (0<<3)|FIXLEN; length_header = (100 << 3) | 0x2 (STRING).
    data = [0x02] + _uvarint((100 << 3) | 0x2)
    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(data), max_string_len=10)


def test_max_string_len_valid_message_roundtrips_without_limit():
    enc = Encoder()
    enc.write_string(3, "x" * 100)
    data = enc.getvalue()

    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(data), max_string_len=64)

    assert values(Decoder, data) == [("str", 3, "x" * 100)]

    within = Encoder()
    within.write_string(3, "y" * 64)  # exactly at the limit: allowed
    assert values(Decoder, within.getvalue(), max_string_len=64) == [
        ("str", 3, "y" * 64)
    ]


def test_max_blob_len_rejects_oversize():
    enc = Encoder()
    enc.write_bytes(1, b"\x00" * 100)
    data = enc.getvalue()

    with pytest.raises(SofaLimitError):
        verdict(Decoder, bytes(data), max_blob_len=16)

    assert values(Decoder, data)[0][2] == b"\x00" * 100


def test_limits_are_independent_per_kind():
    # Each limit governs only its own field kind: a blob is not bound by
    # max_string_len, nor a string by max_blob_len.
    blob = Encoder()
    blob.write_bytes(1, b"z" * 100)
    assert values(Decoder, blob.getvalue(), max_string_len=1)[0][2] == b"z" * 100

    text = Encoder()
    text.write_string(1, "z" * 100)
    assert values(Decoder, text.getvalue(), max_blob_len=1)[0][2] == "z" * 100


def test_limit_error_is_not_a_decode_or_incomplete_error():
    # Part B acceptance: a limit rejection is policy, not wire malformation, so a
    # handler that catches only the invalid-message class must not swallow it.
    enc = Encoder()
    enc.write_unsigned_array(0, list(range(4)))
    data = enc.getvalue()

    with pytest.raises(SofaLimitError) as exc:
        verdict(Decoder, bytes(data), max_array_count=2)
    assert isinstance(exc.value, SofaError)
    assert not isinstance(exc.value, SofaDecodeError)
    assert not isinstance(exc.value, SofaIncompleteError)

    # `except SofaDecodeError` genuinely does not intercept it.
    with pytest.raises(SofaLimitError):
        try:
            verdict(Decoder, data, max_array_count=2)
        except SofaDecodeError:  # pragma: no cover - must not be taken
            pytest.fail("SofaLimitError must not be caught as SofaDecodeError")


# --- encoder-side errors ----------------------------------------------------


def test_encode_id_out_of_range():
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_unsigned(0x80000000, 0)


def test_encode_unsigned_out_of_range():
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_unsigned(0, 1 << 64)


def test_encode_empty_array_is_valid():
    # §4.7/§4.8: zero-count arrays are valid. Integer arrays emit [header][0x00];
    # fixlen arrays emit [header][0x00][fixlen_word] (always present, no payload)
    # so an empty fp32 and fp64 array stay distinguishable on the wire.
    enc = Encoder()
    enc.write_unsigned_array(0, [])
    enc.write_signed_array(0, [])
    enc.write_float32_array(0, [])
    enc.write_float64_array(0, [])
    # u-array (0x03,0x00), s-array (0x04,0x00), fp32-array (0x05,0x00,0x20),
    # fp64-array (0x05,0x00,0x41): 0x20=(4<<3)|fp32, 0x41=(8<<3)|fp64.
    assert enc.getvalue() == bytes(
        [0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x20, 0x05, 0x00, 0x41]
    )


def test_encode_nesting_beyond_max_depth_rejected():
    from sofab import MAX_DEPTH

    enc = Encoder()
    for i in range(MAX_DEPTH):  # 255 nested sequences are allowed
        enc.write_sequence_begin_lazy(i % 100)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin_lazy(0)  # the 256th must be refused


def test_sequence_end_without_begin():
    enc = Encoder()
    with pytest.raises(SofaRangeError):  # §6.3 InvalidArgument — a caller mistake
        enc.write_sequence_end()


def test_buffer_full_without_sink():
    enc = Encoder.over_buffer(bytearray(2))  # too small, no flush sink
    with pytest.raises(SofaBufferError):
        enc.write_unsigned(0, 1 << 60)


def test_wrong_type_read_is_not_an_error():
    """§7.3: the read's type contradicts the wire, so the field is skipped like
    an unknown id — not reported as INVALID, and not an error at all."""
    enc = Encoder()
    enc.write_unsigned(0, 5)
    # A binding that declares field 0 signed contradicts the wire, so the field
    # is skipped like an unknown id and the decode stays COMPLETE.
    b = Binding().signed(0, at=0, count_at=1)
    status, _dec, slots = boundfeed(Decoder, enc.getvalue(), b)
    assert status is Status.COMPLETE
    assert slots.u[1] == 0  # never arrived, as far as the binding is concerned


# --- sticky mode ------------------------------------------------------------


def test_sticky_mode_records_first_error_and_noops():
    enc = Encoder(sticky=True)
    enc.write_unsigned(0, 1 << 64)  # range error, recorded
    enc.write_unsigned(1, 5)  # becomes a no-op
    assert enc.error is not None
    assert isinstance(enc.error, SofaRangeError)
    assert enc.getvalue() == b""


# --- overlong (>64-bit) varint: INVALID, not silently truncated (§4.1/§6.3) --
#
# Regression for issue #43 / Crucible F-0016: a varint whose payload bits spill
# past bit 63 (a 10th byte with any bit above bit 63, or an 11th continuation
# byte) is malformed and must be rejected, not narrowed by `& MASK64`. Two
# distinct malformed inputs must not collapse to distinct wrong values on the
# shared wire. Both engines must agree, so parametrize over _DECODERS.
#
# Reproducer from the issue: a u64 field (id 6) — header 0x30 = (6 << 3) |
# UNSIGNED — carrying the overlong varint.

_OVERLONG_U64 = [
    [0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02],  # 65th bit
    [0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F],  # bits 64..69
    [0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01],  # 11-byte
]


@pytest.mark.parametrize("decoder_cls", _DECODERS)
@pytest.mark.parametrize("data", _OVERLONG_U64)
def test_overlong_varint_rejected_as_invalid(decoder_cls, data):
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes(data))
    # INVALID, never INCOMPLETE — the bytes are garbage, not "need more".
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_max_u64_control_still_decodes(decoder_cls):
    # Control: the valid 10-byte maximum (10th byte 0x01 = only bit 63) must
    # still decode to 2^64-1 — the overlong guard must not reject it.
    data = [0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
    f, ev = pairs(decoder_cls, bytes(data))[0]
    assert f.id == 6
    assert ev == ("u", 6, (1 << 64) - 1)


# --- ID_MAX bounds a sequence-end header's id too ----------------------------
#
# Regression for issue #59 / Crucible F-0054 (spec §4.9/§6.2, documentation#35):
# a sequence end's id is *discarded* (§4.9), but discarded is not unvalidated.
# The header is an ordinary field header, so its id is bounded by ID_MAX (§6.2)
# like every other — an id above the ceiling is INVALID on a sequence end as
# anywhere, validated where the header is read rather than in the branch that
# uses the id. The bound is on the id's *value*, not its spelling (§4.1), so a
# non-minimal encoding of an in-range id stays valid. Both engines must agree.
#
# Isolate: 0x76 opens an unknown sequence (id 14); the next header is a sequence
# end (wire type 7) whose id is 2**31 = ID_MAX + 1 -> INVALID.
_SEQEND_ID_OVER_IDMAX = [0x76, 0x87, 0x80, 0x80, 0x80, 0x40]

# Controls that must stay ACCEPTED — an in-range id decodes as an ordinary
# sequence end (id discarded -> 0). 0x76 opens the sequence each one closes.
_SEQEND_ID_CONTROLS = [
    ([0x76, 0x07], "canonical, id 0"),
    ([0x76, 0x1F], "id 3"),
    ([0x76, 0xFF, 0xFF, 0xFF, 0xFF, 0x3F], "id ID_MAX (2**31 - 1)"),
    ([0x76, 0x87, 0x00], "non-minimal encoding of id 0"),
]


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_seqend_header_id_over_idmax_is_invalid(decoder_cls):
    with pytest.raises(SofaDecodeError) as exc:
        verdict(decoder_cls, bytes(_SEQEND_ID_OVER_IDMAX))
    # INVALID, never INCOMPLETE — all six bytes are present.
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
@pytest.mark.parametrize("data,label", _SEQEND_ID_CONTROLS)
def test_seqend_header_in_range_id_is_accepted(decoder_cls, data, label):
    # Whatever the id said on the wire, it is discarded: the end marker carries
    # no id at all.
    ev = values(decoder_cls, bytes(data))
    assert len(ev) == 2
    assert ev[0][0] == "seq{"       # the sequence start keeps its own id
    assert ev[1] == ("seq}",)       # the end marker carries none


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_fixlen_array_huge_count_is_bounded_on_the_skip_path_too(decoder_cls):
    """The payload size of a fixlen array is a *product*, and both factors come
    off the wire. Walking past such a field without reading it must reach the
    same bound: 2^31-1 fp64 elements claim a 16 GB payload, which is reported as
    truncated rather than attempted."""
    data = bytes([0x05] + _uvarint(0x7FFFFFFF) + _uvarint((8 << 3) | int(FixlenSubtype.FP64)))
    with pytest.raises(SofaIncompleteError):
        # No binding, no visitor hook for it: the field is walked, not read.
        verdict(decoder_cls, data, recorder=Recorder(decline=lambda f: True))
