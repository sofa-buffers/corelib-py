"""Malformed-input tests. Byte vectors transcribed from
corelib-c-cpp/test/c/test_istream.c (SOFAB_RET_E_INVALID_MSG cases)."""

from __future__ import annotations

import pytest
from vectors import reader

from sofab import (
    Decoder,
    Encoder,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaLimitError,
    SofaRangeError,
    SofaStateError,
)


def _decode_fully(data):
    dec = Decoder(reader(data))
    while True:
        f = dec.next()
        if f is None:
            return
        dec.skip()


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
        dec = Decoder(reader([header, 0x00]))
        f = dec.next()
        assert f is not None and f.count == 0
        # No fixlen_word / payload may be consumed; the stream is now at EOF.
        assert dec.next() is None
    # Fixlen array (0x05): [header][0x00][fixlen_word] — the fixlen_word is always
    # present (§4.8), here 0x20 = (4<<3)|fp32, but there is no payload.
    dec = Decoder(reader([0x05, 0x00, 0x20]))
    f = dec.next()
    assert f is not None and f.count == 0 and f.subtype == FixlenSubtype.FP32
    assert dec.next() is None


def test_array_fixlen_count_zero_reads_the_fixlen_word():
    # §4.8: an empty fixlen array still carries its fixlen_word, so the bytes
    # after [0x05, 0x00, <fixlen_word>] must be parsed as the NEXT field.
    # 0x20 = (4<<3)|fp32 fixlen_word; 0x50 = (10 << 3) | UNSIGNED, 0x07 = value 7.
    dec = Decoder(reader([0x05, 0x00, 0x20, 0x50, 0x07]))
    f = dec.next()
    assert f is not None and f.count == 0 and f.subtype == FixlenSubtype.FP32
    assert dec.read_float32_array() == []  # empty fixlen array reads as []
    nxt = dec.next()
    assert nxt is not None and nxt.id == 10 and dec.unsigned() == 7


def test_string_invalid_utf8_raises_decode_error():
    # fixlen STRING (subtype 0x2) of length 2 with invalid UTF-8 bytes.
    # length_header = (2 << 3) | 0x2 = 0x12; payload 0xFF 0xFE is not valid UTF-8.
    data = [0x02, 0x12, 0xFF, 0xFE]
    dec = Decoder(reader(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.string()


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
    dec = Decoder(reader([0x05, 0x01, 0x00]))
    with pytest.raises(SofaDecodeError):
        dec.next()


def test_array_fixlen_element_width_mismatch_overflow():
    # fp32 array claiming an 8-byte element width (fp64's width): even with the
    # payload present, count*8 != count*4, so the fixlen_word is malformed and
    # rejected eagerly at header time.
    # 0x40 = (8<<3)|fp32; eight payload bytes follow the count-1 element.
    data = [0x05, 0x01, 0x40, 0, 0, 0x80, 0x3F, 0, 0, 0, 0]
    dec = Decoder(reader(data))
    with pytest.raises(SofaDecodeError):
        dec.next()


def test_array_fixlen_fp64_width_mismatch():
    # Same defect on the fp64 path: subtype fp64 (1) with a 4-byte element width.
    # 0x05 = ARRAY_FIXLEN, 0x01 = count 1, 0x21 = (4<<3)|fp64; four payload bytes.
    data = [0x05, 0x01, 0x21, 0, 0, 0, 0]
    dec = Decoder(reader(data))
    with pytest.raises(SofaDecodeError):
        dec.next()


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
from sofab.decoder import Decoder as _PyDecoder  # noqa: E402

_DECODERS = [_PyDecoder]
try:  # the native accelerator, when compiled in, must behave identically
    from sofab import _speedups as _sp

    _DECODERS.append(_sp.Decoder)
except ImportError:  # pragma: no cover - pure-Python-only install
    pass


def _decode_one(decoder_cls, data):
    """next() then consume the single fixlen field, surfacing its verdict."""
    dec = decoder_cls(reader(data))
    f = dec.next()
    assert f is not None
    if f.subtype == FixlenSubtype.FP32:
        dec.float32()
    else:
        dec.float64()


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
    dec = decoder_cls(reader([0x75, 0x60, 0x00, 0x0D, 0x0D]))
    with pytest.raises(SofaDecodeError) as exc:
        dec.next()
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("decoder_cls", _DECODERS)
def test_array_fixlen_correct_width_truncated_stays_incomplete(decoder_cls):
    # Control: correct fp32 width (0x20 = (4<<3)|fp32) with count 1 but zero
    # payload bytes → genuinely INCOMPLETE, must NOT be reclassified.
    dec = decoder_cls(reader([0x05, 0x01, 0x20]))
    f = dec.next()
    assert f is not None and f.subtype == FixlenSubtype.FP32
    with pytest.raises(SofaIncompleteError):
        dec.read_float32_array()


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
    dec = Decoder(reader(data))
    dec.next()
    with pytest.raises(SofaIncompleteError):
        dec.string()


# --- three-valued outcome: INCOMPLETE vs INVALID (MESSAGE_SPEC §7) -----------


def test_lone_continuation_byte_is_incomplete_not_malformed():
    # A single dangling 0x80 (continuation bit set, no terminating byte) is the
    # canonical INCOMPLETE case: the bytes end inside a varint, more could follow.
    # It must be neither COMPLETE (next() returning a field / None) nor INVALID.
    dec = Decoder(reader([0x80]))
    with pytest.raises(SofaIncompleteError) as exc:
        dec.next()
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
    dec = Decoder(reader(data))
    f = dec.next()
    assert f is not None and f.count == 0x7FFFFFFF
    with pytest.raises(SofaIncompleteError):
        dec.read_unsigned_array()


def test_signed_array_huge_count_does_not_preallocate():
    # Same hardening on the signed-array path (0x04 = (0<<3)|ARRAY_SIGNED).
    data = [0x04] + _uvarint(0x7FFFFFFF) + [0x01]
    dec = Decoder(reader(data))
    f = dec.next()
    assert f is not None and f.count == 0x7FFFFFFF
    with pytest.raises(SofaIncompleteError):
        dec.read_signed_array()


def test_max_array_count_rejects_oversize_before_alloc():
    # Part B acceptance: with max_array_count=65536 an otherwise-valid message
    # carrying a 65537-element dynamic array raises SofaLimitError at header time
    # (next()) — before read_unsigned_array is ever called. The identical bytes
    # decode unchanged with the limit unset.
    enc = Encoder()
    enc.write_unsigned_array(7, list(range(65537)))
    data = enc.getvalue()

    dec = Decoder(reader(data), max_array_count=65536)
    with pytest.raises(SofaLimitError):
        dec.next()

    dec2 = Decoder(reader(data))
    f = dec2.next()
    assert f is not None and f.count == 65537
    assert dec2.read_unsigned_array() == list(range(65537))


def test_max_array_count_fires_before_any_payload():
    # The cap is enforced on the count varint alone: a header claiming 100
    # elements with NO payload following still raises SofaLimitError (not the
    # truncation that reading the absent elements would give), proving the check
    # runs before allocation/buffering.
    data = [0x03] + _uvarint(100)  # ARRAY_UNSIGNED, count 100, no elements
    dec = Decoder(reader(data), max_array_count=10)
    with pytest.raises(SofaLimitError):
        dec.next()


def test_max_array_count_boundary_is_inclusive():
    # count == max_array_count is allowed; count == max + 1 is rejected.
    ok = Encoder()
    ok.write_unsigned_array(0, list(range(8)))
    dec = Decoder(reader(ok.getvalue()), max_array_count=8)
    f = dec.next()
    assert f is not None and f.count == 8
    assert dec.read_unsigned_array() == list(range(8))

    over = Encoder()
    over.write_unsigned_array(0, list(range(9)))
    dec = Decoder(reader(over.getvalue()), max_array_count=8)
    with pytest.raises(SofaLimitError):
        dec.next()


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
        dec = Decoder(reader(enc.getvalue()), max_array_count=5)
        with pytest.raises(SofaLimitError):
            dec.next()


def test_max_string_len_fires_before_payload():
    # A fixlen STRING header claiming length 100 with NO payload bytes is
    # rejected by max_string_len at next(), before the payload is read/buffered.
    # 0x02 = (0<<3)|FIXLEN; length_header = (100 << 3) | 0x2 (STRING).
    data = [0x02] + _uvarint((100 << 3) | 0x2)
    dec = Decoder(reader(data), max_string_len=10)
    with pytest.raises(SofaLimitError):
        dec.next()


def test_max_string_len_valid_message_roundtrips_without_limit():
    enc = Encoder()
    enc.write_string(3, "x" * 100)
    data = enc.getvalue()

    dec = Decoder(reader(data), max_string_len=64)
    with pytest.raises(SofaLimitError):
        dec.next()

    dec2 = Decoder(reader(data))
    dec2.next()
    assert dec2.string() == "x" * 100

    within = Encoder()
    within.write_string(3, "y" * 64)  # exactly at the limit: allowed
    dec3 = Decoder(reader(within.getvalue()), max_string_len=64)
    dec3.next()
    assert dec3.string() == "y" * 64


def test_max_blob_len_rejects_oversize():
    enc = Encoder()
    enc.write_bytes(1, b"\x00" * 100)
    data = enc.getvalue()

    dec = Decoder(reader(data), max_blob_len=16)
    with pytest.raises(SofaLimitError):
        dec.next()

    dec2 = Decoder(reader(data))
    dec2.next()
    assert dec2.bytes() == b"\x00" * 100


def test_limits_are_independent_per_kind():
    # Each limit governs only its own field kind: a blob is not bound by
    # max_string_len, nor a string by max_blob_len.
    blob = Encoder()
    blob.write_bytes(1, b"z" * 100)
    dec = Decoder(reader(blob.getvalue()), max_string_len=1)
    dec.next()
    assert dec.bytes() == b"z" * 100

    text = Encoder()
    text.write_string(1, "z" * 100)
    dec = Decoder(reader(text.getvalue()), max_blob_len=1)
    dec.next()
    assert dec.string() == "z" * 100


def test_limit_error_is_not_a_decode_or_incomplete_error():
    # Part B acceptance: a limit rejection is policy, not wire malformation, so a
    # handler that catches only the invalid-message class must not swallow it.
    enc = Encoder()
    enc.write_unsigned_array(0, list(range(4)))
    data = enc.getvalue()

    dec = Decoder(reader(data), max_array_count=2)
    with pytest.raises(SofaLimitError) as exc:
        dec.next()
    assert isinstance(exc.value, SofaError)
    assert not isinstance(exc.value, SofaDecodeError)
    assert not isinstance(exc.value, SofaIncompleteError)

    # `except SofaDecodeError` genuinely does not intercept it.
    with pytest.raises(SofaLimitError):
        try:
            Decoder(reader(data), max_array_count=2).next()
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
        enc.write_sequence_begin(i % 100)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin(0)  # the 256th must be refused


def test_sequence_end_without_begin():
    enc = Encoder()
    with pytest.raises(SofaStateError):
        enc.write_sequence_end()


def test_buffer_full_without_sink():
    enc = Encoder.over_buffer(bytearray(2))  # too small, no flush sink
    with pytest.raises(SofaBufferError):
        enc.write_unsigned(0, 1 << 60)


def test_wrong_type_read_raises_state_error():
    enc = Encoder()
    enc.write_unsigned(0, 5)
    dec = Decoder(reader(enc.getvalue()))
    dec.next()
    with pytest.raises(SofaStateError):
        dec.signed()  # field is unsigned


# --- sticky mode ------------------------------------------------------------


def test_sticky_mode_records_first_error_and_noops():
    enc = Encoder(sticky=True)
    enc.write_unsigned(0, 1 << 64)  # range error, recorded
    enc.write_unsigned(1, 5)  # becomes a no-op
    assert enc.error is not None
    assert isinstance(enc.error, SofaRangeError)
    assert enc.getvalue() == b""
