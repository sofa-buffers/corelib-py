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
    # Regression (corelib-py#28): fp32 fixlen array whose fixlen_word declares a
    # 0-byte element width but a non-zero count. The payload is shorter than the
    # count*4 bytes an fp32 unpack reads; the native engine used to trust the
    # count and read off the end of the buffer (SIGSEGV). Both engines must
    # reject it as malformed.
    # 0x05 = (0<<3)|ARRAY_FIXLEN, 0x01 = count 1, 0x00 = fixlen_word (0<<3)|fp32.
    dec = Decoder(reader([0x05, 0x01, 0x00]))
    f = dec.next()
    assert f is not None and f.subtype == FixlenSubtype.FP32
    with pytest.raises(SofaDecodeError):
        dec.read_float32_array()


def test_array_fixlen_element_width_mismatch_overflow():
    # fp32 array claiming an 8-byte element width (fp64's width): enough bytes
    # are present, but count*8 != count*4, so it is still a malformed fixlen_word.
    # 0x40 = (8<<3)|fp32; eight payload bytes follow the count-1 element.
    data = [0x05, 0x01, 0x40, 0, 0, 0x80, 0x3F, 0, 0, 0, 0]
    dec = Decoder(reader(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.read_float32_array()


def test_array_fixlen_fp64_width_mismatch():
    # Same defect on the fp64 path: subtype fp64 (1) with a 4-byte element width.
    # 0x05 = ARRAY_FIXLEN, 0x01 = count 1, 0x21 = (4<<3)|fp64; four payload bytes.
    data = [0x05, 0x01, 0x21, 0, 0, 0, 0]
    dec = Decoder(reader(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.read_float64_array()


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


def test_array_fixlen_payload_size_overflow_rejected_when_skipped():
    # A fixlen array whose count * element_width overflows the payload-size
    # arithmetic must reject as truncated (an unsatisfiable read), not wrap to a
    # small/negative size and drive the cursor off the buffer. Exercises the
    # skip path (which consumes the payload without reading it).
    # count = ARRAY_MAX, element width ~2^61 (fixlen_word low 3 bits 0 => fp32).
    count = 0x7FFFFFFF
    elem_word = 0xFFFFFFFFFFFFFFF8  # (elem_size << 3) | fp32, elem_size ~2^61
    data = [0x05] + _uvarint(count) + _uvarint(elem_word)
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


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
    # fixlen string claims 12 bytes but only 2 follow
    data = [0x02, 0x62, 0x48, 0x65]
    dec = Decoder(reader(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.string()


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
