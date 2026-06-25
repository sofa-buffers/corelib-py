"""Malformed-input tests. Byte vectors transcribed from
corelib-c-cpp/test/c/test_istream.c (SOFAB_RET_E_INVALID_MSG cases)."""

from __future__ import annotations

import pytest
from vectors import reader

from sofab import (
    Decoder,
    Encoder,
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


def test_array_count_zero():
    data = [0x04, 0x00, 0x53]
    with pytest.raises(SofaDecodeError):
        _decode_fully(data)


def test_array_fixlen_invalid_subtype():
    # 0x27 => element_size 4, subtype 7 (reserved) in a fixlen array
    data = [0x05, 0x05, 0x27, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40, 0x00,
            0x00, 0x40, 0x40, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0x7F]
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


def test_encode_empty_array_rejected():
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_unsigned_array(0, [])


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
