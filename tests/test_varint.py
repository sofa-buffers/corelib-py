"""Varint + zigzag codec unit tests."""

from __future__ import annotations

import pytest

from sofab import zigzag_decode, zigzag_encode
from sofab._varint import decode_varint, encode_varint
from sofab.types import SofaDecodeError, SofaIncompleteError


def _reader(data: bytes):
    it = iter(data)

    def read_byte():
        return next(it, None)

    return read_byte


@pytest.mark.parametrize(
    "value,expected",
    [
        (0x0, [0x00]),
        (0x7F, [0x7F]),
        (0x80, [0x80, 0x01]),
        (0x3FFF, [0xFF, 0x7F]),
        (0x4000, [0x80, 0x80, 0x01]),
        ((1 << 64) - 1, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
    ],
)
def test_encode_varint(value, expected):
    assert encode_varint(value) == bytes(expected)


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, (1 << 32), (1 << 63), (1 << 64) - 1])
def test_varint_roundtrip(value):
    assert decode_varint(_reader(encode_varint(value))) == value


@pytest.mark.parametrize("value", [0, 1, -1, 2, -2, 42, -42, (1 << 63) - 1, -(1 << 63)])
def test_zigzag_roundtrip(value):
    assert zigzag_decode(zigzag_encode(value)) == value


def test_zigzag_known_mapping():
    assert [zigzag_encode(v) for v in (0, -1, 1, -2, 2)] == [0, 1, 2, 3, 4]


def test_decode_truncated_is_incomplete():
    # Bytes ending mid-varint (continuation set, no terminator) is INCOMPLETE
    # (§7), NOT malformed — SofaIncompleteError is not a SofaDecodeError, so
    # `except SofaDecodeError` must not catch it.
    with pytest.raises(SofaIncompleteError) as exc:
        decode_varint(_reader(bytes([0x80, 0x80])))  # never terminates
    assert not isinstance(exc.value, SofaDecodeError)


def test_decode_overflow_raises():
    # 10 continuation bytes -> shift reaches 64 with continuation set
    with pytest.raises(SofaDecodeError):
        decode_varint(_reader(bytes([0xFF] * 10 + [0x01])))


def test_decode_max_u64_is_accepted():
    # Control (issue #43): the valid 10-byte maximum whose 10th byte carries
    # only bit 63 (0x01) still decodes to 2^64-1 — the overlong guard must not
    # reject it.
    data = bytes([0xFF] * 9 + [0x01])
    assert decode_varint(_reader(data)) == (1 << 64) - 1


@pytest.mark.parametrize(
    "tenth",
    [
        0x02,  # the 65th bit set
        0x7F,  # bits 64..69 set
        0x03,  # bit 63 (valid) plus the 65th bit
    ],
)
def test_decode_overlong_10th_byte_high_bits_rejected(tenth):
    # Regression (issue #43, Crucible F-0016): a 10-byte varint whose 10th byte
    # sets any bit above bit 63 is an overlong (>64-bit) varint and must be
    # rejected as INVALID — never silently narrowed by `& MASK64` on return.
    data = bytes([0xFF] * 9 + [tenth])
    with pytest.raises(SofaDecodeError):
        decode_varint(_reader(data))
    # And an outright too-long (11th continuation byte) varint is also INVALID.
    with pytest.raises(SofaDecodeError):
        decode_varint(_reader(bytes([0xFF] * 10 + [0x7F])))
