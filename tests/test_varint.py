"""Varint + zigzag codec unit tests."""

from __future__ import annotations

import pytest

from sofab import zigzag_decode, zigzag_encode
from sofab._varint import decode_varint, encode_varint
from sofab.types import SofaDecodeError


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


def test_decode_truncated_raises():
    with pytest.raises(SofaDecodeError):
        decode_varint(_reader(bytes([0x80, 0x80])))  # never terminates


def test_decode_overflow_raises():
    # 10 continuation bytes -> shift reaches 64 with continuation set
    with pytest.raises(SofaDecodeError):
        decode_varint(_reader(bytes([0xFF] * 10 + [0x01])))
