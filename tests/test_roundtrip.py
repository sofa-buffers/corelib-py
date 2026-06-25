"""Round-trip encode -> decode equality across all types and boundaries."""

from __future__ import annotations

import math

import pytest
from vectors import reader

from sofab import (
    ARRAY_MAX,
    ID_MAX,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    Decoder,
    Encoder,
    WireType,
)


def _roundtrip(build):
    enc = Encoder()
    build(enc)
    return Decoder(reader(enc.getvalue()))


@pytest.mark.parametrize("value", [0, 1, 127, 128, 0x4000, UNSIGNED_MAX])
def test_unsigned(value):
    dec = _roundtrip(lambda e: e.write_unsigned(5, value))
    f = dec.next()
    assert f.id == 5 and f.type == WireType.UNSIGNED
    assert dec.unsigned() == value


@pytest.mark.parametrize("value", [0, -1, 1, SIGNED_MIN, SIGNED_MAX])
def test_signed(value):
    dec = _roundtrip(lambda e: e.write_signed(3, value))
    dec.next()
    assert dec.signed() == value


def test_bool():
    dec = _roundtrip(lambda e: (e.write_bool(0, True), e.write_bool(1, False)))
    dec.next(); assert dec.bool() is True
    dec.next(); assert dec.bool() is False


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, math.inf, -math.inf])
def test_float64(value):
    dec = _roundtrip(lambda e: e.write_float64(0, value))
    dec.next()
    got = dec.float64()
    assert got == value or (math.isinf(value) and math.isinf(got))


def test_string_and_bytes_unicode():
    dec = _roundtrip(
        lambda e: (e.write_string(0, "héllo ✓ 日本"), e.write_bytes(1, bytes(range(256))))
    )
    dec.next(); assert dec.string() == "héllo ✓ 日本"
    dec.next(); assert dec.bytes() == bytes(range(256))


def test_arrays():
    def build(e):
        e.write_unsigned_array(0, [0, 1, UNSIGNED_MAX])
        e.write_signed_array(1, [SIGNED_MIN, -1, 0, SIGNED_MAX])
        e.write_float32_array(2, [1.0, 2.0, 3.0])
        e.write_float64_array(3, [1.0, 2.0, 3.0])

    dec = _roundtrip(build)
    dec.next(); assert dec.read_unsigned_array() == [0, 1, UNSIGNED_MAX]
    dec.next(); assert dec.read_signed_array() == [SIGNED_MIN, -1, 0, SIGNED_MAX]
    dec.next(); assert dec.read_float32_array() == [1.0, 2.0, 3.0]
    dec.next(); assert dec.read_float64_array() == [1.0, 2.0, 3.0]


def test_nested_sequences_skip_whole():
    def build(e):
        e.write_unsigned(0, 7)
        e.write_sequence_begin(1)
        e.write_unsigned(0, 1)
        e.write_sequence_begin(2)
        e.write_signed(0, -9)
        e.write_sequence_end()
        e.write_sequence_end()
        e.write_unsigned(9, 99)

    dec = _roundtrip(build)
    dec.next(); assert dec.unsigned() == 7
    f = dec.next(); assert f.type == WireType.SEQUENCE_START and f.id == 1
    dec.skip()  # skip the whole nested sequence
    f = dec.next(); assert f.id == 9 and dec.unsigned() == 99
    assert dec.next() is None


def test_boundary_ids():
    dec = _roundtrip(lambda e: (e.write_unsigned(0, 0), e.write_unsigned(ID_MAX, 1)))
    assert dec.next().id == 0; dec.unsigned()
    assert dec.next().id == ID_MAX; assert dec.unsigned() == 1


def test_max_count_not_required():
    # sanity: ARRAY_MAX is exposed and large
    assert ARRAY_MAX == 0x7FFFFFFF
