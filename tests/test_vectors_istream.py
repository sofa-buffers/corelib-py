"""Decoder tests over the C reference byte vectors: decode the bytes the C suite
asserts on and check the recovered values (incl. the full-scale example)."""

from __future__ import annotations

import math

from vectors import DBL_MAX, FLT_MAX, FULL_SCALE_EXPECTED, reader

from sofab import Decoder, FixlenSubtype, WireType


def test_decode_simple_scalars():
    # 0x00 0x2A => id 0, unsigned 42 (from test_istream simple vectors)
    dec = Decoder(reader([0x00, 0x2A]))
    f = dec.next()
    assert f.id == 0 and f.type == WireType.UNSIGNED
    assert dec.unsigned() == 42
    assert dec.next() is None

    # 0x11 0x53 => id 2, signed -42
    dec = Decoder(reader([0x11, 0x53]))
    f = dec.next()
    assert f.id == 2 and f.type == WireType.SIGNED
    assert dec.signed() == -42


def test_decode_string_vector():
    data = [0x02, 0x62, 0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x20, 0x43, 0x6F, 0x75, 0x63, 0x68, 0x21]
    dec = Decoder(reader(data))
    f = dec.next()
    assert f.type == WireType.FIXLEN and f.subtype == FixlenSubtype.STRING and f.size == 12
    assert dec.string() == "Hello Couch!"


def test_decode_unsigned_array_vector():
    data = [0x03, 0x05, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
    dec = Decoder(reader(data))
    f = dec.next()
    assert f.type == WireType.ARRAY_UNSIGNED and f.count == 5
    assert dec.read_unsigned_array() == [1, 2, 3, 0x80000000, 0xFFFFFFFF]


def test_decode_full_scale_example():
    """Walk every field of the full-scale example bytes and verify values."""
    dec = Decoder(reader(FULL_SCALE_EXPECTED))

    def expect(field_id, wtype):
        f = dec.next()
        assert f is not None, "unexpected EOF"
        assert f.id == field_id and f.type == wtype, f"{f} != ({field_id},{wtype})"
        return f

    expect(0, WireType.UNSIGNED); assert dec.unsigned() == 200
    expect(1, WireType.SIGNED); assert dec.signed() == -100
    expect(2, WireType.UNSIGNED); assert dec.unsigned() == 50000
    expect(3, WireType.SIGNED); assert dec.signed() == -20000
    expect(4, WireType.UNSIGNED); assert dec.unsigned() == 3000000000
    expect(5, WireType.SIGNED); assert dec.signed() == -1000000000
    expect(6, WireType.UNSIGNED); assert dec.unsigned() == 10000000000000
    expect(7, WireType.SIGNED); assert dec.signed() == -5000000000000

    expect(10, WireType.SEQUENCE_START)
    expect(0, WireType.FIXLEN); assert abs(dec.float32() - 3.14) < 1e-6
    expect(1, WireType.FIXLEN); assert dec.float64() == 3.14159265
    expect(2, WireType.FIXLEN); assert dec.string() == "Hello, World!"
    expect(3, WireType.FIXLEN); assert dec.bytes() == bytes([0xDE, 0xAD, 0xBE, 0xEF])
    expect(0, WireType.SEQUENCE_END)

    expect(100, WireType.SEQUENCE_START)
    expect(0, WireType.ARRAY_UNSIGNED); assert dec.read_unsigned_array() == [0, 64, 128, 191, 255]
    expect(1, WireType.ARRAY_SIGNED); assert dec.read_signed_array() == [-128, -64, 0, 63, 127]
    expect(2, WireType.ARRAY_UNSIGNED); assert dec.read_unsigned_array() == [0, 16384, 32768, 49151, 65535]
    expect(3, WireType.ARRAY_SIGNED); assert dec.read_signed_array() == [-32768, -16384, 0, 16383, 32767]
    expect(4, WireType.ARRAY_UNSIGNED); assert dec.read_unsigned_array() == [0, 1073741824, 2147483648, 3221225471, 4294967295]
    expect(5, WireType.ARRAY_SIGNED); assert dec.read_signed_array() == [-2147483648, -1073741824, 0, 1073741823, 2147483647]
    expect(6, WireType.ARRAY_UNSIGNED); assert dec.read_unsigned_array() == [0, 4611686018427387904, 9223372036854775808, 13835058055282163711, 18446744073709551615]
    expect(7, WireType.ARRAY_SIGNED); assert dec.read_signed_array() == [-9223372036854775807, -4611686018427387904, 0, 4611686018427387903, 9223372036854775807]
    expect(10, WireType.SEQUENCE_START)
    expect(0, WireType.ARRAY_FIXLEN); assert dec.read_float32_array() == [1.0, 2.0, 3.0, -FLT_MAX, FLT_MAX]
    expect(1, WireType.ARRAY_FIXLEN); assert dec.read_float64_array() == [1.0, 2.0, 3.0, -DBL_MAX, DBL_MAX]
    expect(0, WireType.SEQUENCE_END)
    expect(0, WireType.SEQUENCE_END)

    expect(200, WireType.SEQUENCE_START)
    expect(0, WireType.FIXLEN); assert dec.string() == "Hello, Sofab!"
    expect(1, WireType.FIXLEN); assert dec.string() == ""
    expect(2, WireType.FIXLEN); assert dec.string() == "1234567890"
    expect(3, WireType.FIXLEN); assert dec.string() == "äöüÄÖÜß"
    expect(4, WireType.FIXLEN); assert dec.string() == "This_is_a_very_long_test_string_with_!@#$%^&*()_+-=[]{}"
    expect(0, WireType.SEQUENCE_END)

    assert dec.next() is None


def test_skip_unwanted_fields():
    dec = Decoder(reader(FULL_SCALE_EXPECTED))
    seen = 0
    while dec.next() is not None:
        seen += 1
        dec.skip()  # never consume; rely on auto/explicit skip
    assert seen > 0


def test_decode_specials_roundtrip_inf():
    data = [0x05, 0x05, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
            0x00, 0x00, 0x80, 0x7F, 0x00, 0x00, 0x80, 0xFF, 0x00, 0x00, 0xC0, 0x7F]
    dec = Decoder(reader(data))
    dec.next()
    vals = dec.read_float32_array()
    assert vals[0] == 0.0 and vals[2] == math.inf and vals[3] == -math.inf
    assert math.isnan(vals[4])
