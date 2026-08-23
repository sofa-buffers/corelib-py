"""Decoder tests over the C reference byte vectors: decode the bytes the C suite
asserts on and check the recovered values (incl. the full-scale example)."""

from __future__ import annotations

import math

from vectors import (
    DBL_MAX,
    FLT_MAX,
    FULL_SCALE_EXPECTED,
    Recorder,
    Status,
    pairs,
    values,
    walk,
)

from sofab import Decoder, FixlenSubtype, WireType


def test_decode_simple_scalars():
    # 0x00 0x2A => id 0, unsigned 42 (from test_istream simple vectors)
    got = pairs(Decoder, bytes([0x00, 0x2A]))
    assert len(got) == 1
    f, ev = got[0]
    assert f.id == 0 and f.type == WireType.UNSIGNED
    assert ev == ("u", 0, 42)

    # 0x11 0x53 => id 2, signed -42
    f, ev = pairs(Decoder, bytes([0x11, 0x53]))[0]
    assert f.id == 2 and f.type == WireType.SIGNED
    assert ev == ("s", 2, -42)


def test_decode_string_vector():
    data = [0x02, 0x62, 0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x20, 0x43, 0x6F, 0x75, 0x63, 0x68, 0x21]
    f, ev = pairs(Decoder, bytes(data))[0]
    assert f.type == WireType.FIXLEN and f.subtype == FixlenSubtype.STRING and f.size == 12
    assert ev == ("str", 0, "Hello Couch!")


def test_decode_unsigned_array_vector():
    data = [0x03, 0x05, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]
    f, ev = pairs(Decoder, bytes(data))[0]
    assert f.type == WireType.ARRAY_UNSIGNED and f.count == 5
    assert ev == ("ua", 0, (1, 2, 3, 0x80000000, 0xFFFFFFFF))


def test_decode_full_scale_example():
    """Walk every field of the full-scale example bytes and verify values."""
    ev = values(Decoder, FULL_SCALE_EXPECTED)

    # fp32 3.14 does not survive a round trip exactly; check it, then pin it so
    # the rest of the message can be compared as one list.
    f32_at = next(i for i, e in enumerate(ev) if e[0] == "f32")
    assert abs(ev[f32_at][2] - 3.14) < 1e-6
    ev[f32_at] = ("f32", ev[f32_at][1], 3.14)

    assert ev == [
        ("u", 0, 200), ("s", 1, -100),
        ("u", 2, 50000), ("s", 3, -20000),
        ("u", 4, 3000000000), ("s", 5, -1000000000),
        ("u", 6, 10000000000000), ("s", 7, -5000000000000),

        ("seq{", 10),
        ("f32", 0, 3.14),
        ("f64", 1, 3.14159265),
        ("str", 2, "Hello, World!"),
        ("blob", 3, bytes([0xDE, 0xAD, 0xBE, 0xEF])),
        ("seq}",),

        ("seq{", 100),
        ("ua", 0, (0, 64, 128, 191, 255)),
        ("sa", 1, (-128, -64, 0, 63, 127)),
        ("ua", 2, (0, 16384, 32768, 49151, 65535)),
        ("sa", 3, (-32768, -16384, 0, 16383, 32767)),
        ("ua", 4, (0, 1073741824, 2147483648, 3221225471, 4294967295)),
        ("sa", 5, (-2147483648, -1073741824, 0, 1073741823, 2147483647)),
        ("ua", 6, (0, 4611686018427387904, 9223372036854775808,
                   13835058055282163711, 18446744073709551615)),
        ("sa", 7, (-9223372036854775807, -4611686018427387904, 0,
                   4611686018427387903, 9223372036854775807)),
        ("seq{", 10),
        ("f32a", 0, (1.0, 2.0, 3.0, -FLT_MAX, FLT_MAX)),
        ("f64a", 1, (1.0, 2.0, 3.0, -DBL_MAX, DBL_MAX)),
        ("seq}",),
        ("seq}",),

        ("seq{", 200),
        ("str", 0, "Hello, Sofab!"),
        ("str", 1, ""),
        ("str", 2, "1234567890"),
        ("str", 3, "äöüÄÖÜß"),
        ("str", 4, "This_is_a_very_long_test_string_with_!@#$%^&*()_+-=[]{}"),
        ("seq}",),
    ]


def test_skip_unwanted_fields():
    """A handler that declines everything: the decoder walks the whole message
    and materialises nothing."""
    status, rec, _dec = walk(
        Decoder, FULL_SCALE_EXPECTED, recorder=Recorder(decline=lambda f: True)
    )
    assert status is Status.COMPLETE
    assert rec.fields and rec.events == [e for e in rec.events if e[0].startswith("seq")]


def test_decode_specials_roundtrip_inf():
    data = [0x05, 0x05, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
            0x00, 0x00, 0x80, 0x7F, 0x00, 0x00, 0x80, 0xFF, 0x00, 0x00, 0xC0, 0x7F]
    vals = values(Decoder, bytes(data))[0][2]
    assert vals[0] == 0.0 and vals[2] == math.inf and vals[3] == -math.inf
    assert math.isnan(vals[4])
