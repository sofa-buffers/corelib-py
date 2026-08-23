"""Round-trip encode -> decode equality across all types and boundaries."""

from __future__ import annotations

import math

import pytest
from vectors import Recorder, Status, pairs, values, walk

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


def _roundtrip(build) -> list[tuple]:
    """Encode, then decode back to the event list the handler was handed."""
    enc = Encoder()
    build(enc)
    return values(Decoder, enc.getvalue())


def _roundtrip_pairs(build) -> list[tuple]:
    """Same, but with each field's wire metadata alongside its value."""
    enc = Encoder()
    build(enc)
    return pairs(Decoder, enc.getvalue())


@pytest.mark.parametrize("value", [0, 1, 127, 128, 0x4000, UNSIGNED_MAX])
def test_unsigned(value):
    f, ev = _roundtrip_pairs(lambda e: e.write_unsigned(5, value))[0]
    assert f.id == 5 and f.type == WireType.UNSIGNED
    assert ev == ("u", 5, value)


@pytest.mark.parametrize("value", [0, -1, 1, SIGNED_MIN, SIGNED_MAX])
def test_signed(value):
    assert _roundtrip(lambda e: e.write_signed(3, value)) == [("s", 3, value)]


def test_bool():
    # A boolean has no wire type (§4.4): it arrives as the unsigned 1 / 0 it was
    # written as, and the caller reads truth from that.
    assert _roundtrip(lambda e: (e.write_bool(0, True), e.write_bool(1, False))) == [
        ("u", 0, 1),
        ("u", 1, 0),
    ]


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, math.inf, -math.inf])
def test_float64(value):
    got = _roundtrip(lambda e: e.write_float64(0, value))[0][2]
    assert got == value or (math.isinf(value) and math.isinf(got))


def test_string_and_bytes_unicode():
    dec = _roundtrip(
        lambda e: (e.write_string(0, "héllo ✓ 日本"), e.write_bytes(1, bytes(range(256))))
    )
    assert dec == [("str", 0, "héllo ✓ 日本"), ("blob", 1, bytes(range(256)))]


def test_fixlen_length_is_on_the_field_not_the_value():
    """The wire byte length of a string/blob is metadata, reported on the Field
    the handler is given — so a schema ``maxlen`` is bounded against the bytes
    the sender declared, without re-encoding the decoded ``str`` to measure it
    (generator #155). A binding does this for you (``maxlen=``); a visitor reads
    it off the Field."""
    s = "héllo ✓ 日本"  # 17 UTF-8 bytes across 1-, 2- and 3-byte code points
    blob = bytes(range(200))
    got = _roundtrip_pairs(lambda e: (e.write_string(0, s), e.write_bytes(1, blob)))

    (f_str, ev_str), (f_blob, ev_blob) = got
    assert f_str.size == len(s.encode("utf-8"))  # byte length, not char count
    assert ev_str == ("str", 0, s)
    assert f_blob.size == len(blob)
    assert ev_blob == ("blob", 1, blob)


def test_a_scalar_field_carries_no_fixlen_length():
    """§7.3: a non-fixlen field simply has none — ``size`` stays 0."""
    f, ev = _roundtrip_pairs(lambda e: e.write_unsigned(0, 42))[0]
    assert f.size == 0
    assert ev == ("u", 0, 42)


def test_arrays():
    def build(e):
        e.write_unsigned_array(0, [0, 1, UNSIGNED_MAX])
        e.write_signed_array(1, [SIGNED_MIN, -1, 0, SIGNED_MAX])
        e.write_float32_array(2, [1.0, 2.0, 3.0])
        e.write_float64_array(3, [1.0, 2.0, 3.0])

    assert _roundtrip(build) == [
        ("ua", 0, (0, 1, UNSIGNED_MAX)),
        ("sa", 1, (SIGNED_MIN, -1, 0, SIGNED_MAX)),
        ("f32a", 2, (1.0, 2.0, 3.0)),
        ("f64a", 3, (1.0, 2.0, 3.0)),
    ]


def test_zero_count_arrays_roundtrip_to_empty():
    # §4.7/§4.8: all four array kinds round-trip an empty list.
    def build(e):
        e.write_unsigned_array(0, [])
        e.write_signed_array(1, [])
        e.write_float32_array(2, [])
        e.write_float64_array(3, [])

    got = _roundtrip_pairs(build)
    assert [f.type for f, _ in got] == [
        WireType.ARRAY_UNSIGNED, WireType.ARRAY_SIGNED,
        WireType.ARRAY_FIXLEN, WireType.ARRAY_FIXLEN,
    ]
    assert all(f.count == 0 for f, _ in got)
    assert [ev[2] for _, ev in got] == [(), (), (), ()]


def test_max_depth_nesting_roundtrips():
    # 255 nested sequences (MAX_DEPTH) must encode and decode cleanly.
    from sofab import MAX_DEPTH

    def build(e):
        for _ in range(MAX_DEPTH):
            e.write_sequence_begin_lazy(0)
        e.write_unsigned(1, 42)
        for _ in range(MAX_DEPTH):
            e.write_sequence_end()

    ev = _roundtrip(build)
    assert ev == (
        [("seq{", 0)] * MAX_DEPTH + [("u", 1, 42)] + [("seq}",)] * MAX_DEPTH
    )


def test_nested_sequences_skip_whole():
    def build(e):
        e.write_unsigned(0, 7)
        e.write_sequence_begin_lazy(1)
        e.write_unsigned(0, 1)
        e.write_sequence_begin_lazy(2)
        e.write_signed(0, -9)
        e.write_sequence_end()
        e.write_sequence_end()
        e.write_unsigned(9, 99)

    # A handler that declines sequence 1 makes the decoder walk the whole
    # sub-tree without reporting anything inside it.
    enc = Encoder()
    build(enc)
    status, rec, _dec = walk(
        Decoder, enc.getvalue(),
        recorder=Recorder(decline=lambda f: False),
    )
    assert status is Status.COMPLETE
    assert rec.events == [
        ("u", 0, 7),
        ("seq{", 1), ("u", 0, 1),
        ("seq{", 2), ("s", 0, -9), ("seq}",), ("seq}",),
        ("u", 9, 99),
    ]


def test_boundary_ids():
    assert _roundtrip(
        lambda e: (e.write_unsigned(0, 0), e.write_unsigned(ID_MAX, 1))
    ) == [("u", 0, 0), ("u", ID_MAX, 1)]


def test_max_count_not_required():
    # sanity: ARRAY_MAX is exposed and large
    assert ARRAY_MAX == 0x7FFFFFFF
