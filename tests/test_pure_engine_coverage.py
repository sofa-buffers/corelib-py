"""Direct exercise of the **pure-Python** engine's error / sticky / edge paths.

The native accelerator shadows ``encoder.py`` / ``decoder.py`` at runtime, so the
main suite (which drives ``sofab.Encoder`` / ``sofab.Decoder``) only exercises
the pure modules when the native extension is absent. Coverage is therefore
measured against the pure engine (see the CI ``coverage`` job / README), and the
pure fallback must be *exactly* as trustworthy as the compiled path.

These tests import the pure classes **directly** (never via ``sofab.Encoder``),
so they cover the pure implementation regardless of which engine is active — the
error branches, sticky-mode latching, fixed-buffer plumbing, and malformed-input
rejections that the happy-path vectors don't reach.
"""

from __future__ import annotations

import io

import pytest
from vectors import ChunkReader

from sofab.decoder import Decoder
from sofab.encoder import Encoder
from sofab.types import (
    ARRAY_MAX,
    ID_MAX,
    SIGNED_MAX,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaIncompleteError,
    SofaRangeError,
    SofaStateError,
    WireType,
)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _hdr(field_id: int, wtype: int) -> bytes:
    return _varint((field_id << 3) | wtype)


# ============================ Encoder — error paths ==========================


def test_write_signed_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_signed(0, SIGNED_MAX + 1)
    with pytest.raises(SofaRangeError):
        Encoder().write_signed(0, -(1 << 70))


def test_write_unsigned_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_unsigned(0, UNSIGNED_MAX + 1)
    with pytest.raises(SofaRangeError):
        Encoder().write_unsigned(0, -1)


def test_write_id_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_unsigned(ID_MAX + 1, 0)


def test_unsigned_array_element_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_unsigned_array(1, [1, 2, UNSIGNED_MAX + 1])


def test_signed_array_element_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_signed_array(1, [0, -(1 << 70)])


def test_sequence_end_without_begin_raises():
    with pytest.raises(SofaStateError):
        Encoder().write_sequence_end()


def test_sequence_nesting_exceeds_max_depth_raises():
    enc = Encoder()
    from sofab.types import MAX_DEPTH

    for i in range(MAX_DEPTH):
        enc.write_sequence_begin(i)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin(0)


# ============================ Encoder — sticky mode ==========================


def test_sticky_latches_first_error_and_turns_writes_into_noops():
    """After a latched error every writer must early-return (no exception, no
    output) and the first error stays readable — covering each writer's sticky
    gate and the ``_fail`` latch path."""
    enc = Encoder(sticky=True)
    assert enc.error is None
    enc.write_signed(0, 1 << 70)  # out of range → latches, does not raise
    first = enc.error
    assert isinstance(first, SofaRangeError)

    # every subsequent writer is now a no-op that must not raise
    enc.write_unsigned(1, 5)
    enc.write_signed(2, -5)
    enc.write_bool(3, True)
    enc.write_float32(4, 1.0)
    enc.write_float64(5, 1.0)
    enc.write_string(6, "x")
    enc.write_bytes(7, b"x")
    enc.write_unsigned_array(8, [1, 2])
    enc.write_signed_array(9, [-1, -2])
    enc.write_float32_array(10, [1.0])
    enc.write_float64_array(11, [1.0])
    enc.write_sequence_begin(12)
    enc.write_sequence_end()

    assert enc.error is first  # first error preserved, not overwritten
    assert enc.getvalue() == b""  # nothing was emitted after the latch


def test_sticky_records_but_does_not_raise_for_each_writer():
    """Each writer, when it is the *first* failure on a fresh sticky encoder,
    latches instead of raising."""
    cases = [
        lambda e: e.write_unsigned(0, UNSIGNED_MAX + 1),
        lambda e: e.write_signed(0, 1 << 70),
        lambda e: e.write_unsigned_array(0, [UNSIGNED_MAX + 1]),
        lambda e: e.write_signed_array(0, [1 << 70]),
        lambda e: e.write_sequence_end(),
        lambda e: e.write_unsigned(ID_MAX + 1, 0),
    ]
    for trigger in cases:
        enc = Encoder(sticky=True)
        trigger(enc)  # must not raise
        assert enc.error is not None


# ====================== Encoder — fixed-buffer plumbing ======================


def test_buffer_set_offset_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder.over_buffer(bytearray(8), offset=8)  # offset == len is invalid


def test_fixed_buffer_full_without_sink_raises():
    enc = Encoder.over_buffer(bytearray(2), offset=0)  # no flush sink
    with pytest.raises(SofaBufferError):
        for i in range(100):
            enc.write_unsigned(i, 0xFFFFFFFF)


def test_getvalue_on_fixed_buffer_raises():
    enc = Encoder.over_buffer(bytearray(16), offset=0)
    enc.write_unsigned(1, 7)
    with pytest.raises(SofaStateError):
        enc.getvalue()


def test_writer_flush_drains_and_clears():
    sink = io.BytesIO()
    enc = Encoder(sink)
    enc.write_unsigned(1, 300)
    enc.write_string(2, "hi")
    n = enc.flush()
    assert n > 0
    assert sink.getvalue()  # bytes reached the writer
    # buffer cleared: a second flush with nothing pending returns 0
    assert enc.flush() == 0


def test_over_buffer_streams_through_flush_sink():
    chunks: list[bytes] = []
    enc = Encoder.over_buffer(bytearray(8), offset=0, flush=chunks.append)
    for i in range(50):
        enc.write_unsigned(i % 100, i)
    enc.flush()
    assert b"".join(chunks)  # streamed out in pieces


# ============================ Decoder — error paths ==========================


def _decode_all(data: bytes, *, chunk: int | None = None):
    src = ChunkReader(data, chunk) if chunk is not None else io.BytesIO(data)
    dec = Decoder(src)
    while dec.next() is not None:
        dec.skip()


def test_decode_id_out_of_range_raises():
    data = _hdr(ID_MAX + 1, WireType.UNSIGNED) + _varint(0)
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_invalid_fixlen_subtype_raises():
    # FIXLEN header, then a length-word whose subtype (low 3 bits) is 4 (> BLOB)
    data = _hdr(1, WireType.FIXLEN) + _varint((0 << 3) | 4)
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_array_count_out_of_range_raises():
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(ARRAY_MAX + 1)
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_fixlen_array_count_out_of_range_raises():
    data = _hdr(1, WireType.ARRAY_FIXLEN) + _varint(ARRAY_MAX + 1)
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_unbalanced_sequence_end_raises():
    data = _varint(WireType.SEQUENCE_END)  # end with nothing open
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_truncated_unbalanced_sequence_raises():
    data = _hdr(1, WireType.SEQUENCE_START)  # opens, never closes, EOF
    with pytest.raises(SofaIncompleteError):  # §7 INCOMPLETE: sequence never closed
        _decode_all(data)


def test_skip_truncated_sequence_raises():
    dec = Decoder(io.BytesIO(_hdr(1, WireType.SEQUENCE_START)))
    f = dec.next()
    assert f.type == WireType.SEQUENCE_START
    with pytest.raises(SofaIncompleteError):
        dec.skip()  # sub-tree never terminates → truncated sequence


def test_decode_truncated_scalar_varint_raises():
    data = _hdr(1, WireType.UNSIGNED) + b"\x80"  # value varint never terminates
    with pytest.raises(SofaIncompleteError):
        _decode_all(data)


def test_decode_truncated_array_element_raises():
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(1) + b"\x80"  # 1 elem, truncated
    with pytest.raises(SofaIncompleteError):
        _decode_all(data)


def test_decode_array_element_overflow_raises():
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(1) + b"\xff" * 10 + b"\x01"
    with pytest.raises(SofaDecodeError):
        _decode_all(data)


def test_decode_fp32_wrong_payload_length_raises():
    # FIXLEN / FP32 subtype but length 3 (not 4). The wrong width is rejected as
    # INVALID at header time (see corelib-py#38), so next() itself raises —
    # before the payload is read — rather than the value reader.
    data = _hdr(1, WireType.FIXLEN) + _varint((3 << 3) | FixlenSubtype.FP32) + b"\x00\x00\x00"
    dec = Decoder(io.BytesIO(data))
    with pytest.raises(SofaDecodeError):
        dec.next()


def test_decode_fp64_wrong_payload_length_raises():
    data = _hdr(1, WireType.FIXLEN) + _varint((4 << 3) | FixlenSubtype.FP64) + b"\x00\x00\x00\x00"
    dec = Decoder(io.BytesIO(data))
    with pytest.raises(SofaDecodeError):
        dec.next()


# --------------- Decoder — wrong-type read → SofaStateError ------------------


def test_read_wrong_type_raises_state_error():
    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_string(2, "hi")
    enc.write_unsigned_array(3, [1, 2])
    enc.write_float32_array(4, [1.0])
    data = enc.getvalue()

    dec = Decoder(io.BytesIO(data))
    dec.next()  # unsigned field
    with pytest.raises(SofaStateError):
        dec.float32()  # fixlen read on a scalar field

    dec.next()  # string (fixlen) field
    with pytest.raises(SofaStateError):
        dec.float64()  # wrong fixlen subtype

    dec.next()  # unsigned array
    with pytest.raises(SofaStateError):
        dec.read_signed_array()  # wrong varint-array wire type

    dec.next()  # fp32 array
    with pytest.raises(SofaStateError):
        dec.read_float64_array()  # wrong fixlen-array subtype


def test_scalar_read_without_matching_field_raises():
    dec = Decoder(io.BytesIO(_hdr(1, WireType.UNSIGNED) + _varint(9)))
    dec.next()
    assert dec.unsigned() == 9
    # no pending value now → asking again is a state error
    with pytest.raises(SofaStateError):
        dec.unsigned()


def test_field_property_tracks_current():
    dec = Decoder(io.BytesIO(_hdr(7, WireType.UNSIGNED) + _varint(1)))
    assert dec.field is None
    f = dec.next()
    assert dec.field is f
    assert dec.field.id == 7


def test_read_exact_overshoot_keeps_remainder():
    """A chunked reader that overshoots a blob payload must keep the surplus for
    the next field (exercises the _read_exact overshoot branch)."""
    enc = Encoder()
    enc.write_bytes(1, b"abcd")
    enc.write_unsigned(2, 42)
    data = enc.getvalue()

    # chunk=5 leaves the 4-byte blob partially buffered, and the refill that
    # completes it also drags in the following field's bytes → overshoot kept.
    dec = Decoder(ChunkReader(data, chunk=5))
    f = dec.next()
    assert f.type == WireType.FIXLEN
    assert dec.bytes() == b"abcd"
    dec.next()
    assert dec.unsigned() == 42
    assert dec.next() is None


# --------- Decoder — paths reached only when *reading* (not skipping) --------


def test_read_array_truncated_element_raises():
    """Truncation inside an array element surfaces while *reading* it (the
    _read_varints refill path, distinct from the skip path)."""
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(2) + _varint(5) + b"\x80"
    dec = Decoder(io.BytesIO(data))
    dec.next()
    with pytest.raises(SofaIncompleteError):
        dec.read_unsigned_array()


def test_read_array_element_overflow_raises():
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(1) + b"\xff" * 10 + b"\x01"
    dec = Decoder(io.BytesIO(data))
    dec.next()
    with pytest.raises(SofaDecodeError):
        dec.read_unsigned_array()


def test_read_signed_array_truncated_across_refill_raises():
    """A signed array read fed one byte at a time, truncated mid-element."""
    data = _hdr(1, WireType.ARRAY_SIGNED) + _varint(1) + b"\x80"
    dec = Decoder(ChunkReader(data, chunk=1))
    dec.next()
    with pytest.raises(SofaIncompleteError):
        dec.read_signed_array()


def test_fixlen_length_word_truncated_raises():
    """A FIXLEN header with a truncated length word (a second varint that runs
    off the end) is INCOMPLETE (§7) — the bytes end inside the field."""
    data = _hdr(1, WireType.FIXLEN) + b"\x80"  # length word never terminates
    with pytest.raises(SofaIncompleteError):
        _decode_all(data)


def test_fixlen_missing_length_word_raises():
    """A FIXLEN header with *no* length word at all — the follow-up varint hits
    EOF on its very first byte (the _varint start-of-buffer refill path)."""
    data = _hdr(1, WireType.FIXLEN)  # header only, nothing after
    with pytest.raises(SofaIncompleteError):
        _decode_all(data)


def test_read_array_missing_element_at_boundary_raises():
    """count says 2 but only one element is present; the read runs out exactly
    at an element boundary (the _read_varints outer-loop refill path)."""
    data = _hdr(1, WireType.ARRAY_UNSIGNED) + _varint(2) + _varint(5)
    dec = Decoder(io.BytesIO(data))
    dec.next()
    with pytest.raises(SofaIncompleteError):
        dec.read_unsigned_array()


def test_read_float_array_on_non_fixlen_array_raises():
    dec = Decoder(io.BytesIO(_hdr(1, WireType.UNSIGNED) + _varint(1)))
    dec.next()
    with pytest.raises(SofaStateError):
        dec.read_float32_array()  # current field is not a fixlen array at all


# ------------- Encoder — error propagation through fixlen writers ------------


def test_write_string_id_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_string(ID_MAX + 1, "x")


def test_write_bytes_id_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_bytes(ID_MAX + 1, b"x")


def test_write_float_array_id_out_of_range_raises():
    with pytest.raises(SofaRangeError):
        Encoder().write_float32_array(ID_MAX + 1, [1.0])
    with pytest.raises(SofaRangeError):
        Encoder().write_float64_array(ID_MAX + 1, [1.0])
