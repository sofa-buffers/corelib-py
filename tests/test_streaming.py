"""Streaming tests: 1-byte-granularity decode and tiny-scratch-buffer encode
must match the one-shot path."""

from __future__ import annotations

from vectors import FULL_SCALE_EXPECTED, ChunkReader, build_full_scale

from sofab import Decoder, Encoder, WireType


def _walk_values(dec: Decoder):
    """Fully consume a decoder, returning a list of (id, type, value?) tuples."""
    out = []
    while (f := dec.next()) is not None:
        if f.type == WireType.UNSIGNED:
            out.append((f.id, f.type, dec.unsigned()))
        elif f.type == WireType.SIGNED:
            out.append((f.id, f.type, dec.signed()))
        elif f.type == WireType.FIXLEN:
            out.append((f.id, f.type, dec.bytes() if f.subtype.name == "BLOB" else
                        (dec.string() if f.subtype.name == "STRING" else
                         (dec.float32() if f.subtype.name == "FP32" else dec.float64()))))
        elif f.type == WireType.ARRAY_UNSIGNED:
            out.append((f.id, f.type, tuple(dec.read_unsigned_array())))
        elif f.type == WireType.ARRAY_SIGNED:
            out.append((f.id, f.type, tuple(dec.read_signed_array())))
        elif f.type == WireType.ARRAY_FIXLEN:
            reader = dec.read_float32_array if f.subtype.name == "FP32" else dec.read_float64_array
            out.append((f.id, f.type, tuple(reader())))
        else:
            out.append((f.id, f.type, None))
    return out


def test_decode_one_byte_at_a_time_matches_oneshot():
    oneshot = _walk_values(Decoder(ChunkReader(FULL_SCALE_EXPECTED, chunk=1 << 20)))
    streamed = _walk_values(Decoder(ChunkReader(FULL_SCALE_EXPECTED, chunk=1)))
    assert streamed == oneshot


def test_encode_through_tiny_scratch_buffer_matches_oneshot():
    # one-shot reference
    ref = Encoder()
    build_full_scale(ref)
    expected = ref.getvalue()

    # stream through a 7-byte scratch buffer + flush sink
    collected = bytearray()
    enc = Encoder.over_buffer(bytearray(7), offset=0, flush=collected.extend)
    build_full_scale(enc)
    enc.flush()
    assert bytes(collected) == expected


def test_reserve_offset_left_untouched_then_flushed():
    # With offset=4 and no overflow, flush emits the reserved bytes + payload.
    buf = bytearray(64)
    buf[0:4] = b"HDR!"
    enc = Encoder.over_buffer(buf, offset=4)
    enc.write_unsigned(0, 42)
    used = enc.bytes_used()
    assert bytes(buf[0:used]) == b"HDR!" + bytes([0x00, 0x2A])
