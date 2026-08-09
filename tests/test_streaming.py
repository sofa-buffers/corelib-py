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


def _stream_with_reserved_header(enc_cls, cap: int, offset: int, filler: int = 0xEE):
    """Encode ``build_full_scale`` through ``cap``-byte buffers whose sink *takes*
    the buffer it was handed and installs a replacement reserving ``offset`` bytes
    of framing room (CORELIB_PLAN §5.1 take-and-replace). Returns the packets.
    """
    packets: list[bytes] = []
    enc = None

    def sink(chunk: bytes) -> None:
        packets.append(bytes(chunk))
        enc.buffer_set(bytearray([filler]) * cap, offset)

    enc = enc_cls.over_buffer(bytearray([filler]) * cap, offset, sink)
    build_full_scale(enc)
    enc.flush()
    return packets


def _assert_every_packet_reserves(packets, offset: int, expected: bytes, filler: int = 0xEE):
    """Every flushed unit must begin at the offset its installation asked for, and
    the payloads behind those reservations must concatenate to the one-shot bytes."""
    head = bytes([filler]) * offset
    for i, packet in enumerate(packets):
        assert packet[:offset] == head, f"packet {i} lost its {offset}-byte reservation"
    assert b"".join(p[offset:] for p in packets) == expected


def test_buffer_set_from_sink_reserves_its_offset_in_every_packet():
    """§5.1: "the start offset belongs to the installation, not to the buffer" —
    a sink that re-arms the reservation on every flush gets header room in *every*
    packet, not just the first.

    The drain used to reset the cursor to 0 unconditionally *after* the sink had
    run, throwing away the offset the sink had just installed, so every packet
    after the first started at byte 0 and the sink's framing header would clobber
    payload bytes with nothing reported.
    """
    ref = Encoder()
    build_full_scale(ref)
    expected = ref.getvalue()

    for cap, offset in ((16, 4), (32, 1), (64, 8), (11, 1)):
        packets = _stream_with_reserved_header(Encoder, cap, offset)
        assert len(packets) > 1, "buffer too large to force a mid-stream flush"
        _assert_every_packet_reserves(packets, offset, expected)


def test_sink_returning_without_installing_resumes_at_zero():
    """The other half of the same rule: the offset is *consumed* by its
    installation, so a copying sink that returns without installing anything
    resumes at 0 — the reservation is not re-armed behind its back."""
    ref = Encoder()
    build_full_scale(ref)
    expected = ref.getvalue()

    collected = bytearray()
    buf = bytearray(b"\xee" * 16)
    enc = Encoder.over_buffer(buf, 4, collected.extend)
    build_full_scale(enc)
    enc.flush()
    # First unit carries the one-time reservation; everything after it is payload.
    assert bytes(collected[:4]) == b"\xee" * 4
    assert bytes(collected[4:]) == expected


def test_reserve_offset_left_untouched_then_flushed():
    # With offset=4 and no overflow, flush emits the reserved bytes + payload.
    buf = bytearray(64)
    buf[0:4] = b"HDR!"
    enc = Encoder.over_buffer(buf, offset=4)
    enc.write_unsigned(0, 42)
    used = enc.bytes_used()
    assert bytes(buf[0:used]) == b"HDR!" + bytes([0x00, 0x2A])
