"""Streaming tests: 1-byte-granularity decode and tiny-scratch-buffer encode
must match the one-shot path."""

from __future__ import annotations

import pytest
from vectors import FULL_SCALE_EXPECTED, build_full_scale, values

import sofab
from sofab import (
    MIN_OUTPUT_BUFFER,
    Decoder,
    Encoder,
    SofaArgumentError,
    SofaBufferError,
)
from sofab.encoder import Encoder as PyEncoder

try:  # the native accelerator is optional — exercise whichever engines exist
    from sofab._speedups import Encoder as NativeEncoder
except ImportError:  # pragma: no cover - pure-Python-only install
    ENCODERS = [PyEncoder]
else:
    ENCODERS = [PyEncoder, NativeEncoder]


def test_decode_one_byte_at_a_time_matches_oneshot():
    """§7.2 item 4: where the chunk boundaries fall must not change anything."""
    oneshot = values(Decoder, FULL_SCALE_EXPECTED)
    streamed = values(Decoder, FULL_SCALE_EXPECTED, chunk=1)
    assert streamed == oneshot


def test_min_output_buffer_is_declared_and_within_the_ceiling():
    """CORELIB_PLAN §5.1: the port MUST expose a documented constant — the
    smallest buffer it accepts for streaming — and the declaration MUST NOT
    exceed 20. This port splits every atomic unit at any byte boundary, so the
    value it may declare is 1."""
    assert isinstance(MIN_OUTPUT_BUFFER, int)
    assert 1 <= MIN_OUTPUT_BUFFER <= 20
    assert "MIN_OUTPUT_BUFFER" in sofab.__all__


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("cap", [MIN_OUTPUT_BUFFER, 7])
def test_encode_through_min_output_buffer_matches_oneshot(enc_cls, cap):
    """§7.2 item 4: encode into a buffer of exactly ``MIN_OUTPUT_BUFFER`` bytes,
    driving the sink repeatedly; the concatenation must be byte-identical to the
    one-shot output. ``build_full_scale`` carries strings and blobs longer than
    the buffer, so the divisible-run split is exercised too."""
    # one-shot reference
    ref = enc_cls()
    build_full_scale(ref)
    expected = ref.getvalue()

    collected = bytearray()
    enc = enc_cls.over_buffer(bytearray(cap), offset=0, flush=collected.extend)
    build_full_scale(enc)
    enc.flush()
    assert bytes(collected) == expected


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_buffer_below_the_minimum_is_rejected_where_it_is_handed_over(enc_cls):
    """§5.1: the minimum binds a buffer installed **with** a flush sink, at
    installation and at every mid-stream buffer-set, and such a buffer is
    rejected *where it is handed over* rather than partway through a message."""
    usable = MIN_OUTPUT_BUFFER - 1  # one byte short of the floor
    sink = bytearray().extend

    with pytest.raises(SofaArgumentError):
        enc_cls.over_buffer(bytearray(usable), 0, sink)
    # the same shortfall produced by the start offset rather than by the length
    with pytest.raises(SofaArgumentError):
        enc_cls.over_buffer(bytearray(16), 16 - usable, sink)
    # a mid-stream buffer-set is the same handover and rejects there too
    enc = enc_cls.over_buffer(bytearray(16), 0, sink)
    with pytest.raises(SofaArgumentError):
        enc.buffer_set(bytearray(16), 16 - usable)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_undersized_buffer_without_a_sink_is_accepted(enc_cls):
    """§5.1: "a buffer installed without a sink is subject to no minimum" — no
    flush can occur, so the constant has nothing to say and the buffer either
    holds the message or reports buffer-full.

    The floor used to be applied to this population instead: ``buffer_set``
    rejected ``offset == len(buffer)`` unconditionally, so a sink-less buffer
    with zero usable bytes failed by accident while an undersized *sink*-backed
    one was never checked against a stated rule.
    """
    usable = MIN_OUTPUT_BUFFER - 1

    for buffer, offset in ((bytearray(usable), 0), (bytearray(16), 16 - usable)):
        enc = enc_cls.over_buffer(buffer, offset)  # accepted: no sink
        with pytest.raises(SofaBufferError):  # ...and reports buffer-full
            enc.write_string(0, "x" * 64)
        enc = enc_cls.over_buffer(bytearray(16), 0)
        enc.buffer_set(buffer, offset)  # mid-stream, also unconstrained


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_message_sized_buffer_stays_exact_without_a_sink(enc_cls):
    """§5.1: sizing from the generated ``MAX_SIZE`` stays exact — "a message that
    encodes to two bytes may be encoded into a two-byte buffer on any port,
    whatever that port declares"."""
    buf = bytearray(2)
    enc = enc_cls.over_buffer(buf, 0)
    enc.write_unsigned(0, 42)
    assert enc.bytes_used() == 2
    assert bytes(buf) == b"\x00\x2a"


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
