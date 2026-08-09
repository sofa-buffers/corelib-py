"""Native-accelerator ↔ pure-Python parity.

These tests are the guarantee behind the dual-implementation design: whatever
``sofab.Encoder`` / ``sofab.Decoder`` resolve to at import time, the native
(Cython) classes and the pure-Python classes must be *byte-for-byte* identical
on encode and produce identical decoded values — otherwise the "runs everywhere,
faster where compiled" promise would silently corrupt data on hosts that fall
back to pure Python.

The whole module is skipped when the compiled extension is not available (e.g.
a pure-Python-only install), since there is nothing to compare against.
"""

from __future__ import annotations

import io
import struct

import pytest

from sofab.decoder import Decoder as PyDecoder
from sofab.encoder import Encoder as PyEncoder

_speedups = pytest.importorskip("sofab._speedups", reason="native extension not built")
NativeEncoder = _speedups.Encoder
NativeDecoder = _speedups.Decoder

from sofab import FixlenSubtype, WireType  # noqa: E402

FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]


def _program(enc) -> None:
    """A single program exercising every write path, including edges."""
    enc.write_unsigned(0, 0)
    enc.write_unsigned(1, (1 << 64) - 1)          # max u64 → 10-byte varint
    enc.write_unsigned(2, 0x7F)                    # varint length boundary
    enc.write_unsigned(3, 0x80)
    enc.write_signed(4, 0)
    enc.write_signed(5, -(1 << 63))                # int64 min
    enc.write_signed(6, (1 << 63) - 1)             # int64 max
    enc.write_bool(7, True)
    enc.write_bool(8, False)
    enc.write_float32(9, 3.14159)
    enc.write_float32(10, -FLT_MAX)
    enc.write_float64(11, 2.718281828459045)
    enc.write_string(12, "")                       # empty string
    enc.write_string(13, "äöü€🎉 mixed")            # multibyte utf-8
    enc.write_bytes(14, b"")                        # empty blob
    enc.write_bytes(15, bytes(range(256)))          # full byte range
    enc.write_unsigned_array(16, [])                # empty varint array
    enc.write_unsigned_array(17, [0, 1, 127, 128, (1 << 64) - 1])
    enc.write_signed_array(18, [-(1 << 63), -1, 0, 1, (1 << 63) - 1])
    enc.write_float32_array(19, [])                 # empty fixlen array (still carries fixlen_word)
    enc.write_float32_array(20, [1.0, -2.0, 3.5])
    enc.write_float64_array(21, [])
    enc.write_float64_array(22, [1e300, -1e-300, 0.0])
    enc.write_sequence_begin_lazy(23)
    enc.write_unsigned(1, 99)
    enc.write_sequence_begin_lazy(2)
    enc.write_signed(1, -7)
    enc.write_sequence_end()
    enc.write_sequence_end()


def test_encode_byte_identical():
    py, na = PyEncoder(), NativeEncoder()
    _program(py)
    _program(na)
    py.flush()
    na.flush()
    assert na.getvalue() == py.getvalue()


def _encode_over_buffer(enc_cls, cap):
    """Encode `_program` through a fixed `cap`-byte buffer whose sink installs a
    FRESH buffer each time, which is the pattern `over_buffer` documents."""
    acc = bytearray()
    enc = None

    def sink(chunk):
        acc.extend(chunk)
        enc.buffer_set(bytearray(cap))

    enc = enc_cls.over_buffer(bytearray(cap), 0, sink)
    _program(enc)
    enc.flush()
    return bytes(acc)


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 8, 16, 64])
def test_over_buffer_byte_identical_with_buffer_set_sink(cap):
    """A caller buffer smaller than the message must produce the one-shot bytes, in
    both engines, at every size — including sizes that force a drain *inside* one
    write.

    The pure engine hoisted the buffer view out of its write loop and kept using it
    after a drain, so once the sink installed a fresh buffer via ``buffer_set``
    everything past the first flush landed in the orphaned one and the fresh buffer
    was emitted zeroed. It surfaced only at small ``cap``; the native engine was
    correct throughout.

    A sink that drains and *reuses* the same buffer was never affected, which is
    what the existing ``over_buffer`` tests do — hence this one installs a new
    buffer, the pattern ``over_buffer``'s own docstring describes.
    """
    ref = PyEncoder()
    _program(ref)
    ref.flush()
    want = ref.getvalue()

    assert _encode_over_buffer(PyEncoder, cap) == want
    assert _encode_over_buffer(NativeEncoder, cap) == want


def _encode_reserving_header(enc_cls, cap, offset, filler=0xEE):
    """Encode `_program` through `cap`-byte buffers whose sink *takes* the buffer it
    was handed and installs a replacement reserving `offset` bytes of framing room —
    the take-and-replace shape of CORELIB_PLAN §5.1."""
    packets = []
    enc = None

    def sink(chunk):
        packets.append(bytes(chunk))
        enc.buffer_set(bytearray([filler]) * cap, offset)

    enc = enc_cls.over_buffer(bytearray([filler]) * cap, offset, sink)
    _program(enc)
    enc.flush()
    return packets


@pytest.mark.parametrize("cap,offset", [(12, 1), (16, 4), (24, 4), (64, 8), (100, 16)])
def test_over_buffer_sink_reinstalled_offset_survives_the_drain(cap, offset):
    """§5.1: the start offset belongs to the installation, not to the buffer — a sink
    that re-arms it on every flush gets header room in *every* packet.

    Both engines used to drop it: the drain reset the cursor to 0 unconditionally
    *after* the sink had run, discarding the offset the sink had just installed, so
    every packet but the first began at byte 0 and the sink's framing header would
    overwrite ``offset`` payload bytes per packet, silently.
    """
    ref = PyEncoder()
    _program(ref)
    ref.flush()
    want = ref.getvalue()

    head = bytes([0xEE]) * offset
    for enc_cls in (PyEncoder, NativeEncoder):
        packets = _encode_reserving_header(enc_cls, cap, offset)
        assert len(packets) > 1, "buffer too large to force a mid-stream flush"
        for i, packet in enumerate(packets):
            assert packet[:offset] == head, f"{enc_cls.__name__} packet {i}: reservation lost"
        assert b"".join(p[offset:] for p in packets) == want

    # ... and both engines cut the stream in exactly the same places.
    assert _encode_reserving_header(PyEncoder, cap, offset) == _encode_reserving_header(
        NativeEncoder, cap, offset
    )


def _walk(dec):
    out = []
    while (f := dec.next()) is not None:
        t = f.type
        if t == WireType.SEQUENCE_END:
            out.append(("end",))
        elif t == WireType.SEQUENCE_START:
            out.append(("seq", f.id))
        elif t == WireType.UNSIGNED:
            out.append(("u", f.id, dec.unsigned()))
        elif t == WireType.SIGNED:
            out.append(("s", f.id, dec.signed()))
        elif t == WireType.FIXLEN:
            st = f.subtype
            if st == FixlenSubtype.FP32:
                out.append(("f32", f.id, dec.float32()))
            elif st == FixlenSubtype.FP64:
                out.append(("f64", f.id, dec.float64()))
            elif st == FixlenSubtype.STRING:
                out.append(("str", f.id, dec.string()))
            else:
                out.append(("blob", f.id, dec.bytes()))
        elif t == WireType.ARRAY_UNSIGNED:
            out.append(("ua", f.id, dec.read_unsigned_array()))
        elif t == WireType.ARRAY_SIGNED:
            out.append(("sa", f.id, dec.read_signed_array()))
        elif t == WireType.ARRAY_FIXLEN:
            if f.subtype == FixlenSubtype.FP32:
                out.append(("f32a", f.id, dec.read_float32_array()))
            else:
                out.append(("f64a", f.id, dec.read_float64_array()))
    return out


def test_decode_values_identical():
    enc = NativeEncoder()
    _program(enc)
    enc.flush()
    data = enc.getvalue()
    assert _walk(NativeDecoder(io.BytesIO(data))) == _walk(PyDecoder(io.BytesIO(data)))


def test_cross_decode():
    """Native encodes → pure decodes, and pure encodes → native decodes."""
    pe, ne = PyEncoder(), NativeEncoder()
    _program(pe)
    _program(ne)
    pe.flush()
    ne.flush()
    pd = pe.getvalue()
    nd = ne.getvalue()
    assert _walk(PyDecoder(io.BytesIO(nd))) == _walk(NativeDecoder(io.BytesIO(pd)))


# --- parity on *malformed* input, not just on valid programs (issue #64) -----
#
# Byte-identical encoding and identical decoded values are only half the
# promise: the two engines must also reject the same bytes. The 64-bit varint
# bound of §4.1 is where they drifted apart — the array read loops inline the
# codec, and the pure one was missing the guard, so the same message decoded to
# a value on a pure-Python host and to INVALID on a compiled one. Walk each
# position a varint can appear in and compare the verdicts, not just the values.

_OVER_64 = [0xFF] * 9 + [0x7F]          # tenth byte's payload lands at bit >= 64
_ELEVEN_BYTES = [0xFF] * 10 + [0x00]    # too long even with a zero surplus byte
_MAX_U64 = [0xFF] * 9 + [0x01]          # the legal boundary: 2^64-1


@pytest.mark.parametrize(
    "data",
    [
        [(1 << 3) | WireType.UNSIGNED] + _OVER_64,                       # scalar value
        [(1 << 3) | WireType.SIGNED] + _OVER_64,
        [(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _OVER_64,           # array element
        [(1 << 3) | WireType.ARRAY_SIGNED, 0x01] + _OVER_64,
        [(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _ELEVEN_BYTES,
        [(1 << 3) | WireType.ARRAY_UNSIGNED, 0x02, 0x01] + _OVER_64,     # behind a good one
        [(1 << 3) | WireType.ARRAY_UNSIGNED] + _OVER_64,                 # element count
        _OVER_64,                                                        # field header
        [(1 << 3) | WireType.ARRAY_UNSIGNED, 0x01] + _MAX_U64,           # legal control
        [(1 << 3) | WireType.ARRAY_SIGNED, 0x01] + _MAX_U64,
    ],
    ids=[
        "scalar-unsigned", "scalar-signed", "elem-unsigned", "elem-signed",
        "elem-eleven-bytes", "elem-behind-valid", "array-count", "field-header",
        "elem-max-u64", "elem-min-i64",
    ],
)
def test_64_bit_bound_verdicts_identical(data):
    def verdict(decoder_cls):
        try:
            return ("ok", _walk(decoder_cls(io.BytesIO(bytes(data)))))
        except Exception as exc:  # the verdict IS the result under test here
            return ("raise", type(exc).__name__)

    assert verdict(PyDecoder) == verdict(NativeDecoder)


def test_active_impl_is_consistent():
    import sofab

    assert sofab.IMPL in {"native", "python"}
    if sofab.IMPL == "native":
        assert sofab.Encoder is NativeEncoder
        assert sofab.Decoder is NativeDecoder
