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
    enc.write_sequence_begin(23)
    enc.write_unsigned(1, 99)
    enc.write_sequence_begin(2)
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


def test_active_impl_is_consistent():
    import sofab

    assert sofab.IMPL in {"native", "python"}
    if sofab.IMPL == "native":
        assert sofab.Encoder is NativeEncoder
        assert sofab.Decoder is NativeDecoder
