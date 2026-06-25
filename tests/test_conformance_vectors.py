"""Conformance tests driven by the shared, language-agnostic test suite.

Loads ``assets/test_vectors.json`` (copied verbatim from the `documentation`
repo — the authoritative ground truth per ARCHITECTURE.md §7) and asserts, for
every vector, that the encoder produces exactly ``serialized.hex`` and that the
decoder recovers the original ``fields``.
"""

from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path

import pytest

from sofab import Decoder, Encoder, FixlenSubtype, WireType

VECTORS_PATH = Path(__file__).resolve().parents[1] / "assets" / "test_vectors.json"
_DATA = json.loads(VECTORS_PATH.read_text())
VECTORS = _DATA["vectors"]
_IDS = [v["name"] for v in VECTORS]

_UNSIGNED_ELEMS = {"u8", "u16", "u32", "u64"}
_SIGNED_ELEMS = {"i8", "i16", "i32", "i64"}


def _fval(v):
    """A JSON float value: a number, or the string literals 'inf' / '-inf'."""
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf}[v]
    return float(v)


def _f32(x: float) -> float:
    """The fp32-rounded value, so encode/decode of an fp32 compares exactly."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# --- encode ------------------------------------------------------------------


def _replay(enc: Encoder, fields) -> None:
    for f in fields:
        op = f["op"]
        if op == "unsigned":
            enc.write_unsigned(f["id"], f["value"])
        elif op == "signed":
            enc.write_signed(f["id"], f["value"])
        elif op == "boolean":
            enc.write_bool(f["id"], f["value"])
        elif op == "fp32":
            enc.write_float32(f["id"], _fval(f["value"]))
        elif op == "fp64":
            enc.write_float64(f["id"], _fval(f["value"]))
        elif op == "string":
            enc.write_string(f["id"], f["value"])
        elif op == "blob":
            enc.write_bytes(f["id"], bytes.fromhex(f["value_hex"]))
        elif op == "array":
            et, vals = f["element_type"], f["values"]
            if et in _UNSIGNED_ELEMS:
                enc.write_unsigned_array(f["id"], vals)
            elif et in _SIGNED_ELEMS:
                enc.write_signed_array(f["id"], vals)
            elif et == "fp32":
                enc.write_float32_array(f["id"], [_fval(x) for x in vals])
            elif et == "fp64":
                enc.write_float64_array(f["id"], [_fval(x) for x in vals])
            else:
                raise AssertionError(f"unknown element_type {et!r}")
        elif op == "sequence_begin":
            enc.write_sequence_begin(f["id"])
        elif op == "sequence_end":
            enc.write_sequence_end()
        else:
            raise AssertionError(f"unknown op {op!r}")


def _encode_vector(vec) -> bytes:
    offset = vec.get("offset", 0)
    if offset == 0:
        enc = Encoder()
        _replay(enc, vec["fields"])
        return enc.getvalue()
    # offset > 0: reserve `offset` bytes at the front, return the payload region
    buf = bytearray(offset + vec["serialized"]["length"] + 16)
    enc = Encoder.over_buffer(buf, offset=offset)
    _replay(enc, vec["fields"])
    return bytes(buf[offset : enc.bytes_used()])


# --- decode ------------------------------------------------------------------


def _decode_stream(data: bytes):
    dec = Decoder(io.BytesIO(data))
    out = []
    while (fld := dec.next()) is not None:
        t = fld.type
        if t == WireType.UNSIGNED:
            out.append(("u", fld.id, dec.unsigned()))
        elif t == WireType.SIGNED:
            out.append(("s", fld.id, dec.signed()))
        elif t == WireType.FIXLEN:
            st = fld.subtype
            if st == FixlenSubtype.FP32:
                out.append(("f32", fld.id, dec.float32()))
            elif st == FixlenSubtype.FP64:
                out.append(("f64", fld.id, dec.float64()))
            elif st == FixlenSubtype.STRING:
                out.append(("str", fld.id, dec.string()))
            else:
                out.append(("blob", fld.id, dec.bytes()))
        elif t == WireType.ARRAY_UNSIGNED:
            out.append(("ua", fld.id, dec.read_unsigned_array()))
        elif t == WireType.ARRAY_SIGNED:
            out.append(("sa", fld.id, dec.read_signed_array()))
        elif t == WireType.ARRAY_FIXLEN:
            if fld.subtype == FixlenSubtype.FP32:
                out.append(("f32a", fld.id, dec.read_float32_array()))
            else:
                out.append(("f64a", fld.id, dec.read_float64_array()))
        elif t == WireType.SEQUENCE_START:
            out.append(("seq", fld.id))
        else:  # SEQUENCE_END
            out.append(("end",))
    return out


def _expected_stream(fields):
    out = []
    for f in fields:
        op = f["op"]
        if op == "unsigned":
            out.append(("u", f["id"], f["value"]))
        elif op == "boolean":  # boolean encodes as unsigned 0/1 on the wire
            out.append(("u", f["id"], 1 if f["value"] else 0))
        elif op == "signed":
            out.append(("s", f["id"], f["value"]))
        elif op == "fp32":
            out.append(("f32", f["id"], _f32(_fval(f["value"]))))
        elif op == "fp64":
            out.append(("f64", f["id"], _fval(f["value"])))
        elif op == "string":
            out.append(("str", f["id"], f["value"]))
        elif op == "blob":
            out.append(("blob", f["id"], bytes.fromhex(f["value_hex"])))
        elif op == "array":
            et, vals = f["element_type"], f["values"]
            if et in _UNSIGNED_ELEMS:
                out.append(("ua", f["id"], list(vals)))
            elif et in _SIGNED_ELEMS:
                out.append(("sa", f["id"], list(vals)))
            elif et == "fp32":
                out.append(("f32a", f["id"], [_f32(_fval(x)) for x in vals]))
            elif et == "fp64":
                out.append(("f64a", f["id"], [_fval(x) for x in vals]))
        elif op == "sequence_begin":
            out.append(("seq", f["id"]))
        elif op == "sequence_end":
            out.append(("end",))
    return out


# --- tests -------------------------------------------------------------------


def test_suite_metadata():
    assert _DATA["format"] == "sofabuffers-test-vectors"
    assert _DATA["version"] == 1
    assert len(VECTORS) >= 40


# The ±0 float-special vectors store both zeros as JSON `0`, which cannot carry
# the sign of -0.0 present in the authoritative `serialized.hex`. So replaying
# `values` legitimately can't reproduce those exact bytes; assert decoded
# equality instead (-0.0 == 0.0). Byte-exact -0.0 encoding is covered by
# test_vectors_ostream.test_fp32_array_specials_finite_prefix.
_SIGN_OF_ZERO_AMBIGUOUS = {"array_fp32_specials", "array_fp64_specials"}


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_encode(vec):
    produced = _encode_vector(vec)
    if vec["name"] in _SIGN_OF_ZERO_AMBIGUOUS:
        expected = bytes.fromhex(vec["serialized"]["hex"])
        assert len(produced) == len(expected)
        assert _decode_stream(produced) == _decode_stream(expected)
    else:
        assert produced.hex() == vec["serialized"]["hex"]


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_decode(vec):
    data = bytes.fromhex(vec["serialized"]["hex"])
    assert _decode_stream(data) == _expected_stream(vec["fields"])
