"""Conformance tests driven by the shared, language-agnostic test suite.

Loads ``assets/test_vectors.json`` (copied verbatim from ``corelib-c-cpp`` — the
authoritative ground truth per ARCHITECTURE.md §7) and runs, for every vector,
the full scenario matrix the harness mandates:

1. **encode** — replay the fields, assert the bytes equal ``serialized.hex``.
2. **chunked-encode** — same, but through a fixed buffer of 1/3/7 bytes with a
   flush sink, exercising mid-stream buffer drains.
3. **decode** — parse the bytes, assert the recovered fields.
4. **chunked-decode** — decode fed one byte at a time (streaming worst case).
5. **skip-ids** — decode while auto-skipping the vector's ``skip_ids`` at every
   nesting depth; assert the surviving fields.
6. **chunked skip-ids** — scenario 5 fed one byte at a time.
7. **roundtrip** — encode then decode, assert the fields survive the trip.
8. **requires gating** — vectors are skipped when they need a capability this
   port does not provide (this pure-Python port provides them all).
"""

from __future__ import annotations

import math
import struct

import pytest
from vectors import VECTOR_DOC as _DATA
from vectors import VECTORS, Recorder, raise_for, walk

from sofab import (
    MIN_OUTPUT_BUFFER,
    Decoder,
    Encoder,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    WireType,
)

_IDS = [v["name"] for v in VECTORS]

# WHICH COLUMN THIS REPO ASSERTS: `serialized` — the dense, primitive-layer ground
# truth, every sequence framed, produced by replaying the vector's op list through
# the raw Encoder. That is the only column a corelib *can* produce.
#
# The vectors also carry `serialized_sparse`, the MESSAGE_SPEC §2 form in which a
# sequence-typed FIELD equal to its declared default is omitted (a wrapper-array
# ELEMENT keeps its frame, §5.1). Omitting is a decision made from *values against
# schema defaults*, and this repo has no message layer and no schema — it never
# sees a default, so it cannot emit that form and no test here reads the column.
# `serialized_sparse` is consumed by the generator's conformance drivers
# (sofabgen, `tests/conformance/<lang>/run.sh`), where generated code knows each
# field's default and picks `write_sequence_end` vs `write_sequence_end_keep`.
# What this repo owes that layer is the two closers behaving exactly as specified
# — proven in `tests/test_lazy_framing.py`, not from these vectors.

# Shared negative UTF-8 vectors (issue #85 / MESSAGE_SPEC §8, CORELIB_PLAN §6.4).
# The `invalid_utf8` array is copied verbatim from corelib-c-cpp (the ground
# truth) and tracks corelib-c-cpp#97. Each entry carries `string_hex` (the raw
# invalid payload, for the encode-reject direction) and `serialized_hex` (a whole
# wire message whose string field must decode → INVALID). Python `str` is a
# Unicode string type, so per §6.4 it is ALWAYS strict — SOFAB_STRICT_UTF8 is a
# no-op and omitted; there is no OFF mode to gate on here.
INVALID_UTF8 = _DATA.get("invalid_utf8", [])
_UTF8_IDS = [e["name"] for e in INVALID_UTF8]

_UNSIGNED_ELEMS = {"u8", "u16", "u32", "u64"}
_SIGNED_ELEMS = {"i8", "i16", "i32", "i64"}

#: Capabilities this port implements. A vector whose ``requires`` names anything
#: outside this set is skipped (scenario 8). The pure-Python runtime is
#: full-featured, so in practice nothing is skipped — but the gate keeps the
#: harness honest for footprint-reduced ports that share these vectors.
SUPPORTED = frozenset({"fixlen", "array", "sequence", "fp64", "int64"})


def _fval(v):
    """A JSON float value: a number, or the string literals 'inf' / '-inf'."""
    if isinstance(v, str):
        return {"inf": math.inf, "-inf": -math.inf}[v]
    return float(v)


def _f32(x: float) -> float:
    """The fp32-rounded value, so encode/decode of an fp32 compares exactly."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _check_requires(vec) -> None:
    missing = set(vec.get("requires", ())) - SUPPORTED
    if missing:
        pytest.skip(f"vector requires unsupported capabilities: {sorted(missing)}")


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
            enc.write_sequence_begin_lazy(f["id"])
        elif op == "sequence_end":
            # The op list is replayed against the RAW encoder and compared to the
            # DENSE `serialized` hex, which always carries the frame — including
            # the three empty-sequence vectors. So the close has to be the
            # frame-keeping one; `write_sequence_end` would drop those frames and
            # encode them to nothing. Sparse omission (MESSAGE_SPEC §2) is a
            # message-layer decision, not one this replay makes.
            enc.write_sequence_end_keep()
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


def _encode_chunked(fields, buf_size: int) -> bytes:
    """Encode through a fixed buffer of ``buf_size`` bytes, draining via a flush
    sink each time it fills — the streaming-encoder worst case."""
    chunks: list[bytes] = []
    buf = bytearray(buf_size)
    enc = Encoder.over_buffer(buf, offset=0, flush=lambda c: chunks.append(bytes(c)))
    _replay(enc, fields)
    enc.flush()
    return b"".join(chunks)


# --- decode ------------------------------------------------------------------


def _decode_stream(data: bytes, *, chunk: int | None = None, skip_ids=()):
    """Decode ``data`` into a list of ``(tag, ...)`` tuples.

    ``chunk`` (if set) feeds the decoder that many bytes at a time — use 1 to
    force byte-at-a-time streaming. ``skip_ids`` are declined wherever they
    appear (a declined sequence-start drops its whole sub-tree)."""
    skip = frozenset(skip_ids)
    status, rec, dec = walk(
        Decoder, data, chunk=chunk, recorder=Recorder(decline=lambda f: f.id in skip)
    )
    raise_for(status, dec)
    # This suite's tuple grammar predates the recorder's: sequence framing is
    # ("seq", id) / ("end",), and array values are lists rather than tuples.
    out = []
    for e in rec.events:
        if e[0] == "seq{":
            out.append(("seq", e[1]))
        elif e[0] == "seq}":
            out.append(("end",))
        elif e[0] in ("ua", "sa", "f32a", "f64a"):
            out.append((e[0], e[1], list(e[2])))
        else:
            out.append(e)
    return out


def _map_field(f):
    """The decoded tuple a single value field is expected to produce."""
    op = f["op"]
    if op == "unsigned":
        return ("u", f["id"], f["value"])
    if op == "boolean":  # boolean encodes as unsigned 0/1 on the wire
        return ("u", f["id"], 1 if f["value"] else 0)
    if op == "signed":
        return ("s", f["id"], f["value"])
    if op == "fp32":
        return ("f32", f["id"], _f32(_fval(f["value"])))
    if op == "fp64":
        return ("f64", f["id"], _fval(f["value"]))
    if op == "string":
        return ("str", f["id"], f["value"])
    if op == "blob":
        return ("blob", f["id"], bytes.fromhex(f["value_hex"]))
    if op == "array":
        et, vals = f["element_type"], f["values"]
        if et in _UNSIGNED_ELEMS:
            return ("ua", f["id"], list(vals))
        if et in _SIGNED_ELEMS:
            return ("sa", f["id"], list(vals))
        if et == "fp32":
            return ("f32a", f["id"], [_f32(_fval(x)) for x in vals])
        return ("f64a", f["id"], [_fval(x) for x in vals])
    raise AssertionError(f"unmappable op {op!r}")


def _expected_stream(fields, skip_ids=()):
    """The tuples a decoder should recover, mirroring the same skip semantics as
    ``_decode_stream``: a skipped sequence-start swallows its whole sub-tree."""
    skip = frozenset(skip_ids)
    out = []
    i, n = 0, len(fields)
    while i < n:
        f = fields[i]
        i += 1
        op = f["op"]
        if op == "sequence_end":
            out.append(("end",))
            continue
        if op == "sequence_begin":
            if f["id"] in skip:
                depth = 1  # consume the balanced sub-tree without emitting it
                while depth:
                    g = fields[i]
                    i += 1
                    if g["op"] == "sequence_begin":
                        depth += 1
                    elif g["op"] == "sequence_end":
                        depth -= 1
            else:
                out.append(("seq", f["id"]))
            continue
        if f["id"] in skip:
            continue
        out.append(_map_field(f))
    return out


# --- tests -------------------------------------------------------------------


def test_suite_metadata():
    assert _DATA["format"] == "sofabuffers-test-vectors"
    assert _DATA["version"] == 1
    assert len(VECTORS) >= 67
    # the new capability/skip features are actually exercised by the suite
    assert any("requires" in v for v in VECTORS)
    assert any("skip_ids" in v for v in VECTORS)
    # every capability a vector asks for is one we know how to gate on
    declared = {tag for v in VECTORS for tag in v.get("requires", ())}
    assert declared <= SUPPORTED, f"unknown capability tags: {declared - SUPPORTED}"


# The ±0 float-special vectors store both zeros as JSON `0`, which cannot carry
# the sign of -0.0 present in the authoritative `serialized.hex`. So replaying
# `values` legitimately can't reproduce those exact bytes; assert decoded
# equality instead (-0.0 == 0.0). Byte-exact -0.0 encoding is covered by
# test_vectors_ostream.test_fp32_array_specials_finite_prefix.
_SIGN_OF_ZERO_AMBIGUOUS = {"array_fp32_specials", "array_fp64_specials"}


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_encode(vec):
    _check_requires(vec)
    produced = _encode_vector(vec)
    if vec["name"] in _SIGN_OF_ZERO_AMBIGUOUS:
        expected = bytes.fromhex(vec["serialized"]["hex"])
        assert len(produced) == len(expected)
        assert _decode_stream(produced) == _decode_stream(expected)
    else:
        assert produced.hex() == vec["serialized"]["hex"]


# The smallest size is the port's declared MIN_OUTPUT_BUFFER (CORELIB_PLAN §5.1):
# every vector must come out byte-identical through a buffer of exactly that many
# bytes, which is what makes the declaration a measured value rather than a claim.
@pytest.mark.parametrize("buf_size", [MIN_OUTPUT_BUFFER, 3, 7])
@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_chunked_encode(vec, buf_size):
    _check_requires(vec)
    if vec["name"] in _SIGN_OF_ZERO_AMBIGUOUS:
        pytest.skip("sign-of-zero is unrepresentable in the vector's JSON values")
    produced = _encode_chunked(vec["fields"], buf_size)
    assert produced.hex() == vec["serialized"]["hex"]


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_decode(vec):
    _check_requires(vec)
    data = bytes.fromhex(vec["serialized"]["hex"])
    assert _decode_stream(data) == _expected_stream(vec["fields"])


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_chunked_decode(vec):
    _check_requires(vec)
    data = bytes.fromhex(vec["serialized"]["hex"])
    assert _decode_stream(data, chunk=1) == _expected_stream(vec["fields"])


@pytest.mark.parametrize("vec", VECTORS, ids=_IDS)
def test_vector_roundtrip(vec):
    _check_requires(vec)
    produced = _encode_vector(vec)
    assert _decode_stream(produced) == _expected_stream(vec["fields"])


# --- skip-ids scenarios (only the vectors that declare skip_ids) -------------

_SKIP_VECTORS = [v for v in VECTORS if v.get("skip_ids")]
_SKIP_IDS = [v["name"] for v in _SKIP_VECTORS]


@pytest.mark.parametrize("vec", _SKIP_VECTORS, ids=_SKIP_IDS)
def test_vector_skip_ids(vec):
    _check_requires(vec)
    data = bytes.fromhex(vec["serialized"]["hex"])
    skip = vec["skip_ids"]
    assert _decode_stream(data, skip_ids=skip) == _expected_stream(vec["fields"], skip)


@pytest.mark.parametrize("vec", _SKIP_VECTORS, ids=_SKIP_IDS)
def test_vector_chunked_skip_ids(vec):
    _check_requires(vec)
    data = bytes.fromhex(vec["serialized"]["hex"])
    skip = vec["skip_ids"]
    assert _decode_stream(data, chunk=1, skip_ids=skip) == _expected_stream(vec["fields"], skip)


# --- strict UTF-8 negative vectors (issue #85) -------------------------------
#
# Every `invalid_utf8` entry declares two symmetric strict outcomes (§6.4):
#   * decode: its `serialized_hex` — a whole wire message carrying the bad
#     string field — must decode to INVALID (SofaDecodeError) once the string is
#     actually read. INVALID, not a length/limit error and not INCOMPLETE, even
#     for the payload-internal truncation cases (the wire frame is complete; the
#     UTF-8 sequence inside it is not).
#   * encode: writing the entry's raw `string_hex` bytes as a string field must
#     be refused with InvalidArgument (SofaArgumentError). Python `str` cannot hold
#     non-UTF-8 bytes, so we reconstruct the offending value with
#     `surrogateescape` (each bad byte → a lone surrogate); the strict encoder
#     (`str.encode("utf-8")`, no errors=) then rejects it — exactly the value a
#     byte-container port would hand to encode, mapped into Python's model.


def test_invalid_utf8_suite_present():
    # The copied vectors carry the negative UTF-8 group (tracks corelib-c-cpp#97).
    assert len(INVALID_UTF8) >= 11
    for e in INVALID_UTF8:
        assert e["decode_outcome"] == "invalid"
        assert e["encode_outcome"] == "invalid_argument"
        assert e["group"] == "invalid/utf8"


def _decode_reading_strings(data: bytes, *, chunk: int | None = None):
    """Decode `data`, materializing every STRING field (so UTF-8 validation runs)
    and skipping everything else. Raises SofaDecodeError on an invalid string."""
    def decline(fld):
        return not (fld.type == WireType.FIXLEN and fld.subtype == FixlenSubtype.STRING)

    status, _rec, dec = walk(Decoder, data, chunk=chunk, recorder=Recorder(decline=decline))
    raise_for(status, dec)


@pytest.mark.parametrize("vec", INVALID_UTF8, ids=_UTF8_IDS)
def test_invalid_utf8_decode_is_invalid(vec):
    data = bytes.fromhex(vec["serialized_hex"])
    with pytest.raises(SofaDecodeError):
        _decode_reading_strings(data)


@pytest.mark.parametrize("vec", INVALID_UTF8, ids=_UTF8_IDS)
def test_invalid_utf8_decode_is_invalid_chunked(vec):
    # INVALID must win over INCOMPLETE even fed one byte at a time.
    data = bytes.fromhex(vec["serialized_hex"])
    with pytest.raises(SofaDecodeError):
        _decode_reading_strings(data, chunk=1)


@pytest.mark.parametrize("vec", INVALID_UTF8, ids=_UTF8_IDS)
def test_invalid_utf8_encode_is_invalid_argument(vec):
    bad = bytes.fromhex(vec["string_hex"]).decode("utf-8", "surrogateescape")
    enc = Encoder()
    with pytest.raises(SofaArgumentError):
        enc.write_string(0, bad)


@pytest.mark.parametrize("vec", INVALID_UTF8, ids=_UTF8_IDS)
def test_invalid_utf8_skipped_field_not_validated(vec):
    # Skipped fields are never UTF-8-validated (§6.4): the same message decodes
    # cleanly when its string field is skip()-ped rather than read.
    data = bytes.fromhex(vec["serialized_hex"])
    # Nothing is materialized, so nothing is validated.
    status, _rec, dec = walk(Decoder, data, recorder=Recorder(decline=lambda f: True))
    raise_for(status, dec)
