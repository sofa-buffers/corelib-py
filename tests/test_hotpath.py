"""Hot-path regression tests.

The encoder/decoder hot paths carry optimizations that are *reformulations* of
the straightforward code: a word-at-a-time varint codec, integer conversion read
straight off CPython's digits, ``Field`` attributes published as slot
descriptors, and fixlen values built directly off the decode buffer. Every one
of them must be indistinguishable from the plain version at the API — same
bytes, same values, same errors, same rejections — so this module pins exactly
that, independently of the shared conformance vectors.

Both engines are exercised wherever both exist, so a divergence introduced on
one side fails here rather than in production on whichever host happens not to
have a compiler.
"""

from __future__ import annotations

import enum
import inspect
import io

import pytest
from vectors import ChunkReader, reader

from sofab.decoder import Decoder as PyDecoder
from sofab.encoder import Encoder as PyEncoder
from sofab.types import (
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaDecodeError,
    SofaRangeError,
    WireType,
)

_speedups = pytest.importorskip("sofab._speedups", reason="native extension not built")
NativeEncoder = _speedups.Encoder
NativeDecoder = _speedups.Decoder

# Stand-in arguments for the keyword-call parity check below, by parameter name.
_KW_SAMPLE = {
    "field_id": 1, "value": 1, "values": [1], "text": "x", "data": b"x",
    "visitor": None, "buffer": bytearray(8), "offset": 0, "flush": None,
}

ENCODERS = [PyEncoder, NativeEncoder]
DECODERS = [PyDecoder, NativeDecoder]

# Every varint length boundary, both sides of it, plus the values that sit
# exactly where the word-at-a-time path changes shape (2**56: the point at
# which the payload no longer fits one 64-bit word, and 2**63).
BOUNDARY_VALUES = sorted(
    {0, 1, 2, 127, 128, 129, 255, 256}
    | {v for k in range(1, 65) for v in ((1 << k) - 1, 1 << k, (1 << k) + 1) if v <= UNSIGNED_MAX}
    | {(1 << (7 * n)) - 1 for n in range(1, 10)}
    | {1 << (7 * n) for n in range(1, 10)}
    | {UNSIGNED_MAX, UNSIGNED_MAX - 1}
)


def _encode_all(cls, write, values):
    enc = cls()
    for v in values:
        write(enc, v)
    enc.flush()
    return enc.getvalue()


def test_unsigned_varint_bytes_match_across_engines():
    """The word-at-a-time codec emits the byte-serial codec's exact bytes."""
    native = _encode_all(NativeEncoder, lambda e, v: e.write_unsigned(1, v), BOUNDARY_VALUES)
    pure = _encode_all(PyEncoder, lambda e, v: e.write_unsigned(1, v), BOUNDARY_VALUES)
    assert native == pure


def test_signed_varint_bytes_match_across_engines():
    values = sorted({0, -1, 1, SIGNED_MIN, SIGNED_MAX, SIGNED_MIN + 1, SIGNED_MAX - 1}
                    | {v - (1 << 62) for v in BOUNDARY_VALUES if v <= UNSIGNED_MAX >> 1})
    native = _encode_all(NativeEncoder, lambda e, v: e.write_signed(2, v), values)
    pure = _encode_all(PyEncoder, lambda e, v: e.write_signed(2, v), values)
    assert native == pure


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("chunk", [0, 1, 3, 7])
def test_varint_boundaries_roundtrip(dec_cls, chunk):
    """Decoding is unaffected by where a chunk boundary falls — the buffered
    fast path and the byte-at-a-time refill path must agree on every value."""
    data = _encode_all(NativeEncoder, lambda e, v: e.write_unsigned(1, v), BOUNDARY_VALUES)
    src = reader(data) if chunk == 0 else ChunkReader(data, chunk)
    dec = dec_cls(src)
    for expected in BOUNDARY_VALUES:
        field = dec.next()
        assert field is not None and field.type == WireType.UNSIGNED
        assert dec.unsigned() == expected
    assert dec.next() is None


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("chunk", [0, 1])
@pytest.mark.parametrize("tenth", [0x02, 0x7F, 0x80, 0xFF])
def test_overlong_varint_rejected(dec_cls, chunk, tenth):
    """A tenth byte above 0x01 carries payload past bit 63 (or continues into an
    eleventh byte): INVALID, on the buffered path and the refilling one alike."""
    data = bytes([0x00]) + bytes([0xFF] * 9) + bytes([tenth])
    src = reader(data) if chunk == 0 else ChunkReader(data, chunk)
    dec = dec_cls(src)
    assert dec.next() is not None
    with pytest.raises(SofaDecodeError):
        dec.unsigned()


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_max_u64_accepted(dec_cls):
    """The 10-byte encoding of 2**64-1 is valid and must not trip the check."""
    data = bytes([0x00]) + bytes([0xFF] * 9) + bytes([0x01])
    dec = dec_cls(reader(data))
    assert dec.next() is not None
    assert dec.unsigned() == UNSIGNED_MAX


# --- integer conversion: the domain edges -----------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", [0, 1, UNSIGNED_MAX, UNSIGNED_MAX - 1, 1 << 63])
def test_unsigned_domain_accepted(enc_cls, value):
    enc = enc_cls()
    enc.write_unsigned(1, value)
    enc.flush()
    dec = NativeDecoder(reader(enc.getvalue()))
    assert dec.next() is not None
    assert dec.unsigned() == value


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", [-1, -(1 << 63), 1 << 64, (1 << 64) + 1, 1 << 200, -(1 << 200)])
def test_unsigned_domain_rejected(enc_cls, value):
    enc = enc_cls()
    with pytest.raises(SofaRangeError):
        enc.write_unsigned(1, value)


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", [0, -1, 1, SIGNED_MIN, SIGNED_MAX])
def test_signed_domain_accepted(enc_cls, value):
    enc = enc_cls()
    enc.write_signed(1, value)
    enc.flush()
    dec = NativeDecoder(reader(enc.getvalue()))
    assert dec.next() is not None
    assert dec.signed() == value


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", [1 << 63, SIGNED_MIN - 1, 1 << 200, -(1 << 200)])
def test_signed_domain_rejected(enc_cls, value):
    enc = enc_cls()
    with pytest.raises(SofaRangeError):
        enc.write_signed(1, value)


# --- what counts as an integer (issue #55) -----------------------------------
#
# Both engines accept exactly what Python accepts where an integer is required:
# an object with __index__. A float has none — it cannot become an integer
# without discarding information — so it is refused rather than truncated.


class _Indexable:
    """Stands in for a third-party integer type (NumPy scalar, etc.)."""

    def __index__(self) -> int:
        return 5


class _FloatOnly:
    """Convertible to a number, but not losslessly to an integer."""

    def __float__(self) -> float:
        return 5.0

    def __int__(self) -> int:      # __int__ is lossy by design; not enough
        return 5


class _Enum(enum.IntEnum):
    FIVE = 5


INTEGRAL = [5, True, False, _Enum.FIVE, _Indexable()]
NON_INTEGRAL = [3.7, 3.0, -0.5, float("nan"), float("inf"), _FloatOnly(), "5", b"5", None, [5]]


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", INTEGRAL)
def test_integral_values_are_accepted(enc_cls, value):
    enc = enc_cls()
    enc.write_unsigned(1, value)
    enc.write_signed(2, value)
    enc.write_unsigned_array(3, [value])
    enc.write_signed_array(4, [value])
    enc.flush()
    dec = NativeDecoder(reader(enc.getvalue()))
    expected = int(value)
    assert (dec.next() is not None) and dec.unsigned() == expected
    assert (dec.next() is not None) and dec.signed() == expected
    assert (dec.next() is not None) and dec.read_unsigned_array() == [expected]
    assert (dec.next() is not None) and dec.read_signed_array() == [expected]


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", NON_INTEGRAL)
def test_non_integral_values_are_refused_not_truncated(enc_cls, value):
    """A value that is not losslessly an integer must never reach the wire as a
    silently truncated one — that would change what the caller asked to send in
    a way the receiver cannot detect."""
    for method, args in (
        ("write_unsigned", (1, value)),
        ("write_signed", (1, value)),
        ("write_unsigned_array", (1, [value])),
        ("write_signed_array", (1, [value])),
        ("write_unsigned", (value, 1)),          # also as a field id
        ("write_sequence_begin_lazy", (value,)),  # the id that is held back
    ):
        enc = enc_cls()
        with pytest.raises(SofaRangeError):
            getattr(enc, method)(*args)


@pytest.mark.parametrize("value", NON_INTEGRAL + [1 << 64, -1])
def test_both_engines_agree_on_every_rejection(value):
    """Whether a value is writable must not depend on which engine is loaded."""
    outcomes = []
    for cls in ENCODERS:
        enc = cls()
        try:
            enc.write_unsigned(1, value)
            enc.flush()
            outcomes.append(("ok", enc.getvalue()))
        except Exception as exc:                 # noqa: BLE001 - the type is the point
            outcomes.append(("raised", type(exc).__name__))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("value", [_Indexable(), 3.7])
def test_integer_rule_holds_over_a_fixed_buffer_too(enc_cls, value):
    """The fixed-buffer model runs its own element loop and its own scalar path;
    both apply the same rule as the growable one."""
    accepted = hasattr(type(value), "__index__")
    for method, args in (
        ("write_unsigned", (1, value)),
        ("write_signed", (1, value)),
        ("write_unsigned_array", (1, [value])),
        ("write_signed_array", (1, [value])),
    ):
        enc = enc_cls.over_buffer(bytearray(64), 0)
        if accepted:
            getattr(enc, method)(*args)
            assert enc.bytes_used() > 0
        else:
            with pytest.raises(SofaRangeError):
                getattr(enc, method)(*args)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_rejection_is_a_sofa_error_so_sticky_mode_latches_it(enc_cls):
    """A refused value has to travel the same path as any other invalid
    argument, or sticky mode would let it escape as a bare TypeError."""
    enc = enc_cls(sticky=True)
    enc.write_unsigned(1, 3.7)
    enc.write_unsigned(2, 5)                     # suppressed: the error latched
    assert isinstance(enc.error, SofaRangeError)
    assert enc.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_array_element_domain_rejected(enc_cls):
    for values in ([1, 1 << 64], [1, -1]):
        enc = enc_cls()
        with pytest.raises(SofaRangeError):
            enc.write_unsigned_array(1, values)
    for values in ([1, 1 << 63], [1, SIGNED_MIN - 1]):
        enc = enc_cls()
        with pytest.raises(SofaRangeError):
            enc.write_signed_array(1, values)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_array_element_domain_rejected_over_fixed_buffer(enc_cls):
    """The fixed-buffer model takes a different element loop; it rejects the
    same values as the growable one."""
    enc = enc_cls.over_buffer(bytearray(64), 0)
    with pytest.raises(SofaRangeError):
        enc.write_unsigned_array(1, [1, 1 << 64])
    enc = enc_cls.over_buffer(bytearray(64), 0)
    with pytest.raises(SofaRangeError):
        enc.write_signed_array(1, [1, 1 << 63])


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_array_bytes_match_over_both_output_models(enc_cls):
    """Growable and fixed-buffer output produce identical bytes."""
    values = [0, 1, 127, 128, (1 << 32) + 7, UNSIGNED_MAX]
    grow = enc_cls()
    grow.write_unsigned_array(1, values)
    grow.write_signed_array(2, [-v // 2 for v in values])
    grow.flush()

    chunks: list[bytes] = []
    fixed = enc_cls.over_buffer(bytearray(8), 0, chunks.append)
    fixed.write_unsigned_array(1, values)
    fixed.write_signed_array(2, [-v // 2 for v in values])
    fixed.flush()
    assert b"".join(chunks) == grow.getvalue()


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_array_input_is_not_consumed_or_mutated(enc_cls):
    """The element loop may read the caller's list directly; it must not alter it."""
    values = [1, 2, 3, UNSIGNED_MAX]
    original = list(values)
    enc = enc_cls()
    enc.write_unsigned_array(1, values)
    enc.write_unsigned_array(2, iter(values))     # a non-list iterable still works
    enc.flush()
    assert values == original
    dec = NativeDecoder(reader(enc.getvalue()))
    for _ in range(2):
        assert dec.next() is not None
        assert dec.read_unsigned_array() == original


# --- fixlen values built off the decode buffer -------------------------------


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("chunk", [0, 1, 5])
def test_fixlen_values_survive_chunking(dec_cls, chunk):
    """String/blob/float reads take a zero-copy path when the payload is already
    buffered and an assembling path when it is not; both must yield the same
    value, and the buffer they borrow from must stay valid afterwards."""
    text = "sofa" * 9 + "äöü€"
    blob = bytes(range(256))
    enc = NativeEncoder()
    enc.write_string(1, text)
    enc.write_bytes(2, blob)
    enc.write_float32(3, 3.14159)
    enc.write_float64(4, 2.718281828459045)
    enc.write_string(5, "")
    enc.write_bytes(6, b"")
    enc.flush()
    data = enc.getvalue()

    src = reader(data) if chunk == 0 else ChunkReader(data, chunk)
    dec = dec_cls(src)
    got = []
    for _ in range(6):
        field = dec.next()
        assert field is not None
        if field.subtype == FixlenSubtype.STRING:
            got.append(dec.string())
        elif field.subtype == FixlenSubtype.BLOB:
            got.append(dec.bytes())
        elif field.subtype == FixlenSubtype.FP32:
            got.append(round(dec.float32(), 5))
        else:
            got.append(dec.float64())
    assert dec.next() is None
    assert got == [text, blob, round(3.14159, 5), 2.718281828459045, "", b""]


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("chunk", [0, 1])
def test_invalid_utf8_string_is_rejected(dec_cls, chunk):
    """Validation applies to the payload however it was assembled."""
    payload = b"\xed\xa0\x80"                       # a surrogate, never valid UTF-8
    data = bytes([0x02, (len(payload) << 3) | FixlenSubtype.STRING]) + payload
    src = reader(data) if chunk == 0 else ChunkReader(data, chunk)
    dec = dec_cls(src)
    assert dec.next() is not None
    with pytest.raises(SofaDecodeError):
        dec.string()


def test_decoded_string_outlives_the_buffer_it_came_from():
    """A decoded value must not alias a buffer the decoder later reuses."""
    enc = NativeEncoder()
    for i in range(64):
        enc.write_string(1, f"value-{i}" * 4)
    enc.flush()
    dec = NativeDecoder(io.BytesIO(enc.getvalue()))
    seen = []
    while dec.next() is not None:
        seen.append(dec.string())
    assert seen == [f"value-{i}" * 4 for i in range(64)]


# --- the Field attribute surface --------------------------------------------


def test_field_attribute_surface():
    """However the five attributes are published, they read back what was stored,
    stay read-only, and keep Field's repr/equality intact."""
    field = _speedups.Field(7, WireType.FIXLEN, 4, 0, FixlenSubtype.FP32)
    assert (field.id, field.type, field.size, field.count, field.subtype) == (
        7, WireType.FIXLEN, 4, 0, FixlenSubtype.FP32)
    assert field == _speedups.Field(7, WireType.FIXLEN, 4, 0, FixlenSubtype.FP32)
    assert field != _speedups.Field(8, WireType.FIXLEN, 4, 0, FixlenSubtype.FP32)
    assert "id=7" in repr(field)
    for name in ("id", "type", "size", "count", "subtype"):
        with pytest.raises(AttributeError):
            setattr(field, name, 1)


def test_decoder_fields_carry_the_expected_metadata():
    enc = NativeEncoder()
    enc.write_unsigned(1, 5)
    enc.write_string(2, "abc")
    enc.write_unsigned_array(3, [1, 2, 3])
    enc.write_float64_array(4, [1.0, 2.0])
    enc.flush()
    dec = NativeDecoder(reader(enc.getvalue()))

    f = dec.next()
    assert (f.id, f.type, f.size, f.count, f.subtype) == (1, WireType.UNSIGNED, 0, 0, None)
    assert dec.unsigned() == 5
    f = dec.next()
    assert (f.id, f.type, f.size, f.subtype) == (2, WireType.FIXLEN, 3, FixlenSubtype.STRING)
    assert dec.string() == "abc"
    f = dec.next()
    assert (f.id, f.type, f.count) == (3, WireType.ARRAY_UNSIGNED, 3)
    assert dec.read_unsigned_array() == [1, 2, 3]
    f = dec.next()
    assert (f.id, f.type, f.count, f.size, f.subtype) == (
        4, WireType.ARRAY_FIXLEN, 2, 8, FixlenSubtype.FP64)
    assert dec.read_float64_array() == [1.0, 2.0]
    assert dec.next() is None


def test_field_stays_valid_after_the_decoder_moves_on():
    """Fields are handed to the caller, so one must not be recycled underneath it."""
    enc = NativeEncoder()
    enc.write_unsigned(1, 1)
    enc.write_unsigned(2, 2)
    enc.flush()
    dec = NativeDecoder(reader(enc.getvalue()))
    first = dec.next()
    assert dec.unsigned() == 1
    second = dec.next()
    assert dec.unsigned() == 2
    assert (first.id, second.id) == (1, 2)
    assert first is not second


# --- the optimizations' own preconditions ------------------------------------


def test_call_shapes_match_the_pure_engine():
    """The calling conventions the accelerator uses are an optimization too — a
    method compiled to a convention that drops keyword support would work
    everywhere the extension is missing and fail where it is present."""
    for pure_cls, native_cls, make in (
        (PyEncoder, NativeEncoder, lambda cls: cls()),
        (PyDecoder, NativeDecoder, lambda cls: cls(reader(b"\x00\x01"))),
    ):
        public = {n for n in dir(pure_cls) if not n.startswith("_")}
        assert public == {n for n in dir(native_cls) if not n.startswith("_")}
        for name in sorted(public):
            attr = getattr(pure_cls, name)
            if not callable(attr):
                continue
            params = [
                p
                for p in inspect.signature(attr).parameters.values()
                if p.name != "self" and p.kind is p.POSITIONAL_OR_KEYWORD
            ]
            if not params:
                continue
            kwargs = {p.name: _KW_SAMPLE.get(p.name, 1) for p in params}
            outcomes = []
            for cls in (pure_cls, native_cls):
                try:
                    getattr(make(cls), name)(**kwargs)
                    outcomes.append("accepted")
                except TypeError as exc:
                    outcomes.append("rejected" if "keyword" in str(exc) else "accepted")
                except Exception:
                    outcomes.append("accepted")  # reached the body: the shape was fine
            assert outcomes[0] == outcomes[1], f"{name} disagrees on keyword arguments"


def test_fast_paths_are_active_on_this_build():
    """Both fast paths verify their own preconditions at import and fall back
    when they do not hold. The fallback is correct but slow, so on a normal
    CPython build it should not be silently in force."""
    assert _speedups.INT_DIGITS_FAST is True
    assert _speedups.FIELD_SLOT_ATTRS is True
