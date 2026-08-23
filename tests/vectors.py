"""Shared test helpers and byte vectors transcribed verbatim from the C
reference suite ``corelib-c-cpp/test/c/test_ostream.c`` / ``test_istream.c``.

Every ``expected`` byte sequence below is copied byte-for-byte from the C tests
(the same approach corelib-rs/java/cs take), so passing these proves byte-exact
interop with the reference implementation.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path
from typing import Callable

import pytest

from sofab import Encoder, Field, SofaIncompleteError, Status, Visitor
from sofab.decoder import Decoder as PyDecoder
from sofab.encoder import Encoder as PyEncoder

# --- the shared conformance vectors -----------------------------------------

#: ``assets/test_vectors.json`` — the shared set, copied verbatim from
#: ``corelib-c-cpp``. Loaded once here rather than re-read and re-parsed by every
#: suite that walks it.
VECTORS_PATH = Path(__file__).resolve().parents[1] / "assets" / "test_vectors.json"
VECTOR_DOC = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
VECTORS = VECTOR_DOC["vectors"]

# --- engine parametrisation --------------------------------------------------
#
# The native accelerator is optional by design (``setup.py`` marks the extension
# ``optional=True``), so every engine-parametrised suite runs the pure engine
# always and adds the native one when it is built. Spelling that once here keeps
# the parameter ids identical everywhere and names them as ``sofab.IMPL`` does;
# ``SOFAB_REQUIRE_ENGINE`` (see ``test_engine_guard``) is what stops a missing
# accelerator from silently halving a run.

try:  # pragma: no cover - depends on whether the extension was built
    from sofab import _speedups as _native
except ImportError:  # pragma: no cover - pure-Python-only install
    _native = None  # type: ignore[assignment]

#: ``pytest.param`` lists for the encoder class, the decoder class, and the pair.
ENCODER_ENGINES = [pytest.param(PyEncoder, id="python")]
DECODER_ENGINES = [pytest.param(PyDecoder, id="python")]
ENGINE_PAIRS = [pytest.param(PyEncoder, PyDecoder, id="python")]
if _native is not None:  # pragma: no cover - native-only branch
    ENCODER_ENGINES.append(pytest.param(_native.Encoder, id="native"))
    DECODER_ENGINES.append(pytest.param(_native.Decoder, id="native"))
    ENGINE_PAIRS.append(pytest.param(_native.Encoder, _native.Decoder, id="native"))


def uvarint(value: int) -> bytes:
    """Encode ``value`` as a base-128 little-endian varint (§4.1)."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def zzvarint(value: int) -> bytes:
    """One ZigZag-encoded signed value on the wire (§4.2)."""
    return uvarint((value << 1) ^ (value >> 63))


# IEEE-754 limits matching the C ``FLT_MAX`` / ``DBL_MAX``.
FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
DBL_MAX = sys.float_info.max

# 3.14159265f (float literal promoted to double) — as written by test_write_fp64.
FP64_FROM_FLOAT = struct.unpack("<f", struct.pack("<f", 3.14159265))[0]


class Recorder(Visitor):
    """Records every field the decoder hands over, in order.

    The decoder is push-only, so this is how a test sees a message: feed the
    bytes, read the events back. ``events`` holds one tuple per field —
    ``(kind, field_id, value)`` for a value, ``("seq{", id)`` / ``("seq}",)``
    for sequence framing — and ``fields`` the :class:`Field` each one arrived
    with, for the tests that assert on wire metadata rather than on values.
    """

    def __init__(self, decline: Callable[[Field], bool] | None = None) -> None:
        self.events: list[tuple] = []
        self.fields: list[Field] = []
        self._decline = decline

    def on_field(self, field: Field) -> bool | None:
        self.fields.append(field)
        return False if self._decline is not None and self._decline(field) else None

    def on_sequence_begin(self, field_id):
        self.events.append(("seq{", field_id))
        return None

    def on_sequence_end(self):
        self.events.append(("seq}",))

    def on_unsigned(self, field_id, value):
        self.events.append(("u", field_id, value))

    def on_signed(self, field_id, value):
        self.events.append(("s", field_id, value))

    def on_float32(self, field_id, value):
        self.events.append(("f32", field_id, value))

    def on_float64(self, field_id, value):
        self.events.append(("f64", field_id, value))

    def on_string(self, field_id, value):
        self.events.append(("str", field_id, value))

    def on_bytes(self, field_id, value):
        self.events.append(("blob", field_id, value))

    def on_unsigned_array(self, field_id, values):
        self.events.append(("ua", field_id, tuple(values)))

    def on_signed_array(self, field_id, values):
        self.events.append(("sa", field_id, tuple(values)))

    def on_float32_array(self, field_id, values):
        self.events.append(("f32a", field_id, tuple(values)))

    def on_float64_array(self, field_id, values):
        self.events.append(("f64a", field_id, tuple(values)))


def walk(dec_cls, data: bytes, chunk: int | None = None, **kw):
    """Feed ``data`` through a :class:`Recorder` and return ``(status, rec)``.

    ``chunk`` feeds the message that many bytes at a time — §7.2 item 4 wants
    the outcome to be the same whatever the chunking, so most tests that care
    run both. Anything the decoder raises (a receiver-cap rejection, a caller
    mistake) propagates; a malformed message comes back as ``Status.INVALID``
    with the reason on ``dec.error``.
    """
    rec = kw.pop("recorder", None) or Recorder()
    dec = dec_cls(visitor=rec, **kw)
    status = Status.COMPLETE
    if chunk is None:
        status = dec.feed(data)
    else:
        for off in range(0, len(data), chunk) or [0]:
            status = dec.feed(data[off : off + chunk])
        if not data:
            status = dec.feed(b"")
    return status, rec, dec


def raise_for(status, dec) -> None:
    """Turn a :meth:`feed` outcome back into the exception the removed pull API
    used to raise.

    Callers do not do this — a returned status *is* the contract (§5.2), and
    nothing in ``sofab`` raises for a truncation any more. It exists so the
    suite can keep asserting the **verdict on a message** with
    ``pytest.raises``, which is what most of these tests are actually about;
    rewriting several hundred of them into status comparisons would change what
    they read like without changing what they check.
    """
    if status is Status.INVALID:
        raise dec.error
    if status is Status.INCOMPLETE:
        raise SofaIncompleteError("truncated")


class Slots:
    """Typed views over a binding's ``words`` buffer, plus its ``objects``.

    Reading a bound decode back means casting the one byte buffer to whatever
    the field's kind is; this keeps that out of every test.
    """

    def __init__(self, words: bytearray, objects: list) -> None:
        mv = memoryview(words)
        self.u = mv.cast("Q")
        self.q = mv.cast("q")
        self.d = mv.cast("d")
        self.objects = objects

    def arr_u(self, at: int, n: int) -> list[int]:
        return list(self.u[at : at + n])

    def arr_q(self, at: int, n: int) -> list[int]:
        return list(self.q[at : at + n])

    def arr_d(self, at: int, n: int) -> list[float]:
        return list(self.d[at : at + n])


def bound(dec_cls, data: bytes, binding, chunk: int | None = None, **kw):
    """Feed ``data`` into ``binding``'s destinations; return ``(status, dec, slots)``.

    The destinations are sized from the table, which is the contract: the caller
    owns and sizes them, never the wire (§6.6).
    """
    words = bytearray(binding.tree_words_required * 8)
    objects: list = [None] * binding.tree_objects_required
    dec = dec_cls(binding=binding, words=words, objects=objects, **kw)
    status = Status.COMPLETE
    if chunk is None:
        status = dec.feed(data)
    else:
        for off in range(0, len(data), chunk):
            status = dec.feed(data[off : off + chunk])
    return status, dec, Slots(words, objects)


def pairs(dec_cls, data: bytes, **kw) -> list[tuple]:
    """``[(Field, event), ...]`` for every value field, in wire order.

    ``on_field`` fires only for the fields that carry a value, so the recorder's
    Fields line up one-for-one with its non-sequence events. Tests that assert
    on wire metadata *and* the decoded value want both halves together.
    """
    status, rec, _dec = walk(dec_cls, data, **kw)
    assert status is Status.COMPLETE, status
    vals = [e for e in rec.events if e[0] not in ("seq{", "seq}")]
    assert len(vals) == len(rec.fields), (len(vals), len(rec.fields))
    return list(zip(rec.fields, vals))


def values(dec_cls, data: bytes, **kw) -> list[tuple]:
    """Just the value events of a message that must decode cleanly."""
    status, rec, _dec = walk(dec_cls, data, **kw)
    assert status is Status.COMPLETE, status
    return rec.events


def reader(data: bytes) -> io.BytesIO:
    return io.BytesIO(bytes(data))


def encode(fn) -> bytes:
    """Run ``fn(encoder)`` against a fresh in-memory encoder, return its bytes."""
    enc = Encoder()
    fn(enc)
    return enc.getvalue()


# --- the full-scale example (test_write_full_scale_example) ------------------


def build_full_scale(enc: Encoder) -> None:
    """Reproduce test_write_full_scale_example exactly.

    Every sequence here receives content, so the lazily held-back header is
    committed by the first child write and the bytes are identical to the eager
    framing the C reference emits — which is exactly what makes this a byte-exact
    check of the commit path.
    """
    enc.write_unsigned(0, 200)
    enc.write_signed(1, -100)
    enc.write_unsigned(2, 50000)
    enc.write_signed(3, -20000)
    enc.write_unsigned(4, 3000000000)
    enc.write_signed(5, -1000000000)
    enc.write_unsigned(6, 10000000000000)
    enc.write_signed(7, -5000000000000)

    enc.write_sequence_begin_lazy(10)
    enc.write_float32(0, 3.14)
    enc.write_float64(1, 3.14159265)
    enc.write_string(2, "Hello, World!")
    enc.write_bytes(3, bytes([0xDE, 0xAD, 0xBE, 0xEF]))
    enc.write_sequence_end()

    enc.write_sequence_begin_lazy(100)
    enc.write_unsigned_array(0, [0, 64, 128, 191, 255])
    enc.write_signed_array(1, [-128, -64, 0, 63, 127])
    enc.write_unsigned_array(2, [0, 16384, 32768, 49151, 65535])
    enc.write_signed_array(3, [-32768, -16384, 0, 16383, 32767])
    enc.write_unsigned_array(4, [0, 1073741824, 2147483648, 3221225471, 4294967295])
    enc.write_signed_array(5, [-2147483648, -1073741824, 0, 1073741823, 2147483647])
    enc.write_unsigned_array(
        6,
        [
            0,
            4611686018427387904,
            9223372036854775808,
            13835058055282163711,
            18446744073709551615,
        ],
    )
    enc.write_signed_array(
        7,
        [
            -9223372036854775807,
            -4611686018427387904,
            0,
            4611686018427387903,
            9223372036854775807,
        ],
    )
    enc.write_sequence_begin_lazy(10)
    enc.write_float32_array(0, [1.0, 2.0, 3.0, -FLT_MAX, FLT_MAX])
    enc.write_float64_array(1, [1.0, 2.0, 3.0, -DBL_MAX, DBL_MAX])
    enc.write_sequence_end()
    enc.write_sequence_end()

    enc.write_sequence_begin_lazy(200)
    enc.write_string(0, "Hello, Sofab!")
    enc.write_string(1, "")
    enc.write_string(2, "1234567890")
    enc.write_string(3, "äöüÄÖÜß")
    enc.write_string(4, "This_is_a_very_long_test_string_with_!@#$%^&*()_+-=[]{}")
    enc.write_sequence_end()


# expected bytes — transcribed verbatim from test_write_full_scale_example
FULL_SCALE_EXPECTED = bytes(
    [
        0x00, 0xC8, 0x01, 0x09, 0xC7, 0x01, 0x10, 0xD0, 0x86, 0x03, 0x19, 0xBF,
        0xB8, 0x02, 0x20, 0x80, 0xBC, 0xC1, 0x96, 0x0B, 0x29, 0xFF, 0xA7, 0xD6,
        0xB9, 0x07, 0x30, 0x80, 0xC0, 0xCA, 0xF3, 0x84, 0xA3, 0x02, 0x39, 0xFF,
        0xBF, 0xCA, 0xF3, 0x84, 0xA3, 0x02, 0x56, 0x02, 0x20, 0xC3, 0xF5, 0x48,
        0x40, 0x0A, 0x41, 0xF1, 0xD4, 0xC8, 0x53, 0xFB, 0x21, 0x09, 0x40, 0x12,
        0x6A, 0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x2C, 0x20, 0x57, 0x6F, 0x72, 0x6C,
        0x64, 0x21, 0x1A, 0x23, 0xDE, 0xAD, 0xBE, 0xEF, 0x07, 0xA6, 0x06, 0x03,
        0x05, 0x00, 0x40, 0x80, 0x01, 0xBF, 0x01, 0xFF, 0x01, 0x0C, 0x05, 0xFF,
        0x01, 0x7F, 0x00, 0x7E, 0xFE, 0x01, 0x13, 0x05, 0x00, 0x80, 0x80, 0x01,
        0x80, 0x80, 0x02, 0xFF, 0xFF, 0x02, 0xFF, 0xFF, 0x03, 0x1C, 0x05, 0xFF,
        0xFF, 0x03, 0xFF, 0xFF, 0x01, 0x00, 0xFE, 0xFF, 0x01, 0xFE, 0xFF, 0x03,
        0x23, 0x05, 0x00, 0x80, 0x80, 0x80, 0x80, 0x04, 0x80, 0x80, 0x80, 0x80,
        0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0x0B, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x2C,
        0x05, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0x00,
        0xFE, 0xFF, 0xFF, 0xFF, 0x07, 0xFE, 0xFF, 0xFF, 0xFF, 0x0F, 0x33, 0x05,
        0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x40, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xBF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0x01, 0x3C, 0x05, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0x7F, 0x00, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0xFE,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x56, 0x05, 0x05,
        0x20, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40,
        0x40, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0x7F, 0x0D, 0x05, 0x41,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xEF, 0x7F, 0x07, 0x07, 0xC6, 0x0C, 0x02, 0x6A, 0x48, 0x65,
        0x6C, 0x6C, 0x6F, 0x2C, 0x20, 0x53, 0x6F, 0x66, 0x61, 0x62, 0x21, 0x0A,
        0x02, 0x12, 0x52, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x30, 0x1A, 0x72, 0xC3, 0xA4, 0xC3, 0xB6, 0xC3, 0xBC, 0xC3, 0x84, 0xC3,
        0x96, 0xC3, 0x9C, 0xC3, 0x9F, 0x22, 0xBA, 0x03, 0x54, 0x68, 0x69, 0x73,
        0x5F, 0x69, 0x73, 0x5F, 0x61, 0x5F, 0x76, 0x65, 0x72, 0x79, 0x5F, 0x6C,
        0x6F, 0x6E, 0x67, 0x5F, 0x74, 0x65, 0x73, 0x74, 0x5F, 0x73, 0x74, 0x72,
        0x69, 0x6E, 0x67, 0x5F, 0x77, 0x69, 0x74, 0x68, 0x5F, 0x21, 0x40, 0x23,
        0x24, 0x25, 0x5E, 0x26, 0x2A, 0x28, 0x29, 0x5F, 0x2B, 0x2D, 0x3D, 0x5B,
        0x5D, 0x7B, 0x7D, 0x07,
    ]
)
