"""MESSAGE_SPEC §7.3 / CORELIB_PLAN §6.3 — a type-mismatched read is not an error.

A read whose declared type contradicts the field on the wire **MUST** be handled
exactly like a field with an unknown id: the field is skipped, the caller's
destination is left untouched, and a decode that meets nothing else stays
COMPLETE. For a value-returning Python API "destination untouched" is ``None``,
returned without consuming anything — so the value is still pending and the
following ``next()`` (or an explicit ``skip()``) discards it.

The other half of what the removed ``SofaStateError`` used to cover is a genuine
caller mistake — a read with **no** pending value at all — for which §6.3 has one
code, ``InvalidArgument`` → :class:`SofaRangeError`.

Both engines carry independent copies of these checks, so every test runs on the
pure-Python classes *and* on the compiled accelerator when it is present: a split
here is a conformance divergence the differential fuzzer would report.
"""

from __future__ import annotations

import io

import pytest
from vectors import ChunkReader

from sofab.decoder import Decoder as PyDecoder
from sofab.encoder import Encoder as PyEncoder
import sofab
from sofab.types import SofaError, SofaLimitError, SofaRangeError

_ENGINES = [(PyEncoder, PyDecoder)]
try:  # the native accelerator, when compiled in, must behave identically
    from sofab import _speedups as _sp

    _ENGINES.append((_sp.Encoder, _sp.Decoder))
except ImportError:  # pragma: no cover - pure-Python-only install
    pass

engine = pytest.mark.parametrize(
    "Encoder,Decoder", _ENGINES, ids=["python", "native"][: len(_ENGINES)]
)


def _decoder(Decoder, build):
    enc = PyEncoder()
    build(enc)
    return Decoder(io.BytesIO(enc.getvalue()))


# --- the taxonomy -----------------------------------------------------------


def test_no_invalid_usage_class_exists():
    """§6.3 fixes the taxonomy at five codes and has none for "invalid usage".
    The old class is gone and so is the alias that briefly stood in for it, so a
    caller mistake is a :class:`SofaRangeError` (§6.3 ``InvalidArgument``) and
    nothing else names the removed category."""
    assert issubclass(SofaRangeError, SofaError)
    assert not hasattr(sofab, "SofaStateError")


# --- §7.3: every wrong-type read, on every wire kind ------------------------

# (write the field, read it with a contradicting type, read it correctly)
_MISMATCHES = [
    ("unsigned", lambda e: e.write_unsigned(1, 5), lambda d: d.signed(), lambda d: d.unsigned(), 5),
    ("signed", lambda e: e.write_signed(1, -5), lambda d: d.unsigned(), lambda d: d.signed(), -5),
    ("bool", lambda e: e.write_signed(1, -5), lambda d: d.bool(), lambda d: d.signed(), -5),
    ("fp32", lambda e: e.write_float32(1, 1.5), lambda d: d.float64(), lambda d: d.float32(), 1.5),
    ("fp64", lambda e: e.write_float64(1, 1.5), lambda d: d.float32(), lambda d: d.float64(), 1.5),
    ("string", lambda e: e.write_string(1, "hi"), lambda d: d.bytes(), lambda d: d.string(), "hi"),
    ("blob", lambda e: e.write_bytes(1, b"hi"), lambda d: d.string(), lambda d: d.bytes(), b"hi"),
    ("string/scalar", lambda e: e.write_string(1, "hi"), lambda d: d.unsigned(), lambda d: d.string(), "hi"),
    ("uarray", lambda e: e.write_unsigned_array(1, [1, 2]), lambda d: d.read_signed_array(), lambda d: d.read_unsigned_array(), [1, 2]),
    ("iarray", lambda e: e.write_signed_array(1, [-1]), lambda d: d.read_unsigned_array(), lambda d: d.read_signed_array(), [-1]),
    ("uarray/scalar", lambda e: e.write_unsigned_array(1, [1, 2]), lambda d: d.unsigned(), lambda d: d.read_unsigned_array(), [1, 2]),
    ("fp32array", lambda e: e.write_float32_array(1, [1.5]), lambda d: d.read_float64_array(), lambda d: d.read_float32_array(), [1.5]),
    ("fp64array", lambda e: e.write_float64_array(1, [1.5]), lambda d: d.read_float32_array(), lambda d: d.read_float64_array(), [1.5]),
    ("fp32array/varray", lambda e: e.write_float32_array(1, [1.5]), lambda d: d.read_unsigned_array(), lambda d: d.read_float32_array(), [1.5]),
    # An *empty* fixlen array still carries its fixlen_word (§4.8), so its
    # subtype is known and can contradict the read like any other field.
    ("fp32array/empty", lambda e: e.write_float32_array(1, []), lambda d: d.read_float64_array(), lambda d: d.read_float32_array(), []),
]
mismatch = pytest.mark.parametrize(
    "write,wrong,right,value", [m[1:] for m in _MISMATCHES], ids=[m[0] for m in _MISMATCHES]
)


@engine
@mismatch
def test_wrong_type_read_returns_none(Encoder, Decoder, write, wrong, right, value):
    dec = _decoder(Decoder, write)
    dec.next()
    assert wrong(dec) is None


@engine
@mismatch
def test_wrong_type_read_consumes_nothing(Encoder, Decoder, write, wrong, right, value):
    """The refusal leaves the value pending, so the field is still there — a
    caller that knows better can read it with the type the wire carries."""
    dec = _decoder(Decoder, write)
    dec.next()
    assert wrong(dec) is None
    assert right(dec) == value
    assert dec.next() is None


@engine
@mismatch
def test_wrong_type_read_leaves_the_decode_complete(Encoder, Decoder, write, wrong, right, value):
    """Ignoring the ``None`` is the §7.3 case proper: the field is skipped like
    an unknown id by the next ``next()``, the fields around it decode normally,
    and the stream ends at clean EOF rather than INVALID or INCOMPLETE."""

    def build(e):
        write(e)
        e.write_unsigned(2, 7)

    dec = _decoder(Decoder, build)
    dec.next()
    assert wrong(dec) is None
    f = dec.next()  # skips the refused field
    assert f is not None and f.id == 2
    assert dec.unsigned() == 7
    assert dec.next() is None


@engine
@mismatch
def test_wrong_type_read_can_be_skipped_explicitly(Encoder, Decoder, write, wrong, right, value):
    dec = _decoder(Decoder, write)
    dec.next()
    assert wrong(dec) is None
    dec.skip()  # the value is still pending, so skip() takes it
    assert dec.next() is None


@engine
def test_wrong_type_read_inside_a_sequence(Encoder, Decoder):
    """§7.3 applies at every nesting level, and the skipped field must not
    disturb the sequence framing."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_string(2, "hi")
        e.write_unsigned(3, 9)
        e.write_sequence_end()

    dec = _decoder(Decoder, build)
    dec.next()  # sequence start
    f = dec.next()
    assert f.id == 2
    assert dec.float64() is None
    f = dec.next()
    assert f.id == 3
    assert dec.unsigned() == 9
    assert dec.next() is not None  # sequence end
    assert dec.next() is None


@engine
def test_fixlen_len_is_a_peek_that_answers_none(Encoder, Decoder):
    dec = _decoder(Decoder, lambda e: e.write_string(1, "hello"))
    dec.next()
    assert dec.fixlen_len() == 5
    assert dec.float32() is None  # a refused read does not disturb the peek
    assert dec.fixlen_len() == 5
    assert dec.string() == "hello"


# --- §6.3: no pending value at all is the caller's mistake ------------------


@engine
def test_read_before_next_raises(Encoder, Decoder):
    dec = _decoder(Decoder, lambda e: e.write_unsigned(1, 5))
    with pytest.raises(SofaRangeError):
        dec.unsigned()


@engine
def test_second_read_of_one_field_raises(Encoder, Decoder):
    dec = _decoder(Decoder, lambda e: e.write_string(1, "hi"))
    dec.next()
    assert dec.string() == "hi"
    with pytest.raises(SofaRangeError):
        dec.string()
    with pytest.raises(SofaRangeError):
        dec.fixlen_len()


@engine
def test_read_on_a_sequence_frame_raises(Encoder, Decoder):
    """A sequence start/end carries no value, so there is nothing pending for a
    typed read to contradict — this is the caller-mistake half, not §7.3."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_unsigned(2, 5)
        e.write_sequence_end_keep()

    dec = _decoder(Decoder, build)
    dec.next()  # sequence start
    with pytest.raises(SofaRangeError):
        dec.unsigned()
    dec.next()
    assert dec.unsigned() == 5
    dec.next()  # sequence end
    with pytest.raises(SofaRangeError):
        dec.read_unsigned_array()


@engine
def test_read_at_eof_raises(Encoder, Decoder):
    dec = _decoder(Decoder, lambda e: e.write_unsigned(1, 5))
    dec.next()
    assert dec.unsigned() == 5
    assert dec.next() is None
    with pytest.raises(SofaRangeError):
        dec.bytes()
    with pytest.raises(SofaRangeError):
        dec.read_float32_array()


# --- §6.3 on the encode side ------------------------------------------------


@engine
def test_sequence_end_without_begin_is_invalid_argument(Encoder, Decoder):
    with pytest.raises(SofaRangeError):
        Encoder().write_sequence_end()
    with pytest.raises(SofaRangeError):
        Encoder().write_sequence_end_keep()


@engine
def test_getvalue_on_a_caller_owned_buffer_is_invalid_argument(Encoder, Decoder):
    enc = Encoder.over_buffer(bytearray(16), offset=0)
    enc.write_unsigned(1, 7)
    with pytest.raises(SofaRangeError):
        enc.getvalue()


@engine
def test_sticky_mode_latches_the_invalid_argument(Encoder, Decoder):
    enc = Encoder(sticky=True)
    enc.write_sequence_end()  # no matching begin → latched, not raised
    assert isinstance(enc.error, SofaRangeError)


def test_array_shrank_mid_encode_is_invalid_argument():
    """Native-engine-only site: the pure encoder materialises its own element
    list, so only the accelerator can observe the caller's list shrinking under
    it (a ``__index__`` that mutates it)."""
    sp = pytest.importorskip("sofab._speedups", reason="native extension not built")

    values = []

    class Evil:
        def __index__(self):
            del values[1:]
            return 1

    values.extend([Evil(), 2, 3])
    with pytest.raises(SofaRangeError):
        sp.Encoder().write_unsigned_array(1, values)


# --- §7.3 against the rest of the decode contract ---------------------------


@engine
def test_a_refused_read_does_not_disturb_a_chunk_fed_decode(Encoder, Decoder):
    """§5.2: a read that answers `None` consumes nothing and suspends nothing —
    the field stays whole for the retry, even when the payload is still arriving
    one byte at a time."""

    def build(e):
        e.write_string(1, "hello world")
        e.write_unsigned(2, 7)

    enc = PyEncoder()
    build(enc)
    dec = Decoder(ChunkReader(enc.getvalue(), chunk=1))
    dec.next()
    assert dec.float64() is None  # not an fp64 → §7.3, nothing consumed
    assert dec.fixlen_len() == 11  # the peek still measures the same field
    assert dec.string() == "hello world"
    dec.next()
    assert dec.unsigned() == 7
    assert dec.next() is None


@engine
def test_a_capped_field_reports_the_cap_rather_than_the_mismatch(Encoder, Decoder):
    """§6.2.1 wins over §7.3 on a *consuming* read: the skip §7.3 asks for would
    have to buffer the payload the cap exists to refuse, and the cap is a
    terminal rejection of the message rather than an answer about one read.
    (`fixlen_len` is the documented exception — it reads and allocates nothing,
    so it answers `None`; see test_schema_bounded.py.)"""
    enc = PyEncoder()
    enc.write_string(1, "x" * 64)
    dec = Decoder(io.BytesIO(enc.getvalue()), max_string_len=8)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.read_unsigned_array()  # the wrong type *and* over the cap
    dec.schema_bounded()  # the caller takes the cap off the field...
    assert dec.read_unsigned_array() is None  # ... and now §7.3 answers
    assert dec.string() == "x" * 64
