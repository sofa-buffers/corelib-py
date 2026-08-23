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

import pytest
from vectors import Status, bound

import sofab
from sofab import Binding
from sofab.decoder import Decoder as PyDecoder
from sofab.encoder import Encoder as PyEncoder
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


def _feed(Decoder, build, binding, **kw):
    """Encode ``build``, decode it against ``binding``; return ``(status, slots)``."""
    enc = PyEncoder()
    build(enc)
    return bound(Decoder, enc.getvalue(), binding, **kw)[::2]


# --- the taxonomy -----------------------------------------------------------


def test_no_invalid_usage_class_exists():
    """§6.3 fixes the taxonomy at five codes and has none for "invalid usage".
    The old class is gone and so is the alias that briefly stood in for it, so a
    caller mistake is a :class:`SofaRangeError` (§6.3 ``InvalidArgument``) and
    nothing else names the removed category."""
    assert issubclass(SofaRangeError, SofaError)
    assert not hasattr(sofab, "SofaStateError")


# --- §7.3: every contradicting binding, on every wire kind ------------------
#
# A binding declares each field's type. When the wire disagrees, §7.3 makes that
# a *skip*, not an error: the field is dropped like an unknown id, its
# destination is left exactly as the caller prepared it, and the decode stays
# COMPLETE. Each row writes a field, binds it wrongly, and binds it rightly.

_AT, _COUNT_AT, _OBJ = 0, 40, 0


def _u(b):     return b.unsigned(1, at=_AT, count_at=_COUNT_AT)
def _s(b):     return b.signed(1, at=_AT, count_at=_COUNT_AT)
def _f32(b):   return b.float32(1, at=_AT, count_at=_COUNT_AT)
def _f64(b):   return b.float64(1, at=_AT, count_at=_COUNT_AT)
def _str(b):   return b.string(1, at=_OBJ, count_at=_COUNT_AT)
def _blob(b):  return b.bytes(1, at=_OBJ, count_at=_COUNT_AT)
def _ua(b):    return b.unsigned_array(1, at=_AT, cap=8, count_at=_COUNT_AT)
def _sa(b):    return b.signed_array(1, at=_AT, cap=8, count_at=_COUNT_AT)
def _f32a(b):  return b.float32_array(1, at=_AT, cap=8, count_at=_COUNT_AT)
def _f64a(b):  return b.float64_array(1, at=_AT, cap=8, count_at=_COUNT_AT)


def _read(slots, kind, n):
    if kind in ("str", "blob"):
        return slots.objects[_OBJ]
    if kind == "u":
        return slots.u[_AT]
    if kind == "s":
        return slots.q[_AT]
    if kind in ("f32", "f64"):
        return slots.d[_AT]
    if kind in ("ua",):
        return slots.arr_u(_AT, n)
    if kind == "sa":
        return slots.arr_q(_AT, n)
    return slots.arr_d(_AT, n)


# (label, write, wrong binder, right binder, right kind, expected value)
_MISMATCHES = [
    ("unsigned", lambda e: e.write_unsigned(1, 5), _s, _u, "u", 5),
    ("signed", lambda e: e.write_signed(1, -5), _u, _s, "s", -5),
    ("fp32", lambda e: e.write_float32(1, 1.5), _f64, _f32, "f32", 1.5),
    ("fp64", lambda e: e.write_float64(1, 1.5), _f32, _f64, "f64", 1.5),
    ("string", lambda e: e.write_string(1, "hi"), _blob, _str, "str", "hi"),
    ("blob", lambda e: e.write_bytes(1, b"hi"), _str, _blob, "blob", b"hi"),
    ("string/scalar", lambda e: e.write_string(1, "hi"), _u, _str, "str", "hi"),
    ("uarray", lambda e: e.write_unsigned_array(1, [1, 2]), _sa, _ua, "ua", [1, 2]),
    ("iarray", lambda e: e.write_signed_array(1, [-1]), _ua, _sa, "sa", [-1]),
    ("uarray/scalar", lambda e: e.write_unsigned_array(1, [1, 2]), _u, _ua, "ua", [1, 2]),
    ("fp32array", lambda e: e.write_float32_array(1, [1.5]), _f64a, _f32a, "f32a", [1.5]),
    ("fp64array", lambda e: e.write_float64_array(1, [1.5]), _f32a, _f64a, "f64a", [1.5]),
    ("fp32array/varray", lambda e: e.write_float32_array(1, [1.5]), _ua, _f32a, "f32a", [1.5]),
    # An *empty* fixlen array still carries its fixlen_word (§4.8), so its
    # subtype is known and can contradict the binding like any other field.
    ("fp32array/empty", lambda e: e.write_float32_array(1, []), _f64a, _f32a, "f32a", []),
]
mismatch = pytest.mark.parametrize(
    "write,wrong,right,kind,value",
    [m[1:] for m in _MISMATCHES],
    ids=[m[0] for m in _MISMATCHES],
)


@engine
@mismatch
def test_a_contradicting_binding_writes_nothing(Encoder, Decoder, write, wrong, right, kind, value):
    status, slots = _feed(Decoder, write, wrong(Binding()))
    assert status is Status.COMPLETE  # §7.3 is not an error
    assert slots.u[_COUNT_AT] == 0    # the field never arrived, as far as it knows


@engine
@mismatch
def test_the_matching_binding_gets_the_value(Encoder, Decoder, write, wrong, right, kind, value):
    status, slots = _feed(Decoder, write, right(Binding()))
    assert status is Status.COMPLETE
    n = slots.u[_COUNT_AT]
    got = _read(slots, kind, n)
    assert got == value if kind not in ("ua", "sa", "f32a", "f64a") else list(got) == list(value)


@engine
@mismatch
def test_a_contradicting_binding_leaves_the_rest_intact(
    Encoder, Decoder, write, wrong, right, kind, value
):
    """The field is skipped like an unknown id; the fields around it decode
    normally and the stream ends at clean EOF, not INVALID or INCOMPLETE."""

    def build(e):
        write(e)
        e.write_unsigned(2, 7)

    b = wrong(Binding()).unsigned(2, at=41, count_at=42)
    status, slots = _feed(Decoder, build, b)
    assert status is Status.COMPLETE
    assert slots.u[_COUNT_AT] == 0
    assert slots.u[41] == 7 and slots.u[42] == 1


@engine
def test_a_contradicting_binding_inside_a_sequence(Encoder, Decoder):
    """Same rule at depth: the id scope is the sequence's, and a mismatch there
    is skipped without disturbing the walk out of it."""

    def build(e):
        e.write_sequence_begin_lazy(9)
        e.write_string(1, "hi")
        e.write_unsigned(2, 4)
        e.write_sequence_end()
        e.write_unsigned(3, 5)

    child = Binding().unsigned(1, at=_AT, count_at=_COUNT_AT).unsigned(2, at=41, count_at=42)
    b = Binding().sequence(9, child).unsigned(3, at=43, count_at=44)
    status, slots = _feed(Decoder, build, b)
    assert status is Status.COMPLETE
    assert slots.u[_COUNT_AT] == 0     # field 1 is a string, the binding says unsigned
    assert slots.u[41] == 4 and slots.u[42] == 1
    assert slots.u[43] == 5 and slots.u[44] == 1


# --- §6.3: the caller-mistake half ------------------------------------------
#
# "No value pending for this field" used to be reachable: a caller could issue a
# typed read before the first next(), twice for one field, or on a sequence
# frame. The decoder owns the reads now — a handler is *handed* values and never
# asks for them — so that whole class of mistake is gone from the surface. What
# is left of §6.3 InvalidArgument on the decode side is binding construction
# (see test_binding) and the encode side below.


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
    """§5.2: the §7.3 skip consumes the field cleanly and suspends nothing — the
    walk carries on even while the payload is still arriving one byte at a
    time."""

    def build(e):
        e.write_string(1, "hello world")
        e.write_unsigned(2, 7)

    b = Binding().float64(1, at=_AT, count_at=_COUNT_AT).unsigned(2, at=41, count_at=42)
    status, slots = _feed(Decoder, build, b, chunk=1)
    assert status is Status.COMPLETE
    assert slots.u[_COUNT_AT] == 0        # field 1 is a string, not an fp64
    assert slots.u[41] == 7 and slots.u[42] == 1


@engine
def test_a_capped_field_reports_the_cap_rather_than_the_mismatch(Encoder, Decoder):
    """§6.2.1 wins over §7.3: the skip §7.3 asks for still has to walk the
    payload the cap exists to refuse, and a cap rejection is terminal for the
    message rather than an answer about one field. Declaring the bound in the
    binding is what takes the cap off — and then §7.3 applies again."""
    enc = PyEncoder()
    enc.write_string(1, "x" * 64)
    wire = enc.getvalue()

    # Bound as an array — the wrong type *and* over the configured cap.
    capped = Binding().unsigned_array(1, at=_AT, cap=8, count_at=_COUNT_AT)
    with pytest.raises(SofaLimitError):
        bound(Decoder, wire, capped, max_dyn_string_len=8)

    # Lifting the cap means declaring the field's own bound — which is only
    # possible by naming its actual kind, so the two can never be combined:
    # §6.2.1 is settled before §7.3 is ever asked.
    declared = Binding().string(1, at=_OBJ, maxlen=64, count_at=_COUNT_AT)
    status, _dec, slots = bound(Decoder, wire, declared, max_dyn_string_len=8)
    assert status is Status.COMPLETE
    assert slots.objects[_OBJ] == "x" * 64
