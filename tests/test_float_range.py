"""fp32 narrowing of a value outside the fp32 range (issue #68).

An ``fp32`` value is handed to the library as a Python ``float`` — a C
``double`` — so a caller can pass a magnitude no ``fp32`` can hold. The wire
form is decided by the IEEE-754 fp64->fp32 narrowing, which is what every
native-``fp32`` corelib in the family performs (a C ``(float)`` cast, Rust
``as f32``, Java ``(float)``): round-to-nearest-even, and a magnitude that
rounds past ``FLT_MAX`` becomes ``±inf``.

The regression this pins: the native accelerator narrowed with a C cast
(``±inf``) while the pure-Python fallback let ``struct.pack("<f", ...)`` raise a
bare ``OverflowError`` — not a :class:`SofaError`, so ``except SofaError`` never
saw it and ``Encoder(sticky=True).error`` never latched it. The same call
therefore either produced bytes or raised, depending on whether a C compiler was
available at install time, breaking the "byte-for-byte identical engines"
contract (README) and the CORELIB_PLAN §6.3 outcome set.

The boundary is exact: the tie between ``FLT_MAX`` (``2**128 - 2**104``) and the
unrepresentable ``2**128`` sits at ``2**128 - 2**103`` and rounds *up* (to an
even significand), i.e. to infinity. Just below it, the value still rounds down
to ``FLT_MAX``.
"""

from __future__ import annotations

import pytest
from vectors import FLT_MAX, values

from sofab import Decoder
from sofab.encoder import Encoder as PyEncoder

_speedups = pytest.importorskip("sofab._speedups", reason="native extension not built")
NativeEncoder = _speedups.Encoder

INF_HEX = "0000807f"
NEG_INF_HEX = "000080ff"
FLT_MAX_HEX = "ffff7f7f"
NEG_FLT_MAX_HEX = "ffff7fff"

# The exact fp64 tie point between FLT_MAX and the next (unrepresentable) fp32.
TIE = 2.0**128 - 2.0**103
# Largest fp64 strictly below the tie — still rounds down to FLT_MAX.
BELOW_TIE = 3.4028235677973362e38

CASES = [
    (1e300, INF_HEX),
    (-1e300, NEG_INF_HEX),
    (float("inf"), INF_HEX),
    (float("-inf"), NEG_INF_HEX),
    (FLT_MAX * (1.0 + 2.0**-23), INF_HEX),
    (-FLT_MAX * (1.0 + 2.0**-23), NEG_INF_HEX),
    (TIE, INF_HEX),
    (-TIE, NEG_INF_HEX),
    (BELOW_TIE, FLT_MAX_HEX),
    (-BELOW_TIE, NEG_FLT_MAX_HEX),
    (FLT_MAX, FLT_MAX_HEX),
    (-FLT_MAX, NEG_FLT_MAX_HEX),
]


@pytest.mark.parametrize("value,want", CASES)
def test_pure_scalar_saturates(value, want):
    """The pure engine narrows a scalar fp32 instead of raising OverflowError."""
    enc = PyEncoder()
    enc.write_float32(1, value)
    assert enc.getvalue()[-4:].hex() == want


@pytest.mark.parametrize("value,want", CASES)
def test_pure_array_element_saturates(value, want):
    """Same at an fp32 *array* element position (§4.8)."""
    enc = PyEncoder()
    enc.write_float32_array(1, [value])
    assert enc.getvalue()[-4:].hex() == want


@pytest.mark.parametrize("value,want", CASES)
def test_native_matches_pure(value, want):
    """Both engines agree byte-for-byte, scalar and array."""
    py, na = PyEncoder(), NativeEncoder()
    for enc in (py, na):
        enc.write_float32(1, value)
        enc.write_float32_array(2, [value, 0.0, value])
    assert py.getvalue() == na.getvalue()
    assert py.getvalue()[-4:].hex() == want


def test_out_of_range_does_not_break_sticky_mode():
    """A saturating write is a normal write: sticky mode stays clean and later
    writes still run (an escaping OverflowError aborted the whole marshal)."""
    for enc_cls in (PyEncoder, NativeEncoder):
        enc = enc_cls(sticky=True)
        enc.write_float32(1, 1e300)
        enc.write_unsigned(2, 7)
        assert enc.error is None
        assert enc.getvalue()[-2:] == b"\x10\x07"


def test_saturated_value_round_trips():
    """The narrowed value decodes as ±inf on the active engine."""
    enc = PyEncoder()
    enc.write_float32(1, 1e300)
    enc.write_float32_array(2, [-1e300])
    assert values(Decoder, enc.getvalue()) == [
        ("f32", 1, float("inf")),
        ("f32a", 2, (float("-inf"),)),
    ]


def test_pure_array_saturates_only_the_out_of_range_elements():
    """An fp32 array that mixes an out-of-fp32-range magnitude with ordinary
    values: the oversized one saturates to ±inf and the rest are untouched.

    A whole-array narrowing has to give the same bytes as a per-element one, so
    the array codec may take a bulk path only where every element survives it.
    """
    enc = PyEncoder()
    enc.write_float32_array(1, [1.0, 1e300, -2.0, -1e300])
    payload = enc.getvalue()[-16:].hex()
    assert payload == "0000803f" + "0000807f" + "000000c0" + "000080ff"

    na = NativeEncoder()
    na.write_float32_array(1, [1.0, 1e300, -2.0, -1e300])
    assert na.getvalue() == enc.getvalue()
