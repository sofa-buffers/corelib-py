"""``sofab.FloatArrayDefault``: a float array default, compared bit for bit.

Generated code builds one per float array field and calls ``matches`` on every
serialize and isDefault. Every case runs on BOTH engines -- the pure class and,
when the extension is built, its native twin -- against the same expectation,
so the two can never disagree; the pure class is additionally driven through
each of the four compare shapes it picks at construction (no zero, one zero,
several zeros, a NaN).
"""

from __future__ import annotations

import math
import random
import struct
from array import array

import pytest

import sofab
from sofab.float_arrays import FloatArrayDefault as PureDefault

NAN = float("nan")
NAN_PAYLOAD = struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_0123))[0]

ENGINES = [pytest.param(PureDefault, id="pure")]
try:
    from sofab._speedups import FloatArrayDefault as NativeDefault

    ENGINES.append(pytest.param(NativeDefault, id="native"))
except ImportError:  # pragma: no cover - only a build without the extension
    NativeDefault = None


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _reference(a, d) -> bool:
    """The element rule, spelled out: equal, same zero sign, identical NaN."""
    if len(a) != len(d):
        return False
    for o, x in zip(a, d):
        if x != x:
            if not (isinstance(o, float) and _bits(o) == _bits(x)):
                return False
        elif not (o == x) or (x == 0 and math.copysign(1.0, o) != math.copysign(1.0, x)):
            return False
    return True


def test_exported_on_the_package_surface() -> None:
    assert "FloatArrayDefault" in sofab.__all__
    expected = NativeDefault if sofab.IMPL == "native" else PureDefault
    assert sofab.FloatArrayDefault is expected


@pytest.mark.parametrize("F", ENGINES)
@pytest.mark.parametrize(
    "default",
    [
        [],
        [1.5],
        [1.5, -2.25, 3.0],  # no zero: the plain list ==
        [0.0],
        [0.0, 1.5],  # one zero
        [-0.0, 1.5],  # one negative zero
        [0.0, 1.5, -0.0, 0.0],  # several zeros, both signs
        [NAN, 1.5],  # a NaN
        [0.0, NAN_PAYLOAD],
    ],
    ids=lambda d: repr(d),
)
def test_the_default_matches_itself_and_nothing_one_bit_away(F, default) -> None:
    d = F(default)
    assert d.values == tuple(default) or any(x != x for x in default)
    assert d.matches(list(default))
    for i, x in enumerate(default):
        flipped = list(default)
        flipped[i] = struct.unpack("<d", struct.pack("<Q", _bits(x) ^ (1 << 63)))[0]
        assert not d.matches(flipped), (default, i, "sign bit")
        flipped[i] = struct.unpack("<d", struct.pack("<Q", _bits(x) ^ 1))[0]
        assert not d.matches(flipped), (default, i, "low bit")
    assert not d.matches(list(default) + [1.5])
    if default:
        assert not d.matches(list(default)[:-1])


@pytest.mark.parametrize("F", ENGINES)
def test_negative_zero_is_not_the_default(F) -> None:
    assert not F([0.0, 1.5]).matches([-0.0, 1.5])
    assert not F([-0.0, 1.5]).matches([0.0, 1.5])
    assert F([-0.0, 1.5]).matches([-0.0, 1.5])


@pytest.mark.parametrize("F", ENGINES)
def test_int_elements_match_the_double_they_equal(F) -> None:
    assert F([0.0, 2.0]).matches([0, 2])
    assert not F([-0.0, 2.0]).matches([0, 2])  # the int 0 is +0.0
    assert F([1.0]).matches([True])
    # Not rounded to a double first: 2**53 + 1 is not 2**53.
    assert not F([2.0**53]).matches([2**53 + 1])
    assert F([2.0**53]).matches([2**53])


@pytest.mark.parametrize("F", ENGINES)
def test_what_is_not_a_matching_number_does_not_match_and_does_not_raise(F) -> None:
    d = F([0.0, 1.5])
    assert not d.matches([10**400, 1.5])  # outside the double range
    assert not d.matches(["0", 1.5])
    assert not d.matches([None, 1.5])
    assert not d.matches([0.0, "1.5"])
    assert not F([NAN]).matches([10**400])
    assert not F([NAN]).matches(["nan"])

    class Weird:
        def __eq__(self, other):
            return True

    # Equal by its own say-so, but it has no sign to read where the default is
    # a zero: no match, and no TypeError escapes.
    assert not d.matches([Weird(), 1.5])
    assert not F([0.0, -0.0]).matches([0.0, Weird()])
    assert F([1.5]).matches([Weird()])


@pytest.mark.parametrize("F", ENGINES)
def test_nan_matches_only_the_identical_nan(F) -> None:
    assert F([NAN]).matches([float("nan")])
    assert not F([NAN]).matches([NAN_PAYLOAD])
    assert F([NAN_PAYLOAD]).matches([NAN_PAYLOAD])
    assert not F([1.5]).matches([NAN])


@pytest.mark.parametrize("F", ENGINES)
@pytest.mark.parametrize(
    "default", [[1.5, 2.5], [0.0, 2.5], [0.0, -0.0, 2.5], [NAN, 2.5]], ids=repr
)
def test_any_sequence_is_compared_element_by_element(F, default) -> None:
    d = F(default)
    assert d.matches(tuple(default))
    assert d.matches(array("d", default))
    assert not d.matches(tuple(default[:-1]))
    assert not d.matches((9.0,) + tuple(default[1:]))
    if 0.0 in default:
        i = default.index(0.0)
        other = list(default)
        other[i] = -0.0 if math.copysign(1.0, default[i]) > 0 else 0.0
        assert not d.matches(tuple(other))


@pytest.mark.parametrize("F", ENGINES)
def test_does_not_mutate_or_keep_its_inputs(F) -> None:
    src = [0.0, 1.5]
    d = F(src)
    src[1] = 9.0
    assert d.matches([0.0, 1.5])
    a = [0.0, 1.5]
    d.matches(a)
    assert a == [0.0, 1.5]
    assert isinstance(d.values, tuple)
    assert repr(d) == "FloatArrayDefault([0.0, 1.5])"


@pytest.mark.parametrize("F", ENGINES)
def test_pseudo_random_cross_check_against_the_reference_rule(F) -> None:
    rng = random.Random(636)
    pool = [0.0, -0.0, 1.5, -1.5, 2.25, NAN, NAN_PAYLOAD, math.inf, -math.inf, 5e-324, 0, 1, 3]
    for _ in range(3000):
        n = rng.randrange(0, 6)
        default = [float(rng.choice(pool)) for _ in range(n)]
        d = F(default)
        for _ in range(4):
            a = [rng.choice(pool + default) for _ in range(n + rng.choice([-1, 0, 0, 0, 1]))]
            assert d.matches(a) == _reference(a, default), (default, a)
            assert d.matches(tuple(a)) == _reference(a, default), (default, a)


@pytest.mark.skipif(NativeDefault is None, reason="native extension not built")
def test_the_engines_agree_everywhere() -> None:  # pragma: no cover - native only
    rng = random.Random(608)
    pool = [0.0, -0.0, 1.5, NAN, NAN_PAYLOAD, 0, 2**53 + 1, 10**400, "x", None, True]
    for _ in range(3000):
        default = [float(rng.choice([0.0, -0.0, 1.5, NAN, 2.0**53])) for _ in range(rng.randrange(0, 5))]
        p, n = PureDefault(default), NativeDefault(default)
        for _ in range(4):
            a = [rng.choice(pool) for _ in range(len(default) + rng.choice([-1, 0, 0, 1]))]
            assert p.matches(a) == n.matches(a), (default, a)
            assert p.matches(tuple(a)) == n.matches(tuple(a)), (default, a)
