"""``sofab.float_arrays``: bit-pattern equality of float arrays (generator#636).

The generated encoder omits a field iff its value equals its default, and floats
round-trip bit-for-bit, so the comparison is on bit patterns: ``[-0.0, 1.5]`` is
not the default ``[0.0, 1.5]``. Every case is checked against a plain reference
bit loop written here, and the IEEE ``==`` the helper replaces is pinned to be
wrong on exactly the cases that matter.
"""

from __future__ import annotations

import math
import random
import struct
from array import array

import pytest

import sofab
from sofab import float_array_bits_equal as eq

NAN = float("nan")
INF = math.inf
SUBNORMAL = 5e-324  # smallest positive double
# A quiet NaN with a payload, and a signaling NaN: both survive struct round trips.
NAN_PAYLOAD = struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_0123))[0]
NAN_SIGNALING = struct.unpack("<d", struct.pack("<Q", 0x7FF0_0000_0000_0001))[0]
NEG_NAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8_0000_0000_0000))[0]


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _reference(a, b) -> bool:
    return len(a) == len(b) and all(_bits(x) == _bits(y) for x, y in zip(a, b))


def test_exported_on_the_package_surface() -> None:
    assert "float_array_bits_equal" in sofab.__all__
    assert sofab.float_array_bits_equal is eq


def test_empty_arrays_are_equal() -> None:
    assert eq([], [])
    assert eq((), [])


def test_one_element() -> None:
    assert eq([1.5], [1.5])
    assert not eq([1.5], [2.5])


def test_equal_arrays() -> None:
    assert eq([0.0, 1.5, -3.25], [0.0, 1.5, -3.25])


def test_same_array_against_itself() -> None:
    xs = [NAN, -0.0, 1.5]
    assert eq(xs, xs)
    assert eq(xs, list(xs))


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_negative_zero_differs_from_positive_zero(pos: int) -> None:
    default = [0.0, 0.0, 0.0]
    value = list(default)
    value[pos] = -0.0
    assert value == default  # the IEEE comparison this helper replaces is wrong here
    assert not eq(value, default)
    assert not eq(default, value)


def test_the_issue_example() -> None:
    assert not eq([-0.0, 1.5], [0.0, 1.5])
    assert eq([0.0, 1.5], [0.0, 1.5])


def test_int_elements_compare_as_their_double() -> None:
    # Generated defaults may be written as ``[0, 1.5]``.
    assert eq([0.0, 1.5], [0, 1.5])
    assert eq([0, 1.5], [0.0, 1.5])
    assert not eq([-0.0, 1.5], [0, 1.5])


def test_same_nan_bit_pattern_is_equal() -> None:
    assert NAN != NAN
    assert eq([NAN], [float("nan")])
    assert eq([NAN_PAYLOAD], [struct.unpack("<d", struct.pack("<d", NAN_PAYLOAD))[0]])
    assert eq([NAN_SIGNALING], [NAN_SIGNALING])


def test_nan_with_a_different_payload_differs() -> None:
    assert not eq([NAN], [NAN_PAYLOAD])
    assert not eq([NAN_PAYLOAD], [NAN_SIGNALING])
    assert not eq([NAN], [NEG_NAN])


def test_infinities() -> None:
    assert eq([INF, -INF], [INF, -INF])
    assert not eq([INF], [-INF])
    assert not eq([INF], [1e308])


def test_subnormals() -> None:
    assert eq([SUBNORMAL, -SUBNORMAL], [SUBNORMAL, -SUBNORMAL])
    assert not eq([SUBNORMAL], [0.0])
    assert not eq([SUBNORMAL], [-SUBNORMAL])
    assert not eq([SUBNORMAL], [2 * SUBNORMAL])


def test_fp32_values_held_as_doubles() -> None:
    # An fp32 value is a double here; the held double's pattern is what counts.
    f32 = struct.unpack("<f", struct.pack("<f", 0.1))[0]
    assert eq([f32], [f32])
    assert not eq([f32], [0.1])
    assert not eq([-0.0], [0.0])


def test_length_mismatch_in_both_directions() -> None:
    assert not eq([1.5], [1.5, 1.5])
    assert not eq([1.5, 1.5], [1.5])
    assert not eq([], [0.0])
    assert not eq([0.0], [])


@pytest.mark.parametrize("n", [65, 100, 4096])
@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_long_arrays_with_one_differing_element(n: int, where: str) -> None:
    i = {"start": 0, "middle": n // 2, "end": n - 1}[where]
    base = [float(k) for k in range(n)]
    assert eq(base, list(base))
    for other in (-0.0 if base[i] == 0.0 else base[i] + 1.0, NAN, INF):
        value = list(base)
        value[i] = other
        assert not eq(value, base)
        assert not eq(base, value)
    zeros = [0.0] * n
    negz = list(zeros)
    negz[i] = -0.0
    assert not eq(negz, zeros)


def test_other_sequence_types() -> None:
    assert eq((0.0, 1.5), [0.0, 1.5])
    assert eq(array("d", [0.0, 1.5]), (0.0, 1.5))
    assert not eq(array("d", [-0.0, 1.5]), [0.0, 1.5])
    assert eq(range(3), [0.0, 1.0, 2.0])


def test_does_not_mutate_its_arguments() -> None:
    a = [NAN, -0.0, 3]
    b = [NAN, -0.0, 3]
    eq(a, b)
    assert [_bits(x) for x in a] == [_bits(NAN), _bits(-0.0), _bits(3.0)]
    assert isinstance(a[2], int)


def test_int_outside_the_double_range_equals_no_double() -> None:
    assert not eq([10**400], [1.0])
    assert not eq([1.0, 10**400], [1.0, 2.0])
    assert not eq([2.0, 1.0], [10**400, 1.0])
    assert eq([10**400], [10**400])


def test_unequal_lists_with_a_nan_elsewhere_take_the_walk() -> None:
    # The C-speed prefilter must not call a pair "different" because of a NaN
    # in a position that is itself equal.
    assert eq([NAN, 1.5], [NAN, 1.5])
    assert not eq([NAN, 1.5], [NAN, 2.5])
    assert not eq([NAN, 1.5], [NAN_PAYLOAD, 1.5])
    assert eq([INF, -INF], [INF, -INF])
    assert not eq([INF, 1.0], [-INF, 1.0])


def test_equal_lists_with_zeros_check_every_sign() -> None:
    assert eq([0.0, 1.5, -0.0], [0.0, 1.5, -0.0])
    assert not eq([0.0, 1.5, -0.0], [0.0, 1.5, 0.0])
    assert not eq([0.0, 1.5, 0.0], [0.0, 1.5, -0.0])
    assert eq([1.5, 2.5], [1.5, 2.5])


@pytest.mark.parametrize("n", [5, 8, 64])
def test_all_zero_defaults_tell_every_sign_apart(n: int) -> None:
    zeros = [0.0] * n
    assert eq(list(zeros), zeros)
    for i in (0, n // 2, n - 1):
        v = list(zeros)
        v[i] = -0.0
        assert not eq(v, zeros)
        assert not eq(zeros, v)
        assert eq(v, list(v))


def test_tuple_against_list_is_decided_by_elements_not_by_container() -> None:
    assert eq((0.0, 1.5), [0.0, 1.5])
    assert not eq((-0.0, 1.5), [0.0, 1.5])
    assert not eq((0.0, 1.5), [0.0, 2.5])


def test_pseudo_random_cross_check_against_a_reference_bit_loop() -> None:
    rng = random.Random(636)
    pool = [0.0, -0.0, 1.5, -1.5, INF, -INF, NAN, NAN_PAYLOAD, NEG_NAN, SUBNORMAL, -SUBNORMAL]
    for _ in range(2000):
        n = rng.choice([0, 1, 2, 3, 16, 64, 65, 200])
        a = [
            rng.choice(pool)
            if rng.random() < 0.7
            else struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            for _ in range(n)
        ]
        b = list(a)
        mode = rng.random()
        if n and mode < 0.5:
            j = rng.randrange(n)
            b[j] = rng.choice(pool)
        elif mode < 0.6:
            b.append(0.0)
        assert eq(a, b) == _reference(a, b)
        assert eq(b, a) == _reference(b, a)
