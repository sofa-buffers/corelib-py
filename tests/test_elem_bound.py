"""An integer array's declared element width, applied while the array is read.

MESSAGE_SPEC §7.1 makes an element outside its declared width invalid, and §5.2
makes INVALID dominate INCOMPLETE: such an element is established by its own
bytes, so truncating the array behind it cannot downgrade the verdict. A caller
scanning the list ``read_unsigned_array`` / ``read_signed_array`` returns can
only decide an array that *arrives* — so the bound travels into the reader
(generator#267, Crucible F-0043).

Both engines are exercised on every case, and both apply the bound AT the
element: §7.1 also says that whether a bound is enforced must not be an emergent
property of the memory model, so the same bytes and the same call must not get
one verdict from an array that completes and another from one that is truncated
behind the offending element (issue #67).
"""

from __future__ import annotations

import io

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, bound, raise_for
from vectors import uvarint as _varint
from vectors import zzvarint as _zz

from sofab.types import SofaDecodeError, SofaIncompleteError

# id 0, unsigned array -> header (0 << 3) | 3
# id 1, signed array   -> header (1 << 3) | 4
U_HDR = bytes([(0 << 3) | 3])
S_HDR = bytes([(1 << 3) | 4])


def _uarray(values: list[int]) -> bytes:
    return U_HDR + _varint(len(values)) + b"".join(_varint(v) for v in values)


def _sarray(values: list[int]) -> bytes:
    return S_HDR + _varint(len(values)) + b"".join(_zz(v) for v in values)


def _read(engine, data: bytes, signed: bool, *bounds):
    """Decode the one array in ``data`` against its declared element width.

    The width lives on the binding now (``elem_min`` / ``elem_max``), which is
    where a schema-declared bound belongs; ``raise_for`` turns the outcome back
    into the verdict these tests assert on.
    """
    b = Binding()
    if signed:
        lo, hi = (list(bounds) + [None, None])[:2]
        # id 1 for the signed header, id 0 for the unsigned one — see S_HDR/U_HDR.
        b.signed_array(1, at=0, cap=64, count_at=100, elem_min=lo, elem_max=hi)
    else:
        hi = (list(bounds) + [None])[0]
        b.unsigned_array(0, at=0, cap=64, count_at=100, elem_max=hi)
    st, dec, slots = bound(engine, data, b)
    raise_for(st, dec)
    n = slots.u[100]
    return slots.arr_q(0, n) if signed else slots.arr_u(0, n)


@pytest.mark.parametrize("engine", ENGINES)
class TestTruncatedArrayIsDecidedByItsElements:
    def test_signed_over_width_then_truncated_is_invalid(self, engine):
        # Crucible width_elem_trunc: count 5, one element = zigzag(5208), end.
        with pytest.raises(SofaDecodeError):
            _read(engine, S_HDR + b"\x05\xb0\x51", True, -128, 127)

    def test_signed_in_range_then_truncated_stays_incomplete(self, engine):
        # ctl_width_elem_inrange_trunc: the same cut, element 1. Nothing is
        # decided yet, so the truncation IS the verdict. This control is what
        # makes the case above an ordering fix rather than a blanket reject.
        with pytest.raises(SofaIncompleteError):
            _read(engine, S_HDR + b"\x05\x02", True, -128, 127)

    def test_unsigned_over_width_then_truncated_is_invalid(self, engine):
        with pytest.raises(SofaDecodeError):
            _read(engine, U_HDR + b"\x05\x80\x04", False, 255)  # 512 > 255

    def test_unsigned_in_range_then_truncated_stays_incomplete(self, engine):
        with pytest.raises(SofaIncompleteError):
            _read(engine, U_HDR + b"\x05\xff\x01", False, 255)  # 255 == bound

    def test_an_unsigned_value_above_2_63_is_out_of_range(self, engine):
        # The largest wire values must not read as "small" through any signed
        # intermediate. u64 max = ten 0xff/0x01 bytes.
        with pytest.raises(SofaDecodeError):
            _read(engine, U_HDR + b"\x05" + b"\xff" * 9 + b"\x01", False, 255)

    def test_no_bound_leaves_the_truncation_alone(self, engine):
        # The additive contract: the vector that is INVALID above stays
        # INCOMPLETE when no bound is passed, which is what a u64 array does.
        with pytest.raises(SofaIncompleteError):
            _read(engine, S_HDR + b"\x05\xb0\x51", True)

    def test_a_complete_in_range_array_is_returned(self, engine):
        # In range: both engines return the values.
        assert _read(engine, U_HDR + b"\x02\x01\xff\x01", False, 255) == [1, 255]

    def test_an_overlong_varint_is_still_invalid(self, engine):
        # The FORMAT bound is unaffected and still outranks everything.
        with pytest.raises(SofaDecodeError):
            _read(engine, S_HDR + b"\x05" + b"\x80" * 11, True, -128, 127)


@pytest.mark.parametrize("engine", ENGINES)
class TestACompleteArrayIsBoundToo:
    """§7.1: the bound binds every element, not only the ones an array happens to
    have decoded before running out of bytes (issue #67)."""

    def test_unsigned_over_width_complete_is_invalid(self, engine):
        # The whole array arrives — 300 is still outside a u8's declared width.
        with pytest.raises(SofaDecodeError):
            _read(engine, _uarray([300]), False, 255)

    def test_unsigned_at_the_bound_is_accepted(self, engine):
        assert _read(engine, _uarray([0, 255]), False, 255) == [0, 255]

    def test_unsigned_one_byte_element_over_a_narrow_bound_is_invalid(self, engine):
        # 100 is a single wire byte and rides the decoder's one-byte fast path;
        # a u7-style bound (max 15) must still catch it.
        with pytest.raises(SofaDecodeError):
            _read(engine, _uarray([1, 100]), False, 15)

    def test_signed_over_width_complete_is_invalid(self, engine):
        with pytest.raises(SofaDecodeError):
            _read(engine, _sarray([5208]), True, -128, 127)

    def test_signed_under_width_complete_is_invalid(self, engine):
        # Below the *lower* bound: the case an unsigned-only check would miss.
        with pytest.raises(SofaDecodeError):
            _read(engine, _sarray([-200]), True, -128, 127)

    def test_signed_at_both_bounds_is_accepted(self, engine):
        assert _read(engine, _sarray([-128, 0, 127]), True, -128, 127) == [-128, 0, 127]

    def test_signed_one_byte_element_over_a_narrow_bound_is_invalid(self, engine):
        # ZigZag(50) = 100, one wire byte, but 50 is outside an i4's width.
        with pytest.raises(SofaDecodeError):
            _read(engine, _sarray([1, 50]), True, -8, 7)

    def test_the_bad_element_decides_even_with_good_ones_behind_it(self, engine):
        with pytest.raises(SofaDecodeError):
            _read(engine, _uarray([1, 2, 999, 3]), False, 255)

    def test_an_unbounded_read_still_returns_the_wide_element(self, engine):
        # The additive contract: no declared width, no rejection.
        assert _read(engine, _uarray([300]), False) == [300]
        assert _read(engine, _sarray([-200]), True) == [-200]


@pytest.mark.parametrize("engine", ENGINES)
class TestOneSidedSignedBounds:
    """``elem_min``/``elem_max`` are guarded independently: supplying one without
    the other bounds that side and leaves the other open — it must not fault on
    the missing half (issue #67)."""

    def test_upper_bound_alone_rejects_an_over_width_element(self, engine):
        with pytest.raises(SofaDecodeError):
            _read(engine, _sarray([300]), True, None, 127)

    def test_upper_bound_alone_rejects_it_when_truncated_too(self, engine):
        # count claims 5, one element arrives: same verdict as the complete case.
        with pytest.raises(SofaDecodeError):
            _read(engine, S_HDR + b"\x05" + _zz(300), True, None, 127)

    def test_upper_bound_alone_leaves_the_lower_side_open(self, engine):
        assert _read(engine, _sarray([-99999, 127]), True, None, 127) == [-99999, 127]

    def test_lower_bound_alone_rejects_an_under_width_element(self, engine):
        with pytest.raises(SofaDecodeError):
            _read(engine, _sarray([-200]), True, -128)

    def test_lower_bound_alone_leaves_the_upper_side_open(self, engine):
        assert _read(engine, _sarray([-128, 99999]), True, -128) == [-128, 99999]


@pytest.mark.parametrize("engine", ENGINES)
def test_bounds_do_not_change_a_valid_array(engine):
    assert _read(engine, S_HDR + b"\x03\x02\x04\x06", True, -128, 127) == [1, 2, 3]


# --- the verdict is exactly "lo <= value <= hi", for every bound and value ---
#
# Each engine restates the declared width in its own terms — the pure one
# normalises an omitted half and hoists the test off its one-byte fast path, the
# native one narrows to typed C integers — so a sweep is what keeps those
# restatements honest: one-byte and multi-byte elements, the widths a schema
# actually declares, and the asymmetric ranges no schema emits but the API takes.

_SIGNED_BOUNDS = [
    (-128, 127),  # i8
    (-32768, 32767),  # i16
    (-(2**31), 2**31 - 1),  # i32
    (-8, 7),  # sub-byte width: bites the one-byte fast path
    (-5, 100),  # asymmetric, both ways: not a two's-complement width
    (-100, 5),
    (0, 10),  # non-negative signed range
    (None, 127),  # one-sided
    (-128, None),
]
_SIGNED_VALUES = [
    0, 1, -1, 7, 8, -8, -9, 63, 64, -64, -65, 100, -100, 127, 128, -128, -129,
    5208, -5208, 32767, -32768, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63),
]

_UNSIGNED_BOUNDS = [255, 65535, 2**32 - 1, 15, 0, 127, 128, None]
_UNSIGNED_VALUES = [
    0, 1, 15, 16, 100, 126, 127, 128, 255, 256, 300, 65535, 65536,
    2**32 - 1, 2**32, 2**63 - 1,
]


@pytest.mark.parametrize("engine", ENGINES)
def test_signed_verdict_is_the_plain_range_check(engine):
    for lo, hi in _SIGNED_BOUNDS:
        for v in _SIGNED_VALUES:
            in_range = (lo is None or v >= lo) and (hi is None or v <= hi)
            data = _sarray([v])
            if in_range:
                assert _read(engine, data, True, lo, hi) == [v], (lo, hi, v)
            else:
                with pytest.raises(SofaDecodeError):
                    _read(engine, data, True, lo, hi)


@pytest.mark.parametrize("engine", ENGINES)
def test_unsigned_verdict_is_the_plain_range_check(engine):
    for hi in _UNSIGNED_BOUNDS:
        for v in _UNSIGNED_VALUES:
            data = _uarray([v])
            if hi is None or v <= hi:
                assert _read(engine, data, False, hi) == [v], (hi, v)
            else:
                with pytest.raises(SofaDecodeError):
                    _read(engine, data, False, hi)
