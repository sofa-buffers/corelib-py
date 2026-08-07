"""An integer array's declared element width, applied while the array is read.

MESSAGE_SPEC §7.1 makes an element outside its declared width invalid, and §5.2
makes INVALID dominate INCOMPLETE: such an element is established by its own
bytes, so truncating the array behind it cannot downgrade the verdict. A caller
scanning the list ``read_unsigned_array`` / ``read_signed_array`` returns can
only decide an array that *arrives* — so the bound travels into the reader
(generator#267, Crucible F-0043).

Both engines are exercised on every case. They apply the bound differently on
purpose — the native one at the element, where a typed compare is free, the pure
one at the truncation, so its decode loop stays a pure decode — and the verdicts
must not diverge for it.
"""

from __future__ import annotations

import io

import pytest

from sofab.decoder import Decoder as PyDecoder
from sofab.types import SofaDecodeError, SofaIncompleteError

_speedups = pytest.importorskip("sofab._speedups", reason="native extension not built")

ENGINES = [pytest.param(PyDecoder, id="pure"), pytest.param(_speedups.Decoder, id="native")]

# id 0, unsigned array -> header (0 << 3) | 3
# id 1, signed array   -> header (1 << 3) | 4
U_HDR = bytes([(0 << 3) | 3])
S_HDR = bytes([(1 << 3) | 4])


def _read(engine, data: bytes, signed: bool, *bounds):
    d = engine(io.BytesIO(data))
    d.next()
    return d.read_signed_array(*bounds) if signed else d.read_unsigned_array(*bounds)


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

    def test_a_complete_array_is_returned_for_the_caller_to_scan(self, engine):
        # In range: both engines return the values.
        assert _read(engine, U_HDR + b"\x02\x01\xff\x01", False, 255) == [1, 255]

    def test_an_overlong_varint_is_still_invalid(self, engine):
        # The FORMAT bound is unaffected and still outranks everything.
        with pytest.raises(SofaDecodeError):
            _read(engine, S_HDR + b"\x05" + b"\x80" * 11, True, -128, 127)


@pytest.mark.parametrize("engine", ENGINES)
def test_bounds_do_not_change_a_valid_array(engine):
    d = engine(io.BytesIO(S_HDR + b"\x03\x02\x04\x06"))
    d.next()
    assert d.read_signed_array(-128, 127) == [1, 2, 3]
