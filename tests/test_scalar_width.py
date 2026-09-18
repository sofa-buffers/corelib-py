"""An integer scalar's declared width, applied where the table stores it (#149).

A ``Binding`` slot is 64 bits whatever the field declares, so MESSAGE_SPEC §1
requires the declared width as an explicit check. It runs at the value, before
the store — which is what keeps §5.2's INVALID-over-INCOMPLETE when the message
is truncated behind an out-of-width value, exactly as for an array element
(``test_elem_bound.py``).
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, Status, bound, encode

from sofab.types import SofaArgumentError


def _u8():
    return Binding().unsigned(1, at=0, count_at=1, max_value=0xFF)


def _i8():
    return Binding().signed(1, at=0, count_at=1, min_value=-128, max_value=127)


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestScalarWidth:
    @pytest.mark.parametrize("value", [0, 255])
    def test_unsigned_inside(self, engine, chunk, value):
        st, _, slots = bound(engine, encode(lambda e: e.write_unsigned(1, value)),
                             _u8(), chunk)
        assert st == Status.COMPLETE
        assert (slots.u[0], slots.u[1]) == (value, 1)

    def test_unsigned_outside_is_invalid_and_unstored(self, engine, chunk):
        st, _, slots = bound(engine, encode(lambda e: e.write_unsigned(1, 256)),
                             _u8(), chunk)
        assert st == Status.INVALID
        assert (slots.u[0], slots.u[1]) == (0, 0)

    @pytest.mark.parametrize("value", [-128, 0, 127])
    def test_signed_inside(self, engine, chunk, value):
        st, _, slots = bound(engine, encode(lambda e: e.write_signed(1, value)),
                             _i8(), chunk)
        assert st == Status.COMPLETE
        assert slots.q[0] == value

    @pytest.mark.parametrize("value", [-129, 128, -(1 << 63), (1 << 63) - 1])
    def test_signed_outside_is_invalid(self, engine, chunk, value):
        st, _, slots = bound(engine, encode(lambda e: e.write_signed(1, value)),
                             _i8(), chunk)
        assert st == Status.INVALID
        assert slots.u[1] == 0

    def test_out_of_width_then_truncated_is_invalid(self, engine, chunk):
        # §5.2: the bad value is established by its own bytes, so the cut behind
        # it cannot downgrade the verdict to INCOMPLETE.
        data = encode(lambda e: (e.write_unsigned(1, 300), e.write_string(2, "abcdef")))
        st, _, _ = bound(engine, data[:-2], _u8(), chunk)
        assert st == Status.INVALID

    def test_one_sided(self, engine, chunk):
        b = Binding().signed(1, at=0, min_value=0)
        st, _, _ = bound(engine, encode(lambda e: e.write_signed(1, -1)), b, chunk)
        assert st == Status.INVALID
        b = Binding().signed(1, at=0, min_value=0)
        st, _, slots = bound(engine, encode(lambda e: e.write_signed(1, 1 << 40)), b, chunk)
        assert (st, slots.q[0]) == (Status.COMPLETE, 1 << 40)

    def test_unbounded_is_unchanged(self, engine, chunk):
        b = Binding().unsigned(1, at=0).signed(2, at=1)
        data = encode(lambda e: (e.write_unsigned(1, (1 << 64) - 1),
                                 e.write_signed(2, -(1 << 63))))
        st, _, slots = bound(engine, data, b, chunk)
        assert (st, slots.u[0], slots.q[1]) == (Status.COMPLETE, (1 << 64) - 1, -(1 << 63))


@pytest.mark.parametrize("bind", [
    lambda b: b.signed(1, at=0, max_value=1 << 63),
    lambda b: b.signed_array(1, at=0, cap=1, elem_max=1 << 63),
    lambda b: b.unsigned(1, at=0, max_value=1 << 64),
    lambda b: b.unsigned(1, at=0, max_value=-1),
])
def test_width_past_the_slot_is_refused(bind):
    # A signed slot cannot hold a maximum above INT64_MAX; accepting one would
    # wrap in the native engine's int64 compare.
    with pytest.raises(SofaArgumentError):
        bind(Binding())
