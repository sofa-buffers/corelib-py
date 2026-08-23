"""``on_array_begin``: the two things only the header can settle.

An integer array reaches a visitor through :meth:`sofab.Visitor.on_unsigned_array`
/ ``on_signed_array``, which receive it *decoded*. By then two decisions have
already been made for the handler, and neither can be revisited:

* **The element width the schema declares.** A handler that checks the list it
  is given can only reject an array that ARRIVED. CORELIB_PLAN §7.1 forbids
  exactly that -- whether a bound is enforced must not be an emergent property
  of the memory model -- and §5.2 wants the INVALID ahead of the INCOMPLETE a
  truncation behind the bad element would otherwise report.
* **Where the elements go.** §6.6.3 has a codec deliver an aggregate either in
  pieces or "into a destination the caller hands back after being told the
  announced count, with the codec refusing a destination too short rather than
  growing it". A list handed over afterwards is a list already built.

``on_array_begin`` runs at the count header, before a single element is read,
and answers both. Every case runs on both engines.
"""

from __future__ import annotations

import ctypes
from array import array

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder, Status, walk

from sofab import Encoder, SofaDecodeError, SofaRangeError, WireType

U16_MAX = 0xFFFF


class Spec(Recorder):
    """Answers on_array_begin with whatever it was constructed with."""

    def __init__(self, spec, **kw):
        super().__init__(**kw)
        self.seen = []
        self._spec = spec

    def on_array_begin(self, field_id, wtype, count):
        self.seen.append((field_id, wtype, count))
        return self._spec


def _msg(values, signed=False):
    enc = Encoder()
    if signed:
        enc.write_signed_array(1, values)
    else:
        enc.write_unsigned_array(1, values)
    enc.flush()
    return enc.getvalue()


# --- the hook itself --------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_the_hook_sees_the_header_before_any_element(engine):
    spec = Spec(None)
    status, rec, _dec = walk(engine, _msg([1, 2, 3]), recorder=spec)
    assert status is Status.COMPLETE
    assert spec.seen == [(1, WireType.ARRAY_UNSIGNED, 3)]


@pytest.mark.parametrize("engine", ENGINES)
def test_returning_none_leaves_the_list_route_alone(engine):
    spec = Spec(None)
    status, rec, _dec = walk(engine, _msg([7, 8]), recorder=spec)
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (7, 8))]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_visitor_that_does_not_override_it_is_unaffected(engine):
    status, rec, _dec = walk(engine, _msg([7, 8]))
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (7, 8))]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_hook_is_not_called_for_a_float_array(engine):
    enc = Encoder()
    enc.write_float32_array(1, [1.5, 2.5])
    enc.flush()
    spec = Spec(None)
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=spec)
    assert status is Status.COMPLETE
    assert spec.seen == []


# --- stating the width (dst is None) ----------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_stated_width_rejects_an_over_wide_element(engine):
    spec = Spec((None, None, U16_MAX))
    status, rec, dec = walk(engine, _msg([1, 1 << 20, 3]), recorder=spec)
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_stated_width_accepts_what_fits(engine):
    spec = Spec((None, None, U16_MAX))
    status, rec, _dec = walk(engine, _msg([1, U16_MAX]), recorder=spec)
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (1, U16_MAX))]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_verdict_does_not_depend_on_the_array_arriving(engine):
    """§7.1: a complete array and one truncated behind the same bad element are
    rejected alike. This is the case a handler checking the list cannot reach --
    the truncated form never produces a list at all."""
    wire = _msg([1, 1 << 20, 3])
    for cut in (len(wire), len(wire) - 1):
        spec = Spec((None, None, U16_MAX))
        status, rec, dec = walk(engine, wire[:cut], recorder=spec)
        assert status is Status.INVALID, cut
        assert isinstance(dec.error, SofaDecodeError), cut


@pytest.mark.parametrize("engine", ENGINES)
def test_a_stated_signed_width_binds_both_halves(engine):
    for values, ok in (([-128, 127], True), ([-129], False), ([128], False)):
        spec = Spec((None, -128, 127))
        status, rec, _dec = walk(engine, _msg(values, signed=True), recorder=spec)
        assert status is (Status.COMPLETE if ok else Status.INVALID), values


@pytest.mark.parametrize("engine", ENGINES)
def test_a_width_that_admits_the_whole_domain_is_no_bound(engine):
    """A u64 field's declared maximum is 2**64-1, which cannot reject anything.
    Both engines must simply accept -- not overflow on the way to finding out."""
    big = (1 << 64) - 1
    spec = Spec((None, None, big))
    status, rec, _dec = walk(engine, _msg([big, 0]), recorder=spec)
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (big, 0))]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_width_does_not_leak_to_the_next_array(engine):
    """The hook is asked per array, so a bound stated for one says nothing about
    the next -- here the second array is told nothing and keeps everything."""

    class Once(Spec):
        def on_array_begin(self, field_id, wtype, count):
            self.seen.append(field_id)
            return (None, None, U16_MAX) if field_id == 1 else None

    enc = Encoder()
    enc.write_unsigned_array(1, [5])
    enc.write_unsigned_array(2, [1 << 20])
    enc.flush()
    spec = Once(None)
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=spec)
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (5,)), ("ua", 2, (1 << 20,))]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [1, 3])
def test_the_hook_survives_a_chunk_boundary(engine, chunk):
    spec = Spec((None, None, U16_MAX))
    status, rec, _dec = walk(engine, _msg([1, 2, 3, 4]), chunk=chunk, recorder=spec)
    assert status is Status.COMPLETE
    assert rec.events == [("ua", 1, (1, 2, 3, 4))]


# --- handing over a destination ---------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_destination_is_filled_and_the_typed_hook_is_not_called(engine):
    dst = array("Q", [0] * 4)
    spec = Spec((dst, None, None))
    status, rec, _dec = walk(engine, _msg([9, 8, 7]), recorder=spec)
    assert status is Status.COMPLETE
    assert list(dst) == [9, 8, 7, 0]
    assert [e for e in rec.events if e[0] == "ua"] == []  # nothing was materialised for a hook to receive


@pytest.mark.parametrize("engine", ENGINES)
def test_a_narrow_destination_needs_a_width_that_fits_it(engine):
    dst = array("H", [0] * 4)
    spec = Spec((dst, None, U16_MAX))
    status, rec, _dec = walk(engine, _msg([1, U16_MAX]), recorder=spec)
    assert status is Status.COMPLETE
    assert list(dst) == [1, U16_MAX, 0, 0]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_signed_destination_is_filled_with_signed_values(engine):
    dst = array("q", [0] * 3)
    spec = Spec((dst, None, None))
    status, rec, _dec = walk(engine, _msg([-5, 6], signed=True), recorder=spec)
    assert status is Status.COMPLETE
    assert list(dst) == [-5, 6, 0]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_destination_survives_a_chunk_boundary(engine):
    dst = array("Q", [0] * 4)
    spec = Spec((dst, None, None))
    status, rec, _dec = walk(engine, _msg([1, 2, 3, 4]), chunk=1, recorder=spec)
    assert status is Status.COMPLETE
    assert list(dst) == [1, 2, 3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_stated_width_still_binds_a_destination(engine):
    dst = array("Q", [0] * 4)
    spec = Spec((dst, None, U16_MAX))
    status, rec, dec = walk(engine, _msg([1, 1 << 20]), recorder=spec)
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)


# --- what a destination is refused for (§6.6: never grown) ------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_short_destination_is_refused_not_grown(engine):
    dst = array("Q", [0] * 2)
    spec = Spec((dst, None, None))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2, 3]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_refusal_comes_before_a_single_element_is_written(engine):
    dst = array("Q", [0] * 2)
    spec = Spec((dst, None, None))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2, 3]), recorder=spec)
    assert list(dst) == [0, 0]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_destination_that_is_not_a_buffer_is_refused(engine):
    spec = Spec(([0, 0, 0], None, None))  # a list has no buffer to write into
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2, 3]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_read_only_destination_is_refused(engine):
    spec = Spec((b"\x00" * 32, None, None))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2, 3]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unsupported_item_size_is_refused(engine):
    """A 3-byte item is contiguous and writable and still no good: the fill
    narrows to 1, 2, 4 or 8 and nothing else."""

    class Three(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("a", ctypes.c_uint8 * 3)]

    spec = Spec((memoryview((Three * 4)()), None, None))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2, 3]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_narrow_destination_with_no_stated_width_is_refused(engine):
    """Nothing says the elements fit, so filling it could truncate silently."""
    dst = array("H", [0] * 4)
    spec = Spec((dst, None, None))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_narrow_destination_with_too_wide_a_width_is_refused(engine):
    dst = array("H", [0] * 4)
    spec = Spec((dst, None, 1 << 20))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2]), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_narrow_signed_destination_needs_both_halves_to_fit(engine):
    dst = array("h", [0] * 4)
    spec = Spec((dst, -(1 << 20), 1 << 20))
    with pytest.raises(SofaRangeError):
        walk(engine, _msg([1, 2], signed=True), recorder=spec)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_narrow_signed_destination_accepts_a_width_that_fits(engine):
    dst = array("h", [0] * 4)
    spec = Spec((dst, -32768, 32767))
    status, rec, _dec = walk(engine, _msg([-5, 32767], signed=True), recorder=spec)
    assert status is Status.COMPLETE
    assert list(dst) == [-5, 32767, 0, 0]
