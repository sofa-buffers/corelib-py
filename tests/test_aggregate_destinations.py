"""Every aggregate has a route that does not size an allocation from the wire.

CORELIB_PLAN §6.6.3 gives a conformant port two shapes for delivering an
aggregate, and requires one of them:

    Ports meeting this section therefore deliver an aggregate either **in
    pieces**, with the payload's total, this piece's offset, and the caller's
    own buffer as arguments; **or into a destination the caller hands back**
    after being told the announced count, with the codec refusing a destination
    too short rather than growing it. That refusal is **`InvalidArgument`**
    (§6.3): the message is well-formed and within every bound it declares --
    what does not fit is the storage this caller offered.

``blob`` and the integer arrays had the destination route already
(``on_blob_begin`` / ``on_array_begin``). ``string``, ``fp32`` array and
``fp64`` array had neither, on the only surface §5.3.1 permits, so the wire
still sized an allocation inside the codec with *both* documented opt-outs
taken. This file covers the two hooks that close that: ``on_string_begin`` and
``on_float_array_begin``. (``fp32`` arrays additionally have §6.5's raw route,
``on_float32_array_bits``, in ``tests/test_float_nan_bits.py``.)
"""

from __future__ import annotations

import array
import tracemalloc

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import ENCODER_ENGINES as ENCODERS
from vectors import Status

from sofab import (
    Encoder,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    Visitor,
)

TEXT = "héllo wörld " * 8  # multi-byte, so bytes != characters


class StringSink(Visitor):
    def __init__(self, dst):
        self.dst = dst
        self.seen: list[tuple[int, int]] = []
        self.materialized: list[str] = []

    def on_string_begin(self, field_id, size):
        self.seen.append((field_id, size))
        return self.dst

    def on_string(self, field_id, value):
        self.materialized.append(value)


class FloatSink(Visitor):
    def __init__(self, dst):
        self.dst = dst
        self.seen: list[tuple[int, FixlenSubtype, int]] = []
        self.materialized: list[list[float]] = []

    def on_float_array_begin(self, field_id, subtype, count):
        self.seen.append((field_id, subtype, count))
        return self.dst

    def on_float32_array(self, field_id, values):
        self.materialized.append(values)

    def on_float64_array(self, field_id, values):
        self.materialized.append(values)


def _string_wire(enc_cls, text=TEXT):
    enc = enc_cls()
    enc.write_string(2, text)
    enc.flush()
    return enc.getvalue()


def _farray_wire(enc_cls, values, width):
    enc = enc_cls()
    if width == 4:
        enc.write_float32_array(3, values)
    else:
        enc.write_float64_array(3, values)
    enc.flush()
    return enc.getvalue()


# --- on_string_begin ---------------------------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_string_lands_in_the_callers_buffer_as_utf8(dec_cls, enc_cls):
    """The hook is told the **wire byte length** — what a schema ``maxlen``
    bounds (MESSAGE_SPEC §1) — and what lands is the payload's own bytes."""
    utf8 = TEXT.encode()
    sink = StringSink(bytearray(len(utf8)))
    assert dec_cls(visitor=sink).feed(_string_wire(enc_cls)) is Status.COMPLETE
    assert sink.seen == [(2, len(utf8))]
    assert len(utf8) != len(TEXT), "the case must distinguish bytes from characters"
    assert bytes(sink.dst) == utf8
    assert sink.materialized == [], "the destination route replaces on_string"


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_returning_none_still_gets_the_str(dec_cls, enc_cls):
    """The hook is an opt-out, not a replacement: a handler that answers
    ``None`` for this field takes the ``str`` as before."""

    class Sometimes(StringSink):
        def on_string_begin(self, field_id, size):
            return None

    sink = Sometimes(bytearray(4))
    assert dec_cls(visitor=sink).feed(_string_wire(enc_cls)) is Status.COMPLETE
    assert sink.materialized == [TEXT]


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_short_destination_is_an_argument_error(dec_cls, enc_cls):
    """§6.3's third tier: the message is well-formed and within every bound it
    declares, so this is `InvalidArgument` and not `InvalidMessage`."""
    sink = StringSink(bytearray(4))
    with pytest.raises(SofaArgumentError):
        dec_cls(visitor=sink).feed(_string_wire(enc_cls))


@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize(
    "bad", [b"\xff", b"\xc3", b"\x80", b"\xe2\x28\xa1", b"\xed\xa0\x80"]
)
def test_the_destination_route_still_validates_utf8(dec_cls, bad):
    """§6.7.2: a field the handler **reads** is materialized *and* validated.
    Skipping the validation would make the route a way to bypass §6.4."""
    enc = Encoder()
    enc.write_bytes(2, bad)
    enc.flush()
    wire = bytearray(enc.getvalue())
    wire[1] = (len(bad) << 3) | FixlenSubtype.STRING  # blob -> string subtype

    sink = StringSink(bytearray(64))
    dec = dec_cls(visitor=sink)
    assert dec.feed(bytes(wire)) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    # And the destination is untouched: no half-written buffer behind a verdict.
    assert bytes(sink.dst) == bytes(64)


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_validation_carries_across_its_own_window(dec_cls):
    """The payload is checked in fixed windows, so a multi-byte sequence
    straddling a window boundary must be carried, not cut (§6.4.4's rule applied
    to the port's own chunking)."""
    from sofab import _core

    window = _core.UTF8_WINDOW
    for pad in range(window - 3, window + 2):
        text = "a" * pad + "é" + "b" * 8
        utf8 = text.encode()
        enc = Encoder()
        enc.write_string(2, text)
        enc.flush()
        sink = StringSink(bytearray(len(utf8)))
        assert dec_cls(visitor=sink).feed(enc.getvalue()) is Status.COMPLETE, pad
        assert bytes(sink.dst) == utf8, pad


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_string_route_survives_a_chunk_boundary(dec_cls, enc_cls):
    wire = _string_wire(enc_cls)
    utf8 = TEXT.encode()
    sink = StringSink(bytearray(len(utf8)))
    dec = dec_cls(visitor=sink, reassembly=bytearray(len(wire) + 16))
    for i in range(0, len(wire), 5):
        dec.feed(wire[i : i + 5])
    assert bytes(sink.dst) == utf8


# --- on_float_array_begin ----------------------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width,subtype", [(4, FixlenSubtype.FP32), (8, FixlenSubtype.FP64)])
def test_a_float_array_lands_in_the_callers_slots(dec_cls, enc_cls, width, subtype):
    """One hook serves both subtypes, and says which it is — so a handler that
    wants only one returns ``None`` for the other."""
    values = [0.5, -1.25, 3.0, 1024.0]
    dst = array.array("d", [0.0] * len(values))
    sink = FloatSink(dst)
    wire = _farray_wire(enc_cls, values, width)
    assert dec_cls(visitor=sink).feed(wire) is Status.COMPLETE
    assert sink.seen == [(3, subtype, len(values))]
    assert list(dst) == values
    assert sink.materialized == []


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width", [4, 8])
def test_declining_one_subtype_still_gets_the_list(dec_cls, enc_cls, width):
    class OnlyFp64(FloatSink):
        def on_float_array_begin(self, field_id, subtype, count):
            if subtype is FixlenSubtype.FP64:
                return self.dst
            return None

    values = [1.0, 2.0]
    sink = OnlyFp64(array.array("d", [0.0, 0.0]))
    assert dec_cls(visitor=sink).feed(_farray_wire(enc_cls, values, width)) is (
        Status.COMPLETE
    )
    if width == 4:
        assert sink.materialized == [values]
    else:
        assert list(sink.dst) == values


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width", [4, 8])
def test_a_short_float_destination_is_an_argument_error(dec_cls, enc_cls, width):
    sink = FloatSink(array.array("d", [0.0]))
    with pytest.raises(SofaArgumentError):
        dec_cls(visitor=sink).feed(_farray_wire(enc_cls, [1.0, 2.0, 3.0], width))


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_narrow_float_destination_is_refused_rather_than_truncated(
    dec_cls, enc_cls
):
    """A Python ``float`` is a double, and so is every value written here; an
    ``array("f")`` would silently narrow, so it is refused."""
    sink = FloatSink(array.array("f", [0.0] * 4))
    with pytest.raises(SofaArgumentError):
        dec_cls(visitor=sink).feed(_farray_wire(enc_cls, [1.0, 2.0], 4))


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width", [4, 8])
def test_an_empty_float_array_fills_nothing_and_is_still_offered(
    dec_cls, enc_cls, width
):
    sink = FloatSink(array.array("d", []))
    assert dec_cls(visitor=sink).feed(_farray_wire(enc_cls, [], width)) is (
        Status.COMPLETE
    )
    assert sink.seen[0][2] == 0
    assert sink.materialized == []


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width", [4, 8])
def test_the_float_route_survives_a_chunk_boundary(dec_cls, enc_cls, width):
    values = [float(i) / 8 for i in range(40)]
    wire = _farray_wire(enc_cls, values, width)
    sink = FloatSink(array.array("d", [0.0] * len(values)))
    dec = dec_cls(visitor=sink, reassembly=bytearray(len(wire) + 16))
    for i in range(0, len(wire), 7):
        dec.feed(wire[i : i + 7])
    assert list(sink.dst) == values


# --- the point of all of it --------------------------------------------------


def _peak(work) -> int:
    work()
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        work()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_string_through_its_destination_does_not_scale(dec_cls, enc_cls):
    """The sharpest of the three cases as measured: with a caller reassembly
    buffer *and* a destination taken, a 1 MiB string used to cost a 1 MiB
    allocation inside the codec, because there was no third opt-out to take."""

    def run(size):
        wire = _string_wire(enc_cls, "x" * size)
        dst = bytearray(size)
        reassembly = bytearray(size + 4096)

        def work():
            sink = StringSink(dst)
            dec = dec_cls(visitor=sink, reassembly=reassembly)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(1 << 10)
    large = run(1 << 20)
    assert large - small < (64 << 10), (
        f"a 1 MiB string cost {large - small} bytes more than a 1 KiB one; the "
        "wire is still sizing an allocation"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("width", [4, 8])
def test_a_float_array_through_its_destination_does_not_scale(
    dec_cls, enc_cls, width
):
    def run(count):
        wire = _farray_wire(enc_cls, [1.5] * count, width)
        dst = array.array("d", [0.0] * count)
        reassembly = bytearray(len(wire) + 16)

        def work():
            sink = FloatSink(dst)
            dec = dec_cls(visitor=sink, reassembly=reassembly)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(64)
    large = run(64 << 10)
    assert large - small < (64 << 10), (
        f"a {64 << 10}-element array cost {large - small} bytes more than a "
        "64-element one; the wire is still sizing an allocation"
    )
