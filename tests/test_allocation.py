"""What the codec allocates, measured (CORELIB_PLAN §6.6.4).

§6.6.4 requires conformance to be checked **both** ways: *read* — no allocation
primitive reachable from a codec entry point, apart from the language-forced
handles §6.6.2 allows — and *measure*. It then names what "measure" means for a
runtime that boxes the codec's values, which CPython does:

    Where it does box them the count is never zero and demanding zero would
    demand the impossible, so the measurable claim is that it **does not grow
    with the message**: the same for a ten-byte and a ten-kilobyte payload of the
    same field shape, and unchanged by a hostile count or length. That is the
    property the prohibition is for, and it is what a port in such a language
    pins with a test.

The *read* half is the source and the README's itemised handle list. This file is
the *measure* half, in exactly that form: a payload a thousand times larger costs
the same, and a hostile count buys nothing. It also pins the other half of §6.6 —
bounded working state **sized at construction**, measured with the codec built
outside the measurement, which is the line that section draws.

Every route now has a shape that does not scale.
``test_a_visitor_that_takes_a_value_pays_for_it`` measures what the *other*
shape costs — a handler that asks for the value rather than supplying the
storage — so the difference between the two is a number and not a claim.
"""

from __future__ import annotations

import tracemalloc

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import ENCODER_ENGINES as ENCODERS
from vectors import NO_CAPS

from sofab import MAX_DEPTH, Binding, Status, Visitor

SMALL = 1 << 10
LARGE = 1 << 20

#: What the handles and the fixed working state may cost, in bytes, independent
#: of the payload. Generous on purpose: the claim under test is that the cost
#: does not *scale*, and a threshold tight enough to argue about would be
#: measuring the interpreter rather than the codec.
FLAT = 96 << 10


def _peak(work) -> int:
    """Peak traced allocation over ``work``, past its own construction."""
    work()  # warm: lazily built state is not what is being measured
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        work()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def _blob_wire(enc_cls, payload: bytes) -> bytes:
    enc = enc_cls()
    enc.write_bytes(1, payload)
    enc.flush()
    return enc.getvalue()


# --- encode ------------------------------------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_encode_does_not_scale_with_the_payload(enc_cls):
    """A blob a thousand times larger streams through the same fixed buffer.

    §5.1.2 leaves the output buffer to the caller and §5.1.6 forbids handing the
    sink anything else, so the only thing between the two sizes is the number of
    flushes — never a bigger allocation.
    """

    def run(payload):
        def work():
            buf = bytearray(4096)
            enc = enc_cls.over_buffer(buf, 0, lambda chunk: None)
            enc.write_bytes(1, payload)
            enc.flush()

        return _peak(work)

    small = run(b"x" * SMALL)
    large = run(b"x" * LARGE)
    assert large - small < FLAT, (
        f"a {LARGE}-byte payload cost {large - small} bytes more than a "
        f"{SMALL}-byte one; the wire is sizing an allocation"
    )


# --- decode, with the caller supplying the storage ---------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_decode_does_not_scale_when_the_caller_supplies_the_storage(dec_cls, enc_cls):
    """``reassembly=`` for the pieces (§6.6.2) and ``on_blob_begin`` for the
    payload (§6.6.3), which between them are what §6.6 asks a port to offer.

    Both buffers are the caller's and both are sized **once**, from the schema,
    before anything is measured — which is §6.6's whole point: "a caller can
    bound a decode's memory by construction instead of by measurement". The
    destination is filled in one piece, so a payload split across chunks is made
    contiguous in the reassembly buffer first; sizing that buffer for the largest
    field the schema allows is the caller's job and not a cost the codec adds.
    """

    class Sink(Visitor):
        def __init__(self, dst):
            self.dst = dst

        def on_blob_begin(self, field_id, size):
            return self.dst

    dst = bytearray(LARGE)
    reassembly = bytearray(LARGE + 4096)

    def run(payload):
        wire = _blob_wire(enc_cls, payload)

        def work():
            dec = dec_cls(**NO_CAPS, visitor=Sink(dst), reassembly=reassembly)
            for i in range(0, len(wire), 4096):
                dec.feed(wire[i : i + 4096])

        return _peak(work)

    small = run(b"x" * SMALL)
    large = run(b"x" * LARGE)
    assert large - small < FLAT, (
        f"a {LARGE}-byte payload cost {large - small} bytes more than a "
        f"{SMALL}-byte one; the wire is sizing an allocation"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_binding_decode_does_not_scale_with_an_arrays_length(dec_cls, enc_cls):
    """A Binding writes elements into slots the caller sized from the schema, so
    a longer array costs wire bytes and no allocation."""

    def run(count):
        enc = enc_cls()
        enc.write_unsigned_array(1, list(range(count)))
        enc.flush()
        wire = enc.getvalue()
        binding = Binding().unsigned_array(1, at=0, cap=LARGE // 8, count_at=1)
        words = bytearray(8 * (LARGE // 8 + 8))

        def work():
            dec = dec_cls(**NO_CAPS, binding=binding, words=words)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(64)
    large = run(64 << 10)
    assert large - small < FLAT, (
        f"a {64 << 10}-element array cost {large - small} bytes more than a "
        "64-element one; the wire is sizing an allocation"
    )


# --- bounded working state is sized at construction (§6.6) -------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_the_hold_back_run_is_sized_at_construction(enc_cls):
    """§6.6: bounded working state "MUST be sized to its **full extent** when
    the codec is constructed. Growing it afterwards is forbidden even where the
    ceiling it grows towards is correct: a pending run that doubles as nesting
    deepens allocates on a `write` path, and that is what this section forbids."

    Both engines grew the hold-back run on demand — the pure one by appending to
    a list, the native one by doubling a ``realloc``. The encoder is built
    *outside* the measurement here, which is exactly the line §6.6 draws.
    """

    def run(depth, keep):
        enc = enc_cls.over_buffer(bytearray(1 << 16), 0, lambda chunk: None)
        close = enc.write_sequence_end_keep if keep else enc.write_sequence_end

        def work():
            for i in range(depth):
                enc.write_sequence_begin_lazy(i)
            if keep:
                enc.write_unsigned(1, 7)  # content: commits the held-back run
            for _ in range(depth):
                close()

        return _peak(work)

    # Read against a control at the same depth, not against a shallower one: on
    # a runtime that boxes every value, writing more fields costs interpreter
    # memory whatever the codec does, and that cost differs by interpreter
    # version. The control opens the same MAX_DEPTH sequences and closes them
    # with the dropping closer, so the run is filled and popped but never
    # committed; the measured leg commits it.
    control = run(MAX_DEPTH - 1, keep=False)
    committed = run(MAX_DEPTH - 1, keep=True)
    assert committed - control < 1024, (
        f"committing a {MAX_DEPTH - 1}-deep run cost {committed - control} "
        "bytes more than filling one; the hold-back run is growing on a write "
        "path"
    )

    # And the runs themselves are the same two lists, at the same length,
    # afterwards. The pair is compared as a set: committing *swaps* them, so a
    # sink that re-enters the encoder writes into the spare rather than over the
    # run being emitted, and neither is ever replaced.
    enc = enc_cls.over_buffer(bytearray(1 << 16), 0, lambda chunk: None)
    if getattr(enc, "_pending", None) is not None:
        before = {(id(enc._pending), len(enc._pending)),
                  (id(enc._spare), len(enc._spare))}
        for i in range(MAX_DEPTH - 1):
            enc.write_sequence_begin_lazy(i)
        enc.write_unsigned(1, 7)
        for _ in range(MAX_DEPTH - 1):
            enc.write_sequence_end_keep()
        after = {(id(enc._pending), len(enc._pending)),
                 (id(enc._spare), len(enc._spare))}
        assert after == before, "a write replaced or extended the hold-back run"
        assert len(enc._pending) == MAX_DEPTH


def _deep_wire(enc_cls, depth):
    enc = enc_cls()
    for i in range(depth):
        enc.write_sequence_begin_lazy(i)
    enc.write_unsigned(1, 7)
    for _ in range(depth):
        enc.write_sequence_end_keep()
    enc.flush()
    return enc.getvalue()


class _Descending(Visitor):
    """Takes each nested scope with a handler, so the suspended-handler stack
    is used at every level."""

    def on_sequence_begin(self, field_id):
        return self


class _Flat(Visitor):
    """Walks the identical bytes without ever handing a scope over, so the
    handler stack is never touched. Everything else — the calls, the cursor
    arithmetic, the ints the interpreter boxes on the way — is the same."""

    def on_sequence_begin(self, field_id):
        return None


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_descending_costs_no_more_than_walking_the_same_bytes(dec_cls, enc_cls):
    """§6.6: the decoder's descent state is sized at construction, so descending
    `MAX_DEPTH` levels must cost nothing a flat walk of the same message does
    not. It was built on the *first descent*, which is inside ``feed``.

    Read against a **control**, not against a shallower message: on a runtime
    that boxes every value, a longer message costs interpreter memory whatever
    the codec does, and that cost is what a depth-1-against-depth-254
    comparison actually measures (it differs by interpreter version). The
    control here walks the identical bytes and makes the identical calls; the
    only difference is whether the handler stack is used at all.
    """
    wire = _deep_wire(enc_cls, MAX_DEPTH - 1)

    def run(handler_cls):
        dec = dec_cls(**NO_CAPS, visitor=handler_cls(), reassembly=bytearray(1 << 12))

        def work():
            dec.reset()
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    flat = run(_Flat)
    descending = run(_Descending)
    assert descending - flat < 512, (
        f"descending {MAX_DEPTH - 1} levels cost {descending - flat} bytes more "
        "than walking the same bytes flat; the handler stack is growing inside "
        "feed"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_decoders_descent_state_is_the_same_containers_afterwards(
    dec_cls, enc_cls
):
    """§6.6.4's *read* half, as an assertion rather than an inspection.

    A measurement cannot separate a container that grew from an interpreter
    that allocated; identity and length can. After a `MAX_DEPTH`-deep descent
    the decoder must hold the **same** stack objects, at the **same** length,
    it was constructed with. (The accelerator keeps its two stacks in one C
    block, which no Python-level check can reach; the byte-for-byte parity
    suite is what pins it to the engine measured here.)
    """
    dec = dec_cls(**NO_CAPS, visitor=_Descending(), reassembly=bytearray(1 << 12))
    stacks = [getattr(dec, n, None) for n in ("_vstack", "_bstack")]
    if all(s is None for s in stacks):
        pytest.skip("the native engine holds these in C memory")
    before = [(id(s), len(s)) for s in stacks if s is not None]

    assert dec.feed(_deep_wire(enc_cls, MAX_DEPTH - 1)) is Status.COMPLETE

    after = [
        (id(getattr(dec, n)), len(getattr(dec, n)))
        for n in ("_vstack", "_bstack")
        if getattr(dec, n, None) is not None
    ]
    assert after == before, "a descent replaced or extended a stack"
    assert all(length == MAX_DEPTH for _ident, length in before)


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_binding_float_array_does_not_scale_with_its_length(dec_cls, enc_cls):
    """The float twin of the unsigned case above.

    It used to build a wire-sized ``list`` *and* a wire-sized ``array`` on the
    way into slots the caller had already sized — 2.8 MB for a 65,536-element
    ``fp32`` array on the pure engine. The payload now crosses in fixed blocks
    (``_core.FARRAY_CHUNK``), straight out of the fed buffer.
    """

    def run(count):
        enc = enc_cls()
        enc.write_float32_array(1, [1.5] * count)
        enc.flush()
        wire = enc.getvalue()
        binding = Binding().float32_array(1, at=0, cap=64 << 10, count_at=1)
        words = bytearray(8 * ((64 << 10) + 8))

        def work():
            dec = dec_cls(**NO_CAPS, binding=binding, words=words)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(64)
    large = run(64 << 10)
    assert large - small < FLAT, (
        f"a {64 << 10}-element fp32 array cost {large - small} bytes more than "
        "a 64-element one; the wire is sizing an allocation"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("kind", ["string", "blob"])
def test_a_declared_byte_destination_does_not_scale_with_the_payload(
    dec_cls, enc_cls, kind
):
    """§6.6.3's third shape, for the two aggregates that had no machine form.

    A ``Binding`` row used to name a slot to *put a value in*, so a whole message
    could decode into caller storage and a 1 MiB string would still cost a 1 MiB
    allocation inside the codec — on the very route §6.6.3 names.
    :meth:`sofab.Binding.string_into` / :meth:`~sofab.Binding.blob_into` name a
    slot that already **holds** the buffer, and the payload is copied into it.
    """

    def run(n):
        enc = enc_cls()
        if kind == "string":
            enc.write_string(1, "x" * n)
            table = Binding().string_into(1, at=0, count_at=0)
        else:
            enc.write_bytes(1, b"x" * n)
            table = Binding().blob_into(1, at=0, count_at=0)
        enc.flush()
        wire = enc.getvalue()
        objs = [bytearray(LARGE)]
        words = bytearray(8 * table.tree_words_required)

        def work():
            dec = dec_cls(**NO_CAPS, binding=table, words=words, objects=objs)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(SMALL)
    large = run(LARGE)
    assert large - small < FLAT, (
        f"a {LARGE}-byte {kind} cost {large - small} bytes more than a "
        f"{SMALL}-byte one; the wire is sizing an allocation"
    )


# --- the gap this port has accepted -----------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_visitor_that_takes_a_value_pays_for_it(dec_cls, enc_cls):
    """What asking for the value costs, with a number on it rather than a claim.

    ``on_bytes`` receives a whole ``bytes``, and the only size the codec can
    build one from is the wire's -- which is exactly what §6.6.3 says a callback
    delivering a materialized aggregate obliges. This port ships that callback
    anyway, beside the ``on_blob_begin`` route that does not, because it is the
    convenient way to read a message. The measurement pins the shape: the cost
    tracks the payload, **once** -- not twice, and not more.
    """

    class Taker(Visitor):
        def __init__(self):
            self.n = 0

        def on_bytes(self, field_id, value):
            self.n = len(value)

    def run(payload):
        wire = _blob_wire(enc_cls, payload)

        def work():
            dec = dec_cls(**NO_CAPS, visitor=Taker())
            dec.feed(wire)

        return _peak(work)

    small = run(b"x" * SMALL)
    large = run(b"x" * LARGE)
    grown = large - small
    assert grown > LARGE * 0.9, "the value must actually be materialized"
    assert grown < 3 * LARGE, (
        f"the payload was materialized {grown / LARGE:.1f} times over; one copy "
        "is the cost this port accepts, more is a defect"
    )


# --- and it costs the value ONCE: nothing else is sized from the wire --------
#
# §6.6.3 obliges the codec to build the aggregate a value-taking callback asks
# for, and this port ships those callbacks. What it must not do is size anything
# *else* from the wire on the way there -- a scratch copy of the payload, or a
# second pass into a second container, is the codec's own allocation, seen by
# nobody, and squarely what §6.6 forbids. Each test below reads the cost of one
# route against a control that delivers the same value without the extra copy,
# so the claim is a comparison and not a threshold anyone has to argue about.


def _peak_visit(dec_cls, wire, sink_cls):
    def work():
        dec_cls(**NO_CAPS, visitor=sink_cls()).feed(wire)

    return _peak(work)


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_on_string_costs_the_str_and_not_a_copy_of_the_payload_as_well(
    dec_cls, enc_cls
):
    """The ``str`` is the value the handler asked for. The ``bytes`` slice the
    payload used to be copied into on the way to it was not: it was scratch the
    wire sized, and a 1 MiB string paid for it twice."""

    class Taker(Visitor):
        def on_string(self, field_id, value):
            self.n = len(value)

    def run(n):
        enc = enc_cls()
        enc.write_string(1, "x" * n)
        enc.flush()
        return _peak_visit(dec_cls, enc.getvalue(), Taker)

    grown = run(LARGE) - run(SMALL)
    assert grown > LARGE * 0.9, "the str must actually be materialized"
    assert grown < LARGE * 1.5, (
        f"a {LARGE}-byte string cost {grown} bytes: the payload is being copied "
        "before the str is built"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_signed_array_costs_what_an_unsigned_array_of_the_same_length_costs(
    dec_cls, enc_cls
):
    """Read against the unsigned twin rather than a threshold: the two deliver
    the same shape — one ``list`` of ``count`` boxed ints — so whatever CPython
    charges for that, it charges both.

    ZigZag used to be undone in a **second pass that built a second list**, so
    the signed route peaked at twice the unsigned one. Folding the transform
    into the element loop costs the same arithmetic where the raw value is
    already in hand, and allocates nothing extra.
    """

    class Taker(Visitor):
        def on_unsigned_array(self, field_id, values):
            self.n = len(values)

        def on_signed_array(self, field_id, values):
            self.n = len(values)

    count = 64 << 10

    def run(signed):
        enc = enc_cls()
        if signed:
            enc.write_signed_array(1, [-1] * count)
        else:
            enc.write_unsigned_array(1, [1] * count)
        enc.flush()
        return _peak_visit(dec_cls, enc.getvalue(), Taker)

    unsigned = run(False)
    signed = run(True)
    assert signed < unsigned * 1.25, (
        f"a {count}-element signed array cost {signed} bytes against the "
        f"unsigned twin's {unsigned}; the ZigZag pass is building a second list"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_an_fp64_array_costs_what_an_fp32_array_of_the_same_length_costs(
    dec_cls, enc_cls
):
    """Both deliver one ``list`` of ``count`` Python floats — a Python float is
    a double whichever subtype it came from — so the two costs must match.

    They did not: the payload was copied into a whole ``bytes`` before the list
    was built, and that copy *is* the difference between the subtypes (4 bytes
    per element against 8). Reading the values straight out of the buffer they
    were fed into removes it, and with it the only term that told the two routes
    apart.
    """

    class Taker(Visitor):
        def on_float32_array(self, field_id, values):
            self.n = len(values)

        def on_float64_array(self, field_id, values):
            self.n = len(values)

    count = 64 << 10

    def run(width):
        enc = enc_cls()
        if width == 4:
            enc.write_float32_array(1, [1.5] * count)
        else:
            enc.write_float64_array(1, [1.5] * count)
        enc.flush()
        return _peak_visit(dec_cls, enc.getvalue(), Taker)

    fp32 = run(4)
    fp64 = run(8)
    # The payload copy was exactly count*width bytes, so the two subtypes
    # differed by count*4. Anything under one byte per element cannot be it.
    assert abs(fp64 - fp32) < count, (
        f"an fp64 array cost {fp64} bytes against an fp32 array's {fp32} at the "
        "same length; the payload is being copied before the list is built"
    )


# --- the handle list is the whole list (§6.6.2) ------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_an_encoder_holds_only_the_handles_the_readme_lists(enc_cls):
    """§6.6.2 charges visibility for the exception: a handle nobody listed must
    fail here rather than hide behind the paragraph."""
    buf = bytearray(64)
    enc = enc_cls.over_buffer(buf, 0, lambda chunk: None)
    enc.write_bytes(1, b"x" * 400)
    enc.flush()
    held = _memoryviews(enc)
    assert held <= 2, (
        f"the encoder holds {held} memoryviews; the README lists one per "
        "installation plus one kept flush slice"
    )


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_binding_decoder_holds_only_the_handles_the_readme_lists(dec_cls):
    binding = Binding().unsigned(1, at=0)
    dec = dec_cls(**NO_CAPS, binding=binding, words=bytearray(64))
    held = _memoryviews(dec)
    assert held <= 3, (
        f"the decoder holds {held} memoryviews; the README lists the views over "
        "the words buffer"
    )


def _memoryviews(obj) -> int:
    """How many memoryviews the object holds, whichever engine it is.

    The pure engine keeps them in ``__slots__``/``__dict__``; the native one in
    cdef attributes, which are not reachable by name -- there the referents are
    counted instead.
    """
    import gc

    seen = {
        id(v)
        for v in gc.get_referents(obj)
        if isinstance(v, memoryview)
    }
    for name in getattr(type(obj), "__slots__", ()):
        v = getattr(obj, name, None)
        if isinstance(v, memoryview):
            seen.add(id(v))
    return len(seen)
