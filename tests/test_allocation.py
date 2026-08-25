"""What the codec allocates, measured (CORELIB_PLAN §6.6.4).

    Source inspection alone is still **not sufficient** [...] Conformance
    therefore requires **both**: *read* — no allocation primitive is reachable
    from a codec entry point, apart from the language-forced handles §6.6.2
    allows; *measure* — an allocation count, or the heap high-water mark, over a
    complete encode and a complete decode.

The *read* half is the source and the README's itemised handle list. This file is
the *measure* half, and it measures the property that list exists to protect:
**no wire number sizes an allocation.** A payload a thousand times larger must
not cost a thousand times the memory — that, not a raw byte count, is what §6.6
is about, and it is the one thing a Python port can state honestly.

Two of the three paths hold. The third does not, by a decision this port has
taken deliberately, and is measured here rather than left as an assertion —
``test_a_visitor_that_takes_a_value_pays_for_it`` is that gap with a number on
it.
"""

from __future__ import annotations

import tracemalloc

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import ENCODER_ENGINES as ENCODERS

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
            dec = dec_cls(visitor=Sink(dst), reassembly=reassembly)
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
            dec = dec_cls(binding=binding, words=words)
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

    def run(depth):
        enc = enc_cls.over_buffer(bytearray(1 << 16), 0, lambda chunk: None)

        def work():
            for i in range(depth):
                enc.write_sequence_begin_lazy(i)
            enc.write_unsigned(1, 7)  # content: commits the whole held-back run
            for _ in range(depth):
                enc.write_sequence_end_keep()

        return _peak(work)

    shallow = run(1)
    deep = run(MAX_DEPTH - 1)
    assert deep - shallow < 1024, (
        f"nesting {MAX_DEPTH - 1} deep cost {deep - shallow} bytes more than "
        "nesting once; the hold-back run is growing on a write path"
    )


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_decoders_descent_state_is_sized_at_construction(dec_cls, enc_cls):
    """The same rule on the decode side: the table stack and the suspended-
    handler stack are the decoder's own working state, so descending must not
    allocate. Both were built on the *first descent*, which is inside ``feed``.
    """

    class Child(Visitor):
        def on_sequence_begin(self, field_id):
            return self

    def wire_for(depth):
        enc = enc_cls()
        for i in range(depth):
            enc.write_sequence_begin_lazy(i)
        enc.write_unsigned(1, 7)
        for _ in range(depth):
            enc.write_sequence_end_keep()
        enc.flush()
        return enc.getvalue()

    def run(depth):
        wire = wire_for(depth)
        dec = dec_cls(visitor=Child(), reassembly=bytearray(1 << 12))

        def work():
            dec.reset()
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    shallow = run(1)
    deep = run(MAX_DEPTH - 1)
    assert deep - shallow < 1024, (
        f"descending {MAX_DEPTH - 1} deep cost {deep - shallow} bytes more than "
        "descending once; the decoder's descent state is growing inside feed"
    )


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
            dec = dec_cls(binding=binding, words=words)
            assert dec.feed(wire) is Status.COMPLETE

        return _peak(work)

    small = run(64)
    large = run(64 << 10)
    assert large - small < FLAT, (
        f"a {64 << 10}-element fp32 array cost {large - small} bytes more than "
        "a 64-element one; the wire is sizing an allocation"
    )


# --- the gap this port has accepted -----------------------------------------


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_a_visitor_that_takes_a_value_pays_for_it(dec_cls, enc_cls):
    """The accepted §6.6.3 gap, with a number on it rather than a claim.

    ``on_bytes`` receives a whole ``bytes``, and the only size the codec can
    build one from is the wire's -- which is exactly what §6.6.3 says a callback
    delivering a materialized aggregate obliges. This port ships that callback
    anyway, alongside the ``on_blob_begin`` route that does not; the README says
    so. The measurement pins the shape: the cost tracks the payload, once.
    """

    class Taker(Visitor):
        def __init__(self):
            self.n = 0

        def on_bytes(self, field_id, value):
            self.n = len(value)

    def run(payload):
        wire = _blob_wire(enc_cls, payload)

        def work():
            dec = dec_cls(visitor=Taker())
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
    dec = dec_cls(binding=binding, words=bytearray(64))
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
