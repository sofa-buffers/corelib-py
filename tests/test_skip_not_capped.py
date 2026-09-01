"""A field the handler never materializes is not capped (#128).

CORELIB_PLAN §6.2.1 fixes both the purpose of a receiver limit and the point it
is enforced at:

    A limit MUST be enforced at the count/length header -- before the allocation
    it is meant to prevent.

A skipped field allocates nothing, so there is no allocation for the cap to
prevent and nothing for it to protect. §6.7.2 says the same from the other side:
the ``skip`` intent "neither materializes nor validates". This port used to raise
on all three routes into a skip, which made it the only member of the family to
reject a well-formed message that every other port accepts under the *same*
configured limits -- and §6.2.1 licenses two receivers to differ only when they
are configured *differently*.

The family, for the record. In C++ the cap is an argument of the read call
(``readArray(dst, schemaCount, dynCap)``), and a skipped field never reaches one;
a §7.3 tag mismatch is settled first, "so neither bound below can be applied to a
field that is not this field's value". In Rust the codec carries no ``max_dyn_*``
at all -- §6.2.1's "the numbers and the allocation are not the codec's", taken
literally. Here the visitor plays the part their generated layer plays: it is
told the announced count or length, and it decides.

What the cap still governs is the default route, where no destination comes back
and the decoder is the one that has to build a ``str``, a ``bytes`` or a list --
sized by the wire, which is exactly the allocation §6.2.1 is about.
``test_receiver_limits.py`` holds that side of the pair.
"""

from __future__ import annotations

import tracemalloc

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import ROOMY_REASSEMBLY, Recorder, Status, bound, walk

from sofab import ARRAY_MAX, FIXLEN_MAX, Binding, Encoder, SofaArgumentError, SofaLimitError

# The issue's own message: a 100-byte blob at id 7 the handler does not want,
# with a cap of 10 -- and a plain field on either side of it, so a skip that
# went wrong shows up as a lost or shifted neighbour rather than only as a
# verdict.
CAP = 10
PAYLOAD = bytes(range(100))


def _msg():
    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_bytes(7, PAYLOAD)
    enc.write_unsigned(3, 6)
    enc.flush()
    return enc.getvalue()


NEIGHBOURS = [("u", 1, 5), ("u", 3, 6)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_binding_that_does_not_name_the_field_skips_it(engine):
    """Route 1: the field is not in the table, so the driver walks past it."""
    b = Binding().unsigned(1, at=0, count_at=2).unsigned(3, at=1, count_at=3)
    status, dec, slots = bound(engine, _msg(), b, max_dyn_blob_len=CAP)
    assert status is Status.COMPLETE
    assert dec.error is None
    assert slots.u[0] == 5 and slots.u[1] == 6
    assert slots.u[2] == 1 and slots.u[3] == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_a_visitor_that_declines_the_field_skips_it(engine):
    """Route 2: ``on_field`` returns False."""
    status, rec, dec = walk(
        engine, _msg(), max_dyn_blob_len=CAP,
        recorder=Recorder(decline=lambda f: f.id == 7),
    )
    assert status is Status.COMPLETE
    assert dec.error is None
    assert rec.events == NEIGHBOURS


@pytest.mark.parametrize("engine", ENGINES)
def test_a_visitor_that_supplies_its_own_buffer_reads_it(engine):
    """Route 3, the sharpest of the three: the handler *wants* the blob and has
    already said where it goes. §6.6.3 calls that the conformant shape, and the
    storage is the caller's -- so the sender dictates no allocation of this
    decoder's and the cap has nothing to refuse."""
    dst = bytearray(4096)
    seen = []

    class Sink(Recorder):
        def on_blob_begin(self, field_id, size):
            seen.append((field_id, size))
            return dst

    status, rec, dec = walk(engine, _msg(), max_dyn_blob_len=CAP, recorder=Sink())
    assert status is Status.COMPLETE
    assert dec.error is None
    assert seen == [(7, len(PAYLOAD))]
    assert dst[: len(PAYLOAD)] == PAYLOAD
    assert rec.events == NEIGHBOURS


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [1, 7, 64], ids=lambda n: f"chunk{n}")
def test_the_skip_survives_a_chunk_boundary(engine, chunk):
    """§5.2: the skip suspends and resumes like any other read, and the cap is
    gone on the retry as it was on the first attempt -- not unparked twice, and
    not re-raised once the payload finally arrives."""
    status, rec, dec = walk(
        engine, _msg(), chunk=chunk, max_dyn_blob_len=CAP,
        recorder=Recorder(decline=lambda f: f.id == 7),
    )
    assert status is Status.COMPLETE
    assert dec.error is None
    assert rec.events == NEIGHBOURS


@pytest.mark.parametrize("engine", ENGINES)
def test_a_skip_does_not_need_the_reassembly_buffer_at_all(engine):
    """The other thing a skip must not be charged for, and the reason it is the
    same rule as the cap (#139).

    A payload being *built* has to be joined somewhere, and §6.6.2 makes that
    somewhere the caller's buffer: it arrives in pieces and the value needs all
    of them at once. A payload being *discarded* needs none of them at once. It
    has no value to rebuild, so nothing has to be replayed from its first byte
    -- what the bytes need is dropping, and dropping is done a chunk at a time.

    This port used to retain a skipped construct anyway, because its resume
    contract replayed every construct alike, and then refused the decode once
    the retained bytes outgrew the buffer. That turned "the receiver ignores
    this field" (MESSAGE_SPEC §7.3) into "this field ends the decode", which is
    the one thing §7.3 rules out -- and made it depend on where the chunks fell,
    which §7.2 item 4 rules out separately.

    So: the smallest buffer the constructor accepts, a payload six times its
    size, and one byte at a time.
    """
    data = _msg()
    rec = Recorder(decline=lambda f: f.id == 7)
    dec = engine(
        max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX,
        max_dyn_blob_len=CAP, visitor=rec, reassembly=bytearray(16),
    )
    status = Status.COMPLETE
    for off in range(0, len(data), 1):
        status = dec.feed(data[off : off + 1])
    assert status is Status.COMPLETE
    assert dec.error is None
    # The values on either side, not merely "it did not raise": a skip that
    # lost its place would take a neighbour with it.
    assert rec.events == NEIGHBOURS


@pytest.mark.parametrize("engine", ENGINES)
def test_a_read_that_spans_a_chunk_still_needs_the_buffer(engine):
    """The control for the case above, so it cannot pass by the buffer having
    stopped mattering. The same bytes, the same 16-byte buffer, a handler that
    *takes* the blob: now there is a value to build, all hundred bytes of it are
    needed at once, and §6.6.2's refusal is the right answer."""
    data = _msg()
    dec = engine(
        max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX,
        max_dyn_blob_len=FIXLEN_MAX, visitor=Recorder(),
        reassembly=bytearray(16),
    )
    with pytest.raises(SofaArgumentError):
        for off in range(0, len(data), 8):
            dec.feed(data[off : off + 8])


@pytest.mark.parametrize("engine", ENGINES)
def test_the_cap_still_fires_on_the_route_that_allocates(engine):
    """The other half of the rule, so this file cannot pass by the cap having
    been removed: the same bytes, the same limit, a handler that takes the
    default. Nothing but a ``bytes`` of the decoder's own can hold the payload,
    and the wire is the only size it could build one from."""
    dec = engine(reassembly=ROOMY_REASSEMBLY, max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Recorder(), max_dyn_blob_len=CAP)
    with pytest.raises(SofaLimitError):
        dec.feed(_msg())


@pytest.mark.parametrize("engine", ENGINES)
def test_an_array_the_handler_skips_is_not_capped_either(engine):
    """The count-header sibling of the length-header case above. Same rule, and
    it has to be the same rule: a handler cannot be told that a blob it declines
    is free while an array it declines is a rejection."""
    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_unsigned_array(7, list(range(100)))
    enc.write_unsigned(3, 6)
    enc.flush()

    status, rec, dec = walk(
        engine, enc.getvalue(), max_dyn_array_count=CAP,
        recorder=Recorder(decline=lambda f: f.id == 7),
    )
    assert status is Status.COMPLETE
    assert dec.error is None
    assert rec.events == NEIGHBOURS


# --- a skipped construct of ANY size, on a chunk-fed receiver (#139) ---------
#
# The measured shape the rule above is really about. Everything up to here uses
# a 100-byte payload, which fits any buffer a test would think to pass, so the
# suite could pin "a skip is not capped" while a skip was still being charged
# for the reassembly space it did not need. What §7.3 actually promises is that
# a field the receiver IGNORES costs it nothing -- so the payload here is chosen
# to be larger than any buffer a receiver would size for the fields it does
# read, and the assertions are on the VALUES of the fields around it.
#
# 4 KiB was the reassembly size this library used to pick for a caller who named
# none, so 4 KiB chunks against a payload past it is exactly the failure that
# was reported: an ordinary two-field message plus one unknown-id field died
# with SofaArgumentError and lost both fields.

SURROUND = [("str", 1, "hi"), ("u", 2, 7)]


def _with_unknown(kind, size, *, first=False):
    """An ordinary in-policy message plus ONE oversized field at an id the
    receiver below does not know."""
    enc = Encoder()
    if not first:
        enc.write_string(1, "hi")
        enc.write_unsigned(2, 7)
    if kind == "blob":
        enc.write_bytes(9000, b"\x5a" * size)
    elif kind == "string":
        enc.write_string(9000, "z" * size)
    elif kind == "array":
        enc.write_unsigned_array(9000, [1] * size)
    elif kind == "sequence":
        enc.write_sequence_begin_lazy(9000)
        enc.write_bytes(3, b"\x5a" * size)
        enc.write_sequence_end_keep()
    if first:
        enc.write_string(1, "hi")
        enc.write_unsigned(2, 7)
    enc.flush()
    return enc.getvalue()


def _known_only(engine, wire, chunk, **kw):
    """Feed ``wire`` to a receiver that knows ids 1 and 2 and nothing else, so
    id 9000 is an unknown id in the §7.3 sense -- not a field a visitor hook
    happens to be offered."""
    rec = Recorder(decline=lambda f: f.id not in (1, 2))
    dec = engine(
        max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX,
        max_dyn_blob_len=FIXLEN_MAX, visitor=rec, **kw
    )
    status = Status.COMPLETE
    for off in range(0, len(wire), chunk):
        status = dec.feed(wire[off : off + chunk])
    return status, rec, dec


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("first", [False, True], ids=["appended", "leading"])
@pytest.mark.parametrize(
    ("kind", "size"),
    [
        ("blob", 5_000),
        ("blob", 200_000),
        ("blob", 1 << 20),
        ("string", 200_000),
        ("array", 50_000),
        ("sequence", 200_000),
    ],
    ids=lambda v: str(v),
)
def test_an_unknown_field_of_any_size_costs_the_decode_nothing(
    engine, kind, size, first
):
    """§7.3 / §6.2.1: the field is ignored, so its size is not the receiver's
    problem. Both positions, because a leading one loses *every* field."""
    wire = _with_unknown(kind, size, first=first)
    status, rec, dec = _known_only(engine, wire, 4096, reassembly=4096)
    assert status is Status.COMPLETE
    assert dec.error is None
    assert rec.events == SURROUND


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [1, 3, 17], ids=lambda n: f"chunk{n}")
def test_the_outcome_does_not_depend_on_where_the_chunks_fall(engine, chunk):
    """§7.2 item 4, which the old behaviour broke outright: the very same bytes
    were COMPLETE in one feed and a raised SofaArgumentError in small pieces."""
    wire = _with_unknown("blob", 40_000)
    whole = _known_only(engine, wire, len(wire), reassembly=64)
    split = _known_only(engine, wire, chunk, reassembly=64)
    assert whole[0] is split[0] is Status.COMPLETE
    assert whole[1].events == split[1].events == SURROUND


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mistyped_field_is_skipped_at_any_size_too(engine):
    """MESSAGE_SPEC §7.3's other half: the id IS declared, under a tag the wire
    contradicts. Same treatment as an unknown id, so the same size must be free
    -- and this route reaches the skip through a Binding, where no visitor hook
    is consulted at all."""
    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_bytes(7, b"\x5a" * 200_000)  # declared unsigned below
    enc.write_unsigned(3, 6)
    enc.flush()

    b = Binding().unsigned(1, at=0).unsigned(7, at=1).unsigned(3, at=2)
    words = bytearray(b.tree_words_required * 8)
    objects: list = [None] * b.tree_objects_required
    dec = engine(
        binding=b, words=words, objects=objects, reassembly=512,
        max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX,
        max_dyn_blob_len=FIXLEN_MAX,
    )
    wire = enc.getvalue()
    status = Status.COMPLETE
    for off in range(0, len(wire), 256):
        status = dec.feed(wire[off : off + 256])
    assert status is Status.COMPLETE
    assert dec.error is None
    u = memoryview(words).cast("Q")
    assert u[0] == 5 and u[2] == 6
    assert u[1] == 0  # the slot the mistyped field must not have touched


@pytest.mark.parametrize("engine", ENGINES)
def test_the_memory_a_skip_costs_does_not_grow_with_the_payload(engine):
    """§6.6's own wording for what "bounded" means -- storage that "does not
    grow with the message". A verdict-only assertion would pass with the payload
    retained in a buffer big enough to hold it, so measure instead: twenty-five
    times the payload, through the same reassembly buffer, for the same money.

    The control is the read of the same field, which must allocate what it
    returns -- without it a broken measurement would pass this vacuously.
    """
    small = _with_unknown("blob", 40_000)
    large = _with_unknown("blob", 1_000_000)

    def peak(fn):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            fn()
            top = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return top - base

    skipped_small = peak(lambda: _known_only(engine, small, 4096, reassembly=4096))
    skipped_large = peak(lambda: _known_only(engine, large, 4096, reassembly=4096))

    # Control: the same bytes with the field taken, in one feed, which has to
    # build the payload it hands over.
    taken = peak(
        lambda: engine(
            max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX,
            max_dyn_blob_len=FIXLEN_MAX, visitor=Recorder(), reassembly=4096,
        ).feed(large)
    )
    if taken < 500_000:
        pytest.skip("tracemalloc is not measuring allocation on this build")

    # 25x the payload, and the skip must not have grown by anything like it.
    assert skipped_large < skipped_small + 100_000
