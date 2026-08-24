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

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder, Status, bound, walk

from sofab import Binding, Encoder, SofaLimitError, SofaRangeError

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
def test_the_callers_reassembly_buffer_is_what_bounds_a_skip(engine):
    """What replaces the cap on this route, and why the swap is the right one.

    A skipped payload that spans a chunk boundary still has to be joined
    somewhere, and §6.6.2 makes that somewhere the caller's buffer. So the
    memory a skip can cost is bounded -- by the buffer the caller sized, not by
    a policy limit -- and a payload that does not fit is refused on
    :class:`SofaRangeError`.

    That is the right channel for it: a capacity fact about the caller's own
    storage, not a policy verdict on the message. The bytes are well formed, and
    the same message skips cleanly through a longer buffer.
    """
    data = _msg()

    roomy = bytearray(4096)
    status, rec, dec = walk(
        engine, data, chunk=8, max_dyn_blob_len=CAP, reassembly=roomy,
        recorder=Recorder(decline=lambda f: f.id == 7),
    )
    assert status is Status.COMPLETE
    assert rec.events == NEIGHBOURS

    dec = engine(
        visitor=Recorder(decline=lambda f: f.id == 7),
        max_dyn_blob_len=CAP,
        reassembly=bytearray(16),
    )
    with pytest.raises(SofaRangeError):
        for off in range(0, len(data), 8):
            dec.feed(data[off : off + 8])


@pytest.mark.parametrize("engine", ENGINES)
def test_the_cap_still_fires_on_the_route_that_allocates(engine):
    """The other half of the rule, so this file cannot pass by the cap having
    been removed: the same bytes, the same limit, a handler that takes the
    default. Nothing but a ``bytes`` of the decoder's own can hold the payload,
    and the wire is the only size it could build one from."""
    dec = engine(visitor=Recorder(), max_dyn_blob_len=CAP)
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
