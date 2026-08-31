"""``reassembly=``: joining a split construct in the caller's memory (§6.6).

CORELIB_PLAN §6.6 makes the codec heap-free after construction, and §6.6.2 is
explicit about the one case that tempts an implementation to allocate:

    A payload split across fed chunks has to be joined somewhere. That somewhere
    is storage the caller supplied [...] A codec MUST NOT grow a private
    accumulator instead.

Passing ``reassembly=bytearray(n)`` is what makes that true here: the pieces are
copied into the caller's buffer as they arrive, and a construct that does not fit
is **refused** rather than accommodated. That is what lets a caller bound a
decode's memory by construction instead of by measurement.

Every case runs on both engines, and every one of them feeds the message in
pieces -- there is nothing to test otherwise.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import NO_CAPS, Recorder, Status

from sofab import DEFAULT_REASSEMBLY, Encoder, SofaArgumentError

BIG = "a" * 300


def _msg():
    enc = Encoder()
    enc.write_unsigned(1, 300)
    enc.write_string(2, BIG)
    enc.write_bytes(3, b"x" * 200)
    enc.write_unsigned_array(4, list(range(50)))
    enc.write_unsigned(5, 7)
    enc.flush()
    return enc.getvalue()


WANT = [
    ("u", 1, 300),
    ("str", 2, BIG),
    ("blob", 3, b"x" * 200),
    ("ua", 4, tuple(range(50))),
    ("u", 5, 7),
]


def _feed(engine, wire, chunk, **kw):
    rec = Recorder()
    dec = engine(**NO_CAPS, visitor=rec, **kw)
    status = Status.COMPLETE
    for i in range(0, len(wire), chunk):
        status = dec.feed(wire[i : i + chunk])
    return status, rec, dec


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 17, 64, 4096])
def test_a_split_message_decodes_out_of_the_callers_buffer(engine, chunk):
    """§7.2 item 4: the outcome must not depend on where the chunks fall."""
    status, rec, _dec = _feed(engine, _msg(), chunk, reassembly=bytearray(1024))
    assert status is Status.COMPLETE
    assert rec.events == WANT


@pytest.mark.parametrize("engine", ENGINES)
def test_the_same_message_decodes_the_same_without_one(engine):
    """The parameter changes where the bytes are joined, nothing else."""
    wire = _msg()
    with_buf = _feed(engine, wire, 3, reassembly=bytearray(1024))
    without = _feed(engine, wire, 3)
    assert with_buf[0] is without[0] is Status.COMPLETE
    assert with_buf[1].events == without[1].events == WANT


@pytest.mark.parametrize("engine", ENGINES)
def test_a_buffer_too_small_is_refused_not_grown(engine):
    """The point of the parameter: a construct larger than the caller allowed
    for is an error, not a bigger buffer."""
    with pytest.raises(SofaArgumentError):
        _feed(engine, _msg(), 8, reassembly=bytearray(32))


@pytest.mark.parametrize("engine", ENGINES)
def test_the_buffer_is_reused_across_messages_not_regrown(engine):
    """A decoder fed message after message must keep working out of the same
    fixed buffer -- that is what 'bounded by construction' means."""
    wire = _msg()
    buf = bytearray(1024)
    rec = Recorder()
    dec = engine(**NO_CAPS, visitor=rec, reassembly=buf)
    for _ in range(3):
        for i in range(0, len(wire), 7):
            dec.feed(wire[i : i + 7])
        assert rec.events == WANT
        rec.events.clear()
        dec.reset()
    assert len(buf) == 1024


@pytest.mark.parametrize("engine", ENGINES)
def test_a_message_that_fits_one_chunk_never_touches_the_buffer(engine):
    """Nothing is carried, so nothing is copied: the chunk is read where it lies
    and the reassembly buffer stays untouched."""
    buf = bytearray(1024)
    status, rec, _dec = _feed(engine, _msg(), 4096, reassembly=buf)
    assert status is Status.COMPLETE
    assert rec.events == WANT
    assert buf == bytearray(1024)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_chunk_may_be_overwritten_the_moment_feed_returns(engine):
    """§6's chunk-lifetime rule, which the buffer is what makes true: after a
    feed that ends mid-construct, the decoder holds nothing of the chunk."""
    wire = _msg()
    rec = Recorder()
    dec = engine(**NO_CAPS, visitor=rec, reassembly=bytearray(1024))
    status = Status.COMPLETE
    for i in range(0, len(wire), 9):
        chunk = bytearray(wire[i : i + 9])
        status = dec.feed(chunk)
        chunk[:] = b"\xff" * len(chunk)  # scribble over what was just fed
    assert status is Status.COMPLETE
    assert rec.events == WANT


@pytest.mark.parametrize("engine", ENGINES)
def test_only_a_bytearray_or_a_byte_count_is_accepted(engine):
    """One buffer type, because both engines index it directly rather than going
    through two buffer protocols with one of them slower (§5.3) — plus an ``int``,
    which is the caller naming a size for the decoder to take at construction.

    Everything else is refused rather than adapted.
    """
    for bad in (b"\x00" * 64, memoryview(bytearray(64)), 64.0, "64", True):
        with pytest.raises(SofaArgumentError):
            engine(**NO_CAPS, visitor=Recorder(), reassembly=bad)
    # A count below what a single spanning construct can need is refused too:
    # a buffer that cannot hold one is not a smaller buffer, it is a broken one.
    with pytest.raises(SofaArgumentError):
        engine(**NO_CAPS, visitor=Recorder(), reassembly=8)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_byte_count_sizes_the_buffer_at_construction(engine):
    """``reassembly=n`` is the same contract as passing ``bytearray(n)``: the
    decoder takes it once, at construction, and never grows it (§6.6)."""
    enc = Encoder()
    enc.write_bytes(1, b"z" * 200)
    enc.flush()
    wire = enc.getvalue()

    status, rec, _dec = _feed(engine, wire, 7, reassembly=300)
    assert status is Status.COMPLETE
    assert rec.events == [("blob", 1, b"z" * 200)]

    # And too small is the caller's mistake, not something to accommodate.
    with pytest.raises(SofaArgumentError):
        _feed(engine, wire, 7, reassembly=64)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_default_buffer_is_bounded_and_never_grows(engine):
    """§6.6.2: "A codec **MUST NOT** grow a private accumulator instead."

    Omitting the parameter used to mean exactly that — a ``bytearray`` of the
    decoder's own, extended to whatever the message declared, so a sender chose
    the receiver's memory. There is now one shape: a buffer sized at
    construction, and a construct that does not fit it is refused.
    """
    enc = Encoder()
    enc.write_bytes(1, b"z" * (DEFAULT_REASSEMBLY * 2))
    enc.flush()
    wire = enc.getvalue()

    with pytest.raises(SofaArgumentError):
        _feed(engine, wire, 512)

    # The same bytes in one call never touch the buffer at all, whatever their
    # size: nothing spans a chunk boundary when there is only one chunk.
    rec = Recorder()
    assert engine(**NO_CAPS, visitor=rec).feed(wire) is Status.COMPLETE
    assert rec.events == [("blob", 1, b"z" * (DEFAULT_REASSEMBLY * 2))]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_buffer_is_slid_back_rather_than_declared_full(engine):
    """Held bytes accumulate at the front as fields are consumed. When the next
    chunk would run off the end, what is still held slides back to the start --
    the buffer is only too small once that has been tried."""
    enc = Encoder()
    for i in range(1, 25):
        enc.write_string(i, f"payload-{i:02d}")
    enc.flush()
    wire = enc.getvalue()

    # 48 bytes holds any one of these fields several times over, but nowhere
    # near the message, so the slide happens on almost every chunk.
    status, rec, _dec = _feed(engine, wire, 13, reassembly=bytearray(48))
    assert status is Status.COMPLETE
    assert rec.events == [("str", i, f"payload-{i:02d}") for i in range(1, 25)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_carry_larger_than_the_buffer_is_refused_on_the_way_out(engine):
    """The refusal also has to happen for a carry left by a single chunk: the
    bytes are still in the caller's chunk, and keeping them would mean holding
    memory past the call (§6) or growing a buffer of our own (§6.6)."""
    wire = _msg()
    rec = Recorder()
    dec = engine(**NO_CAPS, visitor=rec, reassembly=bytearray(8))
    with pytest.raises(SofaArgumentError):
        dec.feed(wire[:60])  # stops deep inside the 300-byte string
