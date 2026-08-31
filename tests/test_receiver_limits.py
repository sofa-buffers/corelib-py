"""§6.2.1: the receiver-side limits are the CALLER's, present, finite, never unset.

    There is no unset state and no unlimited mode. Unbounded by the schema is
    still bounded by the receiver.

and, on the site of the comparison:

    A codec **MUST NOT** hold a limit of its own, **MUST NOT** supply a default
    for one it was not given, **MUST NOT** read an omitted argument as
    *unlimited*, and **MUST NOT** clamp to one.

So a decoder always carries all three of ``max_dyn_array_count``,
``max_dyn_string_len`` and ``max_dyn_blob_len`` — and carries them because a
**caller stated them**. All three are required: omitting one, or passing
``None``, is a defect in the call and lands in §6.3's ``InvalidArgument`` tier
(:class:`SofaArgumentError`), never in ``LimitExceeded``, which would promise a
limit to raise that was never configured.

The format ceilings of §6.2 are **not** a stand-in for an unstated cap. A caller
may state a ceiling as its limit — at the ceiling a limit cannot fire, since a
larger value is already INVALID before the check is reached — but that is then
the caller's number, and a decode with no number at all does not happen.

What the limits *do* once set is the subject of ``test_schema_bounded.py``; this
file is about their provenance, their existence and their domain.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import NO_CAPS, Recorder, capped

from sofab import (
    ARRAY_MAX,
    FIXLEN_MAX,
    Encoder,
    SofaArgumentError,
    SofaError,
    SofaLimitError,
    Status,
)

NAMES = ("max_dyn_array_count", "max_dyn_string_len", "max_dyn_blob_len")
CEILINGS = {
    "max_dyn_array_count": ARRAY_MAX,
    "max_dyn_string_len": FIXLEN_MAX,
    "max_dyn_blob_len": FIXLEN_MAX,
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_an_omitted_limit_is_refused(engine, name):
    """The codec supplies no default for a limit it was not given (§6.2.1).

    The other two are stated, so the only thing missing is this one — and the
    construction fails rather than resolving it to the format ceiling.
    """
    stated = {k: v for k, v in NO_CAPS.items() if k != name}
    with pytest.raises(SofaArgumentError) as exc:
        engine(visitor=Recorder(), **stated)
    assert name in str(exc.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_stating_no_limit_at_all_is_refused(engine):
    with pytest.raises(SofaArgumentError):
        engine(visitor=Recorder())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_an_omitted_limit_is_an_argument_error_not_a_limit_rejection(engine, name):
    """§6.3: the two codes say different things and must stay distinguishable.

    ``LimitExceeded`` means "raise my limit"; there is no limit to raise here,
    because none was ever configured. So the refusal is the ``InvalidArgument``
    tier and nothing else — asserted as *not* a ``SofaLimitError`` rather than
    only as a ``SofaArgumentError``, since a subclass would satisfy both.
    """
    stated = {k: v for k, v in NO_CAPS.items() if k != name}
    with pytest.raises(SofaError) as exc:
        engine(visitor=Recorder(), **stated)
    assert not isinstance(exc.value, SofaLimitError)
    assert isinstance(exc.value, SofaArgumentError)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_none_is_refused_there_is_no_unset_state(engine, name):
    with pytest.raises(SofaArgumentError):
        engine(visitor=Recorder(), **capped(**{name: None}))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_a_limit_outside_its_domain_is_refused(engine, name):
    for bad in (-1, CEILINGS[name] + 1):
        with pytest.raises(SofaArgumentError):
            engine(visitor=Recorder(), **capped(**{name: bad}))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_the_ceiling_itself_is_accepted(engine, name):
    engine(visitor=Recorder(), **capped(**{name: CEILINGS[name]}))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_zero_is_a_limit_like_any_other(engine, name):
    """Zero is a real setting -- 'accept nothing unbounded' -- not an unset
    state wearing a different value."""
    engine(visitor=Recorder(), **capped(**{name: 0}))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_zero_limit_rejects_rather_than_clamps(engine):
    """§6.2.1 'rejected, never clamped'."""
    enc = Encoder()
    enc.write_string(1, "x")
    enc.flush()
    dec = engine(visitor=Recorder(), **capped(max_dyn_string_len=0))
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())


@pytest.mark.parametrize("engine", ENGINES)
def test_a_caller_stated_ceiling_admits_everything_the_format_admits(engine):
    """At the ceiling a limit cannot fire: a longer value is INVALID first, so a
    caller that states the ceilings rejects nothing a looser one would accept.

    This is the caller choosing the widest limit there is, not the codec falling
    back on one — the number is in ``NO_CAPS``, in this suite, and the decoder
    would refuse the construction without it.
    """
    enc = Encoder()
    enc.write_string(1, "z" * 4096)
    enc.write_bytes(2, b"b" * 4096)
    enc.write_unsigned_array(3, list(range(4096)))
    enc.flush()
    rec = Recorder()
    dec = engine(**NO_CAPS, visitor=rec)
    dec.feed(enc.getvalue())
    assert [e[0] for e in rec.events] == ["str", "blob", "ua"]


# --- whose allocation is it? (§6.2.1, #128) --------------------------------


def _uvarint(x):
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        out.append(b | 0x80 if x else b)
        if not x:
            return bytes(out)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_limit_still_governs_the_list_the_decoder_would_build(engine):
    """The default route is the one §6.2.1 is about.

    With no destination back from the handler, the only storage the array can go
    into is a list of the decoder's own, and the only size it could build one
    from is the wire's. That is precisely "the allocation it is meant to
    prevent", so the cap fires -- at the count header, before an element is
    read."""

    class Handler(Recorder):
        def on_array_begin(self, field_id, wtype, count):
            return None  # asked, and declined to name a destination

    wire = bytes([0x0B]) + _uvarint(0x7FFFFFFF) + b"\x01"
    dec = engine(max_dyn_blob_len=FIXLEN_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(), max_dyn_array_count=4)
    with pytest.raises(SofaLimitError):
        dec.feed(wire)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_handlers_own_destination_is_not_the_senders_to_dictate(engine):
    """§6.2.1 exists because an unbounded field "would let the **sender**
    dictate the **receiver's** allocation". A handler that hands back a buffer
    has sized that buffer itself, so the sender dictates nothing and there is no
    allocation of the decoder's for the cap to prevent.

    The handler is therefore asked, and told the announced count first -- which
    is the point at which a receiver that does not want 2**31-1 elements gets to
    say so. Refusing on its behalf, before it has been asked, is what #128 was
    filed about: every other port in the family reads this field.
    """
    from array import array

    asked = []
    dst = array("Q", [0] * 4)

    class Handler(Recorder):
        def on_array_begin(self, field_id, wtype, count):
            asked.append(count)
            return (dst, None, None)

    enc = Encoder()
    enc.write_unsigned_array(1, [7, 8, 9])
    enc.flush()
    dec = engine(max_dyn_blob_len=FIXLEN_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(), max_dyn_array_count=1)
    dec.feed(enc.getvalue())
    assert asked == [3]
    assert list(dst) == [7, 8, 9, 0]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_destination_too_short_is_a_range_error_not_a_limit(engine):
    """With the cap out of the way, the only ceiling left on the caller's own
    buffer is that buffer's size -- and a count it cannot hold is
    :class:`SofaArgumentError`, never a policy rejection. The decoder is protecting
    itself from an overrun, not judging the message: the bytes are fine, and the
    same message fills a longer destination.
    """
    from array import array

    class Handler(Recorder):
        def on_array_begin(self, field_id, wtype, count):
            return (array("Q", [0] * 4), None, None)

    # id 1, unsigned array, count 2**31-1, then one lone element byte.
    wire = bytes([0x0B]) + _uvarint(0x7FFFFFFF) + b"\x01"
    dec = engine(max_dyn_blob_len=FIXLEN_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(), max_dyn_array_count=4)
    with pytest.raises(SofaArgumentError):
        dec.feed(wire)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_same_two_answers_for_a_blob(engine):
    """The blob half of the pair above: `on_blob_begin` is asked, and its
    buffer -- not the cap -- is what the payload has to fit."""
    enc = Encoder()
    enc.write_bytes(1, b"x" * 100)
    enc.flush()
    wire = enc.getvalue()

    asked = []

    class Handler(Recorder):
        def __init__(self, dst):
            super().__init__()
            self._dst = dst

        def on_blob_begin(self, field_id, size):
            asked.append(size)
            return self._dst

    # Its own buffer, big enough: read, cap or no cap.
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(bytearray(4096)), max_dyn_blob_len=10)
    dec.feed(wire)
    assert asked == [100]

    # Its own buffer, too short: the buffer's size is the only ceiling left,
    # and overrunning it is a range error rather than a policy rejection.
    asked.clear()
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(bytearray(8)), max_dyn_blob_len=10)
    with pytest.raises(SofaArgumentError):
        dec.feed(wire)
    assert asked == [100]

    # No buffer back: the decoder would build the ``bytes`` itself, sized by the
    # wire, and that is the allocation the cap exists to prevent.
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Handler(None), max_dyn_blob_len=10)
    with pytest.raises(SofaLimitError):
        dec.feed(wire)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_limit_leaves_a_reassembly_buffer_untouched(engine):
    """The verdict is on the length word alone, so not a byte of payload is
    buffered -- the caller's reassembly buffer never sees the field."""
    buf = bytearray(64)
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_string_len=FIXLEN_MAX, visitor=Recorder(), max_dyn_blob_len=16, reassembly=buf)
    with pytest.raises(SofaLimitError):
        dec.feed(bytes([0x0A]) + _uvarint((1_000_000 << 3) | 0x3))
    assert buf == bytearray(64)


# --- a limit rejection is terminal (§6.3) -----------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_limit_rejection_is_terminal(engine):
    """§6.3: `LimitExceeded` is "a terminal, receiver-local policy rejection".

    Every later feed re-raises it and consumes nothing, so a caller cannot walk
    on past a rejection it caught -- and the fields behind the refused one are
    never delivered.
    """
    enc = Encoder()
    enc.write_string(1, "x" * 2000)
    enc.write_unsigned(2, 7)
    enc.flush()

    rec = Recorder()
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, visitor=rec, max_dyn_string_len=16)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())
    assert rec.events == []

    for _ in range(2):
        with pytest.raises(SofaLimitError):
            dec.feed(b"")
        assert rec.events == []

    tail = Encoder()
    tail.write_unsigned(9, 1)
    tail.flush()
    with pytest.raises(SofaLimitError):
        dec.feed(tail.getvalue())
    assert rec.events == []


@pytest.mark.parametrize("engine", ENGINES)
def test_a_limit_a_handler_raised_is_terminal_too(engine):
    """§6.2.1 gives the codec the report and the visitor the decision -- for a
    sequence array's element index there is no count header, so the decision is
    the handler's. A rejection reached that way is the same terminal rejection.
    """

    class Picky(Recorder):
        def on_sequence_begin(self, field_id):
            if field_id >= 4:
                raise SofaLimitError(f"element index {field_id} is past my cap")
            return super().on_sequence_begin(field_id)

    enc = Encoder()
    enc.write_sequence_begin_lazy(0)
    enc.write_sequence_begin_lazy(9)
    enc.write_unsigned(0, 1)
    enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.flush()

    rec = Picky()
    dec = engine(**NO_CAPS, visitor=rec)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())
    with pytest.raises(SofaLimitError):
        dec.feed(b"")


@pytest.mark.parametrize("engine", ENGINES)
def test_the_rejection_is_on_the_error_channel_not_the_status(engine):
    """§6.3 leaves the surfacing open between "a fourth decode outcome" and "a
    terminal failure carrying the LimitExceeded code on the error channel".
    This port takes the second: the bytes are well formed, so the status never
    becomes INVALID, and `error` is where the rejection is."""
    enc = Encoder()
    enc.write_string(1, "x" * 2000)
    enc.flush()
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, visitor=Recorder(), max_dyn_string_len=16)
    with pytest.raises(SofaLimitError) as caught:
        dec.feed(enc.getvalue())
    assert dec.error is caught.value
    assert dec.status is not Status.INVALID


@pytest.mark.parametrize("engine", ENGINES)
def test_reset_clears_the_rejection(engine):
    enc = Encoder()
    enc.write_string(1, "x" * 2000)
    enc.flush()
    dec = engine(max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, visitor=Recorder(), max_dyn_string_len=16)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())
    dec.reset()
    assert dec.error is None
    small = Encoder()
    small.write_string(1, "ok")
    small.flush()
    assert dec.feed(small.getvalue()) is Status.COMPLETE
