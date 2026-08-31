"""Resume-after-INCOMPLETE tests (CORELIB_PLAN §5.2).

§5.2 requires the decoder to "suspend and resume at **any** byte boundary
without losing state", and forbids folding ``INCOMPLETE`` into either neighbour.
For this pull decoder that means a read which ran the reader dry must leave the
decoder exactly where it started: the partial tail stays buffered, the pending
field stays pending, and re-issuing the same call once more bytes have arrived
must decode the field from its first byte — never from the middle of one.

The reader used here is deliberately hostile in the way a socket is: it hands
out a few bytes and then returns ``b""`` (nothing available *yet*) at the next
call, so every chunk boundary in the message becomes a suspension point.
"""

from __future__ import annotations

import pytest
from vectors import FULL_SCALE_EXPECTED, NO_CAPS, VECTORS, Recorder, Status, values

from sofab import Decoder, Encoder, WireType


class Feed:
    """A decoder plus the bytes handed to it so far.

    Push mode makes §5.2's resume contract direct: hand over a prefix, read the
    outcome, hand over the rest. ``INCOMPLETE`` means "feed me more" and nothing
    of the half-parsed construct is lost or fabricated — which is exactly what
    these tests are about.
    """

    def __init__(self, dec_cls=None, **kw) -> None:
        self.rec = Recorder()
        self.dec = (dec_cls or Decoder)(**NO_CAPS, visitor=self.rec, **kw)
        self.status = Status.COMPLETE

    def push(self, data: bytes) -> Status:
        self.status = self.dec.feed(data)
        return self.status

    @property
    def events(self) -> list:
        return self.rec.events

    @property
    def fields(self) -> list:
        return self.rec.fields


def _drip(data: bytes, chunk: int = 1, dec_cls=None) -> list:
    """Feed ``data`` ``chunk`` bytes at a time, with an empty feed between each.

    Every boundary is therefore an end-of-input as far as the decoder can tell,
    which is the byte boundary §5.2 says it must suspend and resume at.
    """
    f = Feed(dec_cls)
    for off in range(0, len(data), chunk):
        f.push(b"")  # starved: must report INCOMPLETE or COMPLETE, never drift
        f.push(data[off : off + chunk])
    assert f.status is Status.COMPLETE, f.status
    return f.events


# --- the reported repro ------------------------------------------------------


def _three_field_message() -> bytes:
    enc = Encoder()
    enc.write_unsigned(1, 300)  # 08 ac 02 — the varint spans the split
    enc.write_string(2, "hello")
    enc.write_unsigned(3, 7)
    return enc.getvalue()


def test_split_varint_resumes_instead_of_losing_its_head():
    wire = _three_field_message()
    f = Feed()
    assert f.push(wire[:2]) is Status.INCOMPLETE  # the 300 varint spans the split
    assert f.events == []  # nothing of it may be reported yet

    assert f.push(wire[2:]) is Status.COMPLETE  # re-read from its first byte
    assert f.events == [("u", 1, 300), ("str", 2, "hello"), ("u", 3, 7)]


def test_auto_skip_after_incomplete_does_not_fabricate_fields():
    """The issue's exact shape: a bare ``next()`` loop, so the suspension happens
    inside the implicit skip of the previous field's value."""
    wire = _three_field_message()
    f = Feed()
    assert f.push(wire[:2]) is Status.INCOMPLETE
    # The header is whole, so the field is announced; its value is not, so
    # nothing is handed over for it yet.
    assert [(x.id, x.type) for x in f.fields] == [(1, WireType.UNSIGNED)]
    assert f.events == []

    assert f.push(wire[2:]) is Status.COMPLETE
    # Field 1 is announced once, not again on resume, and no field is invented.
    assert [(x.id, x.type) for x in f.fields] == [
        (1, WireType.UNSIGNED),
        (2, WireType.FIXLEN),
        (3, WireType.UNSIGNED),
    ]
    assert f.events == [("u", 1, 300), ("str", 2, "hello"), ("u", 3, 7)]


def test_split_field_header_resumes():
    """The suspension falls inside the field header varint itself."""
    enc = Encoder()
    enc.write_unsigned(1000, 1)  # header 0x1f40 — a two-byte varint
    wire = enc.getvalue()
    f = Feed()
    assert f.push(wire[:1]) is Status.INCOMPLETE
    assert f.push(wire[1:]) is Status.COMPLETE
    assert f.events == [("u", 1000, 1)]


def test_split_fixlen_payload_resumes():
    enc = Encoder()
    enc.write_bytes(4, bytes(range(20)))
    enc.write_unsigned(5, 9)
    wire = enc.getvalue()
    f = Feed()
    assert f.push(wire[:8]) is Status.INCOMPLETE
    assert f.push(wire[8:]) is Status.COMPLETE
    assert f.events == [("blob", 4, bytes(range(20))), ("u", 5, 9)]


def test_split_array_payload_resumes():
    enc = Encoder()
    enc.write_unsigned_array(6, [1, 300, 70000, 4, 5])
    enc.write_unsigned(7, 3)
    wire = enc.getvalue()
    f = Feed()
    assert f.push(wire[:5]) is Status.INCOMPLETE
    assert f.push(wire[5:]) is Status.COMPLETE
    assert f.events == [("ua", 6, (1, 300, 70000, 4, 5)), ("u", 7, 3)]


def test_split_inside_open_sequence_resumes():
    enc = Encoder()
    enc.write_sequence_begin_lazy(3)
    enc.write_unsigned(1, 70000)
    enc.write_sequence_end_keep()
    wire = enc.getvalue()
    f = Feed()
    # Inside an open sequence: truncated, not clean EOF.
    assert f.push(wire[:1]) is Status.INCOMPLETE
    assert f.push(wire[1:]) is Status.COMPLETE
    assert f.events == [("seq{", 3), ("u", 1, 70000), ("seq}",)]


def _nested_sequence_message() -> bytes:
    enc = Encoder()
    enc.write_sequence_begin_lazy(3)
    enc.write_unsigned(1, 70000)
    enc.write_sequence_begin_lazy(2)
    enc.write_string(1, "inner")
    enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.write_unsigned(9, 5)
    return enc.getvalue()


def test_skipping_a_truncated_sequence_resumes():
    """``skip()`` over a sequence spans many fields — it must still be all-or-
    nothing, and re-issuable once the rest of the sequence arrives."""
    wire = _nested_sequence_message()
    for cut in range(2, len(wire) - 2):
        f = Feed()
        # A handler that declines the outer sequence: the decoder walks the whole
        # sub-tree, which spans many fields and so must be all-or-nothing.
        f.rec = Recorder(decline=lambda fld: fld.id == 3)
        f.dec = Decoder(**NO_CAPS, visitor=f.rec)
        f.push(wire[:cut])
        assert f.push(wire[cut:]) is Status.COMPLETE, f"cut={cut}"
        assert ("u", 9, 5) in f.events


def test_nested_sequence_message_survives_a_starving_reader():
    wire = _nested_sequence_message()
    assert _drip(wire) == values(Decoder, wire)


def test_repeated_incomplete_is_stable():
    """Retrying while still starved must keep reporting INCOMPLETE, not drift."""
    wire = _three_field_message()
    f = Feed()
    assert f.push(wire[:2]) is Status.INCOMPLETE
    for _ in range(5):
        assert f.push(b"") is Status.INCOMPLETE
    assert f.push(wire[2:]) is Status.COMPLETE
    assert f.events == [("u", 1, 300), ("str", 2, "hello"), ("u", 3, 7)]


# --- the whole corpus, suspended at every byte boundary ----------------------


def test_full_scale_message_survives_a_starving_reader():
    expected = values(Decoder, FULL_SCALE_EXPECTED)
    for chunk in (1, 3, 7):
        assert _drip(FULL_SCALE_EXPECTED, chunk) == expected, f"chunk={chunk}"


@pytest.mark.parametrize("vec", VECTORS, ids=[v["name"] for v in VECTORS])
def test_every_shared_vector_survives_a_starving_reader(vec):
    wire = bytes.fromhex(vec["serialized"]["hex"])
    assert _drip(wire) == values(Decoder, wire)



