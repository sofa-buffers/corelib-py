"""``on_blob_begin``: a blob into the caller's buffer, not a fresh ``bytes``.

CORELIB_PLAN §6.6.3: a callback carrying a whole aggregate obliges the codec to
build one, and the only size available to build it from is the wire's. A
megabyte blob costs a megabyte allocation per message that way. The section's
other shape is a destination the caller hands back after being told the announced
size, with the codec refusing one too short rather than growing it -- which is
what this hook is.

Strings are deliberately not offered: a string the handler reads must be
validated (§6.7.2), and this port validates by decoding, which builds the ``str``
a destination would exist to avoid.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder, Status, walk

from sofab import Encoder, SofaRangeError

PAYLOAD = bytes(range(256)) * 4


class Sink(Recorder):
    def __init__(self, dst, **kw):
        super().__init__(**kw)
        self.seen = []
        self._dst = dst

    def on_blob_begin(self, field_id, size):
        self.seen.append((field_id, size))
        return self._dst


def _msg(payload=PAYLOAD):
    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_bytes(2, payload)
    enc.write_unsigned(3, 6)
    enc.flush()
    return enc.getvalue()


@pytest.mark.parametrize("engine", ENGINES)
def test_the_hook_sees_the_size_before_the_payload(engine):
    sink = Sink(None)
    status, rec, _dec = walk(engine, _msg(), recorder=sink)
    assert status is Status.COMPLETE
    assert sink.seen == [(2, len(PAYLOAD))]
    assert rec.events == [("u", 1, 5), ("blob", 2, PAYLOAD), ("u", 3, 6)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_visitor_that_does_not_override_it_is_unaffected(engine):
    status, rec, _dec = walk(engine, _msg())
    assert status is Status.COMPLETE
    assert rec.events == [("u", 1, 5), ("blob", 2, PAYLOAD), ("u", 3, 6)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [None, 1, 7, 64])
def test_a_destination_is_filled_and_on_bytes_is_not_called(engine, chunk):
    dst = bytearray(1024)
    sink = Sink(dst)
    status, rec, _dec = walk(engine, _msg(), chunk=chunk, recorder=sink)
    assert status is Status.COMPLETE
    assert dst[: len(PAYLOAD)] == PAYLOAD
    assert rec.events == [("u", 1, 5), ("u", 3, 6)]  # no blob event


@pytest.mark.parametrize("engine", ENGINES)
def test_the_hook_is_not_called_for_a_string(engine):
    enc = Encoder()
    enc.write_string(1, "hello")
    enc.flush()
    sink = Sink(bytearray(64))
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=sink)
    assert status is Status.COMPLETE
    assert sink.seen == []
    assert rec.events == [("str", 1, "hello")]


@pytest.mark.parametrize("engine", ENGINES)
def test_an_empty_blob_fills_nothing(engine):
    dst = bytearray(b"\xff" * 8)
    sink = Sink(dst)
    status, rec, _dec = walk(engine, _msg(b""), recorder=sink)
    assert status is Status.COMPLETE
    assert dst == bytearray(b"\xff" * 8)
    assert rec.events == [("u", 1, 5), ("u", 3, 6)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_short_destination_is_refused_not_grown(engine):
    dst = bytearray(16)
    with pytest.raises(SofaRangeError):
        walk(engine, _msg(), recorder=Sink(dst))
    assert dst == bytearray(16)  # refused before a byte was written


@pytest.mark.parametrize("engine", ENGINES)
def test_a_destination_that_is_not_a_buffer_is_refused(engine):
    with pytest.raises(SofaRangeError):
        walk(engine, _msg(), recorder=Sink([0] * 2048))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_read_only_destination_is_refused(engine):
    with pytest.raises(SofaRangeError):
        walk(engine, _msg(), recorder=Sink(b"\x00" * 2048))


@pytest.mark.parametrize("engine", ENGINES)
def test_a_wide_item_destination_is_refused(engine):
    from array import array

    with pytest.raises(SofaRangeError):
        walk(engine, _msg(), recorder=Sink(array("Q", [0] * 256)))


@pytest.mark.parametrize("engine", ENGINES)
def test_the_answer_may_differ_per_field(engine):
    """Two blobs, one taken into a buffer and one left to on_bytes."""

    class PerField(Sink):
        def on_blob_begin(self, field_id, size):
            self.seen.append(field_id)
            return self._dst if field_id == 2 else None

    enc = Encoder()
    enc.write_bytes(2, b"into")
    enc.write_bytes(4, b"list")
    enc.flush()
    dst = bytearray(16)
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=PerField(dst))
    assert status is Status.COMPLETE
    assert dst[:4] == b"into"
    assert rec.events == [("blob", 4, b"list")]
