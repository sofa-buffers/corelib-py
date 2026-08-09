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

import json
import struct
from pathlib import Path

import pytest
from vectors import FULL_SCALE_EXPECTED

from sofab import Decoder, Encoder, SofaIncompleteError, WireType

VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "assets" / "test_vectors.json").read_text()
)["vectors"]


class Feeder:
    """A ``read(n)`` source fed explicitly by the test; empty when starved."""

    def __init__(self) -> None:
        self.q = bytearray()

    def push(self, data: bytes) -> None:
        self.q += data

    def read(self, n: int) -> bytes:
        out = bytes(self.q[:n])
        del self.q[: len(out)]
        return out


class StarvingReader:
    """Hands out ``chunk`` bytes, then returns ``b""`` once, then the next chunk.

    Every chunk boundary is therefore an end-of-input as far as the decoder can
    tell, which is exactly the byte boundary §5.2 says it must be able to
    suspend and resume at.
    """

    def __init__(self, data: bytes, chunk: int = 1) -> None:
        self._data = bytes(data)
        self._pos = 0
        self._chunk = chunk
        self._starve = False

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self._data)

    def read(self, n: int) -> bytes:
        if self._pos >= len(self._data):
            return b""
        if self._starve:
            self._starve = False
            return b""
        end = min(self._pos + min(n, self._chunk), len(self._data))
        out = self._data[self._pos : end]
        self._pos = end
        self._starve = True
        return out


def _value(dec: Decoder, f) -> object:
    if f.type == WireType.UNSIGNED:
        return dec.unsigned()
    if f.type == WireType.SIGNED:
        return dec.signed()
    if f.type == WireType.FIXLEN:
        name = f.subtype.name
        if name == "BLOB":
            return dec.bytes()
        if name == "STRING":
            return dec.string()
        # Floats compare by bit pattern so NaN payloads stay comparable.
        return struct.pack("<d", dec.float32() if name == "FP32" else dec.float64())
    if f.type == WireType.ARRAY_UNSIGNED:
        return tuple(dec.read_unsigned_array())
    if f.type == WireType.ARRAY_SIGNED:
        return tuple(dec.read_signed_array())
    if f.type == WireType.ARRAY_FIXLEN:
        read = dec.read_float32_array if f.subtype.name == "FP32" else dec.read_float64_array
        return tuple(struct.pack("<d", v) for v in read())
    return None


def _walk(dec: Decoder) -> list:
    """Consume a decoder whose reader never starves."""
    out = []
    while (f := dec.next()) is not None:
        out.append((f.id, f.type, _value(dec, f)))
    return out


def _walk_resuming(dec: Decoder, src: StarvingReader) -> list:
    """Consume a decoder over a starving reader, retrying every suspension.

    This is the loop §5.2 describes for a streaming caller: ``INCOMPLETE`` means
    "feed me the next chunk", so the same call is simply issued again. It is
    only allowed to give up once the source is genuinely exhausted.
    """
    out = []
    while True:
        try:
            f = dec.next()
        except SofaIncompleteError:
            if src.exhausted:
                raise
            continue
        if f is None:
            if src.exhausted:
                return out
            continue  # a field boundary that merely has no bytes yet
        while True:
            try:
                out.append((f.id, f.type, _value(dec, f)))
                break
            except SofaIncompleteError:
                if src.exhausted:
                    raise


# --- the reported repro ------------------------------------------------------


def _three_field_message() -> bytes:
    enc = Encoder()
    enc.write_unsigned(1, 300)  # 08 ac 02 — the varint spans the split
    enc.write_string(2, "hello")
    enc.write_unsigned(3, 7)
    return enc.getvalue()


def test_split_varint_resumes_instead_of_losing_its_head():
    wire = _three_field_message()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:2])  # header + the first byte of the 300 varint

    f = dec.next()
    assert (f.id, f.type) == (1, WireType.UNSIGNED)
    with pytest.raises(SofaIncompleteError):
        dec.unsigned()

    src.push(wire[2:])
    assert dec.unsigned() == 300  # re-read from the varint's first byte
    f = dec.next()
    assert (f.id, f.type) == (2, WireType.FIXLEN)
    assert dec.string() == "hello"
    f = dec.next()
    assert (f.id, f.type) == (3, WireType.UNSIGNED)
    assert dec.unsigned() == 7
    assert dec.next() is None


def test_auto_skip_after_incomplete_does_not_fabricate_fields():
    """The issue's exact shape: a bare ``next()`` loop, so the suspension happens
    inside the implicit skip of the previous field's value."""
    wire = _three_field_message()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:2])

    seen = []
    with pytest.raises(SofaIncompleteError):
        while (f := dec.next()) is not None:
            seen.append((f.id, f.type))
    assert seen == [(1, WireType.UNSIGNED)]

    src.push(wire[2:])
    while (f := dec.next()) is not None:
        seen.append((f.id, f.type))
    assert seen == [
        (1, WireType.UNSIGNED),
        (2, WireType.FIXLEN),
        (3, WireType.UNSIGNED),
    ]


def test_split_field_header_resumes():
    """The suspension falls inside the field header varint itself."""
    enc = Encoder()
    enc.write_unsigned(1000, 1)  # header 0x1f40 — a two-byte varint
    wire = enc.getvalue()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:1])
    with pytest.raises(SofaIncompleteError):
        dec.next()
    src.push(wire[1:])
    f = dec.next()
    assert (f.id, f.type) == (1000, WireType.UNSIGNED)
    assert dec.unsigned() == 1


def test_split_fixlen_payload_resumes():
    enc = Encoder()
    enc.write_bytes(4, bytes(range(20)))
    enc.write_unsigned(5, 9)
    wire = enc.getvalue()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:8])
    f = dec.next()
    assert f.type == WireType.FIXLEN
    with pytest.raises(SofaIncompleteError):
        dec.bytes()
    src.push(wire[8:])
    assert dec.bytes() == bytes(range(20))
    assert dec.next().id == 5
    assert dec.unsigned() == 9


def test_split_array_payload_resumes():
    enc = Encoder()
    enc.write_unsigned_array(6, [1, 300, 70000, 4, 5])
    enc.write_unsigned(7, 3)
    wire = enc.getvalue()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:5])
    f = dec.next()
    assert f.type == WireType.ARRAY_UNSIGNED
    with pytest.raises(SofaIncompleteError):
        dec.read_unsigned_array()
    src.push(wire[5:])
    assert dec.read_unsigned_array() == [1, 300, 70000, 4, 5]
    assert dec.next().id == 7
    assert dec.unsigned() == 3


def test_split_inside_open_sequence_resumes():
    enc = Encoder()
    enc.write_sequence_begin_lazy(3)
    enc.write_unsigned(1, 70000)
    enc.write_sequence_end_keep()
    wire = enc.getvalue()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:1])
    assert dec.next().type == WireType.SEQUENCE_START
    with pytest.raises(SofaIncompleteError):
        dec.next()  # inside an open sequence: truncated, not clean EOF
    src.push(wire[1:])
    f = dec.next()
    assert (f.id, f.type) == (1, WireType.UNSIGNED)
    assert dec.unsigned() == 70000
    assert dec.next().type == WireType.SEQUENCE_END
    assert dec.next() is None


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
        src = Feeder()
        dec = Decoder(src)
        src.push(wire[:cut])
        assert dec.next().type == WireType.SEQUENCE_START
        try:
            dec.skip()
        except SofaIncompleteError:
            src.push(wire[cut:])
            dec.skip()  # re-issued: replays the sequence from its first byte
        else:
            src.push(wire[cut:])
        f = dec.next()
        assert (f.id, f.type) == (9, WireType.UNSIGNED), f"cut={cut}"
        assert dec.unsigned() == 5
        assert dec.next() is None


def test_nested_sequence_message_survives_a_starving_reader():
    wire = _nested_sequence_message()
    expected = _walk(Decoder(io_reader(wire)))
    src = StarvingReader(wire, chunk=1)
    assert _walk_resuming(Decoder(src), src) == expected


def test_repeated_incomplete_is_stable():
    """Retrying while still starved must keep reporting INCOMPLETE, not drift."""
    wire = _three_field_message()
    src = Feeder()
    dec = Decoder(src)
    src.push(wire[:2])
    assert dec.next().id == 1
    for _ in range(5):
        with pytest.raises(SofaIncompleteError):
            dec.unsigned()
    src.push(wire[2:])
    assert dec.unsigned() == 300


# --- the whole corpus, suspended at every byte boundary ----------------------


def test_full_scale_message_survives_a_starving_reader():
    expected = _walk(Decoder(io_reader(FULL_SCALE_EXPECTED)))
    for chunk in (1, 3, 7):
        src = StarvingReader(FULL_SCALE_EXPECTED, chunk=chunk)
        assert _walk_resuming(Decoder(src), src) == expected, f"chunk={chunk}"


@pytest.mark.parametrize("vec", VECTORS, ids=[v["name"] for v in VECTORS])
def test_every_shared_vector_survives_a_starving_reader(vec):
    wire = bytes.fromhex(vec["serialized"]["hex"])
    expected = _walk(Decoder(io_reader(wire)))
    src = StarvingReader(wire, chunk=1)
    assert _walk_resuming(Decoder(src), src) == expected


def io_reader(data: bytes):
    import io

    return io.BytesIO(bytes(data))
