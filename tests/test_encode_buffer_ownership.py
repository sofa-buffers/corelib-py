"""§5.1 output-buffer ownership: no growable, corelib-allocated output buffer.

CORELIB_PLAN §5.1 states that a corelib **MUST NOT** "allocate an output buffer"
and **MUST NOT** "grow or reallocate" one. Both convenience constructors —
``Encoder()`` and ``Encoder(writer)`` — are therefore built on the same
caller-supplied-buffer primitive as :meth:`Encoder.over_buffer`: a **fixed**
scratch buffer installed **with** a flush sink, which is §5.1's "unbounded
schema" shape. Nothing the encoder writes into ever grows, and a message far
larger than the scratch streams out through it instead of accumulating.

Before this, ``Encoder()``/``Encoder(writer)`` owned a ``bytearray`` that grew
for the whole message: a 100 kB message meant a 100 kB encoder-owned buffer, and
``Encoder(writer)`` reached the writer only at the explicit ``flush()`` — so the
"stream straight to the wire" the README promises did not hold.
"""

from __future__ import annotations

import io

import pytest
from vectors import build_full_scale

from sofab import MIN_OUTPUT_BUFFER, SofaRangeError, SofaStateError
from sofab.encoder import _SCRATCH_SIZE as PY_SCRATCH
from sofab.encoder import Encoder as PyEncoder

try:  # the native accelerator is optional — exercise whichever engines exist
    from sofab._speedups import _SCRATCH_SIZE as NATIVE_SCRATCH
    from sofab._speedups import Encoder as NativeEncoder
except ImportError:  # pragma: no cover - pure-Python-only install
    ENCODERS = [PyEncoder]
else:
    ENCODERS = [PyEncoder, NativeEncoder]

#: Enough values that the message is many scratch buffers long.
_VALUES = list(range(20000))


class _Recorder:
    """A ``write(bytes)`` sink that records every chunk it is handed."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(bytes(data))


def _oneshot(enc_cls, build) -> bytes:
    enc = enc_cls()
    build(enc)
    enc.flush()
    return enc.getvalue()


def _write_values(enc) -> None:
    for v in _VALUES:
        enc.write_unsigned(1, v)


def test_both_engines_declare_the_same_scratch_size():
    """The two engines must agree, or "bounded by the scratch" would mean two
    different things depending on which one is loaded."""
    if len(ENCODERS) == 2:
        assert PY_SCRATCH == NATIVE_SCRATCH
    assert PY_SCRATCH >= MIN_OUTPUT_BUFFER


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_writer_model_drains_while_encoding(enc_cls):
    """``Encoder(writer)`` streams: the writer sees bytes *during* the message,
    and the encoder never holds more than one scratch buffer of them."""
    writer = _Recorder()
    enc = enc_cls(writer)
    for v in _VALUES:
        enc.write_unsigned(1, v)
        # The encoder-owned buffer is fixed: it can never hold more than this.
        assert enc.bytes_used() <= PY_SCRATCH
    assert writer.chunks, "no bytes reached the writer before flush(): the encoder buffered the message"
    assert max(len(c) for c in writer.chunks) <= PY_SCRATCH
    enc.flush()
    assert b"".join(writer.chunks) == _oneshot(enc_cls, _write_values)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_in_memory_model_holds_one_fixed_buffer(enc_cls):
    """``Encoder()`` accumulates the *result* — the message it hands back — but
    the buffer it encodes into stays fixed."""
    enc = enc_cls()
    _write_values(enc)
    assert enc.bytes_used() <= PY_SCRATCH
    data = enc.getvalue()
    assert len(data) > 10 * PY_SCRATCH, "message too small to prove anything"
    assert data == _oneshot(enc_cls, _write_values)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_payload_longer_than_the_scratch_splits_across_drains(enc_cls):
    """A `string`/`blob` run is divisible (§5.1): one longer than the whole
    scratch buffer is split across flushes rather than needing contiguous room."""
    def build(enc):
        enc.write_bytes(1, bytes(range(256)) * (4 * PY_SCRATCH // 256))
        enc.write_string(2, "€" * (2 * PY_SCRATCH))
        enc.write_unsigned(3, 7)

    writer = _Recorder()
    enc = enc_cls(writer)
    build(enc)
    enc.flush()
    assert len(writer.chunks) > 4
    expected = _oneshot(enc_cls, build)
    assert b"".join(writer.chunks) == expected

    # The in-memory model takes such a run as one chunk of its result instead of
    # copying it through the buffer; the bytes must be the same either way.
    held = enc_cls()
    build(held)
    assert held.bytes_used() <= PY_SCRATCH
    assert held.getvalue() == expected


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_in_memory_model_drives_a_min_output_buffer(enc_cls):
    """The in-memory model over the smallest buffer §5.1 admits.

    ``MIN_OUTPUT_BUFFER`` is 1 because every atomic unit — a header varint, a
    ``fixlen_word``, an element count, a scalar, one float — splits at any byte
    boundary, so a one-byte buffer must already yield exactly the one-shot
    bytes. Installing one on ``Encoder()`` puts the in-memory sink in the corner
    it never otherwise reaches: every single write overflows the buffer, so the
    result is assembled entirely out of drained pieces, starting with the very
    first one — including the divisible ``string``/``blob`` runs, which are at
    least as long as the whole buffer and are handed to the result directly.
    """
    def build(enc):
        enc.write_unsigned(5, 300)
        enc.write_signed(6, -70000)
        enc.write_string(1, "hello, sofab")
        enc.write_bytes(2, bytes(range(64)))
        enc.write_float64(3, 3.14159265)
        enc.write_unsigned_array(4, [0, 128, 70000])

    expected = _oneshot(enc_cls, build)

    tiny = enc_cls()
    tiny.buffer_set(bytearray(MIN_OUTPUT_BUFFER), 0)
    build(tiny)
    tiny.flush()
    assert tiny.bytes_used() <= MIN_OUTPUT_BUFFER
    assert tiny.getvalue() == expected


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_writer_model_refuses_getvalue(enc_cls):
    """Bytes handed to the writer are gone; returning the undrained tail would be
    "partial output as if it were complete" (§5.1)."""
    enc = enc_cls(io.BytesIO())
    enc.write_unsigned(1, 7)
    with pytest.raises(SofaStateError):
        enc.getvalue()


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_convenience_buffers_are_sink_installed(enc_cls):
    """The scratch is installed *with* a sink, so a buffer handed to the same
    encoder mid-stream is bound by MIN_OUTPUT_BUFFER like any other (§5.1) —
    the convenience models are not a second, unchecked ownership model."""
    for enc in (enc_cls(), enc_cls(io.BytesIO())):
        with pytest.raises(SofaRangeError):
            enc.buffer_set(bytearray(MIN_OUTPUT_BUFFER - 1), 0)
        enc.buffer_set(bytearray(MIN_OUTPUT_BUFFER), 0)  # exactly the minimum: fine


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_convenience_and_caller_buffer_models_agree(enc_cls):
    """The convenience shape is the caller-supplied one with a scratch buffer the
    corelib hands in, so the full-scale vector must come out byte-identical."""
    expected = _oneshot(enc_cls, build_full_scale)

    writer = _Recorder()
    streamed = enc_cls(writer)
    build_full_scale(streamed)
    streamed.flush()
    assert b"".join(writer.chunks) == expected

    chunks: list[bytes] = []
    over = enc_cls.over_buffer(bytearray(MIN_OUTPUT_BUFFER), 0, chunks.append)
    build_full_scale(over)
    over.flush()
    assert b"".join(chunks) == expected
