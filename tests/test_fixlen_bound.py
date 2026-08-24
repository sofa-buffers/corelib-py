"""FIXLEN_MAX bounds what an encoder is willing to frame (CORELIB_PLAN §6.2).

``FIXLEN_MAX`` (2,147,483,647) is a *format-wide ceiling*: §4.6's fixlen word is
``(len << 3) | subtype`` with a length range of ``0 .. 2,147,483,647``, and a
value above it is ``INVALID`` for every decoder — including this port's own,
which rejects it in ``Decoder.next``. An encoder that framed a longer payload
would therefore hand the caller a message no conformant receiver can read, and
report success while doing it (the encode-side form of §5.1's "never present
partial output as complete"). The refusal is §6.3's ``InvalidArgument``, i.e.
:class:`SofaArgumentError`, and it happens *before* any byte of the field reaches
the buffer.

The oversized payload here is never materialised: 2 GiB of real bytes is not
something a test may allocate, and a conformant encoder decides on the length
alone. Both engines are exercised (issue #73).
"""

from __future__ import annotations

import io

import pytest
from vectors import ENCODER_ENGINES as ENGINES

from sofab.encoder import Encoder as PyEncoder
from sofab.types import FIXLEN_MAX, FixlenSubtype, SofaArgumentError, WireType


class OversizedBlob:
    """A bytes-like one byte past ``FIXLEN_MAX``, whose payload is a tripwire.

    Every route to the actual bytes (the buffer protocol on 3.12+, ``__bytes__``,
    iteration) raises: the encoder must decide on the declared length and refuse,
    never copy 2 GiB it is about to reject.
    """

    def __len__(self) -> int:
        return FIXLEN_MAX + 1

    def __buffer__(self, flags: int) -> memoryview:  # Python >= 3.12 (PEP 688)
        raise AssertionError("oversized payload was materialised")

    def __bytes__(self) -> bytes:
        raise AssertionError("oversized payload was materialised")

    def __iter__(self) -> object:
        raise AssertionError("oversized payload was materialised")


@pytest.mark.parametrize("engine", ENGINES)
class TestBlobOverFixlenMax:
    def test_refused_with_a_range_error(self, engine):
        enc = engine()
        with pytest.raises(SofaArgumentError):
            enc.write_bytes(1, OversizedBlob())

    def test_nothing_reaches_the_output(self, engine):
        # Not even the field header: the message must stay exactly what it was
        # before the refused write, so a caller that recovers still has a
        # readable prefix rather than a dangling header.
        enc = engine()
        enc.write_unsigned(0, 7)
        with pytest.raises(SofaArgumentError):
            enc.write_bytes(1, OversizedBlob())
        assert enc.getvalue() == b"\x00\x07"

    def test_nothing_reaches_a_writer_sink(self, engine):
        out = io.BytesIO()
        enc = engine(out)
        with pytest.raises(SofaArgumentError):
            enc.write_bytes(1, OversizedBlob())
        enc.flush()
        assert out.getvalue() == b""

    def test_sticky_latches_it(self, engine):
        enc = engine(sticky=True)
        enc.write_bytes(1, OversizedBlob())
        assert isinstance(enc.error, SofaArgumentError)
        enc.write_unsigned(2, 1)  # skipped, per sticky mode
        assert enc.getvalue() == b""

    def test_in_range_blob_still_writes(self, engine):
        # The control: the bound rejects only what is over it.
        enc = engine()
        enc.write_bytes(1, b"\x01\x02\x03")
        header = bytes([(1 << 3) | WireType.FIXLEN, (3 << 3) | FixlenSubtype.BLOB])
        assert enc.getvalue() == header + b"\x01\x02\x03"


def test_pure_choke_point_bounds_the_materialised_length():
    """The guard sits on the shared fixlen writer, not only on ``write_bytes``.

    ``write_string`` reaches the wire through the same helper with a payload it
    has already encoded, so the bound has to live where the fixlen word is
    emitted — this drives that helper directly, with a length no allocation
    could produce.
    """
    enc = PyEncoder()
    with pytest.raises(SofaArgumentError):
        enc._write_fixlen(1, OversizedBlob(), FixlenSubtype.STRING)
    assert enc.getvalue() == b""
