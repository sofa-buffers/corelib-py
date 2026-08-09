"""Skipping a payload must *consume* it, never materialise it (§5.2).

CORELIB_PLAN §5.2 defines skip as pure consumption — "the field's remaining
bytes ... are consumed and discarded automatically" — and the port's profile is
maxspeed, where skip-heavy decoding (unknown fields, forward compatibility) is a
first-class workload. Building the discarded payload as a ``bytes`` object first
is a full copy of something nobody asked for, so these tests pin the *absence*
of that allocation rather than any visible value.

Allocation is measured with :mod:`tracemalloc`, which counts every allocation
made through CPython's allocators — the compiled accelerator's ``bytes`` objects
included, so the same test binds both engines. Each case carries its own control
measurement: the typed read of the same field, which *must* allocate the payload
it returns. If the control does not see the allocation, the measurement itself
is not working on this interpreter (PyPy, a tracemalloc-less build) and the case
is skipped rather than passing vacuously.

The behavioural half — a skipped payload leaves the cursor in exactly the right
place, on a whole-message reader and on a byte-at-a-time one — is pinned here
too, since the no-copy path is where an off-by-one would land.
"""

from __future__ import annotations

import tracemalloc
from typing import Any, Callable

import pytest
from vectors import ENGINE_PAIRS, ChunkReader, reader

ENGINES = pytest.mark.parametrize(("Encoder", "Decoder"), ENGINE_PAIRS)

# Big enough that a copy of it dwarfs the decoder's own bookkeeping, small
# enough to stay comfortable in CI.
PAYLOAD = 4 << 20


def _peak_bytes(fn: Callable[[], Any]) -> int:
    """Peak extra bytes allocated while ``fn()`` runs."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak - base


def _measure(Decoder: Any, wire: bytes, consume: Callable[[Any], Any]) -> int:
    """Peak allocation of ``consume`` alone: the message is pulled into the
    decoder's buffer by ``next()``, outside the measured region."""
    dec = Decoder(reader(wire), chunk_size=len(wire) + 16)
    dec.next()
    return _peak_bytes(lambda: consume(dec))


def _blob(Encoder: Any) -> bytes:
    enc = Encoder()
    enc.write_bytes(1, b"\xa5" * PAYLOAD)
    enc.write_unsigned(2, 7)
    return enc.getvalue()


def _string(Encoder: Any) -> bytes:
    enc = Encoder()
    enc.write_string(1, "s" * PAYLOAD)
    enc.write_unsigned(2, 7)
    return enc.getvalue()


def _f64_array(Encoder: Any) -> bytes:
    enc = Encoder()
    enc.write_float64_array(1, [1.5] * (PAYLOAD // 8))
    enc.write_unsigned(2, 7)
    return enc.getvalue()


def _control(Decoder: Any, wire: bytes, read_it: Callable[[Any], Any]) -> int:
    """Peak allocation of the typed read of the same field — the payload it
    returns *must* be allocated, so this both scales the assertion and proves
    the measurement works at all on this interpreter."""
    peak = _measure(Decoder, wire, read_it)
    if peak < PAYLOAD // 2:
        pytest.skip("tracemalloc does not account allocations on this interpreter")
    return peak


CASES = pytest.mark.parametrize(
    ("build", "read_it"),
    [
        (_blob, lambda d: d.bytes()),
        (_string, lambda d: d.string()),
        (_f64_array, lambda d: d.read_float64_array()),
    ],
    ids=["blob", "string", "fp64-array"],
)


@ENGINES
@CASES
def test_skip_does_not_copy_the_payload(
    Encoder: Any, Decoder: Any, build: Any, read_it: Any
) -> None:
    """A buffered payload is skipped by advancing the cursor: the skip allocates
    a fraction of what reading the same field allocates."""
    wire = build(Encoder)
    control = _control(Decoder, wire, read_it)
    skipped = _measure(Decoder, wire, lambda d: d.skip())
    assert skipped < PAYLOAD // 8, (
        f"skip allocated {skipped} bytes for a {PAYLOAD}-byte payload "
        f"(reading it allocates {control}) — the payload is being copied"
    )


@ENGINES
@CASES
def test_auto_skip_does_not_copy_the_payload(
    Encoder: Any, Decoder: Any, build: Any, read_it: Any
) -> None:
    """The implicit skip ``next()`` performs over an unconsumed value takes the
    same no-copy path as an explicit ``skip()``."""
    wire = build(Encoder)
    _control(Decoder, wire, read_it)
    dec = Decoder(reader(wire), chunk_size=len(wire) + 16)
    dec.next()
    peak = _peak_bytes(dec.next)
    assert peak < PAYLOAD // 8, f"auto-skip allocated {peak} bytes for a {PAYLOAD}-byte payload"


@ENGINES
@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 1000], ids=str)
@pytest.mark.parametrize("kind", ["blob", "string", "fp64-array"])
def test_skip_lands_on_the_next_field(
    Encoder: Any, Decoder: Any, size: int, kind: str
) -> None:
    """Consuming without materialising must leave the cursor exactly where a
    read would — payloads either side of the buffer boundary, on a whole-message
    reader and on a byte-at-a-time one."""
    enc = Encoder()
    if kind == "blob":
        enc.write_bytes(1, b"\xa5" * size)
    elif kind == "string":
        enc.write_string(1, "s" * size)
    else:
        enc.write_float64_array(1, [1.5] * size)
    enc.write_unsigned(2, 7)
    wire = enc.getvalue()

    for src in (reader(wire), ChunkReader(wire, chunk=1)):
        dec = Decoder(src, chunk_size=64)
        dec.next()
        dec.skip()
        f = dec.next()
        assert f is not None and f.id == 2
        assert dec.unsigned() == 7
        assert dec.next() is None
