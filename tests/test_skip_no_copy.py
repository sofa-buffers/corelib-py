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
from vectors import ENGINE_PAIRS, Binding, Recorder, Status, walk

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


def _measure(Decoder: Any, wire: bytes, *, decline: bool) -> int:
    """Peak allocation of decoding ``wire`` with the payload field taken or
    declined. The bytes are handed over in one feed either way, so the two runs
    differ only in whether the payload is materialised."""
    rec = Recorder(decline=(lambda f: f.id == 1) if decline else None)
    dec = Decoder(visitor=rec)
    return _peak_bytes(lambda: dec.feed(wire))


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


def _control(Decoder: Any, wire: bytes) -> int:
    """Peak allocation when the payload IS materialised — it must be allocated,
    so this both scales the assertion and proves the measurement works at all on
    this interpreter."""
    peak = _measure(Decoder, wire, decline=False)
    if peak < PAYLOAD // 2:
        pytest.skip("tracemalloc does not account allocations on this interpreter")
    return peak


CASES = pytest.mark.parametrize(
    "build", [_blob, _string, _f64_array], ids=["blob", "string", "fp64-array"]
)


@ENGINES
@CASES
def test_a_declined_payload_is_not_copied(Encoder: Any, Decoder: Any, build: Any) -> None:
    """A buffered payload a handler declines is walked by advancing the cursor:
    it allocates a fraction of what materialising the same field allocates."""
    wire = build(Encoder)
    control = _control(Decoder, wire)
    skipped = _measure(Decoder, wire, decline=True)
    assert skipped < PAYLOAD // 8, (
        f"declining allocated {skipped} bytes for a {PAYLOAD}-byte payload "
        f"(materialising it allocates {control}) — the payload is being copied"
    )


@ENGINES
@CASES
def test_an_unbound_payload_is_not_copied(Encoder: Any, Decoder: Any, build: Any) -> None:
    """A field no binding names and no visitor wants is walked the same way —
    nothing is built for bytes nobody asked for."""
    wire = build(Encoder)
    _control(Decoder, wire)
    b = Binding().unsigned(2, at=0, count_at=1)  # field 1, the payload, is unbound
    words = bytearray(b.tree_words_required * 8)
    dec = Decoder(binding=b, words=words)
    peak = _peak_bytes(lambda: dec.feed(wire))
    assert peak < PAYLOAD // 8, (
        f"an unbound payload allocated {peak} bytes for a {PAYLOAD}-byte payload"
    )


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

    for chunk in (None, 1):
        status, rec, _dec = walk(
            Decoder, wire, chunk=chunk, recorder=Recorder(decline=lambda f: f.id == 1)
        )
        assert status is Status.COMPLETE
        assert rec.events == [("u", 2, 7)]
