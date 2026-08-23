"""Receiver-side caps vs. the fields the SCHEMA already bounds (§6.2.1).

``max_array_count`` / ``max_string_len`` / ``max_blob_len`` are *deployment*
configuration: they protect the receiver from a size the SENDER picks freely.
CORELIB_PLAN §6.2.1 therefore forbids applying them "to a field the schema
already bounds. There the schema bound governs and its violation is INVALID",
and §6.3 says the same from the other end — ``LimitExceeded`` is "never raised
for a field the schema bounds".

Only the caller knows which fields those are, and a :class:`sofab.Binding` is
where it says so: ``cap`` on an array and ``maxlen`` on a string or blob *are*
the schema's bound. Declaring one takes the receiver cap off that field and puts
the verdict on the declared bound instead, as INVALID (§7.1). A field with no
declared bound — one the binding does not name, or names without a ``maxlen`` —
stays under the cap.

The verdict is reached at the count/length header, before any payload is read or
any storage written, which is where §6.2.1 requires it to be. Both engines are
exercised on every case.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder, Status, bound, uvarint, walk

from sofab import (
    Binding,
    Encoder,
    SofaDecodeError,
    SofaIncompleteError,
    SofaLimitError,
)

BIG = "x" * 2000


def _uvarint(x: int) -> list[int]:
    return list(uvarint(x))


# --- a declared field is exempt (§6.2.1 "MUST NOT be applied") ---------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_string_is_exempt_from_max_string_len(engine):
    # The issue's repro: a field the schema bounds at maxlen 4194304, carrying
    # 2000 bytes, decoded by a receiver configured max_string_len=1024. The
    # message is well within its schema bound, so the cap must not touch it.
    enc = Encoder()
    enc.write_string(1, BIG)
    b = Binding().string(1, at=0, maxlen=4194304)
    status, _dec, slots = bound(engine, enc.getvalue(), b, max_string_len=1024)
    assert status is Status.COMPLETE
    assert slots.objects[0] == BIG


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_blob_is_exempt_from_max_blob_len(engine):
    enc = Encoder()
    enc.write_bytes(1, b"y" * 2000)
    b = Binding().bytes(1, at=0, maxlen=4194304)
    status, _dec, slots = bound(engine, enc.getvalue(), b, max_blob_len=1024)
    assert status is Status.COMPLETE
    assert slots.objects[0] == b"y" * 2000


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "write,binder",
    [
        (lambda e: e.write_unsigned_array(1, list(range(64))),
         lambda b: b.unsigned_array(1, at=0, cap=128, count_at=200)),
        (lambda e: e.write_signed_array(1, [-1] * 64),
         lambda b: b.signed_array(1, at=0, cap=128, count_at=200)),
        (lambda e: e.write_float32_array(1, [1.5] * 64),
         lambda b: b.float32_array(1, at=0, cap=128, count_at=200)),
        (lambda e: e.write_float64_array(1, [1.5] * 64),
         lambda b: b.float64_array(1, at=0, cap=128, count_at=200)),
    ],
    ids=["unsigned", "signed", "fp32", "fp64"],
)
def test_a_declared_array_is_exempt_from_max_array_count(engine, write, binder):
    enc = Encoder()
    write(enc)
    status, _dec, slots = bound(
        engine, enc.getvalue(), binder(Binding()), max_array_count=8
    )
    assert status is Status.COMPLETE
    assert slots.u[200] == 64


@pytest.mark.parametrize("engine", ENGINES)
def test_the_declaration_covers_that_field_only(engine):
    """Declaring one field's bound says nothing about the next one."""
    enc = Encoder()
    enc.write_string(1, BIG)  # declared
    enc.write_string(2, BIG)  # not
    b = Binding().string(1, at=0, maxlen=4194304).string(2, at=1)
    with pytest.raises(SofaLimitError):
        bound(engine, enc.getvalue(), b, max_string_len=1024)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declaration_inside_a_sequence_is_honoured(engine):
    enc = Encoder()
    enc.write_sequence_begin_lazy(9)
    enc.write_string(1, BIG)
    enc.write_sequence_end()
    child = Binding().string(1, at=0, maxlen=4194304)
    b = Binding().sequence(9, child)
    status, _dec, slots = bound(engine, enc.getvalue(), b, max_string_len=1024)
    assert status is Status.COMPLETE
    assert slots.objects[0] == BIG


@pytest.mark.parametrize("engine", ENGINES)
def test_declaring_a_bound_with_no_cap_configured_changes_nothing(engine):
    enc = Encoder()
    enc.write_string(1, BIG)
    b = Binding().string(1, at=0, maxlen=4194304)
    status, _dec, slots = bound(engine, enc.getvalue(), b)
    assert status is Status.COMPLETE
    assert slots.objects[0] == BIG


# --- an undeclared field is still capped ------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_an_undeclared_string_is_still_rejected(engine):
    enc = Encoder()
    enc.write_string(1, BIG)
    b = Binding().string(1, at=0)  # bound, but with no maxlen declared
    with pytest.raises(SofaLimitError):
        bound(engine, enc.getvalue(), b, max_string_len=1024)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_undeclared_blob_is_still_rejected(engine):
    enc = Encoder()
    enc.write_bytes(1, b"y" * 2000)
    b = Binding().bytes(1, at=0)
    with pytest.raises(SofaLimitError):
        bound(engine, enc.getvalue(), b, max_blob_len=1024)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "write",
    [
        lambda e: e.write_unsigned_array(1, list(range(6))),
        lambda e: e.write_signed_array(1, [-1] * 6),
        lambda e: e.write_float32_array(1, [1.5] * 6),
        lambda e: e.write_float64_array(1, [1.5] * 6),
    ],
    ids=["unsigned", "signed", "fp32", "fp64"],
)
def test_an_unbound_array_is_still_rejected_on_every_array_kind(engine, write):
    """A field the table does not name at all declares nothing, so the cap
    governs it — even though nobody is going to read it."""
    enc = Encoder()
    write(enc)
    with pytest.raises(SofaLimitError):
        walk(engine, enc.getvalue(), max_array_count=5)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_undeclared_field_is_rejected_even_when_declined(engine):
    """Declining a field is a consume too: the walk still passes over the
    payload the cap exists to refuse."""
    enc = Encoder()
    enc.write_string(1, BIG)
    with pytest.raises(SofaLimitError):
        walk(
            engine, enc.getvalue(), max_string_len=1024,
            recorder=Recorder(decline=lambda f: True),
        )


# --- the verdict is reached before any payload ------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_the_cap_is_decided_at_the_header_before_any_payload(engine):
    """A hostile message claiming a huge length with nothing behind it is
    rejected by the cap, not reported as truncated: §6.2.1 wants the verdict on
    the length word alone, before a byte of payload is buffered."""
    # id 1, FIXLEN, length_header = (10_000_000 << 3) | STRING, then nothing.
    data = bytes([0x0A] + _uvarint((10_000_000 << 3) | 0x2))
    with pytest.raises(SofaLimitError) as exc:
        walk(engine, data, max_string_len=1024)
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("engine", ENGINES)
def test_an_array_count_is_capped_before_its_elements(engine):
    # id 1, ARRAY_UNSIGNED, count 2^31-1, then one lone byte.
    data = bytes([0x0B] + _uvarint(0x7FFFFFFF) + [0x01])
    with pytest.raises(SofaLimitError):
        walk(engine, data, max_array_count=5)


# --- past the DECLARED bound is INVALID, never LimitExceeded ----------------


@pytest.mark.parametrize("engine", ENGINES)
def test_over_the_declared_string_bound_is_invalid(engine):
    """What generated code gets for a ``maxlen: 1024`` field: the binding
    declares it, and an over-bound length is INVALID (§7.1) — never the receiver
    cap's SofaLimitError (§6.3)."""
    enc = Encoder()
    enc.write_string(1, BIG)
    b = Binding().string(1, at=0, maxlen=1024)
    status, dec, slots = bound(engine, enc.getvalue(), b, max_string_len=512)
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)
    assert slots.objects[0] is None  # nothing was materialised


@pytest.mark.parametrize("engine", ENGINES)
def test_over_the_declared_array_bound_is_invalid(engine):
    enc = Encoder()
    enc.write_unsigned_array(1, list(range(64)))
    b = Binding().unsigned_array(1, at=0, cap=8, count_at=200)
    status, dec, slots = bound(engine, enc.getvalue(), b, max_array_count=4)
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)
    assert slots.u[0] == 0  # rejected at the count header, before an element


@pytest.mark.parametrize("engine", ENGINES)
def test_exactly_at_the_declared_bound_is_accepted(engine):
    enc = Encoder()
    enc.write_string(1, "z" * 64)
    b = Binding().string(1, at=0, maxlen=64)
    status, _dec, slots = bound(engine, enc.getvalue(), b, max_string_len=8)
    assert status is Status.COMPLETE
    assert slots.objects[0] == "z" * 64


# --- a declared field still gets its ordinary outcomes ----------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_field_can_still_be_incomplete(engine):
    """Lifting the cap does not make the field immune to truncation."""
    enc = Encoder()
    enc.write_string(1, BIG)
    wire = enc.getvalue()
    b = Binding().string(1, at=0, maxlen=4194304)
    status, dec, slots = bound(engine, wire[: len(wire) // 2], b, max_string_len=1024)
    assert status is Status.INCOMPLETE
    assert dec.error is None
    assert slots.objects[0] is None


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_field_can_still_be_invalid_utf8(engine):
    # id 1, FIXLEN STRING of length 2, payload 0xFF 0xFE.
    data = bytes([0x0A, 0x12, 0xFF, 0xFE])
    b = Binding().string(1, at=0, maxlen=4194304)
    status, dec, _slots = bound(engine, data, b, max_string_len=1)
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_capped_field_reports_the_cap_rather_than_the_mismatch(engine):
    """§6.2.1 outranks §7.3: the skip §7.3 asks for still walks the payload the
    cap exists to refuse, so the cap is what the message is rejected with."""
    enc = Encoder()
    enc.write_string(1, BIG)
    b = Binding().unsigned_array(1, at=0, cap=8, count_at=200)  # wrong type
    with pytest.raises(SofaLimitError):
        bound(engine, enc.getvalue(), b, max_string_len=1024)
