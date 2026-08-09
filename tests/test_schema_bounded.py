"""Receiver-side caps vs. the fields the SCHEMA already bounds (§6.2.1).

``max_array_count`` / ``max_string_len`` / ``max_blob_len`` are *deployment*
configuration: they protect the receiver from a size the SENDER picks freely.
CORELIB_PLAN §6.2.1 therefore forbids applying them "to a field the schema
already bounds. There the schema bound governs and its violation is INVALID",
and §6.3 says the same from the other end — ``LimitExceeded`` is "never raised
for a field the schema bounds".

Only the caller knows which fields those are, so it says so per field with
:meth:`Decoder.schema_bounded`, and the cap's verdict is parked on the pending
value at the count/length header until the field is actually consumed — before
any payload is read or any list allocated, which is where §6.2.1 requires it to
be decided. Both engines are exercised on every case.
"""

from __future__ import annotations

import io

import pytest

from sofab import (
    Encoder,
    SofaDecodeError,
    SofaIncompleteError,
    SofaLimitError,
    SofaStateError,
)
from sofab.decoder import Decoder as PyDecoder

ENGINES = [pytest.param(PyDecoder, id="pure")]
try:  # the native accelerator is optional; skip that half where it is absent
    from sofab._speedups import Decoder as NativeDecoder
except ImportError:  # pragma: no cover - pure-Python-only install
    pass
else:
    ENGINES.append(pytest.param(NativeDecoder, id="native"))

BIG = "x" * 2000


def _dec(engine, data, **kw):
    return engine(io.BytesIO(bytes(data)), **kw)


def _uvarint(x: int) -> list[int]:
    out = []
    while True:
        b = x & 0x7F
        x >>= 7
        out.append(b | 0x80 if x else b)
        if not x:
            return out


# --- a declared field is exempt (§6.2.1 "MUST NOT be applied") ---------------


@pytest.mark.parametrize("engine", ENGINES)
def test_schema_bounded_string_is_exempt_from_max_string_len(engine):
    # The issue's repro: a field the schema bounds at maxlen 4194304, carrying
    # 2000 bytes, decoded by a receiver configured max_string_len=1024. The
    # message is well within its schema bound, so the cap must not touch it.
    enc = Encoder()
    enc.write_string(1, BIG)
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    f = dec.next()
    assert f is not None and f.id == 1
    dec.schema_bounded()
    assert dec.string() == BIG


@pytest.mark.parametrize("engine", ENGINES)
def test_schema_bounded_blob_is_exempt_from_max_blob_len(engine):
    payload = bytes(range(256)) * 8
    enc = Encoder()
    enc.write_bytes(2, payload)
    dec = _dec(engine, enc.getvalue(), max_blob_len=16)
    assert dec.next() is not None
    dec.schema_bounded()
    assert dec.bytes() == payload


@pytest.mark.parametrize("engine", ENGINES)
def test_schema_bounded_arrays_are_exempt_from_max_array_count(engine):
    values = list(range(64))
    cases = [
        (lambda e: e.write_unsigned_array(3, values), lambda d: d.read_unsigned_array(), values),
        (lambda e: e.write_signed_array(4, values), lambda d: d.read_signed_array(), values),
        (
            lambda e: e.write_float32_array(5, [1.0] * 64),
            lambda d: d.read_float32_array(),
            [1.0] * 64,
        ),
        (
            lambda e: e.write_float64_array(6, [2.0] * 64),
            lambda d: d.read_float64_array(),
            [2.0] * 64,
        ),
    ]
    for write, read, want in cases:
        enc = Encoder()
        write(enc)
        dec = _dec(engine, enc.getvalue(), max_array_count=8)
        assert dec.next() is not None
        dec.schema_bounded()
        assert read(dec) == want


@pytest.mark.parametrize("engine", ENGINES)
def test_declaration_covers_the_current_field_only(engine):
    # §6.2.1 is a per-field statement, so the declaration is too: the next
    # next() starts an undeclared — and therefore capped — field again.
    enc = Encoder()
    enc.write_string(1, BIG)
    enc.write_string(2, BIG)
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    assert dec.next() is not None
    dec.schema_bounded()
    assert dec.string() == BIG
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.string()


@pytest.mark.parametrize("engine", ENGINES)
def test_declaration_inside_a_sequence_is_honoured(engine):
    enc = Encoder()
    enc.write_sequence_begin_lazy(9)
    enc.write_string(1, BIG)
    enc.write_sequence_end()
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    assert dec.next() is not None  # sequence start
    assert dec.next() is not None  # the bounded string inside it
    dec.schema_bounded()
    assert dec.string() == BIG


@pytest.mark.parametrize("engine", ENGINES)
def test_schema_bounded_on_an_uncapped_field_is_a_no_op(engine):
    enc = Encoder()
    enc.write_string(1, "small")
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    assert dec.next() is not None
    dec.schema_bounded()
    dec.schema_bounded()  # idempotent
    assert dec.string() == "small"


@pytest.mark.parametrize("engine", ENGINES)
def test_schema_bounded_before_any_field_is_harmless(engine):
    enc = Encoder()
    enc.write_unsigned(1, 7)
    dec = _dec(engine, enc.getvalue())
    dec.schema_bounded()
    assert dec.next() is not None
    assert dec.unsigned() == 7


# --- an UNDECLARED field is still capped -------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_undeclared_string_is_still_rejected(engine):
    enc = Encoder()
    enc.write_string(1, BIG)
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.string()


@pytest.mark.parametrize("engine", ENGINES)
def test_undeclared_blob_is_still_rejected(engine):
    enc = Encoder()
    enc.write_bytes(1, b"y" * 100)
    dec = _dec(engine, enc.getvalue(), max_blob_len=10)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.bytes()


@pytest.mark.parametrize("engine", ENGINES)
def test_undeclared_array_is_still_rejected_on_every_array_kind(engine):
    cases = [
        (lambda e: e.write_unsigned_array(1, list(range(6))), lambda d: d.read_unsigned_array()),
        (lambda e: e.write_signed_array(1, list(range(6))), lambda d: d.read_signed_array()),
        (lambda e: e.write_float32_array(1, [1.0] * 6), lambda d: d.read_float32_array()),
        (lambda e: e.write_float64_array(1, [1.0] * 6), lambda d: d.read_float64_array()),
    ]
    for write, read in cases:
        enc = Encoder()
        write(enc)
        dec = _dec(engine, enc.getvalue(), max_array_count=5)
        assert dec.next() is not None
        with pytest.raises(SofaLimitError):
            read(dec)


@pytest.mark.parametrize("engine", ENGINES)
def test_undeclared_field_is_rejected_when_skipped(engine):
    # A skip still buffers the payload, so the cap protects it too: skip() and
    # the auto-skip the following next() performs both reject the field.
    enc = Encoder()
    enc.write_bytes(1, b"z" * 100)
    data = enc.getvalue()

    dec = _dec(engine, data, max_blob_len=10)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.skip()

    dec2 = _dec(engine, data, max_blob_len=10)
    assert dec2.next() is not None
    with pytest.raises(SofaLimitError):
        dec2.next()


@pytest.mark.parametrize("engine", ENGINES)
def test_cap_is_decided_at_the_length_header_before_any_payload(engine):
    # §6.2.1: enforced at the count/length header, before the allocation it is
    # meant to prevent. A header claiming 100 elements with NO payload behind it
    # is a limit rejection, not the truncation reading those elements would give.
    dec = _dec(engine, [0x03] + _uvarint(100), max_array_count=10)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.read_unsigned_array()

    # 0x02 = (0<<3)|FIXLEN; length header = (100 << 3) | 0x2 (STRING).
    dec2 = _dec(engine, [0x02] + _uvarint((100 << 3) | 0x2), max_string_len=10)
    assert dec2.next() is not None
    with pytest.raises(SofaLimitError):
        dec2.string()


@pytest.mark.parametrize("engine", ENGINES)
def test_declared_field_still_gets_its_ordinary_outcomes(engine):
    # Declaring a field schema-bounded waives the CAP, nothing else: a truncated
    # payload is still INCOMPLETE and malformed bytes are still INVALID.
    dec = _dec(engine, [0x02] + _uvarint((100 << 3) | 0x2), max_string_len=10)
    assert dec.next() is not None
    dec.schema_bounded()
    with pytest.raises(SofaIncompleteError):
        dec.string()

    bad = [0x02] + _uvarint((2 << 3) | 0x2) + [0xC3, 0x28]  # invalid UTF-8
    dec2 = _dec(engine, bad, max_string_len=1)
    assert dec2.next() is not None
    dec2.schema_bounded()
    with pytest.raises(SofaDecodeError):
        dec2.string()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_capped_field_read_as_the_wrong_type_reports_the_cap(engine):
    # The field is being rejected either way, so the read that would consume it
    # reports why it is rejected rather than which type it isn't.
    enc = Encoder()
    enc.write_string(1, BIG)
    dec = _dec(engine, enc.getvalue(), max_string_len=1024)
    assert dec.next() is not None
    with pytest.raises(SofaLimitError):
        dec.read_unsigned_array()


@pytest.mark.parametrize("engine", ENGINES)
def test_fixlen_len_still_refuses_a_capped_array(engine):
    # The peek answers for a fixlen field only: a capped ARRAY states a count,
    # not a payload byte length, so it is the wrong shape for it either way.
    enc = Encoder()
    enc.write_unsigned_array(1, list(range(6)))
    dec = _dec(engine, enc.getvalue(), max_array_count=5)
    assert dec.next() is not None
    with pytest.raises(SofaStateError):
        dec.fixlen_len()


# --- the schema bound is the caller's to enforce, as INVALID -----------------


@pytest.mark.parametrize("engine", ENGINES)
def test_fixlen_len_peeks_through_a_capped_field(engine):
    # fixlen_len() is a pure peek — it allocates nothing — and it is how
    # generated code enforces the SCHEMA maxlen (as INVALID, §7.1). It therefore
    # answers whether or not the receiver cap has spoken on the field, so the
    # schema bound can be decided first, in either order.
    enc = Encoder()
    enc.write_string(1, BIG)
    data = enc.getvalue()

    dec = _dec(engine, data, max_string_len=1024)
    assert dec.next() is not None
    assert dec.fixlen_len() == 2000
    dec.schema_bounded()
    assert dec.fixlen_len() == 2000
    assert dec.string() == BIG


@pytest.mark.parametrize("engine", ENGINES)
def test_over_schema_bound_is_invalid_not_limit_exceeded(engine):
    # What generated code does for a `maxlen: 1024` field: declare it, then
    # reject an over-bound length itself. The verdict is INVALID
    # (SofaDecodeError), never the receiver cap's SofaLimitError (§6.3).
    enc = Encoder()
    enc.write_string(1, BIG)
    dec = _dec(engine, enc.getvalue(), max_string_len=512)
    assert dec.next() is not None
    dec.schema_bounded()
    with pytest.raises(SofaDecodeError) as exc:
        if dec.fixlen_len() > 1024:
            raise SofaDecodeError("s: string byte length above schema maxlen 1024")
        dec.string()
    assert not isinstance(exc.value, SofaLimitError)
