"""A caller's declared bound, handed to the writer that already measures the value.

Generated code refuses a value past its schema bound at encode. Where a writer takes
the value whole, the bound rides the call as its optional last argument and the
writer compares it in the pass it already makes:

* ``write_string(id, text, maxlen)`` -- a schema's ``maxlen`` counts UTF-8 bytes,
  and a ``str`` does not know that length: only ``write_string`` produces it;
* ``write_bytes(id, data, maxlen)`` -- the blob length;
* ``write_unsigned(id, value, max_value)`` / ``write_signed(id, value, min_value,
  max_value)`` -- the declared width of a narrower integer, which Python's
  unbounded ``int`` does not carry;
* every native array writer's ``cap`` -- the declared capacity, against the
  element count.

A value past it is :class:`SofaArgumentError` (CORELIB_PLAN §6.3) before any byte
of the field is written. The library holds no bound: ``None`` (the default) checks
none, and the bytes are exactly those of the call without it.

Both engines must answer every case identically: the refusal, its text, the bytes,
and what the buffer holds afterwards.
"""

from __future__ import annotations

import io

import pytest
from vectors import ENCODER_ENGINES

from sofab import SofaArgumentError


def _buf_encoder(enc_cls, **kw):
    out = io.BytesIO()
    return enc_cls(out, **kw), out


def _bytes(enc_cls, fn) -> bytes:
    enc, out = _buf_encoder(enc_cls)
    fn(enc)
    enc.flush()
    return out.getvalue()


#: (text, maxlen): every one fits. The bound is in BYTES, so "éé" (4 bytes,
#: 2 characters) fits maxlen 4 exactly and a 0 bound admits only "".
FITS = [
    ("", 0),
    ("", 4),
    ("abcd", 4),
    ("éé", 4),
    ("€", 3),
    ("🎉", 4),
    ("x" * 1000, 1000),
]

#: (text, maxlen): every one is one byte or more past its bound. "xxxé" is 5
#: bytes: a cut at 4 would land inside the "é", which is why it is refused and
#: never shortened. "éé" is 2 characters but 4 bytes -- over a bound of 3 even
#: though len() is 2.
OVER = [
    ("a", 0),
    ("xxxxx", 4),
    ("xxxé", 4),
    ("é" * 40, 4),
    ("éé", 3),
    ("🎉", 3),
    ("x" * 1001, 1000),
]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("text,maxlen", FITS, ids=range(len(FITS)))
def test_at_or_under_the_bound_writes_the_same_bytes_as_no_bound(enc_cls, text, maxlen):
    with_bound = _bytes(enc_cls, lambda e: e.write_string(1, text, maxlen))
    without = _bytes(enc_cls, lambda e: e.write_string(1, text))
    assert with_bound == without


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("text,maxlen", OVER, ids=range(len(OVER)))
def test_over_the_bound_is_refused_and_writes_nothing(enc_cls, text, maxlen):
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="exceeds maxlen"):
        enc.write_string(1, text, maxlen)
    enc.flush()
    # Only the field before it: no header, no length, no partial payload.
    assert out.getvalue() == b"\x00\x07"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_over_the_bound_in_a_fixed_buffer_leaves_the_cursor(enc_cls):
    buf = bytearray(16)
    enc = enc_cls.over_buffer(buf, 0)
    with pytest.raises(SofaArgumentError):
        enc.write_string(3, "xxxxx", 4)
    assert enc.bytes_used() == 0


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_none_is_no_bound(enc_cls):
    big = "y" * 5000
    assert _bytes(enc_cls, lambda e: e.write_string(1, big, None)) == \
        _bytes(enc_cls, lambda e: e.write_string(1, big))


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_mode_latches_the_refusal(enc_cls):
    enc, out = _buf_encoder(enc_cls, sticky=True)
    enc.write_string(1, "xxxxx", 4)  # latched, not raised
    enc.write_unsigned(2, 1)          # skipped: the encoder is already failed
    assert isinstance(enc.error, SofaArgumentError)
    assert "exceeds maxlen" in str(enc.error)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_invalid_utf8_still_wins_over_the_bound(enc_cls):
    """A lone surrogate is refused as invalid UTF-8 whatever the bound says: there
    are no bytes to measure."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="UTF-8"):
        enc.write_string(1, "\ud800", 100)


def test_both_engines_refuse_the_same_cases():
    """Parity: the verdict and the bytes agree case for case."""
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    for text, maxlen in FITS + OVER:
        got = []
        for cls in (PyEncoder, native.Encoder):
            enc, out = _buf_encoder(cls)
            try:
                enc.write_string(1, text, maxlen)
                enc.flush()
                got.append(("ok", out.getvalue()))
            except SofaArgumentError as exc:
                got.append(("refused", str(exc)))
        assert got[0] == got[1], (text, maxlen)


# --- every other bounded writer -----------------------------------------------
#
# (name, call(enc, bound), bound that fits exactly, bound one below it, call without
# a bound). The value is the same in all three calls; only the bound moves.

WRITERS = [
    ("unsigned", lambda e, b: e.write_unsigned(1, 255, b), 255, 254,
     lambda e: e.write_unsigned(1, 255)),
    ("unsigned zero", lambda e, b: e.write_unsigned(1, 0, b), 0, None,
     lambda e: e.write_unsigned(1, 0)),
    ("bytes", lambda e, b: e.write_bytes(1, b"1234", b), 4, 3,
     lambda e: e.write_bytes(1, b"1234")),
    ("bytes bytearray", lambda e, b: e.write_bytes(1, bytearray(b"12"), b), 2, 1,
     lambda e: e.write_bytes(1, bytearray(b"12"))),
    ("unsigned array", lambda e, b: e.write_unsigned_array(1, [1, 2, 3], b), 3, 2,
     lambda e: e.write_unsigned_array(1, [1, 2, 3])),
    ("signed array", lambda e, b: e.write_signed_array(1, [-1, 2], b), 2, 1,
     lambda e: e.write_signed_array(1, [-1, 2])),
    ("bool array", lambda e, b: e.write_bool_array(1, [True, False, True], b), 3, 2,
     lambda e: e.write_bool_array(1, [True, False, True])),
    ("float32 array", lambda e, b: e.write_float32_array(1, [1.5, 2.5], b), 2, 1,
     lambda e: e.write_float32_array(1, [1.5, 2.5])),
    ("float64 array", lambda e, b: e.write_float64_array(1, [1.5], b), 1, 0,
     lambda e: e.write_float64_array(1, [1.5])),
    ("empty array", lambda e, b: e.write_unsigned_array(1, [], b), 0, None,
     lambda e: e.write_unsigned_array(1, [])),
    ("tuple array", lambda e, b: e.write_signed_array(1, (5, 6, 7), b), 3, 2,
     lambda e: e.write_signed_array(1, (5, 6, 7))),
]
_IDS = [w[0] for w in WRITERS]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,plain", WRITERS, ids=_IDS)
def test_a_bound_that_fits_changes_no_byte(enc_cls, name, call, fits, under, plain):
    assert _bytes(enc_cls, lambda e: call(e, fits)) == _bytes(enc_cls, plain)
    assert _bytes(enc_cls, lambda e: call(e, None)) == _bytes(enc_cls, plain)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,plain",
                         [w for w in WRITERS if w[3] is not None],
                         ids=[w[0] for w in WRITERS if w[3] is not None])
def test_one_past_the_bound_is_refused_and_writes_nothing(enc_cls, name, call, fits, under, plain):
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="exceeds|outside"):
        call(enc, under)
    enc.flush()
    assert out.getvalue() == b"\x00\x07"


#: (value, min_value, max_value, refused?) for write_signed.
SIGNED = [
    (-128, -128, 127, False),
    (127, -128, 127, False),
    (-129, -128, 127, True),
    (128, -128, 127, True),
    (-32769, -32768, 32767, True),
    (40000, None, 32767, True),
    (-40000, None, 32767, False),
    (-40000, -32768, None, True),
    (40000, -32768, None, False),
    (0, None, None, False),
]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("value,lo,hi,refused", SIGNED, ids=range(len(SIGNED)))
def test_signed_range(enc_cls, value, lo, hi, refused):
    if refused:
        enc, out = _buf_encoder(enc_cls)
        with pytest.raises(SofaArgumentError, match="outside"):
            enc.write_signed(1, value, lo, hi)
        enc.flush()
        assert out.getvalue() == b""
    else:
        assert _bytes(enc_cls, lambda e: e.write_signed(1, value, lo, hi)) == \
            _bytes(enc_cls, lambda e: e.write_signed(1, value))


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_64_bit_range_is_checked_before_the_bound(enc_cls):
    """A value no 64-bit field can hold reports that, not the declared bound."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_unsigned(1, -1, 255)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_signed(1, 1 << 63, -128, 127)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_index_object_is_bounded_by_its_value(enc_cls):
    """An IntEnum/IntFlag member is an integer by __index__; the bound applies to it."""
    import enum

    class E(enum.IntEnum):
        BIG = 300

    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds"):
        enc.write_unsigned(1, E.BIG, 255)
    with pytest.raises(SofaArgumentError, match="outside"):
        enc.write_signed(1, E.BIG, -128, 127)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_mode_latches_every_bound(enc_cls):
    for call in (lambda e: e.write_unsigned(1, 256, 255),
                 lambda e: e.write_signed(1, 128, -128, 127),
                 lambda e: e.write_bytes(1, b"12345", 4),
                 lambda e: e.write_unsigned_array(1, [1, 2], 1),
                 lambda e: e.write_float32_array(1, [1.0, 2.0], 1)):
        enc, out = _buf_encoder(enc_cls, sticky=True)
        call(enc)
        enc.write_unsigned(2, 1)  # skipped: already failed
        assert isinstance(enc.error, SofaArgumentError)
        enc.flush()
        assert out.getvalue() == b""


def test_both_engines_word_every_refusal_alike():
    """Parity: the verdict, its text and the bytes agree call for call."""
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    calls = [lambda e, w=w: w[1](e, w[2]) for w in WRITERS]
    calls += [lambda e, w=w: w[1](e, w[3]) for w in WRITERS if w[3] is not None]
    calls += [lambda e, v=v: e.write_signed(1, v[0], v[1], v[2]) for v in SIGNED]
    for i, call in enumerate(calls):
        got = []
        for cls in (PyEncoder, native.Encoder):
            enc, out = _buf_encoder(cls)
            try:
                call(enc)
                enc.flush()
                got.append(("ok", out.getvalue()))
            except SofaArgumentError as exc:
                got.append(("refused", str(exc)))
        assert got[0] == got[1], i
