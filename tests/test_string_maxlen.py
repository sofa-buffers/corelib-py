"""``write_string(id, text, maxlen)``: a caller's bound on the UTF-8 length.

A schema's ``maxlen`` counts UTF-8 bytes, and a ``str`` does not know that length:
only ``write_string`` produces it. So the bound is passed into the call, which
compares it in the same pass that produces the bytes and refuses an over-long
payload with :class:`SofaArgumentError` (CORELIB_PLAN §6.3) before any byte of the
field is written. The library holds no bound: ``None`` (the default) checks none.

Both engines must answer every case identically: the refusal, the bytes, and what
the buffer holds afterwards.
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
