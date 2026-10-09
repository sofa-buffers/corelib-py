"""A caller's declared bound, handed to the writer that already measures the value.

Generated code refuses a value past its schema bound at encode. Every writer that
measures its value has a ``*_bounded`` twin taking the bound as an argument, and
compares it in the pass the writer already makes:

* ``write_string_bounded(id, text, maxlen)`` -- a schema's ``maxlen`` counts UTF-8
  bytes, and a ``str`` does not know that length: only the writer produces it;
* ``write_bytes_bounded(id, data, maxlen)`` -- the blob length;
* ``write_unsigned_bounded(id, value, max_value)`` /
  ``write_signed_bounded(id, value, min_value, max_value)`` -- the declared width
  of a narrower integer, which Python's unbounded ``int`` does not carry;
* ``write_{unsigned,signed}_array_bounded(id, values, cap, [elem_min,] elem_max)``
  -- the declared capacity against the element count, and the element's declared
  width against each element, in the range check every element already passes;
* ``write_{bool,float32,float64}_array_bounded(id, values, cap)``.

A value past a bound is :class:`SofaArgumentError` (CORELIB_PLAN §6.3). A scalar,
string, blob or array count is refused before any byte of the field is written; an
element is refused where the 64-bit range check refuses one, mid-array. The library
holds no bound: ``None`` checks none on that side, and the bytes are exactly those
of the plain writer. A bound is compared as a number, as Python compares it (see
ODD_BOUNDS below).

Both engines must answer every case identically: the refusal, its text, the bytes,
and what the buffer holds afterwards.
"""

from __future__ import annotations

import enum
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
    with_bound = _bytes(enc_cls, lambda e: e.write_string_bounded(1, text, maxlen))
    without = _bytes(enc_cls, lambda e: e.write_string(1, text))
    assert with_bound == without


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("text,maxlen", OVER, ids=range(len(OVER)))
def test_over_the_bound_is_refused_and_writes_nothing(enc_cls, text, maxlen):
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="exceeds maxlen"):
        enc.write_string_bounded(1, text, maxlen)
    enc.flush()
    # Only the field before it: no header, no length, no partial payload.
    assert out.getvalue() == b"\x00\x07"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_over_the_bound_in_a_fixed_buffer_leaves_the_cursor(enc_cls):
    buf = bytearray(16)
    enc = enc_cls.over_buffer(buf, 0)
    with pytest.raises(SofaArgumentError):
        enc.write_string_bounded(3, "xxxxx", 4)
    assert enc.bytes_used() == 0


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_none_is_no_bound(enc_cls):
    big = "y" * 5000
    assert _bytes(enc_cls, lambda e: e.write_string_bounded(1, big, None)) == \
        _bytes(enc_cls, lambda e: e.write_string(1, big))


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_mode_latches_the_refusal(enc_cls):
    enc, out = _buf_encoder(enc_cls, sticky=True)
    enc.write_string_bounded(1, "xxxxx", 4)  # latched, not raised
    enc.write_unsigned(2, 1)                  # skipped: the encoder is already failed
    assert isinstance(enc.error, SofaArgumentError)
    assert "exceeds maxlen" in str(enc.error)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_invalid_utf8_still_wins_over_the_bound(enc_cls):
    """A lone surrogate is refused as invalid UTF-8 whatever the bound says: there
    are no bytes to measure."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="UTF-8"):
        enc.write_string_bounded(1, "\ud800", 100)


# --- every other bounded writer -----------------------------------------------
#
# (name, call(enc, bound), bound that fits exactly, bound one below it, the plain
# writer's call). The value is the same in all calls; only the bound moves.

WRITERS = [
    ("unsigned", lambda e, b: e.write_unsigned_bounded(1, 255, b), 255, 254,
     lambda e: e.write_unsigned(1, 255)),
    ("unsigned zero", lambda e, b: e.write_unsigned_bounded(1, 0, b), 0, None,
     lambda e: e.write_unsigned(1, 0)),
    ("bytes", lambda e, b: e.write_bytes_bounded(1, b"1234", b), 4, 3,
     lambda e: e.write_bytes(1, b"1234")),
    ("bytes bytearray", lambda e, b: e.write_bytes_bounded(1, bytearray(b"12"), b), 2, 1,
     lambda e: e.write_bytes(1, bytearray(b"12"))),
    ("unsigned array cap",
     lambda e, b: e.write_unsigned_array_bounded(1, [1, 2, 3], b, None), 3, 2,
     lambda e: e.write_unsigned_array(1, [1, 2, 3])),
    ("signed array cap",
     lambda e, b: e.write_signed_array_bounded(1, [-1, 2], b, None, None), 2, 1,
     lambda e: e.write_signed_array(1, [-1, 2])),
    ("bool array cap",
     lambda e, b: e.write_bool_array_bounded(1, [True, False, True], b), 3, 2,
     lambda e: e.write_bool_array(1, [True, False, True])),
    ("float32 array cap",
     lambda e, b: e.write_float32_array_bounded(1, [1.5, 2.5], b), 2, 1,
     lambda e: e.write_float32_array(1, [1.5, 2.5])),
    ("float64 array cap",
     lambda e, b: e.write_float64_array_bounded(1, [1.5], b), 1, 0,
     lambda e: e.write_float64_array(1, [1.5])),
    ("empty array cap",
     lambda e, b: e.write_unsigned_array_bounded(1, [], b, None), 0, None,
     lambda e: e.write_unsigned_array(1, [])),
    ("tuple array cap",
     lambda e, b: e.write_signed_array_bounded(1, (5, 6, 7), b, None, None), 3, 2,
     lambda e: e.write_signed_array(1, (5, 6, 7))),
    ("unsigned array elem_max",
     lambda e, b: e.write_unsigned_array_bounded(1, [7, 300, 2], None, b), 300, 299,
     lambda e: e.write_unsigned_array(1, [7, 300, 2])),
    ("signed array elem_max",
     lambda e, b: e.write_signed_array_bounded(1, [-3, 127], None, None, b), 127, 126,
     lambda e: e.write_signed_array(1, [-3, 127])),
    ("signed array elem_min",
     lambda e, b: e.write_signed_array_bounded(1, [5, -128], None, b, None), -128, -127,
     lambda e: e.write_signed_array(1, [5, -128])),
]
_IDS = [w[0] for w in WRITERS]
_REFUSING = [w for w in WRITERS if w[3] is not None]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,plain", WRITERS, ids=_IDS)
def test_a_bound_that_fits_changes_no_byte(enc_cls, name, call, fits, under, plain):
    assert _bytes(enc_cls, lambda e: call(e, fits)) == _bytes(enc_cls, plain)
    assert _bytes(enc_cls, lambda e: call(e, None)) == _bytes(enc_cls, plain)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,plain", _REFUSING,
                         ids=[w[0] for w in _REFUSING])
def test_one_past_the_bound_is_refused(enc_cls, name, call, fits, under, plain):
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="exceeds|outside"):
        call(enc, under)
    if "elem" in name:
        return  # an element is refused mid-array: see test_an_element_*
    enc.flush()
    assert out.getvalue() == b"\x00\x07"


#: (value, min_value, max_value, refused?) for write_signed_bounded.
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
            enc.write_signed_bounded(1, value, lo, hi)
        enc.flush()
        assert out.getvalue() == b""
    else:
        assert _bytes(enc_cls, lambda e: e.write_signed_bounded(1, value, lo, hi)) == \
            _bytes(enc_cls, lambda e: e.write_signed(1, value))


#: (elements, elem_min, elem_max, index refused or None) for
#: write_signed_array_bounded -- an enum array bounded by its implied i8 width.
SIGNED_ELEMS = [
    ([-128, 0, 127], -128, 127, None),
    ([0, 128], -128, 127, 1),
    ([-129], -128, 127, 0),
    ([1, 2, 200, 3], -128, 127, 2),
    ([-40000], None, 32767, None),
    ([40000], -32768, None, None),
    ([-40000], -32768, None, 0),
]

#: (elements, elem_max, index refused or None) for write_unsigned_array_bounded --
#: an array<u8> and a bitfield array bounded by its implied u16 width.
UNSIGNED_ELEMS = [
    ([0, 255], 255, None),
    ([300], 255, 0),
    ([1, 256], 255, 1),
    ([65535, 65536], 65535, 1),
    ([1 << 63], None, None),
]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("elems,lo,hi,bad", SIGNED_ELEMS, ids=range(len(SIGNED_ELEMS)))
def test_an_element_outside_its_signed_width(enc_cls, elems, lo, hi, bad):
    if bad is None:
        assert _bytes(enc_cls, lambda e: e.write_signed_array_bounded(1, elems, 4, lo, hi)) \
            == _bytes(enc_cls, lambda e: e.write_signed_array(1, elems))
        return
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError,
                       match=f"signed array value {elems[bad]} outside {lo}..{hi}"):
        enc.write_signed_array_bounded(1, elems, 4, lo, hi)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("elems,hi,bad", UNSIGNED_ELEMS, ids=range(len(UNSIGNED_ELEMS)))
def test_an_element_over_its_unsigned_width(enc_cls, elems, hi, bad):
    if bad is None:
        assert _bytes(enc_cls, lambda e: e.write_unsigned_array_bounded(1, elems, None, hi)) \
            == _bytes(enc_cls, lambda e: e.write_unsigned_array(1, elems))
        return
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError,
                       match=f"unsigned array value {elems[bad]} exceeds elem_max {hi}"):
        enc.write_unsigned_array_bounded(1, elems, None, hi)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_element_past_64_bits_keeps_its_own_words(enc_cls):
    """The 64-bit range is checked first and worded as the plain writer words it."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="unsigned array value -1 out of range"):
        enc.write_unsigned_array_bounded(1, [-1], None, 255)
    with pytest.raises(SofaArgumentError, match="signed array value .* out of range"):
        enc.write_signed_array_bounded(1, [1 << 63], None, -128, 127)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_element_refusal_fails_the_message(enc_cls):
    """Mid-array, so the header is already out -- in sticky mode the message is
    failed and nothing after it is written."""
    enc, out = _buf_encoder(enc_cls, sticky=True)
    enc.write_unsigned_array_bounded(1, [1, 300], 4, 255)
    enc.write_unsigned(2, 1)  # skipped
    assert isinstance(enc.error, SofaArgumentError)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_64_bit_range_is_checked_before_the_bound(enc_cls):
    """A value no 64-bit field can hold reports that, not the declared bound."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_unsigned_bounded(1, -1, 255)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_signed_bounded(1, 1 << 63, -128, 127)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_index_object_is_bounded_by_its_value(enc_cls):
    """An IntEnum/IntFlag member is an integer by __index__; the bound applies to it."""

    class E(enum.IntEnum):
        BIG = 300

    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds"):
        enc.write_unsigned_bounded(1, E.BIG, 255)
    with pytest.raises(SofaArgumentError, match="outside"):
        enc.write_signed_bounded(1, E.BIG, -128, 127)
    with pytest.raises(SofaArgumentError, match="exceeds"):
        enc.write_unsigned_array_bounded(1, [E.BIG], None, 255)
    with pytest.raises(SofaArgumentError, match="outside"):
        enc.write_signed_array_bounded(1, [E.BIG], None, -128, 127)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_mode_latches_every_bound(enc_cls):
    for call in (lambda e: e.write_unsigned_bounded(1, 256, 255),
                 lambda e: e.write_signed_bounded(1, 128, -128, 127),
                 lambda e: e.write_bytes_bounded(1, b"12345", 4),
                 lambda e: e.write_unsigned_array_bounded(1, [1, 2], 1, None),
                 lambda e: e.write_signed_array_bounded(1, [1, 2], 1, None, None),
                 lambda e: e.write_bool_array_bounded(1, [True, True], 1),
                 lambda e: e.write_float32_array_bounded(1, [1.0, 2.0], 1),
                 lambda e: e.write_float64_array_bounded(1, [1.0, 2.0], 1)):
        enc, out = _buf_encoder(enc_cls, sticky=True)
        call(enc)
        enc.write_unsigned(2, 1)  # skipped: already failed
        assert isinstance(enc.error, SofaArgumentError)
        enc.flush()
        assert out.getvalue() == b""


# --- a bound outside the 64-bit domain, or no integer at all -------------------
#
# Generated code passes schema literals only, but the writers are public API. A
# bound is compared as a number, exactly as Python compares it: a bound no value
# can meet refuses every value (SofaArgumentError, latched in sticky mode), one
# past the 64-bit range refuses none, and a bound that is no number raises
# Python's own TypeError. Both engines, case for case -- never an OverflowError
# from a C conversion.

#: (name, call, refused?) -- every one of these bounds is outside the C range of
#: what it bounds, or not an int.
ODD_BOUNDS = [
    ("unsigned max -1", lambda e: e.write_unsigned_bounded(0, 5, -1), True),
    ("unsigned max 2**70", lambda e: e.write_unsigned_bounded(0, 5, 1 << 70), False),
    ("unsigned max float under", lambda e: e.write_unsigned_bounded(0, 5, 4.5), True),
    ("unsigned max float over", lambda e: e.write_unsigned_bounded(0, 5, 5.5), False),
    ("unsigned max True", lambda e: e.write_unsigned_bounded(0, 1, True), False),
    ("signed min -2**70", lambda e: e.write_signed_bounded(0, 5, -(1 << 70), 10), False),
    ("signed min 2**63", lambda e: e.write_signed_bounded(0, 5, 1 << 63, None), True),
    ("signed max 2**63", lambda e: e.write_signed_bounded(0, 5, None, 1 << 63), False),
    ("signed max -2**70", lambda e: e.write_signed_bounded(0, 5, None, -(1 << 70)), True),
    ("string maxlen float", lambda e: e.write_string_bounded(0, "ab", 1.5), True),
    ("string maxlen -1", lambda e: e.write_string_bounded(0, "", -1), True),
    ("bytes maxlen 2**64", lambda e: e.write_bytes_bounded(0, b"ab", 1 << 64), False),
    ("cap -1", lambda e: e.write_unsigned_array_bounded(0, [], -1, None), True),
    ("cap float", lambda e: e.write_float64_array_bounded(0, [1.0], 1.0), False),
    ("cap 2**64", lambda e: e.write_bool_array_bounded(0, [True], 1 << 64), False),
    ("elem_max -1", lambda e: e.write_unsigned_array_bounded(0, [0], None, -1), True),
    ("elem_max -1 empty", lambda e: e.write_unsigned_array_bounded(0, [], None, -1), False),
    ("elem_max 2**64", lambda e: e.write_unsigned_array_bounded(0, [(1 << 64) - 1], None, 1 << 64), False),
    ("elem_max float", lambda e: e.write_unsigned_array_bounded(0, [1, 2], None, 1.5), True),
    ("elem_min 2**63", lambda e: e.write_signed_array_bounded(0, [1], None, 1 << 63, None), True),
    ("elem_min -2**70", lambda e: e.write_signed_array_bounded(0, [-(1 << 63)], None, -(1 << 70), None), False),
    ("elem_max -2**70", lambda e: e.write_signed_array_bounded(0, [1], None, None, -(1 << 70)), True),
    ("elem_max float", lambda e: e.write_signed_array_bounded(0, [0, 1], None, None, 0.5), True),
    ("elem both float", lambda e: e.write_signed_array_bounded(0, [0], None, -0.5, 0.5), False),
]

#: Bounds that are no number: Python's own TypeError, from either engine.
NOT_NUMBERS = [
    ("unsigned max str", lambda e: e.write_unsigned_bounded(0, 5, "9")),
    ("signed min str", lambda e: e.write_signed_bounded(0, 5, "x", None)),
    ("signed max str", lambda e: e.write_signed_bounded(0, 5, None, "9")),
    ("string maxlen str", lambda e: e.write_string_bounded(0, "ab", "4")),
    ("bytes maxlen list", lambda e: e.write_bytes_bounded(0, b"ab", [4])),
    ("cap str", lambda e: e.write_bool_array_bounded(0, [True], "1")),
    ("elem_max str", lambda e: e.write_unsigned_array_bounded(0, [], None, "1")),
    ("elem_min str", lambda e: e.write_signed_array_bounded(0, [], None, "1", None)),
    ("elem_max object", lambda e: e.write_signed_array_bounded(0, [], None, None, object())),
]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,refused", ODD_BOUNDS, ids=[b[0] for b in ODD_BOUNDS])
def test_a_bound_is_compared_as_a_number(enc_cls, name, call, refused):
    enc, out = _buf_encoder(enc_cls)
    if not refused:
        call(enc)
        return
    with pytest.raises(SofaArgumentError, match="exceeds|outside"):
        call(enc)
    enc, _ = _buf_encoder(enc_cls, sticky=True)
    call(enc)  # latched, not raised
    assert isinstance(enc.error, SofaArgumentError)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call", NOT_NUMBERS, ids=[b[0] for b in NOT_NUMBERS])
def test_a_bound_that_is_no_number_is_a_type_error(enc_cls, name, call):
    enc, out = _buf_encoder(enc_cls)
    with pytest.raises(TypeError):
        call(enc)
    enc.flush()
    assert out.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_index_object_bound_compares_by_value(enc_cls):
    """An IntEnum bound is the number it is."""

    class Cap(enum.IntEnum):
        TWO = 2

    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds cap"):
        enc.write_unsigned_array_bounded(0, [1, 2, 3], Cap.TWO, None)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_plain_writers_take_no_bound(enc_cls):
    """The plain writers keep their signatures: a bound goes to the *_bounded twin."""
    enc, _ = _buf_encoder(enc_cls)
    for call in (lambda: enc.write_string(0, "ab", 1),
                 lambda: enc.write_unsigned(0, 5, 3),
                 lambda: enc.write_unsigned_array(0, [1], 1)):
        with pytest.raises(TypeError):
            call()


# --- parity --------------------------------------------------------------------


def _outcome(cls, call):
    enc, out = _buf_encoder(cls)
    try:
        call(enc)
        enc.flush()
        return ("ok", out.getvalue())
    except SofaArgumentError as exc:
        return ("refused", str(exc))
    except TypeError:
        return ("type error",)


def test_both_engines_word_every_refusal_alike():
    """Parity: the verdict, its text and the bytes agree call for call."""
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    calls = [lambda e, t=t, m=m: e.write_string_bounded(1, t, m) for t, m in FITS + OVER]
    calls += [lambda e, w=w: w[1](e, w[2]) for w in WRITERS]
    calls += [lambda e, w=w: w[1](e, w[3]) for w in _REFUSING]
    calls += [lambda e, v=v: e.write_signed_bounded(1, v[0], v[1], v[2]) for v in SIGNED]
    calls += [lambda e, v=v: e.write_signed_array_bounded(1, v[0], 4, v[1], v[2])
              for v in SIGNED_ELEMS]
    calls += [lambda e, v=v: e.write_unsigned_array_bounded(1, v[0], None, v[1])
              for v in UNSIGNED_ELEMS]
    calls += [b[1] for b in ODD_BOUNDS]
    calls += [b[1] for b in NOT_NUMBERS]
    for i, call in enumerate(calls):
        assert _outcome(PyEncoder, call) == _outcome(native.Encoder, call), i


def test_both_engines_have_the_same_bounded_writers():
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    def bounded(cls):
        return sorted(n for n in dir(cls) if n.startswith("write_") and n.endswith("_bounded"))

    assert bounded(PyEncoder) == bounded(native.Encoder)
    assert len(bounded(PyEncoder)) == 9


# --- the paths every writer shares ----------------------------------------------


class _Index:
    """An integer only by ``__index__`` (a NumPy scalar, say): not an ``int``."""

    def __init__(self, v: int) -> None:
        self.v = v

    def __index__(self) -> int:
        return self.v


class _OversizedBlob:
    """One byte past ``FIXLEN_MAX`` by its declared length; never materialised."""

    def __len__(self) -> int:
        from sofab.types import FIXLEN_MAX

        return FIXLEN_MAX + 1

    def __bytes__(self) -> bytes:  # pragma: no cover - the tripwire
        raise AssertionError("oversized payload was materialised")


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_index_value_is_bounded(enc_cls):
    assert _bytes(enc_cls, lambda e: e.write_unsigned_bounded(1, _Index(7), 255)) == \
        _bytes(enc_cls, lambda e: e.write_unsigned(1, 7))
    assert _bytes(enc_cls, lambda e: e.write_signed_bounded(1, _Index(-7), -8, 7)) == \
        _bytes(enc_cls, lambda e: e.write_signed(1, -7))
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds max_value"):
        enc.write_unsigned_bounded(1, _Index(256), 255)
    with pytest.raises(SofaArgumentError, match="must be an integer"):
        enc.write_signed_bounded(1, 1.0, -8, 7)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_fixlen_max_is_checked_before_the_bound(enc_cls):
    enc, out = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="FIXLEN_MAX"):
        enc.write_bytes_bounded(1, _OversizedBlob(), 4)
    enc.flush()
    assert out.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_latches_invalid_utf8_and_fixlen_max(enc_cls):
    for call in (lambda e: e.write_string_bounded(1, "\ud800", 8),
                 lambda e: e.write_bytes_bounded(1, _OversizedBlob(), 4)):
        enc, _ = _buf_encoder(enc_cls, sticky=True)
        call(enc)
        assert isinstance(enc.error, SofaArgumentError)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_a_failed_encoder_skips_every_bounded_writer(enc_cls):
    """Sticky mode: after the first failure every writer is a no-op, bounded ones too."""
    enc, out = _buf_encoder(enc_cls, sticky=True)
    enc.write_unsigned_bounded(0, 300, 255)  # the failure
    first = enc.error
    enc.write_unsigned_bounded(1, 1, 255)
    enc.write_signed_bounded(1, 1, -8, 7)
    enc.write_string_bounded(1, "a", 8)
    enc.write_bytes_bounded(1, b"a", 8)
    enc.write_unsigned_array_bounded(1, [1], 4, 255)
    enc.write_signed_array_bounded(1, [1], 4, -8, 7)
    enc.write_bool_array_bounded(1, [True], 4)
    enc.write_float32_array_bounded(1, [1.0], 4)
    enc.write_float64_array_bounded(1, [1.0], 4)
    assert enc.error is first
    enc.flush()
    assert out.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_bounded_writers_are_positional_only(enc_cls):
    """Positional-only on both engines: the native wrapper then parses no keywords,
    which is what keeps the bound argument cheap."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(TypeError):
        enc.write_string_bounded(0, "ab", maxlen=4)
    with pytest.raises(TypeError):
        enc.write_unsigned_array_bounded(0, [1], cap=2, elem_max=None)
