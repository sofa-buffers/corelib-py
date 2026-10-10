"""A caller's declared bound, handed to the writer that already measures the value.

Generated code refuses a value past its schema bound at encode. A schema's maxlen
or count is the caller's number and rides the call; a declared integer width is a
type and is in the writer's name:

* ``write_string_bounded(id, text, maxlen)`` -- a schema's ``maxlen`` counts UTF-8
  bytes, and a ``str`` does not know that length: only the writer produces it;
* ``write_bytes_bounded(id, data, maxlen)`` -- the blob length;
* ``write_{u8,u16,u32,i8,i16,i32}(id, value)`` -- the declared width of a narrower
  integer, enum or bitfield, which Python's unbounded ``int`` does not carry;
* ``write_{u8..u64,i8..i64}_array(id, values, cap)`` -- the declared capacity
  against the element count (negative: none), and the element's width against
  each element, in the range check every element already passes;
* ``write_{bool,float32,float64}_array_bounded(id, values, cap)``.

A value past a bound is :class:`SofaArgumentError` (CORELIB_PLAN §6.3). A scalar,
string, blob or array count is refused before any byte of the field is written; an
element is refused where the 64-bit range check refuses one, mid-array, after the
header and the elements before it -- the message is then failed and must be
discarded. The library holds no bound: a negative ``cap`` checks no count (``None``
too, on a writer that is not width-typed), and the bytes are exactly those of the
plain writer. A bound is compared as a number, as Python compares it (see
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


# --- the other bounded writers: a schema maxlen or count as an argument --------
#
# (name, call(enc, bound), bound that fits exactly, bound one below it, the bound
# that checks nothing, the plain writer's call). The value is the same in all
# calls; only the bound moves.

WRITERS = [
    ("bytes", lambda e, b: e.write_bytes_bounded(1, b"1234", b), 4, 3, None,
     lambda e: e.write_bytes(1, b"1234")),
    ("bytes bytearray", lambda e, b: e.write_bytes_bounded(1, bytearray(b"12"), b), 2, 1,
     None, lambda e: e.write_bytes(1, bytearray(b"12"))),
    ("bool array cap",
     lambda e, b: e.write_bool_array_bounded(1, [True, False, True], b), 3, 2, None,
     lambda e: e.write_bool_array(1, [True, False, True])),
    ("float32 array cap",
     lambda e, b: e.write_float32_array_bounded(1, [1.5, 2.5], b), 2, 1, None,
     lambda e: e.write_float32_array(1, [1.5, 2.5])),
    ("float64 array cap",
     lambda e, b: e.write_float64_array_bounded(1, [1.5], b), 1, 0, None,
     lambda e: e.write_float64_array(1, [1.5])),
    ("u8 array cap", lambda e, b: e.write_u8_array(1, [1, 2, 3], b), 3, 2, -1,
     lambda e: e.write_unsigned_array(1, [1, 2, 3])),
    ("u64 array cap", lambda e, b: e.write_u64_array(1, [1 << 63], b), 1, 0, -1,
     lambda e: e.write_unsigned_array(1, [1 << 63])),
    ("i16 array cap", lambda e, b: e.write_i16_array(1, [-1, 2], b), 2, 1, -1,
     lambda e: e.write_signed_array(1, [-1, 2])),
    ("i64 array cap", lambda e, b: e.write_i64_array(1, [-(1 << 63)], b), 1, 0, -1,
     lambda e: e.write_signed_array(1, [-(1 << 63)])),
    ("empty array cap", lambda e, b: e.write_u32_array(1, [], b), 0, None, -1,
     lambda e: e.write_unsigned_array(1, [])),
    ("tuple array cap", lambda e, b: e.write_i32_array(1, (5, 6, 7), b), 3, 2, -1,
     lambda e: e.write_signed_array(1, (5, 6, 7))),
]
_IDS = [w[0] for w in WRITERS]
_REFUSING = [w for w in WRITERS if w[3] is not None]


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,none,plain", WRITERS, ids=_IDS)
def test_a_bound_that_fits_changes_no_byte(enc_cls, name, call, fits, under, none, plain):
    assert _bytes(enc_cls, lambda e: call(e, fits)) == _bytes(enc_cls, plain)
    assert _bytes(enc_cls, lambda e: call(e, none)) == _bytes(enc_cls, plain)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("name,call,fits,under,none,plain", _REFUSING,
                         ids=[w[0] for w in _REFUSING])
def test_one_past_the_bound_is_refused(enc_cls, name, call, fits, under, none, plain):
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="exceeds"):
        call(enc, under)
    enc.flush()
    assert out.getvalue() == b"\x00\x07"  # nothing of the refused field


# --- the width-typed writers: a declared width in the writer's name -------------

#: name -> (lo, hi) of every width-typed writer, scalar and array alike.
WIDTHS = {
    "u8": (0, 255), "u16": (0, 65535), "u32": (0, (1 << 32) - 1),
    "i8": (-128, 127), "i16": (-32768, 32767), "i32": (-(1 << 31), (1 << 31) - 1),
}
ARRAY_WIDTHS = dict(WIDTHS, u64=(0, (1 << 64) - 1), i64=(-(1 << 63), (1 << 63) - 1))

#: (writer, value, refused?): every edge of every scalar width, both sides.
SCALARS = []
for _w, (_lo, _hi) in WIDTHS.items():
    SCALARS += [(_w, _lo, False), (_w, _hi, False), (_w, _hi + 1, True), (_w, 0, False)]
    if _lo < 0:
        SCALARS += [(_w, _lo - 1, True)]


def _plain_scalar(width):
    return "write_signed" if width.startswith("i") else "write_unsigned"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("width,value,refused", SCALARS,
                         ids=[f"{w}={v}" for w, v, _ in SCALARS])
def test_a_scalar_within_its_width(enc_cls, width, value, refused):
    call = lambda e: getattr(e, f"write_{width}")(1, value)  # noqa: E731
    if not refused:
        assert _bytes(enc_cls, call) == \
            _bytes(enc_cls, lambda e: getattr(e, _plain_scalar(width))(1, value))
        return
    enc, out = _buf_encoder(enc_cls)
    enc.write_unsigned(0, 7)
    word = "outside" if width.startswith("i") else "exceeds"
    with pytest.raises(SofaArgumentError, match=f"value {value} {word} {width}$"):
        call(enc)
    enc.flush()
    assert out.getvalue() == b"\x00\x07"


#: (writer, elements, index refused or None): every array width at its edges.
ELEMS = []
for _w, (_lo, _hi) in ARRAY_WIDTHS.items():
    ELEMS += [(_w, [_lo, 0, _hi], None)]
    if _w not in ("u64", "i64"):
        ELEMS += [(_w, [0, _hi + 1], 1)]
        if _lo < 0:
            ELEMS += [(_w, [_lo - 1, 0], 0)]


def _plain_array(width):
    return "write_signed_array" if width.startswith("i") else "write_unsigned_array"


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("width,elems,bad", ELEMS, ids=[f"{w}-{i}" for i, (w, _, _) in enumerate(ELEMS)])
def test_an_array_element_within_its_width(enc_cls, width, elems, bad):
    call = lambda e: getattr(e, f"write_{width}_array")(1, elems, -1)  # noqa: E731
    if bad is None:
        assert _bytes(enc_cls, call) == \
            _bytes(enc_cls, lambda e: getattr(e, _plain_array(width))(1, elems))
        return
    enc, _ = _buf_encoder(enc_cls)
    word = "outside" if width.startswith("i") else "exceeds"
    with pytest.raises(SofaArgumentError, match=f"array value {elems[bad]} {word} {width}$"):
        call(enc)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_no_cap_still_checks_the_element_width(enc_cls):
    """A negative cap bounds no count, but an unbounded array<u8> is still u8."""
    assert _bytes(enc_cls, lambda e: e.write_u8_array(1, list(range(256)) * 4, -1)) == \
        _bytes(enc_cls, lambda e: e.write_unsigned_array(1, list(range(256)) * 4))
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds u8"):
        enc.write_u8_array(1, [256], -1)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_64_bit_range_is_checked_before_the_width(enc_cls):
    """A value no 64-bit field can hold reports that, not the declared width."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="unsigned value -1 out of range"):
        enc.write_u8(1, -1)
    with pytest.raises(SofaArgumentError, match="signed value .* out of range"):
        enc.write_i8(1, 1 << 63)
    with pytest.raises(SofaArgumentError, match="unsigned array value -1 out of range"):
        enc.write_u8_array(1, [-1], -1)
    with pytest.raises(SofaArgumentError, match="signed array value .* out of range"):
        enc.write_i8_array(1, [1 << 63], -1)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_u64_array(1, [1 << 64], -1)
    with pytest.raises(SofaArgumentError, match="out of range"):
        enc.write_i64_array(1, [1 << 63], -1)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_index_object_is_held_to_its_width(enc_cls):
    """An IntEnum/IntFlag member is an integer by __index__; the width applies to it."""

    class E(enum.IntEnum):
        BIG = 300

    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(SofaArgumentError, match="exceeds u8"):
        enc.write_u8(1, E.BIG)
    with pytest.raises(SofaArgumentError, match="outside i8"):
        enc.write_i8(1, E.BIG)
    with pytest.raises(SofaArgumentError, match="exceeds u8"):
        enc.write_u8_array(1, [E.BIG], -1)
    with pytest.raises(SofaArgumentError, match="outside i8"):
        enc.write_i8_array(1, [E.BIG], -1)
    assert _bytes(enc_cls, lambda e: e.write_u16(1, _Index(7))) == \
        _bytes(enc_cls, lambda e: e.write_unsigned(1, 7))
    with pytest.raises(SofaArgumentError, match="must be an integer"):
        enc.write_i16(1, 1.0)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_an_element_refusal_fails_the_message(enc_cls):
    """Mid-array, so the header is already out -- in sticky mode the message is
    failed and nothing after it is written."""
    enc, out = _buf_encoder(enc_cls, sticky=True)
    enc.write_u8_array(1, [1, 300], 4)
    enc.write_unsigned(2, 1)  # skipped
    assert isinstance(enc.error, SofaArgumentError)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
@pytest.mark.parametrize("sticky", [False, True])
def test_an_element_refusal_leaves_the_field_half_written(enc_cls, sticky):
    """The documented contract, pinned: an element is refused where the loop
    reaches it, so the header (which states every element) and the elements
    before it are already in the buffer. The output is no valid message and
    must be discarded -- nothing rewinds it."""
    enc = enc_cls(sticky=sticky)
    enc.write_u8(1, 5)
    before = enc.bytes_used()
    if sticky:
        enc.write_u8_array(2, [1, 2, 300, 4], -1)
        assert isinstance(enc.error, SofaArgumentError)
    else:
        with pytest.raises(SofaArgumentError, match="exceeds u8"):
            enc.write_u8_array(2, [1, 2, 300, 4], -1)
    # 08 05 = field 1; 13 04 01 02 = field 2's header, count 4, two elements.
    assert enc.bytes_used() == before + 4
    assert bytes(enc.getvalue()) == bytes.fromhex("080513040102")


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_sticky_mode_latches_every_bound(enc_cls):
    for call in (lambda e: e.write_u8(1, 256),
                 lambda e: e.write_u16(1, 1 << 16),
                 lambda e: e.write_u32(1, 1 << 32),
                 lambda e: e.write_i8(1, 128),
                 lambda e: e.write_i16(1, -32769),
                 lambda e: e.write_i32(1, 1 << 31),
                 lambda e: e.write_bytes_bounded(1, b"12345", 4),
                 lambda e: e.write_u64_array(1, [1, 2], 1),
                 lambda e: e.write_i64_array(1, [1, 2], 1),
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
# maxlen or count given as an object is compared as a number, exactly as Python
# compares it: a bound no value can meet refuses every value (SofaArgumentError,
# latched in sticky mode), one past the 64-bit range refuses none, and a bound
# that is no number raises Python's own TypeError. An array writer's negative
# ``cap`` checks no count, on every array writer alike. A width-typed array writer's
# ``cap`` is an integer: not one is TypeError, one outside a C ssize_t is
# OverflowError. Both engines, case for case.

#: (name, call, refused?)
ODD_BOUNDS = [
    ("string maxlen float", lambda e: e.write_string_bounded(0, "ab", 1.5), True),
    ("string maxlen -1", lambda e: e.write_string_bounded(0, "", -1), True),
    ("bytes maxlen 2**64", lambda e: e.write_bytes_bounded(0, b"ab", 1 << 64), False),
    ("cap float", lambda e: e.write_float64_array_bounded(0, [1.0], 1.0), False),
    ("cap 2**64", lambda e: e.write_bool_array_bounded(0, [True], 1 << 64), False),
    ("cap -1 bool", lambda e: e.write_bool_array_bounded(0, [True, False], -1), False),
    ("cap -1 fp32", lambda e: e.write_float32_array_bounded(0, [1.0, 2.0], -1), False),
    ("cap -2**70 fp64", lambda e: e.write_float64_array_bounded(0, [1.0], -(1 << 70)), False),
    ("cap -0.5 bool", lambda e: e.write_bool_array_bounded(0, [True], -0.5), False),
    ("typed cap -1", lambda e: e.write_u8_array(0, [1, 2], -1), False),
    ("typed cap -2**63", lambda e: e.write_i8_array(0, [1], -(1 << 63)), False),
    ("typed cap True", lambda e: e.write_u16_array(0, [1, 2], True), True),
]

#: Bounds that are no number: Python's own TypeError, from either engine.
NOT_NUMBERS = [
    ("string maxlen str", lambda e: e.write_string_bounded(0, "ab", "4")),
    ("bytes maxlen list", lambda e: e.write_bytes_bounded(0, b"ab", [4])),
    ("cap str", lambda e: e.write_bool_array_bounded(0, [True], "1")),
    ("typed cap str", lambda e: e.write_u8_array(0, [], "1")),
    ("typed cap float", lambda e: e.write_i32_array(0, [], 1.0)),
    ("typed cap None", lambda e: e.write_u64_array(0, [], None)),
]

#: A typed cap outside a C ssize_t: OverflowError, from either engine.
OVERFLOWS = [
    ("typed cap 2**63", lambda e: e.write_u8_array(0, [], 1 << 63)),
    ("typed cap -2**70", lambda e: e.write_i64_array(0, [], -(1 << 70))),
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
@pytest.mark.parametrize("name,call", OVERFLOWS, ids=[b[0] for b in OVERFLOWS])
def test_a_typed_cap_outside_ssize_t_overflows(enc_cls, name, call):
    enc, out = _buf_encoder(enc_cls)
    with pytest.raises(OverflowError):
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
        enc.write_u32_array(0, [1, 2, 3], Cap.TWO)
    with pytest.raises(SofaArgumentError, match="exceeds cap"):
        enc.write_bool_array_bounded(0, [1, 2, 3], Cap.TWO)


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_plain_writers_take_no_bound(enc_cls):
    """The plain writers keep their signatures: a bound goes to the bounded or
    width-typed writer."""
    enc, _ = _buf_encoder(enc_cls)
    for call in (lambda: enc.write_string(0, "ab", 1),
                 lambda: enc.write_unsigned(0, 5, 3),
                 lambda: enc.write_unsigned_array(0, [1], 1),
                 lambda: enc.write_u8(0, 5, 255)):
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
    except OverflowError:
        return ("overflow",)


def test_both_engines_word_every_refusal_alike():
    """Parity: the verdict, its text and the bytes agree call for call."""
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    calls = [lambda e, t=t, m=m: e.write_string_bounded(1, t, m) for t, m in FITS + OVER]
    calls += [lambda e, w=w: w[1](e, w[2]) for w in WRITERS]
    calls += [lambda e, w=w: w[1](e, w[3]) for w in _REFUSING]
    calls += [lambda e, v=v: getattr(e, f"write_{v[0]}")(1, v[1]) for v in SCALARS]
    calls += [lambda e, v=v: getattr(e, f"write_{v[0]}_array")(1, v[1], 4) for v in ELEMS]
    calls += [lambda e, v=v: getattr(e, f"write_{v[0]}_array")(1, v[1], 1) for v in ELEMS]
    calls += [b[1] for b in ODD_BOUNDS]
    calls += [b[1] for b in NOT_NUMBERS]
    calls += [b[1] for b in OVERFLOWS]
    for i, call in enumerate(calls):
        assert _outcome(PyEncoder, call) == _outcome(native.Encoder, call), i


def test_both_engines_have_the_same_bounded_writers():
    native = pytest.importorskip("sofab._speedups", reason="native extension not built")
    from sofab.encoder import Encoder as PyEncoder

    def bounded(cls):
        return sorted(n for n in dir(cls) if n.startswith("write_") and n.endswith("_bounded"))

    def typed(cls):
        return sorted(n for n in dir(cls)
                      if n.startswith(("write_u", "write_i")) and n[7:8].isdigit())

    assert bounded(PyEncoder) == bounded(native.Encoder)
    assert bounded(PyEncoder) == ["write_bool_array_bounded", "write_bytes_bounded",
                                  "write_float32_array_bounded",
                                  "write_float64_array_bounded", "write_string_bounded"]
    assert typed(PyEncoder) == typed(native.Encoder)
    assert len(typed(PyEncoder)) == 14


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
    enc.write_u8(0, 300)  # the failure
    first = enc.error
    for name in ("u8", "u16", "u32", "i8", "i16", "i32"):
        getattr(enc, f"write_{name}")(1, 1)
    for name in ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"):
        getattr(enc, f"write_{name}_array")(1, [1], 4)
    enc.write_string_bounded(1, "a", 8)
    enc.write_bytes_bounded(1, b"a", 8)
    enc.write_bool_array_bounded(1, [True], 4)
    enc.write_float32_array_bounded(1, [1.0], 4)
    enc.write_float64_array_bounded(1, [1.0], 4)
    assert enc.error is first
    enc.flush()
    assert out.getvalue() == b""


@pytest.mark.parametrize("enc_cls", ENCODER_ENGINES)
def test_the_bounded_writers_are_positional_only(enc_cls):
    """Positional-only on both engines: the native wrapper then parses no keywords,
    which is what keeps the call cheap."""
    enc, _ = _buf_encoder(enc_cls)
    with pytest.raises(TypeError):
        enc.write_string_bounded(0, "ab", maxlen=4)
    with pytest.raises(TypeError):
        enc.write_u8_array(0, [1], cap=2)
    with pytest.raises(TypeError):
        enc.write_i16(0, value=1)
