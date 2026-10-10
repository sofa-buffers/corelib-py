"""What each writer takes as a value, on both engines (CORELIB_PLAN §6.3).

A generated ``serialize()`` hands every value to its writer exactly as the
message holds it, with no ``int(...)`` or ``bytes(...)`` around it, so the
writer's own contract is the only one there is:

* An integer writer takes what Python takes as an integer -- anything with
  ``__index__``: ``int``, ``bool``, an ``IntEnum`` or ``IntFlag`` member -- and
  refuses a ``float`` rather than truncating it. A held subclass encodes to the
  same bytes as the plain int it is, and the encoding works on that plain int:
  an ``IntFlag`` whose ``&`` / ``|`` / ``>>`` build new members must not run
  that Python code once per varint byte.
* A blob writer takes a byte string -- ``bytes``, ``bytearray`` or a
  one-dimensional ``memoryview`` of format ``'B'``, the values that compare
  equal to a ``bytes`` of the same content -- and measures it in bytes.
  Anything else is refused, never converted: a list or an int is not a payload,
  and a buffer of wider items has a ``len()`` that counts items, which would
  let an over-``maxlen`` blob through.
"""

from __future__ import annotations

import array
import enum

import pytest
from vectors import ENCODER_ENGINES as ENGINES

from sofab.types import SofaArgumentError


class Color(enum.IntEnum):
    RED = 1
    BIG = 300


class Flags(enum.IntFlag):
    E = 0x01
    X = 0x02
    H = 0x80


class NoArith(int):
    """An int whose arithmetic is a tripwire: the writer must encode the plain
    int it is, not do its varint math on the subclass."""

    def _trip(self, *_: object) -> int:
        raise AssertionError("writer did arithmetic on the int subclass")

    __and__ = __rand__ = __or__ = __ror__ = __rshift__ = __lshift__ = _trip
    __xor__ = __rxor__ = __lt__ = __gt__ = __le__ = __ge__ = _trip


class Indexable:
    """Not an int, but losslessly one (``__index__``), like a NumPy integer."""

    def __init__(self, v: int) -> None:
        self.v = v

    def __index__(self) -> int:
        return self.v


def _bytes(engine, write) -> bytes:
    enc = engine()
    write(enc)
    return enc.getvalue()


# --- integers ---------------------------------------------------------------

UNSIGNED_SCALARS = ["write_u8", "write_u16", "write_u32", "write_unsigned"]
SIGNED_SCALARS = ["write_i8", "write_i16", "write_i32", "write_signed"]
UNSIGNED_ARRAYS = ["write_u8_array", "write_u16_array", "write_u32_array",
                   "write_u64_array"]
SIGNED_ARRAYS = ["write_i8_array", "write_i16_array", "write_i32_array",
                 "write_i64_array"]

#: (held value, the plain int it is). 0x83 needs two varint bytes.
HELD = [
    (True, 1),
    (False, 0),
    (Color.RED, 1),
    (Flags.E | Flags.X | Flags.H, 0x83),
    (NoArith(0x83), 0x83),
    (NoArith(5), 5),
    (Indexable(0x83), 0x83),
]
HELD_IDS = ["True", "False", "IntEnum", "IntFlag", "NoArith-2byte", "NoArith", "index"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("held,plain", HELD, ids=HELD_IDS)
@pytest.mark.parametrize("name", UNSIGNED_SCALARS + SIGNED_SCALARS)
def test_scalar_held_int_encodes_as_the_plain_int(engine, name, held, plain):
    if name == "write_i8" and plain > 127:
        pytest.skip("outside i8; the range check is test_held_subclass_still_range_checked")
    got = _bytes(engine, lambda e: getattr(e, name)(3, held))
    assert got == _bytes(engine, lambda e: getattr(e, name)(3, plain))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", UNSIGNED_ARRAYS + SIGNED_ARRAYS)
def test_array_held_ints_encode_as_the_plain_ints(engine, name):
    pairs = [(v, p) for v, p in HELD if name != "write_i8_array" or p <= 127]
    held = [v for v, _ in pairs]
    plain = [p for _, p in pairs]
    got = _bytes(engine, lambda e: getattr(e, name)(3, held, -1))
    assert got == _bytes(engine, lambda e: getattr(e, name)(3, plain, -1))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["write_unsigned_array", "write_signed_array"])
def test_generic_array_held_ints(engine, name):
    held = [v for v, _ in HELD] + [Color.BIG]
    plain = [p for _, p in HELD] + [300]
    assert (_bytes(engine, lambda e: getattr(e, name)(3, held))
            == _bytes(engine, lambda e: getattr(e, name)(3, plain)))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", UNSIGNED_SCALARS + SIGNED_SCALARS)
@pytest.mark.parametrize("bad", [3.0, 255.9, "3", None], ids=["3.0", "255.9", "str", "None"])
def test_scalar_non_integer_refused(engine, name, bad):
    enc = engine()
    with pytest.raises(SofaArgumentError, match="must be an integer"):
        getattr(enc, name)(3, bad)
    assert enc.getvalue() == b""


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", UNSIGNED_ARRAYS + SIGNED_ARRAYS)
def test_array_non_integer_element_refused(engine, name):
    with pytest.raises(SofaArgumentError, match="must be an integer"):
        getattr(engine(), name)(3, [1, Flags.E, 2.5], -1)


@pytest.mark.parametrize("engine", ENGINES)
def test_held_subclass_still_range_checked(engine):
    with pytest.raises(SofaArgumentError):
        engine().write_u8(1, Color.BIG)
    with pytest.raises(SofaArgumentError):
        engine().write_u8_array(1, [Color.RED, Color.BIG], -1)
    with pytest.raises(SofaArgumentError):
        engine().write_i8_array(1, [NoArith(-129)], -1)


# --- blobs ------------------------------------------------------------------


class SubBytes(bytes):
    pass


class SubArray(bytearray):
    pass


ACCEPTED = [
    b"\x01\x02\x03",
    bytearray(b"\x01\x02\x03"),
    memoryview(b"\x01\x02\x03"),
    memoryview(b"\x01\x00\x02\x00\x03")[::2],          # non-contiguous, format B
    memoryview(bytearray(b"\x01\x02\x03")),
    SubBytes(b"\x01\x02\x03"),
    SubArray(b"\x01\x02\x03"),
]
ACCEPTED_IDS = ["bytes", "bytearray", "memoryview", "strided", "mv-bytearray",
                "bytes-sub", "bytearray-sub"]

REFUSED = [
    3,
    [1, 2, 3],
    (1, 2, 3),
    [],
    array.array("B", [1, 2, 3]),
    array.array("H", [1, 2, 3]),
    memoryview(b"abcdef").cast("H"),
    memoryview(b"\x01\x02\x03").cast("b"),
    memoryview(b"\x01\x02\x03").cast("c"),
    memoryview(b"\x01\x02\x03\x04").cast("B", (2, 2)),
    "abc",
    None,
]
REFUSED_IDS = ["int", "list", "tuple", "empty-list", "array-B", "array-H",
               "memoryview-H", "memoryview-b", "memoryview-c", "memoryview-2d",
               "str", "None"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("data", ACCEPTED, ids=ACCEPTED_IDS)
def test_blob_byte_strings_accepted(engine, data):
    expected = _bytes(engine, lambda e: e.write_bytes(1, b"\x01\x02\x03"))
    assert _bytes(engine, lambda e: e.write_bytes(1, data)) == expected
    assert _bytes(engine, lambda e: e.write_bytes_bounded(1, data, 3)) == expected
    with pytest.raises(SofaArgumentError, match="blob of 3 bytes exceeds maxlen 2"):
        engine().write_bytes_bounded(1, data, 2)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("data", REFUSED, ids=REFUSED_IDS)
@pytest.mark.parametrize("bounded", [False, True], ids=["write_bytes", "bounded"])
def test_blob_non_byte_strings_refused(engine, data, bounded):
    enc = engine()
    enc.write_unsigned(0, 7)
    with pytest.raises(SofaArgumentError, match="blob must be bytes"):
        if bounded:
            enc.write_bytes_bounded(1, data, 1 << 20)
        else:
            enc.write_bytes(1, data)
    assert enc.getvalue() == b"\x00\x07"


@pytest.mark.parametrize("engine", ENGINES)
def test_blob_refusal_names_the_format(engine):
    with pytest.raises(SofaArgumentError, match="memoryview of format 'H'"):
        engine().write_bytes(1, memoryview(b"abcdef").cast("H"))
    with pytest.raises(SofaArgumentError, match="not list$"):
        engine().write_bytes(1, [1])


@pytest.mark.parametrize("engine", ENGINES)
def test_blob_refusal_is_sticky(engine):
    enc = engine(sticky=True)
    enc.write_bytes(1, array.array("H", [1, 2, 3]))
    assert isinstance(enc.error, SofaArgumentError)
    enc.write_bytes(2, b"x")   # skipped, per sticky mode
    assert enc.getvalue() == b""
    enc = engine(sticky=True)
    enc.write_bytes_bounded(1, [1, 2], 4)
    assert isinstance(enc.error, SofaArgumentError)
    assert enc.getvalue() == b""
