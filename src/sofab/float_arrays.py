"""The static helper layer: bit-pattern equality of float arrays.

A float array field is omitted on the wire iff its value equals its default
(MESSAGE_SPEC §2), and floats round-trip bit-for-bit (CORELIB_PLAN §4.6). The
comparison that decides the omission therefore has to be a comparison of **bit
patterns**, not IEEE ``==``: the array ``[-0.0, 1.5]`` is *not* the default
``[0.0, 1.5]``, and dropping it would lose the sign of the zero.

The helper is the same for every schema -- its schema dependence is only the two
sequences it is handed -- so it lives here, written once, and generated code
calls it with the field's list and a constant default list.

:func:`float_array_bits_equal` returns ``True`` iff the lengths are equal and,
at every index, the 64-bit IEEE-754 pattern of the held double is identical:

* ``+0.0`` and ``-0.0`` are different;
* a NaN equals another NaN only when the patterns match, payload included;
* there is no IEEE ``==`` anywhere, so ``nan`` is equal to itself.

Python holds every float, ``fp32`` fields included, as a double, so the pattern
compared is that of the held double for both element widths. (The ``fp32``
signaling-NaN raw-bytes path of CORELIB_PLAN §6.5 is separate and untouched.)
An ``int`` element is compared as the double it converts to, which is what the
encoder would write for it; one outside the double range is not a
float at all and raises ``struct.error``.

It works under both engines: it is plain Python and never touches the codec.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import copysign
from struct import pack


def float_array_bits_equal(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether ``a`` and ``b`` have the same length and identical double bit patterns."""
    if a is b:
        return True
    if len(a) != len(b):
        return False
    # Allocation-free walk with an early exit on the first differing element.
    # Two values that compare ``==`` share one double, except the zeros, whose
    # sign is checked; the ones that do not (including every NaN) fall back to
    # their packed bytes, which is the only place a bit pattern is built.
    for x, y in zip(a, b):
        if x is y:
            continue
        if x == y:
            if x == 0 and copysign(1.0, x) != copysign(1.0, y):
                return False
        elif pack("=d", x) != pack("=d", y):
            return False
    return True


# --- a declared default, compared many times ---------------------------------
#
# Generated code compares a float array field with the SAME declared default on
# every serialize and every isDefault. FloatArrayDefault is built once per field
# (a module-level constant in the generated module) and decides at construction
# how that comparison is made, so the per-call path does no analysis of the
# default: a default with no zero and no NaN is told apart by the plain list
# `==`, which runs at C speed and is already bit-exact for every other double; a
# default with zeros adds one sign read per zero position, and only after the
# lists compared equal; a default holding a NaN takes the element-wise path.
#
# The element rule, for every engine and every path: an element `o` matches the
# default element `d` iff `o == d` and, where `d` is a zero, `o` carries the
# same sign; where `d` is a NaN, iff `o` is a float with the identical bit
# pattern. An `int` therefore matches the double it equals exactly (`0` is
# `+0.0`); anything that is not equal, including an `int` outside the double
# range, does not match. Nothing raises.


def _elem_matches(o: object, d: float) -> bool:
    """Whether one held element ``o`` matches the default element ``d``."""
    if d != d:
        return isinstance(o, float) and pack("=d", o) == pack("=d", d)
    try:
        if not (o == d):
            return False
        return d != 0 or copysign(1.0, o) == copysign(1.0, d)  # type: ignore[arg-type,unused-ignore]
    except (TypeError, ValueError, OverflowError):
        return False


def _seq_matches(a: Sequence[float], v: list[float]) -> bool:
    """The element-wise path: any sequence, any default."""
    if len(a) != len(v):
        return False
    for o, d in zip(a, v):
        if not _elem_matches(o, d):
            return False
    return True


def _matcher(v: list[float]) -> Callable[[Sequence[float]], bool]:
    """The compare a default of these elements is told apart with."""
    if any(d != d for d in v):

        def matches_nan(a: Sequence[float]) -> bool:
            return _seq_matches(a, v)

        return matches_nan
    zeros = tuple((i, copysign(1.0, d)) for i, d in enumerate(v) if d == 0)
    if not zeros:

        def matches(a: Sequence[float]) -> bool:
            if a.__class__ is list:
                return a == v
            return _seq_matches(a, v)

        return matches
    if len(zeros) == 1:
        (i0, s0), = zeros

        def matches_zero(a: Sequence[float]) -> bool:
            if a.__class__ is list:
                try:
                    return a == v and copysign(1.0, a[i0]) == s0
                except (TypeError, ValueError):
                    pass  # equal by its own say-so, but not a number: the rule decides
            return _seq_matches(a, v)

        return matches_zero

    def matches_zeros(a: Sequence[float]) -> bool:
        if a.__class__ is not list:
            return _seq_matches(a, v)
        if a != v:
            return False
        try:
            for i, s in zeros:
                if copysign(1.0, a[i]) != s:
                    return False
        except (TypeError, ValueError):
            return _seq_matches(a, v)
        return True

    return matches_zeros


class FloatArrayDefault:
    """A float array field's declared default, compared bit for bit.

    Built once per field; :meth:`matches` is then the omission test generated
    code runs on every serialize. ``matches(a)`` is true iff ``a`` has the
    default's length and every element matches (see the element rule above):
    ``+0.0`` and ``-0.0`` differ, a NaN matches only the identical NaN, an
    ``int`` matches the double it equals. Any sequence is accepted; a ``list``
    takes the fast path. :attr:`values` is the default, as a tuple of floats.
    """

    __slots__ = ("values", "matches")

    values: tuple[float, ...]
    matches: Callable[[Sequence[float]], bool]

    def __init__(self, values: Iterable[float]) -> None:
        v = [float(x) for x in values]
        self.values = tuple(v)
        self.matches = _matcher(v)

    def __repr__(self) -> str:
        return f"FloatArrayDefault({list(self.values)!r})"
