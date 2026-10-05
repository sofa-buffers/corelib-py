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
encoder would write for it; one outside the double range equals no double.

It works under both engines: it is plain Python and never touches the codec.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import copysign
from struct import pack


def float_array_bits_equal(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether ``a`` and ``b`` have the same length and identical double bit patterns."""
    if a is b:
        return True
    if len(a) != len(b):
        return False
    # Allocation-free walk with an early exit on the first differing element.
    for x, y in zip(a, b):
        if x is y:
            continue
        if x == y:
            # Equal values share one double, except the zeros, whose sign is read.
            if x == 0 and copysign(1.0, x) != copysign(1.0, y):
                return False
        elif x == x or y == y:
            # Unequal and at least one is not a NaN: different doubles. This is
            # the answer for every ordinary difference, before any bytes are built.
            return False
        elif pack("=d", x) != pack("=d", y):
            # Both are NaN: the payload and sign decide, and only bytes show them.
            return False
    return True
