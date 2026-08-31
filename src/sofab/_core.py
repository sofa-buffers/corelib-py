"""Shared fixlen (IEEE-754) pack/unpack helpers — part of the hot path.

Little-endian via ``struct`` with ``<``; this is correct on big-endian hosts
too (struct handles the byte order), matching the explicit LE handling in the
other core libs.

fp32 values are carried through a Python ``float`` (a C ``double``). A *hardware*
fp32->fp64 widening (or fp64->fp32 narrowing) quiets a signaling NaN — it sets
the mantissa's is-quiet bit — which MESSAGE_SPEC / CORELIB_PLAN §4.6 forbids:
every float payload, NaN included, must round-trip bit-for-bit with no
normalization. So for a NaN we do the width conversion by hand on the raw bits,
preserving the sign, the payload, and the signaling bit. Non-NaN values take
the plain ``struct`` path (their conversion is exact and never quiets).
"""

from __future__ import annotations

import struct
from array import array as _array
from codecs import utf_8_decode as _utf8_decode
from typing import Any

_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

# fp32 field masks.
_F32_EXP = 0x7F800000  # all exponent bits (NaN/inf when set)
_F32_MANT = 0x007FFFFF  # 23-bit mantissa (nonzero => NaN, zero => inf)


def _unpack_f32_bits(bits: int) -> float:
    """Widen a raw little-endian fp32 bit pattern to a Python float.

    NaN is widened bit-for-bit (see module docstring); every other value goes
    through ``struct`` where the fp32->fp64 conversion is exact.
    """
    if (bits & _F32_EXP) == _F32_EXP and (bits & _F32_MANT):
        # NaN: build the fp64 bit pattern directly. The 23-bit fp32 mantissa
        # (top bit = is-quiet) maps to the top 23 bits of the 52-bit fp64
        # mantissa (<< 29), so the signaling bit and payload survive.
        dbits = ((bits >> 31) << 63) | (0x7FF << 52) | ((bits & _F32_MANT) << 29)
        wide: float = _F64.unpack(_U64.pack(dbits))[0]
        return wide
    value: float = _F32.unpack(_U32.pack(bits))[0]
    return value


def _pack_f32_bits(value: float) -> int:
    """Narrow a Python float to a raw little-endian fp32 bit pattern.

    NaN is narrowed bit-for-bit (inverse of :func:`_unpack_f32_bits`); every
    other value goes through ``struct``.

    A magnitude too large for fp32 **overflows to ±inf**, which is the IEEE-754
    fp64->fp32 narrowing every native-``fp32`` corelib performs (a C ``(float)``
    cast, Rust ``as f32``, Java ``(float)``) and what the native accelerator's
    ``_pack_f32`` does. ``struct`` reports that overflow as ``OverflowError``
    instead of producing the bytes, and it does so *exactly* at the rounding
    boundary — a value between ``FLT_MAX`` and the tie point still packs, as
    ``FLT_MAX`` — so turning the exception into the ±inf pattern reproduces the
    native narrowing bit-for-bit. Letting it escape instead would both break
    that engine parity and leave the §6.3 outcome set (``OverflowError`` is no
    ``SofaError``, so sticky mode could not latch it).
    """
    if value != value:  # NaN — only a NaN is unequal to itself
        dbits: int = _U64.unpack(_F64.pack(value))[0]
        # Recover the top 23 mantissa bits (>> 29), keeping sign + signaling bit.
        bits = ((dbits >> 63) << 31) | _F32_EXP | ((dbits >> 29) & _F32_MANT)
        if (bits & _F32_MANT) == 0:
            # A NaN whose payload lived only in the dropped low bits would
            # collapse to inf; force it back to a (quiet) NaN instead.
            bits |= 0x00400000
        return bits
    try:
        narrowed: int = _U32.unpack(_F32.pack(value))[0]
        return narrowed
    except OverflowError:
        # Cold: only an out-of-fp32-range magnitude gets here. float() re-raises
        # for a value that is not even a double (a huge int), which is what the
        # native path does with it too.
        return _F32_EXP | (0x80000000 if float(value) < 0.0 else 0)


# The bit-level fp32 helpers above exist for NaN alone: only a NaN payload has
# to survive the trip through a Python ``float`` unchanged (§4.6/§6.5). Every
# other value converts exactly through ``struct``'s own ``f`` code, in ONE call
# rather than the unpack->pack->unpack round trip the bit path costs — and
# whether a value is a NaN is answered by the conversion itself (``v != v``),
# so the fast path needs no test of its own before it runs.


def pack_f32(value: float) -> bytes:
    """Pack a single fp32 value to 4 little-endian bytes."""
    if value == value:  # not NaN: struct narrows exactly, except on overflow
        try:
            return _F32.pack(value)
        except OverflowError:
            pass
    return _U32.pack(_pack_f32_bits(value))


def pack_f64(value: float) -> bytes:
    """Pack a single fp64 value to 8 little-endian bytes."""
    return _F64.pack(value)


def unpack_f32(data: bytes) -> float:
    """Decode a single little-endian fp32 value from 4 bytes."""
    value: float = _F32.unpack(data)[0]
    if value != value:  # NaN: re-derive it from the raw bits, payload intact
        return _unpack_f32_bits(_U32.unpack(data)[0])
    return value


def unpack_u32(data: Any) -> int:
    """The raw little-endian 32 bits of an ``fp32`` payload, untouched.

    CORELIB_PLAN §6.5 requires a double-only target to carry an ``fp32`` to a
    bit-exact consumer as **wire bits**, never as the widened value: the IEEE
    widening sets the quiet bit, and a signaling NaN's payload is gone the
    instant it passes through a wider float. This is that channel.
    """
    bits: int = _U32.unpack(data)[0]
    return bits


def unpack_f64(data: bytes) -> float:
    """Decode a single little-endian fp64 value from 8 bytes."""
    value: float = _F64.unpack(data)[0]
    return value


def unpack_f32_array(data: Any, count: int, start: int = 0) -> list[float]:
    """Decode ``count`` little-endian fp32 values (NaN-bit-preserving).

    ``data`` is any buffer and ``start`` the byte offset the payload begins at,
    so a caller holding the payload inside a larger buffer never slices a copy
    out first (CORELIB_PLAN §6.6: nothing the wire sizes on the way to a value).
    """
    values = list(struct.unpack_from(f"<{count}f", data, start))
    if not any(v != v for v in values):
        return values
    bits = struct.unpack_from(f"<{count}I", data, start)
    return [_unpack_f32_bits(b) if v != v else v for v, b in zip(values, bits)]


def unpack_f64_array(data: Any, count: int, start: int = 0) -> list[float]:
    """Decode ``count`` little-endian fp64 values in one ``struct`` call; see
    :func:`unpack_f32_array` for ``start``."""
    return list(struct.unpack_from(f"<{count}d", data, start))


def pack_f32_array(values: list[float]) -> bytes:
    """Encode a list of fp32 values (NaN-bit-preserving), little-endian."""
    if not any(v != v for v in values):
        try:
            return struct.pack(f"<{len(values)}f", *values)
        except OverflowError:
            pass
    return struct.pack(f"<{len(values)}I", *[_pack_f32_bits(v) for v in values])


def pack_f64_array(values: list[float]) -> bytes:
    """Encode a list of fp64 values in one ``struct`` call (little-endian)."""
    return struct.pack(f"<{len(values)}d", *values)


# --- fixlen arrays straight into a caller destination -----------------------
#
# The array forms above build a ``list`` the wire sizes, which is what the
# visitor's default route needs. Where the destination already exists — a
# binding's ``words`` slots, a buffer a handler returned — CORELIB_PLAN §6.6
# forbids sizing anything from the wire on the way there, so the payload is
# moved through a **landing zone of a fixed number of elements**: §6.6.2's
# "landing zone for a scalar", widened to a block so the copy still runs at C
# speed. Peak allocation is then the block, whatever the count is.

#: Elements converted per pass. Large enough that the per-pass overhead is
#: amortised, small enough that the staging tuple/array is a constant.
FARRAY_CHUNK = 64

_F32_CHUNK = struct.Struct(f"<{FARRAY_CHUNK}f")
_F64_CHUNK = struct.Struct(f"<{FARRAY_CHUNK}d")
_U32_CHUNK = struct.Struct(f"<{FARRAY_CHUNK}I")


def unpack_farray_into(
    dst: Any, at: int, data: Any, count: int, width: int, start: int = 0
) -> None:
    """Move ``count`` little-endian fp32/fp64 values from ``data`` into ``dst``.

    ``dst`` is a writable ``d``-format buffer (a ``memoryview`` cast, an
    ``array``) and ``at`` the element index to start at; ``data`` is any buffer
    and ``start`` the byte offset the payload begins at, so the caller need not
    slice a copy out first. Nothing sized by ``count`` is allocated: the payload
    crosses in blocks of :data:`FARRAY_CHUNK`.
    """
    fp32 = width == 4
    off = 0
    while off < count:
        n = count - off
        if n > FARRAY_CHUNK:
            n = FARRAY_CHUNK
        pos = start + off * width
        if fp32:
            if n == FARRAY_CHUNK:
                values = _F32_CHUNK.unpack_from(data, pos)
            else:
                values = struct.unpack_from(f"<{n}f", data, pos)
            if any(v != v for v in values):
                # A NaN in the block: re-derive the whole block from its raw
                # bits so a signaling payload survives the widening (§6.5).
                if n == FARRAY_CHUNK:
                    bits = _U32_CHUNK.unpack_from(data, pos)
                else:
                    bits = struct.unpack_from(f"<{n}I", data, pos)
                values = tuple(
                    _unpack_f32_bits(b) if v != v else v
                    for v, b in zip(values, bits)
                )
        elif n == FARRAY_CHUNK:
            values = _F64_CHUNK.unpack_from(data, pos)
        else:
            values = struct.unpack_from(f"<{n}d", data, pos)
        dst[at + off : at + off + n] = _array("d", values)
        off += n


# --- UTF-8 validation without materializing the string ----------------------

#: Bytes validated per pass. The transient ``str`` each pass builds is bounded
#: by this, so a payload of any length costs a constant.
UTF8_WINDOW = 4096


def utf8_valid(data: Any, start: int, size: int) -> bool:
    """Are ``size`` bytes of ``data`` from ``start`` valid UTF-8?

    CORELIB_PLAN §6.4.3's primitive, for the one caller that needs the answer
    without the value: a `string` read into a destination the caller supplied
    (§6.6.3) still has to be validated (§6.7.2), and decoding it to find out
    would build the very ``str`` the destination exists to avoid.

    The bytes are checked in windows of :data:`UTF8_WINDOW`, with a sequence
    straddling a window boundary carried into the next pass exactly as §6.4.4
    carries one across a fed chunk. Peak allocation is therefore the window,
    whatever the payload's length.
    """
    view = memoryview(data)
    try:
        pos = 0
        while pos < size:
            n = size - pos
            if n > UTF8_WINDOW:
                n = UTF8_WINDOW
            final = pos + n >= size
            try:
                _text, used = _utf8_decode(
                    view[start + pos : start + pos + n], "strict", final
                )
            except UnicodeDecodeError:
                return False
            if used <= 0:
                # An incomplete sequence at the end of a non-final window. UTF-8
                # sequences are at most 4 bytes and the window is far wider, so
                # this can only mean the window *is* the tail — which `final`
                # already covered.
                return False  # pragma: no cover
            pos += used
        return True
    finally:
        view.release()
