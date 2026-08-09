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


def unpack_f64(data: bytes) -> float:
    """Decode a single little-endian fp64 value from 8 bytes."""
    value: float = _F64.unpack(data)[0]
    return value


def unpack_f32_array(data: bytes, count: int) -> list[float]:
    """Decode ``count`` little-endian fp32 values (NaN-bit-preserving)."""
    values = list(struct.unpack(f"<{count}f", data))
    if not any(v != v for v in values):
        return values
    bits = struct.unpack(f"<{count}I", data)
    return [_unpack_f32_bits(b) if v != v else v for v, b in zip(values, bits)]


def unpack_f64_array(data: bytes, count: int) -> list[float]:
    """Decode ``count`` little-endian fp64 values in one ``struct`` call."""
    return list(struct.unpack(f"<{count}d", data))


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
