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
        return float(_F64.unpack(_U64.pack(dbits))[0])
    return float(_F32.unpack(_U32.pack(bits))[0])


def _pack_f32_bits(value: float) -> int:
    """Narrow a Python float to a raw little-endian fp32 bit pattern.

    NaN is narrowed bit-for-bit (inverse of :func:`_unpack_f32_bits`); every
    other value goes through ``struct``.
    """
    if value != value:  # NaN — only a NaN is unequal to itself
        dbits = int(_U64.unpack(_F64.pack(value))[0])
        # Recover the top 23 mantissa bits (>> 29), keeping sign + signaling bit.
        bits = ((dbits >> 63) << 31) | _F32_EXP | ((dbits >> 29) & _F32_MANT)
        if (bits & _F32_MANT) == 0:
            # A NaN whose payload lived only in the dropped low bits would
            # collapse to inf; force it back to a (quiet) NaN instead.
            bits |= 0x00400000
        return bits
    return int(_U32.unpack(_F32.pack(value))[0])


def pack_f32(value: float) -> bytes:
    """Pack a single fp32 value to 4 little-endian bytes."""
    return _U32.pack(_pack_f32_bits(value))


def pack_f64(value: float) -> bytes:
    """Pack a single fp64 value to 8 little-endian bytes."""
    return _F64.pack(value)


def unpack_f32(data: bytes) -> float:
    """Decode a single little-endian fp32 value from 4 bytes."""
    return _unpack_f32_bits(_U32.unpack(data)[0])


def unpack_f64(data: bytes) -> float:
    """Decode a single little-endian fp64 value from 8 bytes."""
    return float(_F64.unpack(data)[0])


def unpack_f32_array(data: bytes, count: int) -> list[float]:
    """Decode ``count`` little-endian fp32 values (NaN-bit-preserving)."""
    bits = struct.unpack(f"<{count}I", data)
    return [_unpack_f32_bits(b) for b in bits]


def unpack_f64_array(data: bytes, count: int) -> list[float]:
    """Decode ``count`` little-endian fp64 values in one ``struct`` call."""
    return list(struct.unpack(f"<{count}d", data))


def pack_f32_array(values: list[float]) -> bytes:
    """Encode a list of fp32 values (NaN-bit-preserving), little-endian."""
    return struct.pack(f"<{len(values)}I", *[_pack_f32_bits(v) for v in values])


def pack_f64_array(values: list[float]) -> bytes:
    """Encode a list of fp64 values in one ``struct`` call (little-endian)."""
    return struct.pack(f"<{len(values)}d", *values)
