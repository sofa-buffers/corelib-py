"""Shared fixlen (IEEE-754) pack/unpack helpers — part of the hot path.

Little-endian via ``struct`` with ``<``; this is correct on big-endian hosts
too (struct handles the byte order), matching the explicit LE handling in the
other core libs.
"""

from __future__ import annotations

import struct

_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")


def pack_f32(value: float) -> bytes:
    return _F32.pack(value)


def pack_f64(value: float) -> bytes:
    return _F64.pack(value)


def unpack_f32(data: bytes) -> float:
    return float(_F32.unpack(data)[0])


def unpack_f64(data: bytes) -> float:
    return float(_F64.unpack(data)[0])
