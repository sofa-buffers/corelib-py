"""Varint (base-128 LE) and ZigZag codec — the hot path.

This module is deliberately small and free of higher-level concepts so it can
later be replaced by a compiled accelerator (mypyc/Cython, or a PyO3 binding
over ``corelib-rs``) without touching the public API. Keep the signatures
stable.
"""

from __future__ import annotations

from typing import Callable

from .types import MASK64, SofaDecodeError


def zigzag_encode(v: int) -> int:
    """Map a signed int to unsigned: ``(n << 1) ^ (n >> 63)`` (64-bit)."""
    return ((v << 1) ^ (v >> 63)) & MASK64


def zigzag_decode(u: int) -> int:
    """Inverse of :func:`zigzag_encode`: ``(z >> 1) ^ -(z & 1)``."""
    return (u >> 1) ^ -(u & 1)


def encode_varint(value: int) -> bytes:
    """Encode an unsigned 64-bit value as a base-128 little-endian varint.

    Matches the C ``_varint_encode`` do/while loop: ``0`` encodes to a single
    ``0x00`` byte.
    """
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(read_byte: Callable[[], int | None], first: int | None = None) -> int:
    """Decode a varint by pulling one byte at a time from ``read_byte``.

    ``read_byte`` returns the next byte as an ``int`` or ``None`` at EOF. Pass
    ``first`` to supply an already-read leading byte. Raises
    :class:`SofaDecodeError` on truncation or >64-bit overflow, mirroring the C
    decoder (overflow once the shift reaches 64 bits with continuation set).
    """
    value = 0
    shift = 0
    while True:
        byte = first if first is not None else read_byte()
        first = None
        if byte is None:
            raise SofaDecodeError("truncated varint")
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            return value & MASK64
        if shift >= 64:
            raise SofaDecodeError("varint overflow")
