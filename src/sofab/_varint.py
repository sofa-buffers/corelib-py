"""Varint (base-128 LE) and ZigZag codec — the hot path.

This module is deliberately small and free of higher-level concepts so it can
later be replaced by a compiled accelerator (mypyc/Cython, or a PyO3 binding
over ``corelib-rs``) without touching the public API. Keep the signatures
stable.
"""

from __future__ import annotations

from typing import Callable

from .types import MASK64, SofaDecodeError, SofaIncompleteError


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
    :class:`SofaIncompleteError` when the bytes end mid-varint (truncation, §7
    INCOMPLETE) and :class:`SofaDecodeError` on a >64-bit overflow (§7 INVALID),
    mirroring the C decoder (overflow once the shift reaches 64 bits with
    continuation set).
    """
    value = 0
    shift = 0
    while True:
        byte = first if first is not None else read_byte()
        first = None
        if byte is None:
            raise SofaIncompleteError("truncated varint")
        # Reject an overlong (>64-bit) varint before OR-ing: if this byte's 7
        # payload bits would spill past bit 63 they are unrepresentable in u64
        # and must be INVALID, not silently masked away on return (§4.1/§6.3,
        # issue #43). ``room`` is the bits left below 64.
        room = 64 - shift
        if room < 7 and (byte & 0x7F) >> room:
            raise SofaDecodeError("overlong varint")
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            return value & MASK64
        if shift >= 64:
            raise SofaDecodeError("overlong varint")
