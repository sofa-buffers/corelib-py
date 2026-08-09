"""Varint (base-128 LE) encoding and the ZigZag mapping — the hot path.

This module is deliberately small and free of higher-level concepts so it can
later be replaced by a compiled accelerator (mypyc/Cython, or a PyO3 binding
over ``corelib-rs``) without touching the public API. Keep the signatures
stable.

It holds the **encode** half only. Decoding a varint means advancing a cursor
over a refillable buffer, so each engine inlines the codec into its own decode
loops (``Decoder._varint`` / ``Decoder._read_varints``, and their C twins in
``_speedups``); a byte-at-a-time decoder here would be a third copy of §4.1 that
no shipped path can reach, kept in sync by hand (issue #75).
"""

from __future__ import annotations

from .types import MASK64


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
