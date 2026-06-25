"""SofaBuffers — pure-Python runtime for the SofaBuffers binary wire format.

Byte-for-byte compatible with the C/Rust/Go/Java/C# core libraries. Import the
:class:`Encoder` and :class:`Decoder` and the wire-format types from here.
"""

from __future__ import annotations

from ._varint import zigzag_decode, zigzag_encode
from .decoder import Decoder
from .encoder import Encoder
from .types import (
    API_VERSION,
    ARRAY_MAX,
    ID_MAX,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    Field,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaError,
    SofaRangeError,
    SofaStateError,
    WireType,
)

__version__ = "0.1.0"

__all__ = [
    "Encoder",
    "Decoder",
    "Field",
    "WireType",
    "FixlenSubtype",
    "SofaError",
    "SofaDecodeError",
    "SofaRangeError",
    "SofaStateError",
    "SofaBufferError",
    "API_VERSION",
    "ID_MAX",
    "ARRAY_MAX",
    "UNSIGNED_MAX",
    "SIGNED_MIN",
    "SIGNED_MAX",
    "zigzag_encode",
    "zigzag_decode",
    "__version__",
]
