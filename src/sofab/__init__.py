"""SofaBuffers — runtime for the SofaBuffers binary wire format.

Byte-for-byte compatible with the C/C++/Rust/Go/Java/C# core libraries. Import
the :class:`Encoder` and :class:`Decoder` and the wire-format types from here.

``Encoder`` / ``Decoder`` resolve to the compiled native accelerator
(``sofab._speedups``, built by Cython) when it is available, and to the
pure-Python implementations otherwise — the two are byte-for-byte interchangeable
(see :data:`IMPL`).
"""

from __future__ import annotations

import os as _os
from typing import TYPE_CHECKING

from ._varint import zigzag_decode, zigzag_encode
from .types import (
    API_VERSION,
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    MAX_DEPTH,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    Field,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaRangeError,
    SofaStateError,
    WireType,
)
from .visitor import Visitor

# --- implementation selection -----------------------------------------------
#
# ``Encoder`` / ``Decoder`` / ``Field`` come from the compiled native
# accelerator (``sofab._speedups``, built by Cython) when it is available and
# produce the exact same bytes / values as the pure-Python classes. If the
# extension was never built (no compiler / unsupported platform) or
# ``SOFAB_PUREPYTHON=1`` is set, the pure-Python implementations are used
# instead. Both are byte-for-byte compatible and validated by the same shared
# conformance vectors, so callers and generated code never need to care which
# one is active. ``Field`` is re-exported from the active engine so that
# ``isinstance(decoder.next(), sofab.Field)`` holds in both modes.
if TYPE_CHECKING:
    # For static analysis the pure-Python classes are the reference definitions;
    # the native accelerator mirrors their public API exactly.
    from .decoder import Decoder as Decoder
    from .encoder import Encoder as Encoder
    from .types import Field as Field

    #: Which implementation ``Encoder``/``Decoder`` resolve to: ``"native"`` when
    #: the compiled ``sofab._speedups`` extension is in use, else ``"python"``.
    IMPL = "python"
elif _os.environ.get("SOFAB_PUREPYTHON") == "1":
    from .decoder import Decoder
    from .encoder import Encoder

    IMPL = "python"
else:  # pragma: no cover - native branch: exercised only when the compiled
    #                        extension is present; coverage runs force pure Python.
    try:
        from ._speedups import Decoder, Encoder, Field  # Field shadows types.Field

        IMPL = "native"
    except ImportError:
        from .decoder import Decoder
        from .encoder import Encoder

        IMPL = "python"

__version__ = "0.1.0"

__all__ = [
    "Encoder",
    "Decoder",
    "Visitor",
    "Field",
    "WireType",
    "FixlenSubtype",
    "SofaError",
    "SofaDecodeError",
    "SofaIncompleteError",
    "SofaRangeError",
    "SofaStateError",
    "SofaBufferError",
    "API_VERSION",
    "ID_MAX",
    "ARRAY_MAX",
    "FIXLEN_MAX",
    "MAX_DEPTH",
    "UNSIGNED_MAX",
    "SIGNED_MIN",
    "SIGNED_MAX",
    "zigzag_encode",
    "zigzag_decode",
    "IMPL",
    "__version__",
]
