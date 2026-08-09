"""Wire-format constants, enums, the :class:`Field` descriptor, and errors.

These mirror the shared SofaBuffers definitions used by ``corelib-c-cpp``,
``corelib-rs``, ``corelib-go``, ``corelib-java`` and ``corelib-cs`` so the
Python runtime produces byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# --- limits (from corelib-c-cpp/src/include/sofab/sofab.h) -------------------

#: SofaBuffers API version (mirrors C ``SOFAB_API_VERSION``). Callers and the
#: code generator use this to verify wire compatibility.
API_VERSION = 1

#: Highest valid field ID (``INT32_MAX``).
ID_MAX = 0x7FFF_FFFF
#: Largest unsigned wire value (``UINT64_MAX``).
UNSIGNED_MAX = (1 << 64) - 1
#: Signed wire value range (``INT64_MIN`` .. ``INT64_MAX``).
SIGNED_MIN = -(1 << 63)
SIGNED_MAX = (1 << 63) - 1
#: Largest fixlen payload length in bytes (``INT32_MAX``).
FIXLEN_MAX = 0x7FFF_FFFF
#: Largest array element count (``INT32_MAX``).
ARRAY_MAX = 0x7FFF_FFFF
#: Maximum nested-sequence depth. An encoder must not open more than this many
#: nested sequences; a decoder rejects a message nesting deeper.
MAX_DEPTH = 255

#: Smallest output buffer this port accepts **for streaming** (CORELIB_PLAN
#: §5.1). It is ``1`` because the encoder splits every atomic unit — a header
#: varint, a ``fixlen_word``, an element count, a scalar, one float — at any byte
#: boundary, so a one-byte scratch buffer already produces exactly the one-shot
#: bytes. The declaration binds a buffer installed **with** a flush sink, at
#: installation and at every mid-stream :meth:`~sofab.Encoder.buffer_set`:
#: ``len(buffer) - offset`` must be at least this. A buffer installed **without**
#: a sink is subject to no minimum — no flush can occur, so it simply holds the
#: message or reports ``SofaBufferError``, and a caller sizing from a generated
#: ``MAX_SIZE`` gets an exact fit.
MIN_OUTPUT_BUFFER = 1

#: 64-bit mask used by varint/zigzag wrap-around to match the C ``uint64_t``.
MASK64 = (1 << 64) - 1


class WireType(IntEnum):
    """The 3 low bits of a field header."""

    UNSIGNED = 0x0
    SIGNED = 0x1
    FIXLEN = 0x2
    ARRAY_UNSIGNED = 0x3
    ARRAY_SIGNED = 0x4
    ARRAY_FIXLEN = 0x5
    SEQUENCE_START = 0x6
    SEQUENCE_END = 0x7


class FixlenSubtype(IntEnum):
    """The 3 low bits of a fixlen length header."""

    FP32 = 0x0
    FP64 = 0x1
    STRING = 0x2
    BLOB = 0x3


@dataclass
class Field:
    """Describes the field the decoder is currently positioned on.

    Mirrors the C field callback's ``(id, size, count)`` plus the wire type.
    ``size`` is the fixlen byte length (or the per-element size of a fixlen
    array); ``count`` is the element count of an array; ``subtype`` is set for
    fixlen and fixlen-array fields.
    """

    id: int
    type: WireType
    size: int = 0
    count: int = 0
    subtype: FixlenSubtype | None = None


# --- errors -----------------------------------------------------------------


class SofaError(Exception):
    """Base class for all SofaBuffers errors."""


class SofaDecodeError(SofaError):
    """Malformed input — invalid *regardless* of what bytes might follow: an
    overflowing (>64-bit) varint, a bad fixlen subtype, an out-of-range
    id/count/length, invalid UTF-8, nesting past ``MAX_DEPTH``, or a dangling
    sequence end (``MESSAGE_SPEC`` §7 INVALID).

    This is deliberately **not** raised for truncation — bytes that simply end
    inside a field are :class:`SofaIncompleteError` (§7 INCOMPLETE), a distinct
    non-error outcome that is not a subclass of this class, so
    ``except SofaDecodeError`` does not catch it.
    """


class SofaIncompleteError(SofaError):
    """Truncated input — the bytes end *inside* a field (``MESSAGE_SPEC`` §7
    INCOMPLETE): an unterminated varint, a fixlen/array payload shorter than its
    declared length, an array element that runs off the end, or a nested
    sequence that is never closed.

    This is **not** malformed: more bytes could complete the message, and the
    caller owns end-of-input. It is a sibling of :class:`SofaDecodeError` under
    :class:`SofaError`, *not* a subclass of it, so callers can tell "need more
    bytes" apart from "these bytes are garbage".
    """


class SofaLimitError(SofaError):
    """A wire-declared array count or fixlen (string/blob) length exceeded a
    **receiver-configured** decode limit (``Decoder(max_array_count=…,
    max_string_len=…, max_blob_len=…)``).

    This is a *policy* rejection, not wire malformation: the bytes are perfectly
    well-formed and would decode fine with the limit unset — the receiver simply
    declined to allocate for them. It is therefore a sibling of
    :class:`SofaDecodeError` under :class:`SofaError`, **not** a subclass of it,
    so ``except SofaDecodeError`` does not catch it and differential fuzzing does
    not see a limit rejection as a conformance divergence from another engine.

    It is raised only for a field the **schema** leaves unbounded: where the
    schema states a ``count:``/``maxlen:`` the caller says so with
    :meth:`sofab.Decoder.schema_bounded` and that bound governs instead, an
    over-bound value being :class:`SofaDecodeError` (CORELIB_PLAN §6.2.1/§6.3,
    MESSAGE_SPEC §7.1).
    """


class SofaRangeError(SofaError):
    """A value (or id/count) is not writable on encode — the ``InvalidArgument``
    outcome of CORELIB_PLAN §6.3.

    Either it is outside the permitted range, or it is not an integer at all:
    integer fields accept whatever Python considers losslessly an integer (any
    object with ``__index__`` — ``int``, ``bool``, ``IntEnum``, NumPy integers),
    and refuse everything else rather than silently truncating it. A ``float`` is
    therefore rejected, ``3.0`` included; convert explicitly with ``int(x)`` if
    that is what you mean.
    """


class SofaStateError(SofaError):
    """API misuse, e.g. reading a value of the wrong type for the current
    field, or ending a sequence that was never started."""


class SofaBufferError(SofaError):
    """A fixed encoder buffer filled up and no flush sink was provided."""
