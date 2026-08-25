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

#: Bytes of reassembly space a :class:`sofab.Decoder` takes when the caller
#: names no size (see ``Decoder(reassembly=…)``).
#:
#: A construct split across fed chunks has to be joined somewhere, and
#: CORELIB_PLAN §6.6.2 puts that somewhere in the caller's hands: "A codec
#: **MUST NOT** grow a private accumulator instead." So the decoder holds one
#: buffer, sized **once at construction** and never grown — a sender cannot make
#: it bigger by sending different bytes, which is the property §6.6 protects —
#: and a construct that does not fit it is refused with
#: :class:`SofaArgumentError` rather than accommodated.
#:
#: This number is the corelib's, not the specification's: §6.6.2 names no
#: default because the storage is meant to be the caller's outright. It is a
#: size that lets ordinary messages stream without configuration; a receiver
#: that takes larger `string`/`blob`/array payloads **across chunk boundaries**
#: passes its own buffer, or a byte count for the decoder to size one from.
#: (Whole messages fed in one call never touch it, whatever their size.)
DEFAULT_REASSEMBLY = 4096

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


class Status(IntEnum):
    """The three-valued decode outcome CORELIB_PLAN §5.2 requires.

    Returned by every :meth:`sofab.Decoder.feed`, describing the bytes consumed
    **so far** — not a verdict on the message as a whole:

    * :attr:`COMPLETE` — the consumed bytes end exactly at a field boundary. A
      valid message *may* end here; more fields may also still follow.
    * :attr:`INCOMPLETE` — the bytes end *inside* a construct. **This is not an
      error.** The partial tail is retained and the next ``feed`` continues from
      it. Whether an incomplete message is acceptable is the caller's decision:
      only its framing (a length prefix, a datagram boundary, EOF) knows whether
      more bytes can still come.
    * :attr:`INVALID` — the bytes are malformed regardless of what follows.
      Terminal; the reason is on :attr:`sofab.Decoder.error`.

    There is deliberately **no** ``finish``/``end`` step that could reclassify
    :attr:`INCOMPLETE` as an error (§5.2): the status ``feed`` returned *is* the
    answer. A receiver-side limit rejection (§6.2.1) is not one of these three —
    it is a well-formed message the receiver declined, so it arrives on the error
    channel as :class:`SofaLimitError`, never as :attr:`INVALID` (§6.3).
    """

    COMPLETE = 0
    INCOMPLETE = 1
    INVALID = 2


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
    **receiver-configured** decode limit (``Decoder(max_dyn_array_count=…,
    max_dyn_string_len=…, max_dyn_blob_len=…)``).

    This is a *policy* rejection, not wire malformation: the bytes are perfectly
    well-formed and would decode fine with the limit unset — the receiver simply
    declined to allocate for them. It is therefore a sibling of
    :class:`SofaDecodeError` under :class:`SofaError`, **not** a subclass of it,
    so ``except SofaDecodeError`` does not catch it and differential fuzzing does
    not see a limit rejection as a conformance divergence from another engine.

    It is raised only for a field the **schema** leaves unbounded: where the
    schema states a ``count:``/``maxlen:`` the caller says so by declaring that
    bound on its :class:`sofab.Binding` entry, and that bound governs instead,
    an over-bound value being :class:`SofaDecodeError` (CORELIB_PLAN §6.2.1/§6.3,
    MESSAGE_SPEC §7.1). Nor is it raised for a field nothing materializes — a
    skipped one, or one read into storage the handler returned from
    :meth:`sofab.Visitor.on_blob_begin` / :meth:`sofab.Visitor.on_array_begin`.
    """


class SofaArgumentError(SofaError):
    """The caller's own request is invalid — the ``InvalidArgument`` outcome of
    CORELIB_PLAN §6.3, which is the *only* code that taxonomy has for a caller
    mistake (every remaining malformed input is :class:`SofaDecodeError`).

    Named after the code it carries. §6.3 lets a port "adapt casing and idiom",
    and this class used to take that as far as ``SofaRangeError`` — which read
    narrower than the code is: a destination too short for what a hook was told
    is a caller mistake, not a value out of range. Every other port keeps the
    word (``Error::Argument`` in Rust, ``Error::InvalidArgument`` in C++), so
    this one does too. ``SofaRangeError`` remains as an alias.

    On encode, a value (or id/count) is not writable: either it is outside the
    permitted range, or it is not an integer at all: integer fields accept
    whatever Python considers losslessly an integer (any object with
    ``__index__`` — ``int``, ``bool``, ``IntEnum``, NumPy integers), and refuse
    everything else rather than silently truncating it. A ``float`` is therefore
    rejected, ``3.0`` included; convert explicitly with ``int(x)`` if that is
    what you mean. The same code covers the encoder's other invalid calls:
    :meth:`~sofab.Encoder.getvalue` on a caller-owned fixed buffer, and a
    sequence end without a matching begin.

    On decode it covers the caller's own storage not fitting what the message
    announced — a **destination that cannot hold what was announced**: a buffer handed back from
    :meth:`sofab.Visitor.on_blob_begin` or :meth:`sofab.Visitor.on_array_begin`
    shorter than the size those hooks were told, or a ``reassembly=`` buffer too
    small for a construct spanning a chunk. The handler was given the count or
    length first and answered with storage that does not fit it, so the mistake
    is the caller's; §6.6.3 has the codec refuse such a destination "rather than
    growing it", and §6.2.1 forbids clamping into it.

    That second case is the *only* ceiling left on a destination the caller
    supplies. A receiver's configured ``max_dyn_*`` limit does not also apply to
    it: the limit exists to stop the sender dictating the receiver's allocation,
    and a handler that returns a buffer has sized that buffer itself
    (:class:`SofaLimitError`).

    It is *not* raised when a field's wire type merely contradicts the type a
    binding declares for it: that is MESSAGE_SPEC §7.3, which the decoder answers
    by skipping the field (see :class:`sofab.Decoder`).
    """


#: Deprecated alias for :class:`SofaArgumentError`, kept so existing
#: ``except SofaRangeError`` and ``isinstance`` checks keep working. It is the
#: same class, not a subclass, so either name catches what the other raises.
SofaRangeError = SofaArgumentError


class SofaBufferError(SofaError):
    """A fixed encoder buffer filled up and no flush sink was provided."""
