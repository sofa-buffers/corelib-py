"""Visitor-pattern decode driver — the language-idiomatic pull alternative.

ARCHITECTURE.md lists the visitor pattern as a recommended decoder shape:

    "The decoder calls typed visitor methods on a user-supplied object.
     Pull-reading becomes the visitor receives the value or chooses to skip."

:class:`Visitor` is a base class whose hooks all default to *no-op* (the value
is still consumed, so an unhandled field is transparently skipped). Subclass it
and override only the fields you care about, then hand it to
:meth:`sofab.Decoder.drive`.

Two control hooks let a visitor decline work *before* the value is decoded — so
skipping a 10k-element array or a deep sub-tree costs nothing:

* :meth:`Visitor.on_field` — return ``False`` to skip a scalar/fixlen/array
  field instead of decoding it.
* :meth:`Visitor.on_sequence_begin` — return ``False`` to skip the entire
  nested sequence (its matching end is consumed too, so ``on_sequence_end`` is
  *not* called for a skipped sequence).

The driver itself is layered on the public pull API, so it inherits the same
"advance a cursor over a contiguous buffer" hot path as direct pull decoding.
"""

from __future__ import annotations

from typing import Any

from .types import Field, WireType


class Visitor:
    """Base visitor: override the hooks for the fields you handle.

    Every hook is keyed by the wire type the decoder recovered. ``field_id`` is
    the decoded field id. Unhandled hooks default to a no-op, which still
    consumes the value (so unknown fields are skipped safely)."""

    # --- control hooks (return False to skip before decoding) ---------------

    def on_field(self, field: Field) -> bool | None:
        """Called for every non-sequence field before its value is decoded.
        Return ``False`` to skip the value entirely; any other return proceeds
        to decode it and dispatch to the typed hook below."""
        return None

    def on_sequence_begin(self, field_id: int) -> bool | None:
        """A nested sequence is opening. Return ``False`` to skip the whole
        sub-tree (its end is consumed, ``on_sequence_end`` is not called)."""
        return None

    def on_sequence_end(self) -> None:
        """The current nested sequence closed."""

    def on_array_begin(
        self, field_id: int, wtype: WireType, count: int
    ) -> tuple[Any, int | None, int | None] | None:
        """An integer array's header has been read; no element has been decoded.

        This is the only place a handler can say anything about the array's
        elements, because the typed hook below receives them already decoded.
        Return ``None`` to take the default — a list, handed to
        :meth:`on_unsigned_array` / :meth:`on_signed_array` — or a
        ``(dst, elem_min, elem_max)`` triple:

        ``dst``
            Somewhere to put the elements, or ``None`` to keep the list. A
            writable buffer of at least ``count`` slots: an ``array`` of the
            right typecode, a ``memoryview`` over one, or any object supporting
            the buffer protocol. The decoder writes into it and does **not**
            call the typed hook — the handler already has the values where it
            wanted them, and none of them was ever a Python object. A buffer
            too short is :class:`sofab.SofaRangeError`; the decoder never grows
            one (CORELIB_PLAN §6.6).
        ``elem_min`` / ``elem_max``
            The element width the schema declares, or ``None`` for an open
            side. The decoder applies it **at each element**, so a value outside
            it is INVALID whether the array completes or is truncated behind it
            (§7.1), which is also §5.2's INVALID-over-INCOMPLETE for free. A
            handler cannot do this itself: by the time it holds the list, an
            array that never arrived is indistinguishable from one that did.

        Called again for the same array if a chunk boundary suspends the read,
        so return the same answer each time; the decoder restarts the array from
        its first element and fills ``dst`` from the beginning.

        Not called for float arrays, which carry no declared width to state and
        are already moved into a destination in one piece.
        """
        return None

    # --- typed value hooks --------------------------------------------------

    def on_unsigned(self, field_id: int, value: int) -> None:
        """Handle a decoded unsigned-integer field."""

    def on_signed(self, field_id: int, value: int) -> None:
        """Handle a decoded signed-integer field."""

    def on_float32(self, field_id: int, value: float) -> None:
        """Handle a decoded 32-bit float field."""

    def on_float64(self, field_id: int, value: float) -> None:
        """Handle a decoded 64-bit float field."""

    def on_string(self, field_id: int, value: str) -> None:
        """Handle a decoded UTF-8 string field."""

    def on_bytes(self, field_id: int, value: bytes) -> None:
        """Handle a decoded raw byte-blob field."""

    def on_unsigned_array(self, field_id: int, values: list[int]) -> None:
        """Handle a decoded unsigned-integer array field."""

    def on_signed_array(self, field_id: int, values: list[int]) -> None:
        """Handle a decoded signed-integer array field."""

    def on_float32_array(self, field_id: int, values: list[float]) -> None:
        """Handle a decoded 32-bit float array field."""

    def on_float64_array(self, field_id: int, values: list[float]) -> None:
        """Handle a decoded 64-bit float array field."""
