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

from .types import Field


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

    # --- typed value hooks --------------------------------------------------

    def on_unsigned(self, field_id: int, value: int) -> None: ...
    def on_signed(self, field_id: int, value: int) -> None: ...
    def on_float32(self, field_id: int, value: float) -> None: ...
    def on_float64(self, field_id: int, value: float) -> None: ...
    def on_string(self, field_id: int, value: str) -> None: ...
    def on_bytes(self, field_id: int, value: bytes) -> None: ...
    def on_unsigned_array(self, field_id: int, values: list[int]) -> None: ...
    def on_signed_array(self, field_id: int, values: list[int]) -> None: ...
    def on_float32_array(self, field_id: int, values: list[float]) -> None: ...
    def on_float64_array(self, field_id: int, values: list[float]) -> None: ...
