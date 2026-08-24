"""The static helper layer: turning a wrapper-array's events back into a list.

CORELIB_PLAN §6.6.1 names it — "the reassembly buffers, **sequence collectors
and array builders** a port holds so the generator need not emit them into every
generated package". It ships beside the codec and is **not part of it**: the
generated layer calls a collector, the collector calls the codec, never the other
way round, and *this* layer is the one allowed to allocate (§6.6).

A wrapper-sequence array (MESSAGE_SPEC §5.1) is an array whose elements are not
native scalars — strings, blobs, structs — and it reaches a visitor as a nested
sequence **whose child ids are the array indices**. Turning that back into a list
is the same code for every schema: place at the id, fill the gap the omitted
interior elements left, refuse an id past the bound. Only the bounds differ, and
a bound is an argument.

Two conventions run through the file.

**Place, never append.** An interior element equal to the element default is
omitted on the wire (MESSAGE_SPEC §2), so appending would shorten the array by
every such gap and would take a reopened id as a second element instead of
overwriting the first. The array's *last* element is always written, which is
what makes the decoded length — highest present id + 1 — exact.

**Which bound applies is the schema's choice.** ``cap`` is the schema's declared
element count: an id at or past it is a schema-bound violation and therefore
INVALID (§7.1). ``max_dyn_array_count`` is the receiver's limit and applies only
where the schema declares none — §6.2.1 forbids a receiver limit on a field the
schema already bounds — and exceeding it is a policy rejection, not INVALID.
Either way the id is judged **before** the list grows, so an index near 2**31
costs a comparison and not an allocation.
"""

from __future__ import annotations

from typing import Any, Callable

from .types import ARRAY_MAX, SofaDecodeError, SofaLimitError
from .visitor import Visitor

__all__ = [
    "BytesSeq",
    "Float32Seq",
    "Float64Seq",
    "NestedSeq",
    "SequenceCollector",
    "SignedSeq",
    "StringSeq",
    "UnsignedSeq",
]


class SequenceCollector(Visitor):
    """Base for the collectors: the bound check and the gap-filling placement.

    ``out`` is the caller's list and is written in place. ``cap`` is the schema's
    declared count, or ``None`` where the schema declares none;
    ``max_dyn_array_count`` is the receiver limit that then applies. ``default``
    is what an omitted interior element leaves behind.
    """

    default: Any = None

    def __init__(
        self,
        out: list[Any],
        *,
        cap: int | None = None,
        max_dyn_array_count: int = ARRAY_MAX,
    ) -> None:
        self.out = out
        self.cap = cap
        self.max_dyn_array_count = max_dyn_array_count

    def _slot(self, index: int) -> None:
        """Judge the index, then make room for it — in that order (§6.2.1)."""
        if self.cap is not None:
            if index >= self.cap:
                # The schema bounded this array, so an id past it is a statement
                # about validity, not about capacity (§7.1).
                raise SofaDecodeError(
                    f"element index {index} exceeds the {self.cap} the schema declares"
                )
        elif index >= self.max_dyn_array_count:
            raise SofaLimitError(
                f"element index {index} exceeds "
                f"max_dyn_array_count {self.max_dyn_array_count}"
            )
        out = self.out
        while len(out) <= index:
            out.append(self.default)

    def _place(self, index: int, value: Any) -> None:
        self._slot(index)
        self.out[index] = value


class _LeafSeq(SequenceCollector):
    """Elements that arrive as a single value in the wrapper's own scope.

    Like every collector, an instance handles **one wrapper scope** — the object
    holding the array field returns it from ``on_sequence_begin`` — so the ids it
    sees are that array's indices and nothing else's.
    """


class StringSeq(_LeafSeq):
    """``string`` elements. ``elem_max`` is the schema's ``maxlen``, if any."""

    default = ""

    def __init__(
        self, out: list[Any], *, elem_max: int | None = None, **kw: Any
    ) -> None:
        super().__init__(out, **kw)
        self.elem_max = elem_max

    def on_string(self, field_id: int, value: str) -> None:
        if self.elem_max is not None and len(value) > self.elem_max:
            raise SofaDecodeError(
                f"string length {len(value)} exceeds the {self.elem_max} "
                "the schema declares"
            )
        self._place(field_id, value)


class BytesSeq(_LeafSeq):
    """``blob`` elements — the string twin, with no UTF-8 to check."""

    default = b""

    def __init__(
        self, out: list[Any], *, elem_max: int | None = None, **kw: Any
    ) -> None:
        super().__init__(out, **kw)
        self.elem_max = elem_max

    def on_bytes(self, field_id: int, value: bytes) -> None:
        if self.elem_max is not None and len(value) > self.elem_max:
            raise SofaDecodeError(
                f"blob length {len(value)} exceeds the {self.elem_max} "
                "the schema declares"
            )
        self._place(field_id, value)


class UnsignedSeq(_LeafSeq):
    """Unsigned elements, for the wrapper form an integer array takes when its
    elements are not packed into one ARRAY_UNSIGNED field."""

    default = 0

    def on_unsigned(self, field_id: int, value: int) -> None:
        self._place(field_id, value)


class SignedSeq(_LeafSeq):
    default = 0

    def on_signed(self, field_id: int, value: int) -> None:
        self._place(field_id, value)


class Float32Seq(_LeafSeq):
    default = 0.0

    def on_float32(self, field_id: int, value: float) -> None:
        self._place(field_id, value)


class Float64Seq(_LeafSeq):
    default = 0.0

    def on_float64(self, field_id: int, value: float) -> None:
        self._place(field_id, value)


class NestedSeq(SequenceCollector):
    """Elements that are themselves framed — a struct, a union, a nested row.

    Each element reaches the decoder as a sub-sequence whose id is the array
    index, so this hands that scope to a handler of its own: ``factory()`` is
    called once per element and its return value both receives the element's
    fields and is what lands in ``out``.

    Like every collector it is the handler **for the wrapper's scope**, which the
    object holding the field hands over::

        class Doc(Visitor):
            def __init__(self):
                self.rows: list[Row] = []
            def on_sequence_begin(self, field_id):
                if field_id == 4:
                    return NestedSeq(self.rows, factory=Row, cap=16)
                return None

    The element is placed **before** it is filled, so the list's shape is settled
    at the index — which is where §6.2.1 wants the judgement — and a handler that
    keeps a reference sees the same object the list holds.
    """

    def __init__(
        self, out: list[Any], *, factory: Callable[[], Visitor], **kw: Any
    ) -> None:
        super().__init__(out, **kw)
        self.factory = factory

    def on_sequence_begin(self, field_id: int) -> Visitor | None:
        element = self.factory()
        self._place(field_id, element)
        return element
