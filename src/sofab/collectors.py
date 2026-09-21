"""The static helper layer: a wrapper array's element placement and growth.

CORELIB_PLAN §6.6.1 names it — "the reassembly buffers, **sequence collectors
and array builders** a port holds so the generator need not emit them into every
generated package". It ships beside the codec and is **not part of it**: nothing
here touches the wire, the codec never calls into it, and *this* layer is the one
allowed to allocate (§6.6). Generated code calls these functions from inside its
own flat :class:`~sofab.Visitor` callbacks; a hand-written visitor calls them the
same way.

A wrapper-sequence array (MESSAGE_SPEC §5.1) is an array whose elements are not
native scalars — strings, blobs, structs, unions, nested rows — and it reaches a
visitor as a nested sequence **whose child ids are the array indices**. Turning
that back into a list has the same shape for every schema. Its schema dependence
is exactly three things, and each arrives as an argument: a **bound** (the
schema's ``count``, or the receiver cap where there is none), an **element type**
(the factory) and an **element default**. That is why the code lives here,
written once, rather than being re-emitted into every generated package.

Three reservations, one shape
-----------------------------

Every wrapper array a schema can declare reaches this module through one of three
calls, which differ only in what a new slot holds:

:func:`reserve_leaf`
    a ``string`` or ``blob`` element: the gap value is a **shared immutable
    default** (``""`` / ``b""``), so a gap costs no allocation.
:func:`reserve_elem`
    a ``struct`` / ``union`` element, or a native-integer matrix row: each new
    slot gets **its own object** from ``make()``, because a shared mutable
    default would alias every element of the array onto one object.
:func:`reserve_row`
    a row that is itself a wrapper array (``array<array<string>>`` and kin):
    gaps are fresh empty rows and the row at the id is **replaced** by a fresh
    empty list, because an array wrapper *is* the array's value (§7.4).

Each call bounds the index, then grows the list. None of them places the value:
generated code stores ``out[id] = value`` from the value hook, or binds its own
element-index register and keeps routing the element's fields into ``out[id]``.
**The helper owns growth and the bound; the routing stays generated**, because
the routing is what differs per schema.

The index rules of MESSAGE_SPEC §5.1
------------------------------------

None of these is visible in the bytes — two implementations can disagree about
every one and still emit an identical message — which is why they are written
here once (CORELIB_PLAN §7.2 item 8 asks for them separately for the same reason).

* **Ids are positions; gaps are legal.** An interior element equal to the element
  default is omitted on the wire (MESSAGE_SPEC §2), so a missing id fills a gap
  rather than shifting every later element down by one. Place, never append.
* **The length is highest present id + 1.** The last element is never elided,
  so growing to ``id + 1`` is exactly right and no trailing fill is ever needed.
* **A repeated id replaces** (§7.4). An indexed store does that by construction.
  :func:`reserve_leaf` and :func:`reserve_elem` never overwrite a slot already
  present — the value store that follows does, and a re-opened framed element
  *merges* into the object its earlier fields built.
* **A rejected id extends nothing.** The index is judged **before** the list
  grows, so an index near 2**31 costs a comparison and not an allocation, and a
  refused id leaves the list exactly as it was — a lower id delivered afterwards
  still lands at its own index (§7.2 item 8).

The two bounds of CORELIB_PLAN §6.2.1
-------------------------------------

Every call takes **both** numbers, and exactly one of them applies. The schema
picks which:

``cap``
    the array's schema ``count``, or :data:`UNBOUNDED` where the schema declares
    none. A ``count`` is a **capacity**, not a length: the list starts empty and
    the wire carries the length. An id at or past a declared ``cap`` contradicts
    the schema both peers agreed on — malformed input,
    :class:`~sofab.SofaDecodeError`, the decoder's ``Status.INVALID`` (§7.1). A
    receiver cap is then **not applied at all** (§6.2.1 forbids a receiver limit
    on a field the schema already bounds).
``rcap``
    the receiver's ``max_dyn_array_count``, compared only where ``cap`` is
    :data:`UNBOUNDED`. An id at or past it is well-formed input this receiver
    declines: :class:`~sofab.SofaLimitError`, the policy category (§6.3) — the
    same element decodes for a receiver configured more loosely.

Neither number is this module's. Both are **passed in** on every call, used for
that one comparison and never retained. §6.2.1: a port "**MUST NOT** hold a limit
of its own, **MUST NOT** supply a default for one it was not given, **MUST NOT**
read an omitted argument as *unlimited*, and **MUST NOT** clamp to one". So
``rcap`` is a required positional argument, and on a schema-unbounded array a
value that states no cap — a negative number, ``None``, ``float('inf')``,
anything but an ``int`` — admits no element and is refused as
:class:`~sofab.SofaArgumentError` (§6.3's ``InvalidArgument``): the mistake is in
the **call**, and :class:`~sofab.SofaLimitError` would name a receiver policy
nobody configured.

What these bounds do not cover
------------------------------

A ``string`` or ``blob`` element's own ``maxlen`` is not an argument here. It is
a bound on the payload's **wire byte length** and must be judged at the length
word, before a payload byte is read — so that a message truncated right after
that word is INVALID rather than INCOMPLETE (MESSAGE_SPEC §5.2), and so that a
``str`` is never measured in code points. The codec owns that check: a visitor
declares the bound from :meth:`~sofab.Visitor.on_schema_bound`, and the decoder
applies it at the length word.

Where to call
-------------

A leaf element is reserved from :meth:`~sofab.Visitor.on_field` — at the
element's **header** — and its value stored from ``on_string`` / ``on_bytes``.
The index verdict therefore lands at the header, where §5.2 wants it: a check in
the value hook would never fire for a payload the message truncates behind, and
INVALID outranks INCOMPLETE. Reserving at the header is safe to repeat (the call
is idempotent), and a reserved slot whose value never arrives exists only in a
decode that has already failed. A framed element or a row is reserved from
:meth:`~sofab.Visitor.on_sequence_begin`. The caller runs its own §7.3
type-mismatch decline **before** the reserve, so a mistyped element at an
over-bound index is skipped rather than refused.

Why functions, not collector objects
------------------------------------

A generated visitor is flat (CORELIB_PLAN §5.3.1): it routes every scope itself
and hands no child visitor back, so a collector *object* would have to be held
and forwarded to. On CPython a plain module function costs less per element than
a bound method on a held instance, and allocates nothing per array opening — so
this layer is three functions, and a flat visitor is its only intended caller.

Two implementations, one contract
---------------------------------

These are the reference definitions, and what the pure engine uses. The compiled
accelerator (``sofab._speedups``) carries a twin of each, and ``sofab``
re-exports the twins whenever the native engine is active — the same selection
``Encoder`` and ``Decoder`` get. The reason is measured: generated code calls one
of these per wrapper-array element, and on ``vehicle_telemetry`` a Python-level
call there cost the native engine +2.5% Ir per decode where the compiled twin
costs nothing measurable. The twins build their refusals with :func:`_refusal`
below, and ``tests/test_collectors.py`` runs every direct case against both.
"""

from __future__ import annotations

from typing import Any, Callable, Final, TypeVar

from .types import SofaArgumentError, SofaDecodeError, SofaError, SofaLimitError

__all__ = [
    "UNBOUNDED",
    "reserve_elem",
    "reserve_leaf",
    "reserve_row",
]

T = TypeVar("T")

#: The ``cap`` a caller passes where the schema declares no ``count`` for the
#: array — the receiver cap ``rcap`` then applies instead.
UNBOUNDED: Final[int] = -1


def reserve_leaf(out: list[T], id: int, default: T, cap: int, rcap: int) -> None:
    """**Reserve a leaf element's slot** — a wrapper array's ``string`` or
    ``blob`` — at the index its wire id names.

    Bounds ``id`` first, then grows ``out`` to at least ``id + 1``, filling every
    new slot with ``default``. A slot already present is left alone; the caller
    stores the value with ``out[id] = value`` once it has arrived.

    ``default`` is **shared, not copied**: ``""`` or ``b""``, immutable, so a gap
    costs no allocation. An element whose default is mutable belongs in
    :func:`reserve_elem`.

    :param out: the destination list, which this grows
    :param id: the element's wire id, which is its index
    :param default: the element default, filling any gap up to ``id``
    :param cap: the array's schema ``count``, or :data:`UNBOUNDED`
    :param rcap: the receiver's ``max_dyn_array_count``, compared only where
        ``cap`` is :data:`UNBOUNDED`
    :raises SofaDecodeError: ``id`` reaches a declared ``cap`` (INVALID, §7.1)
    :raises SofaLimitError: ``id`` reaches ``rcap`` on a schema-unbounded array
    :raises SofaArgumentError: a schema-unbounded array was handed no cap
    """
    if cap >= 0:
        if id >= cap:
            raise _refusal(id, cap, rcap)
    elif rcap.__class__ is not int or id >= rcap:
        raise _refusal(id, cap, rcap)
    while len(out) <= id:
        out.append(default)


def reserve_elem(
    out: list[T], id: int, make: Callable[[], T], cap: int, rcap: int
) -> None:
    """**Reserve a framed element's slot** — a wrapper array's ``struct`` or
    ``union`` element, or a native-integer matrix row.

    Bounds ``id`` first, then grows ``out`` to at least ``id + 1``, giving each
    new slot **its own** ``make()`` — the element class, or ``list`` for a row.
    A shared default would alias every element onto one object, which is the
    single reason this is a second function rather than :func:`reserve_leaf`.

    A slot already present is left alone: a framed element's fields arrive one
    at a time into the object reserved here, so a re-opened element id merges
    into what its earlier fields built (§7.4). ``make`` is never called for a
    refused ``id``.

    Nothing is returned: the caller binds ``id`` in its own element-index
    register and reaches the element as ``out[id]``.

    :param out: the destination list, which this grows
    :param id: the element's wire id, which is its index
    :param make: the element factory, called once per slot this creates
    :param cap: the array's schema ``count``, or :data:`UNBOUNDED`
    :param rcap: the receiver's ``max_dyn_array_count``, compared only where
        ``cap`` is :data:`UNBOUNDED`
    :raises SofaDecodeError: ``id`` reaches a declared ``cap`` (INVALID, §7.1)
    :raises SofaLimitError: ``id`` reaches ``rcap`` on a schema-unbounded array
    :raises SofaArgumentError: a schema-unbounded array was handed no cap
    """
    if cap >= 0:
        if id >= cap:
            raise _refusal(id, cap, rcap)
    elif rcap.__class__ is not int or id >= rcap:
        raise _refusal(id, cap, rcap)
    while len(out) <= id:
        out.append(make())


def reserve_row(rows: list[list[Any]], id: int, cap: int, rcap: int) -> None:
    """**Reserve a matrix row** whose elements are themselves a wrapper array.

    Bounds ``id`` first, fills every gap below it with a fresh empty row, and
    then **rebinds** ``rows[id]`` to a fresh empty list: an array wrapper *is*
    the array's value, so a later occurrence of its id replaces the row whole
    (§7.4) rather than merging into it. Rebinding rather than ``clear()`` leaves
    a list a caller took out of an earlier decode intact.

    The row's own elements are then reserved with :func:`reserve_leaf` or
    :func:`reserve_elem` against ``rows[id]`` and the row's own bound.

    :param rows: the outer list, one entry per row
    :param id: the row's wire id, which is its index
    :param cap: the outer array's schema ``count``, or :data:`UNBOUNDED`
    :param rcap: the receiver's ``max_dyn_array_count``, compared only where
        ``cap`` is :data:`UNBOUNDED`
    :raises SofaDecodeError: ``id`` reaches a declared ``cap`` (INVALID, §7.1)
    :raises SofaLimitError: ``id`` reaches ``rcap`` on a schema-unbounded array
    :raises SofaArgumentError: a schema-unbounded array was handed no cap
    """
    if cap >= 0:
        if id >= cap:
            raise _refusal(id, cap, rcap)
    elif rcap.__class__ is not int or id >= rcap:
        raise _refusal(id, cap, rcap)
    while len(rows) < id:
        rows.append([])
    if len(rows) == id:
        rows.append([])
    else:
        rows[id] = []


def _refusal(id: int, cap: int, rcap: Any) -> SofaError:
    """The refusal a rejected ``id`` has earned, built out of line so the three
    hot bodies above stay one comparison and one ``raise``.

    The categories are not interchangeable. INVALID (``SofaDecodeError``) says
    the message contradicts its schema. ``SofaLimitError`` says *"raise my limit,
    or the sender must send less"*. ``SofaArgumentError`` says the call stated no
    receiver cap at all — the absence of the number §6.2.1 requires, which is
    neither a small limit nor an unlimited one.
    """
    if cap >= 0:
        return SofaDecodeError(
            f"element index {id} exceeds the count {cap} the schema declares"
        )
    if rcap.__class__ is not int or rcap < 0:
        return SofaArgumentError(
            f"max_dyn_array_count not stated ({rcap!r}) for element index {id} "
            "(§6.2.1): the helper holds no limit of its own and reads no "
            "missing cap as unlimited"
        )
    return SofaLimitError(
        f"element index {id} exceeds the receiver limit max_dyn_array_count {rcap}"
    )
