"""The decode surface. CORELIB_PLAN §5.3.1 makes it the only one:

    "A corelib exposes exactly one decode surface: the visitor. The decoder
     calls typed visitor methods on a caller-supplied object; pull-reading
     becomes 'the visitor writes the decoded value into one of the object's own
     members and skips what it does not recognise'."

:class:`Visitor` is a base class whose hooks all default to *no-op* (the value
is still consumed, so an unhandled field is transparently skipped). Subclass it
and override only the fields you care about, then pass it to
:class:`sofab.Decoder` as ``visitor=``.

Two control hooks let a visitor decline work *before* the value is decoded — so
skipping a 10k-element array or a deep sub-tree costs nothing:

* :meth:`Visitor.on_field` — return ``False`` to skip a scalar/fixlen/array
  field instead of decoding it.
* :meth:`Visitor.on_sequence_begin` — return ``False`` to skip the entire
  nested sequence (its matching end is consumed too, so ``on_sequence_end`` is
  *not* called for a skipped sequence).

A third hook carries the one fact the codec cannot know and the schema does:
:meth:`Visitor.on_schema_bound` names the ``count``/``maxlen`` the schema puts
on a field, which is what takes the receiver-side ``max_dyn_*`` cap off it
(§6.2.1) and makes exceeding it ``INVALID`` rather than a policy rejection. It
is told the wire's tag alongside the id, because it is the only hook that spans
more than one kind — every other one fires for a single wire type, so the
decoder has already matched the tag before calling it.

A fourth hook is asked **once**, when the decoder is built, and never again:
:meth:`Visitor.destinations` names the slots the handler wants its fields
written into, as a :class:`sofab.Binding` plus the storage it addresses. It is
the same bargain :meth:`Visitor.on_array_begin`, :meth:`~Visitor.on_string_begin`
and :meth:`~Visitor.on_blob_begin` already strike — name a destination and the
codec writes there instead of calling you back — declared once for the whole
message instead of per field.

Because this is the *only* decode surface (§5.3.1), a table is reached *through*
it and never beside it: there is one handler object, one walk, and one
implementation of every rule, so nothing can be right on one route and wrong on
another.

Which hooks a handler overrides is read off its **type**, once, when the
decoder binds it — so a hook nobody overrides costs nothing per field, and a
child handler returned from :meth:`Visitor.on_sequence_begin` is measured on its
own type rather than its parent's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import Field, FixlenSubtype, WireType

if TYPE_CHECKING:
    from .binding import Binding


class Visitor:
    """Base visitor: override the hooks for the fields you handle.

    Every hook is keyed by the wire type the decoder recovered. ``field_id`` is
    the decoded field id. Unhandled hooks default to a no-op, which still
    consumes the value (so unknown fields are skipped safely)."""

    # --- declared once, at construction -------------------------------------

    def destinations(self) -> tuple[Binding, Any, list[Any] | None] | None:
        """The slots this handler wants its fields written into, or ``None``.

        Asked **once**, when the :class:`sofab.Decoder` is built, and never
        again — so nothing the wire says can change the answer, which is what
        §6.6 asks of a decode's storage. Return
        ``(binding, words, objects)``: a :class:`sofab.Binding` mapping field
        ids to slots, a writable 8-byte-aligned buffer for the scalar and array
        slots, and a list for ``string``/``blob`` slots (or ``None`` when the
        table names none).

        A field the table names is written straight into its slot and **no**
        typed hook fires for it — the same bargain :meth:`on_array_begin`,
        :meth:`on_string_begin` and :meth:`on_blob_begin` strike per field, made
        once for the whole message. A field the table does not name reaches this
        visitor's hooks exactly as it would have without a table, and the
        ``count``/``maxlen`` a table entry declares is answered from the table
        rather than from :meth:`on_schema_bound`.

        This is **not** a second decode surface (§5.3.1). The decoder still
        drives, the walk is the same walk, and every rule — the receiver cap,
        the schema bound, the §7.3 tag test, the UTF-8 check, the declared
        element width, the resume transaction — has one implementation that runs
        for a mapped field and an unmapped one alike. The table says *where* a
        value goes; it never says *how* it is decoded.

        ``Decoder(binding=…, words=…, objects=…)`` is the constructor shorthand
        for a handler that declares exactly this and nothing else.
        """
        return None

    # --- control hooks (return False to skip before decoding) ---------------

    def on_field(self, field: Field) -> bool | None:
        """Called for every non-sequence field before its value is decoded.
        Return ``False`` to skip the value entirely; any other return proceeds
        to decode it and dispatch to the typed hook below."""
        return None

    def on_schema_bound(
        self,
        field_id: int,
        n: int,
        wtype: WireType,
        subtype: FixlenSubtype | None,
    ) -> int:
        """The count or length the **schema** declares for this field, or ``-1``.

        Asked once, at the count/length header — after :meth:`on_field`, before a
        payload byte is read or any storage is written — for a ``string``, a
        ``blob`` or an array this handler has accepted, and for nothing else. A
        scalar carries neither a count nor a length, so none is asked for, and a
        field the handler skipped is never asked (§6.7.2).

        ``n`` is what the **wire** announced: the byte length for a ``string`` or
        ``blob``, the element count for an array.

        ``wtype``/``subtype`` are the **tag the wire carried**, and they are here
        so that a handler can apply MESSAGE_SPEC §7.3 to its own declaration
        before answering. This is the only hook that spans more than one kind —
        :meth:`on_string_begin` fires for a ``string`` and nothing else,
        :meth:`on_array_begin` for an integer array and nothing else, so the
        decoder has already matched the tag for them. Here it has not, and an id
        the schema bounds can arrive under a tag the schema never declared for
        it. §7.3 says such a field is skipped like an unknown id, so **a handler
        must answer** ``-1`` **for a tag it did not declare** — a bound answered
        for someone else's field is a bound applied to a length that was never
        the handler's, and the ``INVALID`` that follows contradicts §7.3. A
        table entry (:meth:`destinations`) gets the same test run for it by the
        decoder, which is why the two routes agree.

        ``subtype`` is the fixlen subtype for a ``string``, a ``blob`` and a
        fixlen array, and ``None`` for an integer array, which carries none.
        Both are enum members recovered by index, not built, so overriding this
        hook still costs no allocation per field.

        Returning ``n >= 0``:

        * a wire count/length above it is ``INVALID`` — :class:`sofab.SofaDecodeError`
          (MESSAGE_SPEC §7.1): the message contradicts the schema;
        * the receiver-side ``max_dyn_*`` cap **stops applying** to the field
          (§6.2.1) — a schema bound is a statement about *validity*, a receiver
          limit about *capacity*.

        Returning ``-1`` (the default) leaves the field to the receiver caps,
        unchanged.

        A handler that declares destinations (:meth:`destinations`) answers this
        from its table for every field the table names; this hook is what the
        rest go through, and both reach the same rule in the same place.

        .. code-block:: python

            def on_schema_bound(self, field_id, n, wtype, subtype):
                if (field_id == 0 and wtype is WireType.FIXLEN
                        and subtype is FixlenSubtype.STRING):
                    return 32          # the schema's maxlen for this field
                return -1              # not the field the schema declared
        """
        return -1

    def on_sequence_begin(self, field_id: int) -> bool | Visitor | None:
        """A nested sequence is opening; nothing inside it has been decoded.

        Three answers:

        ``False``
            skip the whole sub-tree — its end marker is consumed and
            :meth:`on_sequence_end` is not called.
        another :class:`Visitor`
            **descend into it**: every field of that sub-tree goes to the visitor
            returned, its :meth:`on_sequence_end` fires when the scope closes,
            and this visitor resumes afterwards. That is how a wrapper array's
            elements each get a handler of their own (see
            :mod:`sofab.collectors`), and how a generated object hands a nested
            message to the object that models it.
        anything else
            decode the sub-tree into this same visitor, as a flat event stream.

        A sub-tree opens a fresh id scope (§4.9), so the ids inside it mean what
        the nested schema says, not what the enclosing one does.
        """
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
            too short is :class:`sofab.SofaArgumentError`; the decoder never grows
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

        Not called for float arrays, which carry no declared width to state.
        Their destination hook is :meth:`on_float_array_begin`.

        **A configured** ``max_dyn_array_count`` **does not gate this hook**, on
        the same reasoning as :meth:`on_blob_begin`: it is asked first and told
        ``count``, and a ``dst`` it hands back is storage it sized itself. The
        cap governs the list the decoder would otherwise build — the ``None``
        answer — not a destination of the handler's own.
        """
        return None

    def on_blob_begin(self, field_id: int, size: int) -> Any:
        """A blob's length has been read; no payload byte has been copied yet.

        Return ``None`` to take the default — a ``bytes``, handed to
        :meth:`on_bytes`. Return a writable, contiguous buffer of at least
        ``size`` bytes and the decoder copies the payload straight into it and
        does **not** call :meth:`on_bytes`. One too short is
        :class:`sofab.SofaArgumentError`; the decoder never grows one
        (CORELIB_PLAN §6.6), and the refusal comes at the length word, before a
        byte is written.

        This is §6.6.3's second shape for an aggregate: a callback carrying a
        whole blob obliges the codec to build one, and the only size available
        to build it from is the wire's. A megabyte blob costs a megabyte
        allocation per message that way; into a destination it costs none.

        Called again for the same blob if a chunk boundary suspends the copy, so
        return the same answer each time; the decoder restarts the payload from
        its first byte.

        The string twin is :meth:`on_string_begin`.

        **A configured** ``max_dyn_blob_len`` **does not gate this hook.** It is
        asked first, and asked whatever the announced size is. The limit is
        there to stop the *sender* dictating the *receiver's* allocation
        (§6.2.1), and a handler that hands back a buffer has sized that buffer
        itself — there is no allocation of the decoder's left to prevent. It is
        told ``size`` before a byte is copied precisely so that a receiver
        unwilling to take that many can refuse it here, which is its call to
        make. Return ``None`` and the cap applies again, because then the
        ``bytes`` is the decoder's to build and the wire is its only size.
        """
        return None

    def on_string_begin(self, field_id: int, size: int) -> Any:
        """A string's byte length has been read; no payload byte has been
        copied yet, and none has been validated.

        Return ``None`` to take the default — a ``str``, handed to
        :meth:`on_string`. Return a writable, contiguous buffer of at least
        ``size`` **bytes** and the decoder validates the payload as UTF-8, copies
        the wire bytes straight into it, and does **not** call
        :meth:`on_string`. One too short is :class:`sofab.SofaArgumentError`;
        the decoder never grows one (CORELIB_PLAN §6.6), and the refusal comes at
        the length word, before a byte is written.

        ``size`` is the **wire byte length**, which is what a schema ``maxlen``
        bounds (MESSAGE_SPEC §1) — not a character count. What lands in the
        buffer is UTF-8, so a target that wants Python text decodes it itself;
        what this saves is the ``str`` the decoder would otherwise have had to
        build, sized by the wire.

        This is §6.6.3's second shape for the third aggregate, and it is the
        sharpest of the three: with a caller ``reassembly=`` buffer *and* a
        ``Binding``, a 1 MiB string still cost a 1 MiB allocation inside the
        codec, because there was no third opt-out to take.

        **The payload is still validated** (§6.7.2: a field the handler reads is
        materialized *and* validated). Validation walks the bytes — §6.4.3's
        ``utf8_valid`` primitive — so nothing the wire sizes is built to check
        them. Invalid UTF-8 is ``INVALID`` and the destination is left
        untouched.

        Called again for the same string if a chunk boundary suspends the copy,
        so return the same answer each time; the decoder restarts the payload
        from its first byte.

        **A configured** ``max_dyn_string_len`` **does not gate this hook**, on
        the same reasoning as :meth:`on_blob_begin`.
        """
        return None

    def on_float_array_begin(
        self, field_id: int, subtype: FixlenSubtype, count: int
    ) -> Any:
        """A fixlen (``fp32``/``fp64``) array's count has been read; no element
        has been decoded.

        Return ``None`` to take the default — a ``list``, handed to
        :meth:`on_float32_array` / :meth:`on_float64_array` — or a writable
        buffer of at least ``count`` **8-byte** slots (an ``array("d")``, a
        ``memoryview`` over one, a NumPy ``float64`` array). The decoder widens
        each element into it and does **not** call the typed hook. A buffer too
        short is :class:`sofab.SofaArgumentError`; the decoder never grows one
        (CORELIB_PLAN §6.6).

        ``subtype`` is :attr:`sofab.FixlenSubtype.FP32` or
        :attr:`~sofab.FixlenSubtype.FP64`, so one hook serves both and a handler
        that only wants one returns ``None`` for the other.

        Slots are 8 bytes for both subtypes because a Python ``float`` is a
        double and that is what the values become. A consumer that needs an
        ``fp32``'s **wire bits** intact takes :meth:`on_float32_array_bits`
        instead (§6.5).

        Called again for the same array if a chunk boundary suspends the read,
        so return the same answer each time.

        **A configured** ``max_dyn_array_count`` **does not gate this hook**, on
        the same reasoning as :meth:`on_blob_begin`.
        """
        return None

    # --- typed value hooks --------------------------------------------------

    def on_unsigned(self, field_id: int, value: int) -> None:
        """Handle a decoded unsigned-integer field."""

    def on_signed(self, field_id: int, value: int) -> None:
        """Handle a decoded signed-integer field."""

    def on_float32(self, field_id: int, value: float) -> None:
        """Handle a decoded 32-bit float field.

        The value is a Python ``float`` — a C ``double`` — because Python has no
        other float. CORELIB_PLAN §6.5 permits that for a **value** consumer;
        a consumer that has to reproduce the wire bytes takes
        :meth:`on_float32_bits` instead.
        """

    def on_float32_bits(self, field_id: int, bits: int) -> None:
        """The **raw wire bits** of a 32-bit float field, as an ``int``.

        Override this and the decoder calls it *instead of* :meth:`on_float32`
        for every scalar ``fp32``. ``bits`` is the little-endian payload read as
        an unsigned 32-bit integer, exactly as it lay on the wire, and
        :meth:`sofab.Encoder.write_float32_bits` puts it back verbatim.

        This is CORELIB_PLAN §6.5's required channel for a **double-only**
        target, which Python is. IEEE widening ``fp32`` to a double **sets the
        quiet bit**, so a signaling NaN's payload is destroyed the instant the
        value passes through the wider float — and no later code can recover it.
        A port on such a target therefore "**MUST** provide a raw-wire-bytes
        path for bit-exact consumers (transcode, round-trip, any re-encode) that
        re-emits those bytes **verbatim**" and "**MUST NOT** re-encode an
        ``fp32`` from the widened value".

        (This port *also* preserves an sNaN through the widened ``float``, by
        doing the conversion on the bit pattern by hand rather than letting the
        hardware quiet it — so :meth:`on_float32` is bit-exact here today. That
        is a property of this implementation on this platform; the raw channel
        is the guarantee.)

        The array twin is :meth:`on_float32_array_bits`.
        """

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

    def on_float32_array_bits(self, field_id: int, count: int, payload: Any) -> None:
        """The **raw wire bytes** of a 32-bit float array, undecoded.

        Override this and the decoder calls it *instead of*
        :meth:`on_float32_array`. ``payload`` is a read-only ``memoryview`` of
        exactly ``4 * count`` little-endian bytes — the array's payload as it
        lay on the wire — and :meth:`sofab.Encoder.write_float32_array_bits`
        puts it back verbatim.

        §6.5's requirement is stated over "**every** ``fp32`` position — a
        **scalar** ``fp32`` (§4.6) **and** each element of an ``fp32`` array
        (§4.8)", so the scalar channel alone would not meet it.

        **The bytes do not outlive the call.** They are the caller's own input,
        borrowed for the duration of this callback exactly as a fed chunk is
        (§6, chunk lifetime), and a handler that still needs them afterwards
        copies them. That is §6.7's second route — "the codec passes the value
        through the callback … and the caller copies it. The second route is not
        a view" — and it is why nothing is allocated to deliver an array of any
        length.
        """

    def on_float64_array(self, field_id: int, values: list[float]) -> None:
        """Handle a decoded 64-bit float array field."""
