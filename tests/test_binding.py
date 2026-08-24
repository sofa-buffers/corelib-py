"""Bound destinations: decoding straight into caller-owned storage.

:class:`sofab.Binding` is the fast path behind :meth:`sofab.Decoder.feed` — the
handler declares once where each field belongs and the decoder writes there
itself. These tests are about what that has to *mean*, which is more than
"faster": the caller owns and sizes the storage (§6.6), a declared bound is the
schema's (§7.1, §6.2.1), a contradicting wire tag is skipped rather than
rejected (§7.3), a sequence opens a fresh id scope (§4.9), and the two engines
agree on all of it.
"""

from __future__ import annotations

import math
import struct

import pytest
from vectors import DECODER_ENGINES, ENGINE_PAIRS

from sofab import Binding, SofaArgumentError, Status


def storage(b: Binding):
    """The caller's storage, sized from the table — never from the wire."""
    words = bytearray(b.tree_words_required * 8)
    objects = [None] * b.tree_objects_required
    mv = memoryview(words)
    return words, objects, mv.cast("Q"), mv.cast("q"), mv.cast("d")


# --- the destinations --------------------------------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_every_kind_lands_in_its_slot(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_unsigned(1, (1 << 64) - 1)
    enc.write_signed(2, -(1 << 63))
    enc.write_float32(3, 1.5)
    enc.write_float64(4, -2.25)
    enc.write_string(5, "grüß dich")
    enc.write_bytes(6, b"\x00\xff\x10")
    enc.write_unsigned_array(7, [1, 2, 300])
    enc.write_signed_array(8, [-1, 2, -3])
    enc.write_float32_array(9, [1.5, -2.5])
    enc.write_float64_array(10, [1e300, -0.5])
    enc.flush()

    b = (
        Binding()
        .unsigned(1, at=0)
        .signed(2, at=1)
        .float32(3, at=2)
        .float64(4, at=3)
        .string(5, at=0)
        .bytes(6, at=1)
        .unsigned_array(7, at=8, cap=4, count_at=4)
        .signed_array(8, at=16, cap=4, count_at=5)
        .float32_array(9, at=24, cap=4, count_at=6)
        .float64_array(10, at=32, cap=4, count_at=7)
    )
    words, objects, u, q, f = storage(b)
    assert dec_cls(binding=b, words=words, objects=objects).feed(enc.getvalue()) is (
        Status.COMPLETE
    )

    assert u[0] == (1 << 64) - 1
    assert q[1] == -(1 << 63)
    assert f[2] == 1.5
    assert f[3] == -2.25
    assert objects[0] == "grüß dich"
    assert objects[1] == b"\x00\xff\x10"
    assert list(u[8 : 8 + u[4]]) == [1, 2, 300]
    assert list(q[16 : 16 + u[5]]) == [-1, 2, -3]
    assert list(f[24 : 24 + u[6]]) == [1.5, -2.5]
    assert list(f[32 : 32 + u[7]]) == [1e300, -0.5]


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_fp32_round_trips_bit_exactly_including_nan(enc_cls, dec_cls):
    """§6.5: a float must survive bit-for-bit, and widening an fp32 into the
    slot must not be the thing that loses it."""
    payload = struct.unpack("<f", b"\x01\x00\xc0\x7f")[0]  # a signaling NaN
    enc = enc_cls()
    enc.write_float32(1, payload)
    enc.flush()
    b = Binding().float32(1, at=0)
    words, objects, _u, _q, f = storage(b)
    dec_cls(binding=b, words=words).feed(enc.getvalue())
    assert math.isnan(f[0])
    assert struct.pack("<f", f[0]) == b"\x01\x00\xc0\x7f"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_an_absent_field_leaves_its_slot_untouched(enc_cls, dec_cls):
    """No sentinel is invented: absence is "the caller's value is still there",
    which is why count_at exists."""
    enc = enc_cls()
    enc.write_unsigned(1, 5)
    enc.flush()
    b = Binding().unsigned(1, at=0, count_at=2).unsigned(9, at=1, count_at=3)
    words, objects, u, _q, _f = storage(b)
    u[1] = 0xDEAD
    dec_cls(binding=b, words=words).feed(enc.getvalue())
    assert u[0] == 5 and u[2] == 1
    assert u[1] == 0xDEAD and u[3] == 0


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_count_slots_report_arrival(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_unsigned_array(1, [7, 8])
    enc.flush()
    b = Binding().unsigned_array(1, at=0, cap=4, count_at=8)
    words, objects, u, _q, _f = storage(b)
    dec_cls(binding=b, words=words).feed(enc.getvalue())
    assert u[8] == 2


# --- the caller owns the size (§6.6 / §7.1) ---------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_an_array_past_the_declared_cap_is_invalid(enc_cls, dec_cls):
    """The destination's size is the schema's bound, so a longer array is a
    malformed message (§7.1) — and it is rejected at the count header, before
    any element is read."""
    enc = enc_cls()
    enc.write_unsigned_array(1, [1, 2, 3, 4, 5])
    enc.flush()
    b = Binding().unsigned_array(1, at=0, cap=3, count_at=8)
    words, objects, u, _q, _f = storage(b)
    dec = dec_cls(binding=b, words=words)
    assert dec.feed(enc.getvalue()) is Status.INVALID
    assert u[0] == 0, "nothing may be written before the verdict"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_string_past_its_declared_maxlen_is_invalid(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_string(1, "far too long")
    enc.flush()
    b = Binding().string(1, at=0, maxlen=4)
    words, objects, *_ = storage(b)
    dec = dec_cls(binding=b, words=words, objects=objects)
    assert dec.feed(enc.getvalue()) is Status.INVALID
    assert objects[0] is None


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_declared_element_width_is_checked_at_the_element(enc_cls, dec_cls):
    """§7.1: an element outside the schema's declared width is INVALID, and the
    verdict is reached at that element — not after the array completes."""
    enc = enc_cls()
    enc.write_unsigned_array(1, [1, 2, 999])
    enc.flush()
    b = Binding().unsigned_array(1, at=0, cap=8, elem_max=255)
    words, objects, _u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words).feed(enc.getvalue()) is Status.INVALID

    enc = enc_cls()
    enc.write_signed_array(1, [-1, -200])
    enc.flush()
    b2 = Binding().signed_array(1, at=0, cap=8, elem_min=-128, elem_max=127)
    words2, _o, _u, _q, _f = storage(b2)
    assert dec_cls(binding=b2, words=words2).feed(enc.getvalue()) is Status.INVALID


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_storage_must_be_big_enough_and_writable(dec_cls):
    b = Binding().unsigned_array(1, at=0, cap=4).string(2, at=0)
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, words=bytearray(8), objects=[None])
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, words=bytearray(b.tree_words_required * 8), objects=[])
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, words=bytes(b.tree_words_required * 8), objects=[None])
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, words=bytearray(b.tree_words_required * 8 + 3), objects=[None])
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, objects=[None])


# --- §7.3: a contradicting tag is skipped, not rejected ----------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_wrong_wire_tag_is_skipped_and_the_decode_stays_complete(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_string(1, "not a number")
    enc.write_unsigned(2, 9)
    enc.flush()
    b = Binding().unsigned(1, at=0, count_at=2).unsigned(2, at=1, count_at=3)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words).feed(enc.getvalue()) is Status.COMPLETE
    assert u[0] == 0 and u[2] == 0, "the mismatched field is untouched"
    assert u[1] == 9 and u[3] == 1, "and the walk carries on"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_wrong_fixlen_subtype_is_skipped_too(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_string(1, "text")
    enc.write_unsigned(2, 4)
    enc.flush()
    b = Binding().bytes(1, at=0).unsigned(2, at=0, count_at=1)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words, objects=objects).feed(
        enc.getvalue()
    ) is Status.COMPLETE
    assert objects[0] is None
    assert u[0] == 4


# --- sequences (§4.9) --------------------------------------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_bound_sequence_decodes_into_the_same_storage(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_unsigned(1, 11)
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(1, 22)
    enc.write_string(2, "inner")
    enc.write_sequence_end()
    enc.flush()

    child = Binding().unsigned(1, at=4, count_at=5).string(2, at=0, count_at=6)
    b = Binding().unsigned(1, at=0, count_at=1).sequence(2, child, count_at=2)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words, objects=objects).feed(
        enc.getvalue()
    ) is Status.COMPLETE
    assert u[0] == 11 and u[1] == 1
    assert u[2] == 1, "the sequence occurred once"
    assert u[4] == 22 and u[5] == 1
    assert objects[0] == "inner"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_sequence_opens_a_fresh_id_scope(enc_cls, dec_cls):
    """§4.9. Field 1 inside the sequence must not land in the outer field 1's
    slot — which is exactly what a table that layered instead of replacing
    would do."""
    enc = enc_cls()
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(1, 99)
    enc.write_sequence_end()
    enc.flush()
    child = Binding().unsigned(1, at=4)
    b = Binding().unsigned(1, at=0, count_at=1).sequence(2, child)
    words, objects, u, _q, _f = storage(b)
    dec_cls(binding=b, words=words).feed(enc.getvalue())
    assert u[0] == 0 and u[1] == 0
    assert u[4] == 99


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_an_unbound_sequence_is_skipped_whole(enc_cls, dec_cls):
    enc = enc_cls()
    enc.write_sequence_begin_lazy(7)
    enc.write_unsigned(1, 5)
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(1, 6)
    enc.write_sequence_end()
    enc.write_sequence_end()
    enc.write_unsigned(1, 8)
    enc.flush()
    b = Binding().unsigned(1, at=0, count_at=1)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words).feed(enc.getvalue()) is Status.COMPLETE
    assert u[0] == 8 and u[1] == 1, "only the outer field 1 was bound"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
@pytest.mark.parametrize("chunk", [1, 3])
def test_a_bound_decode_survives_any_chunking(enc_cls, dec_cls, chunk):
    enc = enc_cls()
    enc.write_unsigned(1, 300)
    enc.write_string(2, "a payload long enough to straddle chunks")
    enc.write_unsigned_array(3, list(range(20)))
    enc.write_sequence_begin_lazy(4)
    enc.write_float64(1, 2.5)
    enc.write_sequence_end()
    enc.flush()
    msg = enc.getvalue()

    child = Binding().float64(1, at=4)
    b = (
        Binding()
        .unsigned(1, at=0)
        .string(2, at=0, count_at=1)
        .unsigned_array(3, at=8, cap=32, count_at=2)
        .sequence(4, child)
    )
    words, objects, u, _q, f = storage(b)
    dec = dec_cls(binding=b, words=words, objects=objects)
    st = None
    for off in range(0, len(msg), chunk):
        st = dec.feed(msg[off : off + chunk])
    assert st is Status.COMPLETE
    assert u[0] == 300
    assert objects[0] == "a payload long enough to straddle chunks"
    assert list(u[8 : 8 + u[2]]) == list(range(20))
    assert f[4] == 2.5


# --- the visitor is the fallback, not a replacement --------------------------


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_unbound_fields_go_to_the_visitor(enc_cls, dec_cls):
    from test_push_feed import Collect

    enc = enc_cls()
    enc.write_unsigned(1, 5)
    enc.write_unsigned(9, 6)
    enc.flush()
    b = Binding().unsigned(1, at=0)
    words, objects, u, _q, _f = storage(b)
    v = Collect()
    assert dec_cls(binding=b, visitor=v, words=words).feed(
        enc.getvalue()
    ) is Status.COMPLETE
    assert u[0] == 5
    assert v.events == [("u", 9, 6)]


# --- the table is build-once -------------------------------------------------


def test_a_binding_freezes_when_a_decoder_takes_it():
    child = Binding().unsigned(1, at=1)
    b = Binding().sequence(2, child)
    assert b.tree_words_required == 2
    with pytest.raises(SofaArgumentError):
        b.unsigned(3, at=0)
    with pytest.raises(SofaArgumentError):
        child.unsigned(4, at=9), "freezing the root must close the whole tree"


def test_binding_rejects_nonsense_at_bind_time():
    with pytest.raises(SofaArgumentError):
        Binding().unsigned(-1, at=0)
    with pytest.raises(SofaArgumentError):
        Binding().unsigned(1 << 40, at=0)
    with pytest.raises(SofaArgumentError):
        Binding().unsigned(1, at=-1)
    with pytest.raises(SofaArgumentError):
        Binding().unsigned(1, at=0).signed(1, at=1)
    with pytest.raises(SofaArgumentError):
        Binding().sequence(1, "not a binding")  # type: ignore[arg-type]
    with pytest.raises(SofaArgumentError):
        Binding().unsigned("one", at=0)  # type: ignore[arg-type]


def test_binding_reports_the_storage_it_needs():
    b = Binding().unsigned(1, at=0).unsigned_array(2, at=4, cap=6, count_at=20)
    assert b.words_required == 21
    assert b.objects_required == 0
    b2 = Binding().string(1, at=3)
    assert b2.objects_required == 4

    child = Binding().unsigned(1, at=99)
    parent = Binding().sequence(1, child)
    assert parent.words_required == 0
    assert parent.tree_words_required == 100


def test_a_recursive_binding_terminates():
    """A schema may legitimately nest a message inside itself."""
    b = Binding().unsigned(1, at=0)
    b.sequence(2, b)
    assert b.tree_words_required == 1
    assert len(b.freeze()) == 1


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_one_binding_serves_many_decoders(enc_cls, dec_cls):
    """The compiled table is cached on the Binding, so two decoders over
    different storage must still decode independently."""
    enc = enc_cls()
    enc.write_unsigned(1, 7)
    enc.flush()
    msg = enc.getvalue()
    b = Binding().unsigned(1, at=0, count_at=1)
    w1, o1, u1, _q1, _f1 = storage(b)
    w2, o2, u2, _q2, _f2 = storage(b)
    dec_cls(binding=b, words=w1).feed(msg)
    assert u1[0] == 7 and u2[0] == 0
    dec_cls(binding=b, words=w2).feed(msg)
    assert u2[0] == 7


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_sparse_ids_decode_the_same_as_dense_ones(enc_cls, dec_cls):
    """Above a threshold the lookup falls back from a direct index to a scan.
    Both must find the same field."""
    far = 1 << 20
    enc = enc_cls()
    enc.write_unsigned(1, 11)
    enc.write_unsigned(far, 22)
    enc.flush()
    b = Binding().unsigned(1, at=0).unsigned(far, at=1)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words).feed(enc.getvalue()) is Status.COMPLETE
    assert u[0] == 11 and u[1] == 22


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_an_undeclared_string_is_still_subject_to_the_receiver_cap(enc_cls, dec_cls):
    """§6.2.1 cuts both ways: binding a string *without* a maxlen declares no
    schema bound, so the configured cap still governs it — and a bound read must
    not be the way around that."""
    from sofab import SofaLimitError

    enc = enc_cls()
    enc.write_string(1, "0123456789")
    enc.flush()
    b = Binding().string(1, at=0)
    words, objects, *_ = storage(b)
    dec = dec_cls(binding=b, words=words, objects=objects, max_dyn_string_len=4)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())
    assert objects[0] is None


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_declared_maxlen_lifts_the_receiver_cap(enc_cls, dec_cls):
    """And declaring one does take the cap off the field, which is the whole
    point of §6.2.1."""
    enc = enc_cls()
    enc.write_string(1, "0123456789")
    enc.flush()
    b = Binding().string(1, at=0, maxlen=64)
    words, objects, *_ = storage(b)
    dec = dec_cls(binding=b, words=words, objects=objects, max_dyn_string_len=4)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert objects[0] == "0123456789"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_limit_rejection_stays_rejected(enc_cls, dec_cls):
    """§6.3: terminal. The decoder must not shrug it off and carry on.

    Field 1 is bound and declares no ``maxlen``, so the configured cap governs
    it and the ``str`` the decoder would build -- sized by the wire -- is the
    allocation §6.2.1 refuses.
    """
    from sofab import SofaLimitError

    enc = enc_cls()
    enc.write_string(1, "x" * 64)
    enc.write_unsigned(2, 7)
    enc.flush()
    b = Binding().string(1, at=0).unsigned(2, at=0, count_at=1)
    words, objects, u, _q, _f = storage(b)
    dec = dec_cls(binding=b, words=words, objects=objects, max_dyn_string_len=2)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())
    with pytest.raises(SofaLimitError):
        dec.feed(b"")
    assert u[1] == 0, "nothing past the rejection may be decoded"


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_a_field_the_table_does_not_name_is_skipped_uncapped(enc_cls, dec_cls):
    """#128: a cap does not apply to a field the handler never materializes.

    Field 1 is not in the table, so it is skipped -- and §6.2.1 puts the limit
    "before the allocation it is meant to prevent", of which a skip makes none.
    The message is well formed, so it decodes; the port that rejected it here
    was the only one in the family that did.
    """
    enc = enc_cls()
    enc.write_unsigned_array(1, [1, 2, 3, 4, 5])
    enc.write_bytes(3, b"z" * 64)
    enc.write_unsigned(2, 7)
    enc.flush()
    b = Binding().unsigned(2, at=0, count_at=1)
    words, objects, u, _q, _f = storage(b)
    dec = dec_cls(
        binding=b,
        words=words,
        max_dyn_array_count=2,
        max_dyn_blob_len=2,
    )
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert dec.error is None
    assert u[0] == 7 and u[1] == 1


# --- the table's own surface -------------------------------------------------


def test_binding_exposes_its_rows():
    b = Binding().unsigned(1, at=0).signed(2, at=1)
    assert len(b) == 2
    assert [e.field_id for e in b.entries] == [1, 2]
    assert "2 fields" in repr(b)


@pytest.mark.parametrize(("enc_cls", "dec_cls"), ENGINE_PAIRS)
def test_boolean_binds_as_the_unsigned_it_is_on_the_wire(enc_cls, dec_cls):
    """§4.4: a boolean has no wire type. The slot gets the 0/1 the sender wrote
    and the caller tests it for truth."""
    enc = enc_cls()
    enc.write_bool(1, True)
    enc.write_bool(2, False)
    enc.flush()
    b = Binding().boolean(1, at=0, count_at=2).boolean(2, at=1, count_at=3)
    words, objects, u, _q, _f = storage(b)
    assert dec_cls(binding=b, words=words).feed(enc.getvalue()) is Status.COMPLETE
    assert (u[0], u[2]) == (1, 1)
    assert (u[1], u[3]) == (0, 1)


def test_a_frozen_child_cannot_be_bound_into_a_new_tree():
    """Reaching a closed table through a fresh parent would extend it by the
    back door."""
    child = Binding().unsigned(1, at=0)
    Binding().sequence(2, child).freeze()
    with pytest.raises(SofaArgumentError):
        Binding().sequence(3, child)


def test_binding_rejects_out_of_range_sizes():
    with pytest.raises(SofaArgumentError):
        Binding().unsigned_array(1, at=0, cap=1 << 40)
    with pytest.raises(SofaArgumentError):
        Binding().string(1, at=0, maxlen=1 << 40)
    with pytest.raises(SofaArgumentError):
        Binding().unsigned(1, at=0, count_at=-1)
    with pytest.raises(SofaArgumentError):
        Binding().unsigned_array(1, at=0, cap=4, elem_max=1 << 70)
    with pytest.raises(SofaArgumentError):
        Binding().signed_array(1, at=0, cap=4, elem_min=-(1 << 70))


@pytest.mark.parametrize("dec_cls", DECODER_ENGINES)
def test_a_binding_with_object_fields_needs_an_objects_list(dec_cls):
    b = Binding().string(1, at=0)
    with pytest.raises(SofaArgumentError):
        dec_cls(binding=b, words=bytearray(b.tree_words_required * 8 or 8))
