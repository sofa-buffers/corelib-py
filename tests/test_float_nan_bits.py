"""fp32/fp64 NaN payloads round-trip bit-for-bit (issue #49, Crucible F-0031).

CORELIB_PLAN / MESSAGE_SPEC §4.6: float payloads are stored as raw IEEE-754
little-endian bytes, so every value — including ``±0``, ``±inf`` and ``NaN`` —
round-trips **bit-for-bit**. The corelib never inspects or normalizes the value;
there is no signaling-NaN carve-out, so a *signaling* NaN must survive like any
other payload.

The regression: an fp32 value is carried through a Python ``float`` (a C
``double``), and a hardware fp32<->fp64 conversion *quiets* a signaling NaN — it
sets the mantissa is-quiet bit, turning ``0x7F800001`` into ``0x7FC00001``. These
tests pin the exact wire bytes so any re-normalization is caught, on whichever
engine (native/pure) is active.
"""

from __future__ import annotations

import struct

import pytest
from vectors import DECODER_ENGINES as DECODERS
from vectors import ENCODER_ENGINES as ENCODERS
from vectors import Status, values

from sofab import Decoder, Encoder, SofaArgumentError, Visitor
from sofab._core import _unpack_f32_bits, unpack_f32

# We must feed the corelib the *literal* payload bytes: struct.unpack("<f", ...)
# would itself quiet a signaling NaN before the library ever sees it (that
# widening is exactly the bug). So we build a real single-field frame and splice
# the raw payload into its trailing bytes (the fixlen payload is last on the
# wire), then let the corelib's own decoder recover the value.


def _f32_frame(hexbits: str) -> bytes:
    enc = Encoder()
    enc.write_float32(7, 0.0)
    return enc.getvalue()[:-4] + bytes.fromhex(hexbits)


def _f64_frame(hexbits: str) -> bytes:
    enc = Encoder()
    enc.write_float64(3, 0.0)
    return enc.getvalue()[:-8] + bytes.fromhex(hexbits)


def _f32_encode_hex(hexbits: str) -> str:
    """decode ``hexbits`` (4 LE bytes) -> float -> re-encode; return payload hex.

    This is the exact decode -> re-encode round-trip the §4.6 oracle checks.
    """
    out = Encoder()
    out.write_float32(7, values(Decoder, _f32_frame(hexbits))[0][2])
    return out.getvalue()[-4:].hex()


def _f64_encode_hex(hexbits: str) -> str:
    """decode ``hexbits`` (8 LE bytes) -> float -> re-encode; return payload hex."""
    out = Encoder()
    out.write_float64(3, values(Decoder, _f64_frame(hexbits))[0][2])
    return out.getvalue()[-8:].hex()


# fp32 payloads that must survive decode -> re-encode unchanged. The signaling
# NaN is the F-0031 finding; the rest guard the neighbours the fix must not
# disturb (quiet NaN, negative NaN, ±inf, and ordinary values).
FP32_PAYLOADS = {
    "sNaN 0x7F800001": "0100807f",
    "sNaN hi payload 0x7FFFFFFF": "ffff7f7f",
    "qNaN 0x7FC00001": "0100c07f",
    "neg qNaN 0xFFC00000": "0000c0ff",
    "+inf 0x7F800000": "0000807f",
    "-inf 0xFF800000": "000080ff",
    "1.0 0x3F800000": "0000803f",
    "-0.0 0x80000000": "00000080",
}

# fp64 NaNs — the issue reports fp64 signaling NaN already round-trips; pin it so
# it stays that way.
FP64_PAYLOADS = {
    "sNaN 0x7FF0000000000001": "0100000000f0ff7f",
    "qNaN 0x7FF8000000000001": "0100000000f8ff7f",
}


@pytest.mark.parametrize("name,hexbits", list(FP32_PAYLOADS.items()), ids=list(FP32_PAYLOADS))
def test_fp32_payload_roundtrips_bit_for_bit(name, hexbits):
    assert _f32_encode_hex(hexbits) == hexbits, f"{name} was normalized"


@pytest.mark.parametrize("name,hexbits", list(FP64_PAYLOADS.items()), ids=list(FP64_PAYLOADS))
def test_fp64_payload_roundtrips_bit_for_bit(name, hexbits):
    assert _f64_encode_hex(hexbits) == hexbits, f"{name} was normalized"


def test_fp32_signaling_nan_in_fixlen_array():
    # The materialized element walk (raw-bits oracle) must preserve a signaling
    # NaN inside an fp32 array, not just a scalar field. Splice the literal
    # element payloads into a real array frame (see _f32_frame for why).
    payloads = list(FP32_PAYLOADS.values())
    enc = Encoder()
    enc.write_float32_array(2, [0.0] * len(payloads))
    frame = enc.getvalue()[: -4 * len(payloads)] + bytes.fromhex("".join(payloads))

    got = values(Decoder, frame)[0][2]

    out = Encoder()
    out.write_float32_array(2, got)
    reencoded = out.getvalue()[-4 * len(payloads):].hex()
    assert reencoded == "".join(payloads)


def test_f_0031_reproduce_wire_roundtrips():
    # The exact message from issue #49: nested.f32 = 0x7F800001 (wire 01 00 80 7f).
    # A structural decode -> re-encode must reproduce the input byte-for-byte.
    wire = bytes.fromhex("5602200100807f07a606560707c60c07ce0c07")
    assert _reencode_message(wire) == wire


def _reencode_message(wire: bytes) -> bytes:
    """Structural decode -> re-encode of a whole message (mirrors the harness).

    A transcoder must reproduce the input byte-for-byte, and the input carries
    three contentless frames — ``56 07`` (id 10, nested inside the ``a6 06`` …
    ``07`` frame of id 100), ``c6 0c 07`` (id 200) and ``ce 0c 07`` (id 201) — so
    every sequence it copies closes with ``write_sequence_end_keep``: it must
    preserve a frame it was handed, never decide it away. Dropping one is the
    message layer's call (MESSAGE_SPEC §2), made from the values, not the bytes.
    """
    class Transcoder(Visitor):
        """Writes back every field it is handed, in order."""

        def __init__(self) -> None:
            self.enc = Encoder()

        def on_sequence_begin(self, field_id):
            self.enc.write_sequence_begin_lazy(field_id)
            return None

        def on_sequence_end(self):
            self.enc.write_sequence_end_keep()

        def on_unsigned(self, field_id, value):
            self.enc.write_unsigned(field_id, value)

        def on_signed(self, field_id, value):
            self.enc.write_signed(field_id, value)

        def on_float32(self, field_id, value):
            self.enc.write_float32(field_id, value)

        def on_float64(self, field_id, value):
            self.enc.write_float64(field_id, value)

        def on_string(self, field_id, value):
            self.enc.write_string(field_id, value)

        def on_bytes(self, field_id, value):
            self.enc.write_bytes(field_id, value)

        def on_unsigned_array(self, field_id, values_):
            self.enc.write_unsigned_array(field_id, values_)

        def on_signed_array(self, field_id, values_):
            self.enc.write_signed_array(field_id, values_)

        def on_float32_array(self, field_id, values_):
            self.enc.write_float32_array(field_id, values_)

        def on_float64_array(self, field_id, values_):
            self.enc.write_float64_array(field_id, values_)

    t = Transcoder()
    assert Decoder(visitor=t).feed(wire) is Status.COMPLETE
    enc = t.enc
    return enc.getvalue()


def test_fp32_array_mixes_nan_and_ordinary_elements():
    """A NaN and an ordinary value in the *same* fp32 array.

    Only a NaN needs the raw-bits path — every other value converts exactly
    through ``struct``'s own ``f`` code — so the array codec decides per element.
    An array that mixes the two is the case where getting that split wrong shows:
    a NaN handed to the exact path would come back quieted, and an ordinary value
    handed to the bit path must still come back identical.
    """
    payloads = ["0000803f", "0100807f", "000000c0", "ffff7f7f", "00000080"]

    enc = Encoder()
    enc.write_float32_array(4, [0.0] * len(payloads))
    frame = enc.getvalue()[: -4 * len(payloads)] + bytes.fromhex("".join(payloads))

    got = values(Decoder, frame)[0][2]
    assert got[0] == 1.0 and got[2] == -2.0 and got[4] == 0.0

    out = Encoder()
    out.write_float32_array(4, got)
    assert out.getvalue()[-4 * len(payloads):].hex() == "".join(payloads)


# --- narrowing a NaN whose payload fp32 cannot hold --------------------------
#
# An fp32 mantissa is the top 23 bits of the fp64 one, so the low 29 bits are
# dropped by the narrowing `write_float32` performs on a Python float (a C
# double). A NaN whose *whole* payload lives in those bits would narrow to an
# all-zero mantissa — which is no longer a NaN but ±inf. §4.6 allows the payload
# to be lost (fp32 cannot represent it) but not the value's class: a NaN must
# stay a NaN, so the narrowing forces the is-quiet bit back on. Both engines do
# it, and CI runs this file under each.


def _f64_hex(bits: int) -> str:
    return struct.pack("<Q", bits).hex()


#: fp64 bit pattern -> the fp32 payload `write_float32` must produce.
NARROWED_NAN = {
    "sNaN, payload 1 (lost)": (0x7FF0000000000001, "0000c07f"),
    "negative sNaN, payload 1 (lost)": (0xFFF0000000000001, "0000c0ff"),
    "sNaN, all 29 dropped bits set": (0x7FF000001FFFFFFF, "0000c07f"),
    "negative sNaN, all 29 dropped bits set": (0xFFF000001FFFFFFF, "0000c0ff"),
    # Control: the lowest payload bit fp32 *can* hold (bit 29) survives, and the
    # value stays signaling — the rescue above must not fire here.
    "sNaN, lowest surviving payload bit": (0x7FF0000020000000, "0100807f"),
}


@pytest.mark.parametrize("name,case", list(NARROWED_NAN.items()), ids=list(NARROWED_NAN))
def test_nan_narrowed_to_fp32_stays_a_nan(name, case):
    bits, expected = case
    value = values(Decoder, _f64_frame(_f64_hex(bits)))[0][2]
    assert value != value, "the fp64 NaN did not survive the trip through a float"

    out = Encoder()
    out.write_float32(7, value)
    got = out.getvalue()[-4:].hex()
    assert got != "0000807f" and got != "000080ff", f"{name} collapsed to inf"
    assert got == expected, f"{name} narrowed to {got}"


def test_unpack_f32_bits_matches_struct_on_ordinary_values():
    """The raw-bit widening agrees with the ordinary one on every non-NaN value.

    ``_unpack_f32_bits`` exists for NaN payloads (§6.5): the public helpers reach
    its bit path only for a NaN and take ``struct``'s exact conversion otherwise,
    so its own non-NaN fallback is what any future caller that skips that test
    would land in. Pin the two to each other — a bit path that quietly disagreed
    on ordinary values would be a landmine for the next caller — comparing the
    raw doubles, so ``-0.0`` and the infinities are compared by bits and not by
    ``==``.
    """
    for hexbits in ("0000803f", "00000080", "0000807f", "000080ff", "ffff7f7f", "00000000"):
        raw = bytes.fromhex(hexbits)
        via_bits = _unpack_f32_bits(int.from_bytes(raw, "little"))
        assert struct.pack("<d", via_bits) == struct.pack("<d", unpack_f32(raw)), hexbits


# --- §6.5's raw-wire-bytes path ---------------------------------------------
#
# The tests above prove the *outcome*: an fp32 payload survives decode ->
# re-encode. It holds today only because ``_core`` does the width conversion by
# hand on the bit pattern instead of letting the hardware quiet the NaN — a
# property of this implementation on this platform, not a mechanism.
#
# §6.5 requires the mechanism: a double-only target "**MUST** provide a
# **raw-wire-bytes** path for bit-exact consumers (transcode, round-trip, any
# re-encode) that re-emits those bytes **verbatim**" and "**MUST NOT** re-encode
# an ``fp32`` from the widened value", at "**every** ``fp32`` position — a
# **scalar** ``fp32`` (§4.6) **and** each element of an ``fp32`` array (§4.8)".
# These pin that channel.


class _BitTaker(Visitor):
    """A bit-exact consumer: it never sees a ``float``."""

    def __init__(self) -> None:
        self.scalars: list[tuple[int, int]] = []
        self.arrays: list[tuple[int, int, bytes]] = []
        self.values: list[float] = []

    def on_float32(self, field_id, value):  # pragma: no cover - must not fire
        self.values.append(value)

    def on_float32_array(self, field_id, values):  # pragma: no cover
        self.values.extend(values)

    def on_float32_bits(self, field_id, bits):
        self.scalars.append((field_id, bits))

    def on_float32_array_bits(self, field_id, count, payload):
        # The bytes stop being valid when this returns (§6.7), so copy.
        self.arrays.append((field_id, count, bytes(payload)))


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
@pytest.mark.parametrize("name,hexbits", list(FP32_PAYLOADS.items()), ids=list(FP32_PAYLOADS))
def test_a_scalar_fp32_reaches_a_bit_consumer_as_wire_bits(
    name, hexbits, dec_cls, enc_cls
):
    raw = bytes.fromhex(hexbits)
    enc = enc_cls()
    enc.write_float32(7, 0.0)
    enc.flush()
    wire = enc.getvalue()[:-4] + raw

    taker = _BitTaker()
    assert dec_cls(visitor=taker).feed(wire) is Status.COMPLETE
    assert taker.values == [], "the raw channel must replace the value one"
    assert taker.scalars == [(7, int.from_bytes(raw, "little"))]

    # And back out verbatim: no float is constructed anywhere on this path.
    out = enc_cls()
    out.write_float32_bits(7, taker.scalars[0][1])
    out.flush()
    assert out.getvalue() == wire


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_every_element_of_an_fp32_array_reaches_it_as_wire_bits(dec_cls, enc_cls):
    """§6.5 is stated over array positions too, so the scalar channel alone
    would not meet it."""
    payload = b"".join(bytes.fromhex(h) for h in FP32_PAYLOADS.values())
    enc = enc_cls()
    enc.write_float32_array(2, [0.0] * len(FP32_PAYLOADS))
    enc.flush()
    wire = enc.getvalue()[: -len(payload)] + payload

    taker = _BitTaker()
    assert dec_cls(visitor=taker).feed(wire) is Status.COMPLETE
    assert taker.values == []
    assert taker.arrays == [(2, len(FP32_PAYLOADS), payload)]

    out = enc_cls()
    out.write_float32_array_bits(2, payload)
    out.flush()
    assert out.getvalue() == wire


@pytest.mark.parametrize("enc_cls", ENCODERS)
@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_raw_array_payload_survives_a_chunk_boundary(dec_cls, enc_cls):
    """A payload split across fed chunks reaches the handler whole, and the
    handler still gets the same bytes."""
    payload = b"".join(bytes.fromhex(h) for h in FP32_PAYLOADS.values())
    enc = enc_cls()
    enc.write_float32_array(2, [0.0] * len(FP32_PAYLOADS))
    enc.flush()
    wire = enc.getvalue()[: -len(payload)] + payload

    taker = _BitTaker()
    dec = dec_cls(visitor=taker, reassembly=bytearray(256))
    for i in range(0, len(wire), 3):
        dec.feed(wire[i : i + 3])
    assert taker.arrays == [(2, len(FP32_PAYLOADS), payload)]


@pytest.mark.parametrize("dec_cls", DECODERS)
def test_the_raw_array_view_does_not_outlive_the_callback(dec_cls):
    """§6.7: "validity ends when the callback returns". The view is released on
    the way out, so a handler that kept it holds a released one rather than a
    window onto the decoder's buffer."""
    kept: list = []

    class Keeper(Visitor):
        def on_float32_array_bits(self, field_id, count, payload):
            kept.append(payload)

    enc = Encoder()
    enc.write_float32_array(2, [1.0, 2.0])
    enc.flush()
    assert dec_cls(visitor=Keeper()).feed(enc.getvalue()) is Status.COMPLETE
    with pytest.raises(ValueError):
        bytes(kept[0])


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_the_raw_writes_refuse_what_they_cannot_represent(enc_cls):
    """§6.3: a caller mistake is ``InvalidArgument``."""
    enc = enc_cls()
    with pytest.raises(SofaArgumentError):
        enc.write_float32_bits(1, 1 << 32)
    with pytest.raises(SofaArgumentError):
        enc.write_float32_bits(1, -1)
    with pytest.raises(SofaArgumentError):
        enc.write_float32_bits(1, 1.0)
    with pytest.raises(SofaArgumentError):
        enc.write_float32_array_bits(1, b"\x00\x00\x00")
    with pytest.raises(SofaArgumentError):
        enc.write_float32_array_bits(1, b"\x00\x00\x00\x00", count=2)


@pytest.mark.parametrize("enc_cls", ENCODERS)
def test_an_empty_raw_fp32_array_still_carries_its_fixlen_word(enc_cls):
    """§4.8: the ``fixlen_word`` is present even when the array is empty, so an
    empty fp32 and an empty fp64 array stay distinguishable."""
    raw = enc_cls()
    raw.write_float32_array_bits(1, b"")
    raw.flush()
    ordinary = enc_cls()
    ordinary.write_float32_array(1, [])
    ordinary.flush()
    assert raw.getvalue() == ordinary.getvalue()
