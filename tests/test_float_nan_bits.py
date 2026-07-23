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

import pytest
from vectors import reader

from sofab import Decoder, Encoder, FixlenSubtype, WireType

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
    dec = Decoder(reader(_f32_frame(hexbits)))
    dec.next()
    out = Encoder()
    out.write_float32(7, dec.float32())
    return out.getvalue()[-4:].hex()


def _f64_encode_hex(hexbits: str) -> str:
    """decode ``hexbits`` (8 LE bytes) -> float -> re-encode; return payload hex."""
    dec = Decoder(reader(_f64_frame(hexbits)))
    dec.next()
    out = Encoder()
    out.write_float64(3, dec.float64())
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

    dec = Decoder(reader(frame))
    dec.next()
    got = dec.read_float32_array()

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
    """Structural decode -> re-encode of a whole message (mirrors the harness)."""
    dec = Decoder(reader(wire))
    enc = Encoder()

    def copy() -> None:
        while True:
            f = dec.next()
            if f is None:
                return
            t = f.type
            if t == WireType.SEQUENCE_START:
                enc.write_sequence_begin(f.id)
                copy()
            elif t == WireType.SEQUENCE_END:
                enc.write_sequence_end()
                return
            elif t == WireType.UNSIGNED:
                enc.write_unsigned(f.id, dec.unsigned())
            elif t == WireType.SIGNED:
                enc.write_signed(f.id, dec.signed())
            elif t == WireType.FIXLEN:
                st = f.subtype
                if st == FixlenSubtype.FP32:
                    enc.write_float32(f.id, dec.float32())
                elif st == FixlenSubtype.FP64:
                    enc.write_float64(f.id, dec.float64())
                elif st == FixlenSubtype.STRING:
                    enc.write_string(f.id, dec.string())
                else:
                    enc.write_bytes(f.id, dec.bytes())
            elif t == WireType.ARRAY_UNSIGNED:
                enc.write_unsigned_array(f.id, dec.read_unsigned_array())
            elif t == WireType.ARRAY_SIGNED:
                enc.write_signed_array(f.id, dec.read_signed_array())
            else:  # ARRAY_FIXLEN
                if f.subtype == FixlenSubtype.FP32:
                    enc.write_float32_array(f.id, dec.read_float32_array())
                else:
                    enc.write_float64_array(f.id, dec.read_float64_array())

    copy()
    return enc.getvalue()
