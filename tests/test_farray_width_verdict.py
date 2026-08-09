"""A fixlen array's element width is decided once, at its ``fixlen_word``.

§4.8 fixes the element width of a fixlen array to its subtype (fp32 → 4, fp64 →
8), and §5.2 wants that INVALID verdict reached at the header — before any
payload byte — so it outranks the INCOMPLETE a truncated payload would raise.
``next()`` is therefore the single decision point, and the property below is
what the rest of the decoder may rely on: **whatever ``next()`` returns for a
fixlen array already carries the subtype's exact width**.

That makes a second, post-read width check on the typed read path unreachable
code (issue #75), so these tests pin both halves: the verdict really is complete
at the header for every width a wire can claim, and neither engine keeps a
re-check behind it that no input can reach.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from vectors import DECODER_ENGINES as ENGINES

import sofab
from sofab.types import FixlenSubtype, SofaDecodeError, SofaIncompleteError, WireType

# The two fixlen-array element kinds and the only width each may declare (§4.8).
SUBTYPES = [
    pytest.param(FixlenSubtype.FP32, 4, "<f", id="fp32"),
    pytest.param(FixlenSubtype.FP64, 8, "<d", id="fp64"),
]

_HDR = bytes([(1 << 3) | WireType.ARRAY_FIXLEN])


def _farray(subtype: int, elem_size: int, count: int, payload: bytes = b"") -> bytes:
    """A fixlen-array field: header, count, ``fixlen_word``, then the payload."""
    assert count < 0x80 and elem_size < 0x10  # single-byte varints suffice here
    return _HDR + bytes([count, (elem_size << 3) | int(subtype)]) + payload


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("subtype,width,fmt", SUBTYPES)
@pytest.mark.parametrize("elem_size", list(range(0, 16)))
def test_next_settles_the_element_width_for_every_declared_size(
    engine, subtype, width, fmt, elem_size
):
    """Sweep every element width a one-byte ``fixlen_word`` can carry.

    Either ``next()`` rejects it, or the field it returns declares exactly the
    subtype's width — there is no third outcome for a typed read to clean up
    afterwards. The payload matches what the *wire* claims, so a mismatch is
    never merely "too few bytes": the wrong width has to be caught on its own.
    """
    data = _farray(subtype, elem_size, 1, b"\x00" * elem_size)
    dec = engine(io.BytesIO(data))
    if elem_size != width:
        with pytest.raises(SofaDecodeError) as exc:
            dec.next()
        # INVALID, not "need more bytes" — the whole payload is present here.
        assert not isinstance(exc.value, SofaIncompleteError)
        return
    field = dec.next()
    assert field is not None and field.size == width and field.subtype == subtype
    read = dec.read_float32_array if width == 4 else dec.read_float64_array
    assert read() == [0.0]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("subtype,width,fmt", SUBTYPES)
@pytest.mark.parametrize("elem_size", [0, 1, 3, 5, 8, 12])
def test_a_wrong_width_is_rejected_before_any_payload_arrives(
    engine, subtype, width, fmt, elem_size
):
    """Same verdict with nothing behind the header at all (§5.2 precedence)."""
    if elem_size == width:
        pytest.skip("that width is the legal one for this subtype")
    dec = engine(io.BytesIO(_farray(subtype, elem_size, 3)))
    with pytest.raises(SofaDecodeError) as exc:
        dec.next()
    assert not isinstance(exc.value, SofaIncompleteError)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("subtype,width,fmt", SUBTYPES)
def test_an_empty_array_still_declares_its_width(engine, subtype, width, fmt):
    """§4.8: a zero-count array carries its ``fixlen_word`` too, so its width is
    decided at the header like any other — and the empty read that follows has
    no payload to disagree with."""
    dec = engine(io.BytesIO(_farray(subtype, width, 0)))
    field = dec.next()
    assert field is not None and field.size == width and field.count == 0
    read = dec.read_float32_array if width == 4 else dec.read_float64_array
    assert read() == []
    # ... and a zero-count array with a wrong width is rejected all the same.
    dec = engine(io.BytesIO(_farray(subtype, width + 1, 0)))
    with pytest.raises(SofaDecodeError):
        dec.next()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("subtype,width,fmt", SUBTYPES)
def test_a_correct_width_with_a_short_payload_is_incomplete(engine, subtype, width, fmt):
    """Control: with the width right, a missing payload is INCOMPLETE — the read
    path decides *arrival*, and nothing else."""
    dec = engine(io.BytesIO(_farray(subtype, width, 2, b"\x00" * (width - 1))))
    assert dec.next() is not None
    read = dec.read_float32_array if width == 4 else dec.read_float64_array
    with pytest.raises(SofaIncompleteError):
        read()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("subtype,width,fmt", SUBTYPES)
def test_values_survive_the_single_check(engine, subtype, width, fmt):
    values = [1.0, -2.5, 0.0]
    payload = b"".join(struct.pack(fmt, v) for v in values)
    dec = engine(io.BytesIO(_farray(subtype, width, len(values), payload)))
    assert dec.next() is not None
    read = dec.read_float32_array if width == 4 else dec.read_float64_array
    assert read() == values


def _sources() -> list[tuple[str, str]]:
    """The two engines' sources: the pure decoder and the accelerator's ``.pyx``
    (shipped as package data, so it is readable from an installed package too)."""
    out = []
    for name in ("decoder.py", "_speedups.pyx"):
        path = Path(sofab.__file__).with_name(name)
        if path.exists():
            out.append((name, path.read_text(encoding="utf-8")))
    return out


def test_no_engine_keeps_an_unreachable_width_recheck():
    """No typed read may re-decide the element width after the payload is in.

    ``next()`` has already bound it to the subtype (the sweep above proves it for
    every declared size), the payload read returns exactly the bytes the
    ``fixlen_word`` claims or raises, so such a branch can never be taken — it is
    dead weight on the hot path and a second place for §4.8 to be spelled out
    (issue #75).
    """
    sources = _sources()
    assert [name for name, _ in sources], "no engine source found to check"
    dead = "element width does not match"
    offenders = [name for name, src in sources if dead in src]
    assert not offenders, f"unreachable element-width re-check still present in {offenders}"
