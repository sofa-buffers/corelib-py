"""§6.2.1: the receiver-side limits are present, finite, and never unset.

    There is no unset state and no unlimited mode. Unbounded by the schema is
    still bounded by the receiver.

A decoder therefore always carries all three of ``max_dyn_array_count``,
``max_dyn_string_len`` and ``max_dyn_blob_len``, and ``None`` is refused rather
than read as "no limit". The defaults are the format-wide ceilings of §6.2, above
which the value is already INVALID — the widest a limit can be while still being
one, and not a policy the codec invented.

What the limits *do* once set is the subject of ``test_schema_bounded.py``; this
file is about their existence and their domain.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder

from sofab import ARRAY_MAX, FIXLEN_MAX, Encoder, SofaLimitError, SofaRangeError

NAMES = ("max_dyn_array_count", "max_dyn_string_len", "max_dyn_blob_len")
CEILINGS = {
    "max_dyn_array_count": ARRAY_MAX,
    "max_dyn_string_len": FIXLEN_MAX,
    "max_dyn_blob_len": FIXLEN_MAX,
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_none_is_refused_there_is_no_unset_state(engine, name):
    with pytest.raises(SofaRangeError):
        engine(visitor=Recorder(), **{name: None})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_a_limit_outside_its_domain_is_refused(engine, name):
    for bad in (-1, CEILINGS[name] + 1):
        with pytest.raises(SofaRangeError):
            engine(visitor=Recorder(), **{name: bad})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_the_ceiling_itself_is_accepted(engine, name):
    engine(visitor=Recorder(), **{name: CEILINGS[name]})


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", NAMES)
def test_zero_is_a_limit_like_any_other(engine, name):
    """Zero is a real setting -- 'accept nothing unbounded' -- not an unset
    state wearing a different value."""
    engine(visitor=Recorder(), **{name: 0})


@pytest.mark.parametrize("engine", ENGINES)
def test_a_zero_limit_rejects_rather_than_clamps(engine):
    """§6.2.1 'rejected, never clamped'."""
    enc = Encoder()
    enc.write_string(1, "x")
    enc.flush()
    dec = engine(visitor=Recorder(), max_dyn_string_len=0)
    with pytest.raises(SofaLimitError):
        dec.feed(enc.getvalue())


@pytest.mark.parametrize("engine", ENGINES)
def test_the_defaults_admit_everything_the_format_admits(engine):
    """At the ceiling a limit cannot fire: a longer value is INVALID first, so
    the default configuration rejects nothing a looser one would accept."""
    enc = Encoder()
    enc.write_string(1, "z" * 4096)
    enc.write_bytes(2, b"b" * 4096)
    enc.write_unsigned_array(3, list(range(4096)))
    enc.flush()
    rec = Recorder()
    dec = engine(visitor=rec)
    dec.feed(enc.getvalue())
    assert [e[0] for e in rec.events] == ["str", "blob", "ua"]
