"""A struct option's members must start from *their* declared defaults (#169).

MESSAGE_SPEC §7.4.1 says a re-selected union option "starts from its own
default". For a ``struct`` option that default is not one value but its rows':
every member the arriving frame leaves out has to read as the default the schema
declared for *it*.

§4.2 forbids a non-empty ``default`` only for an option **of** type ``string``,
``blob`` or ``array`` — its stated reason being that a reset then "needs no copy
of the caller's storage". A **member** of a struct option carries no such
restriction: the schema's ``string`` default (length ≤ ``maxlen``) and ``array``
default (length ≤ ``items.count``) are ordinary field defaults, and §2's
empty-frame table lists an array field "whose declared default is non-empty" as
a case it has to distinguish.

:meth:`sofab.Binding.freeze` builds the reset plan from each row's ``own_words``
/ ``own_objects``. Those held the *empty* value for a ``string``/``blob`` row and
a count of ``0`` for an array whatever the schema declared, and no binder for
those kinds took a ``default=`` at all — corelib-py#169. They do now: the
prepared value goes into the same flat reset list the scalar default already
used, so nothing on the decode path changed shape.

``TestTheWiring`` holds what was already right, and is what says a failure below
is about the defaults rather than about the plumbing.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Binding, Status, capped, encode

from sofab import SofaArgumentError

# The union field and its two options -- the issue's `cmd`: a `u16` (the
# default_id) and a `struct` whose members carry declared defaults.
UNION_ID = 21
OPT_U16, OPT_CFG = 0, 1
DEFAULT_ID = OPT_U16
# The struct option's members: `name` (string), `vals` (array), `lvl` (scalar).
M_NAME, M_VALS, M_LVL = 0, 1, 2

# words: 0 = which, 1 = u16 option, 2 = lvl, 3 = vals count, 4..7 = vals
W_WHICH, W_U16, W_LVL, W_VCOUNT, W_VALS = 0, 1, 2, 3, 4
O_NAME = 0  # objects

NAME_DEFAULT = "abc"
VALS_DEFAULT = (1, 2)  # two elements of a four-element capacity
LVL_SENT = 7


def _table(defaults: bool) -> Binding:
    """The issue's schema. ``defaults`` declares the members' defaults through
    the API the fix has to add — today those binders take no ``default=``."""
    member = Binding(closed=True)
    if defaults:
        member.string(M_NAME, at=O_NAME, maxlen=8, default=NAME_DEFAULT)
        member.unsigned_array(M_VALS, at=W_VALS, cap=4, count_at=W_VCOUNT,
                              elem_max=0xFF, default=VALS_DEFAULT)
    else:
        member.string(M_NAME, at=O_NAME, maxlen=8)
        member.unsigned_array(M_VALS, at=W_VALS, cap=4, count_at=W_VCOUNT,
                              elem_max=0xFF)
    member.unsigned(M_LVL, at=W_LVL, max_value=0xFF)
    option = Binding(closed=True, which_at=W_WHICH, default_id=DEFAULT_ID)
    option.unsigned(OPT_U16, at=W_U16, max_value=0xFFFF)
    option.sequence(OPT_CFG, member)
    return Binding(closed=True).sequence(UNION_ID, option)


def _cfg_name(e, value: str) -> None:
    """``seq[cmd]( seq[cfg]( [0:str] <value> ) )``"""
    e.write_sequence_begin_lazy(UNION_ID)
    e.write_sequence_begin_lazy(OPT_CFG)
    e.write_string(M_NAME, value)
    e.write_sequence_end()
    e.write_sequence_end()


def _u16(e, value: int) -> None:
    """``seq[cmd]( [0:u16] <value> )`` — discards whatever `cfg` held."""
    e.write_sequence_begin_lazy(UNION_ID)
    e.write_unsigned(OPT_U16, value)
    e.write_sequence_end()


def _cfg_lvl(e, value: int) -> None:
    """``seq[cmd]( seq[cfg]( [2:u8] <value> ) )`` — `cfg` again, carrying only
    `lvl`, so `name` and `vals` must come from their declared defaults."""
    e.write_sequence_begin_lazy(UNION_ID)
    e.write_sequence_begin_lazy(OPT_CFG)
    e.write_unsigned(M_LVL, value)
    e.write_sequence_end()
    e.write_sequence_end()


def _decode(engine, message, *, defaults: bool, chunk: int | None = None):
    """Decode ``message``, with every member prepared at its declared default —
    §2's "a new message has every field at its schema default", which is the
    caller's job for the first arrival and the table's for a re-selection."""
    root = _table(defaults)
    words = bytearray(root.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    view[W_WHICH] = DEFAULT_ID
    objects: list[object] = [None] * root.tree_objects_required
    objects[O_NAME] = NAME_DEFAULT
    for i, v in enumerate(VALS_DEFAULT):
        view[W_VALS + i] = v
    view[W_VCOUNT] = len(VALS_DEFAULT)

    data = encode(message)
    dec = engine(binding=root, words=words, objects=objects, **capped())
    step = chunk or len(data)
    status = None
    for i in range(0, len(data), step):
        status = dec.feed(data[i:i + step])
    vals = tuple(view[W_VALS + i] for i in range(view[W_VCOUNT]))
    return status, view, objects, vals


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestTheWiring:
    """Green today: what a switched-to struct option gets right. Without this
    the xfails below would not say which half is broken."""

    def test_the_payload_of_a_switched_to_option_lands(self, engine, chunk):
        status, view, _objects, _vals = _decode(
            engine, lambda e: (_cfg_name(e, "zz"), _u16(e, 5), _cfg_lvl(e, LVL_SENT)),
            defaults=False, chunk=chunk,
        )
        assert status == Status.COMPLETE
        assert view[W_WHICH] == OPT_CFG
        assert view[W_LVL] == LVL_SENT

    def test_a_member_the_reselected_frame_omits_does_not_keep_the_old_value(
        self, engine, chunk
    ):
        # The half of §7.4.1 that #167 did land: the discarded option's `name`
        # must not survive into the new one. It reads empty rather than its
        # default -- which is #169 -- but it is not "zz" either.
        _status, _view, objects, _vals = _decode(
            engine, lambda e: (_cfg_name(e, "zz"), _u16(e, 5), _cfg_lvl(e, LVL_SENT)),
            defaults=False, chunk=chunk,
        )
        assert objects[O_NAME] != "zz", "the discarded option's name survived"

    def test_the_reset_overwrites_what_the_caller_prepared(self, engine, chunk):
        # The mechanism #169 is about, isolated from the missing API: the reset
        # *writes* these slots, it does not merely leave them alone. With no
        # default declared the empty value is the right thing to write (§2: the
        # type's zero value when no default is given), so this stays true after
        # the fix -- and it is why preparing the members by hand cannot stand in
        # for a declared default, not even on the option's first arrival.
        _status, _view, objects, vals = _decode(
            engine, lambda e: _cfg_lvl(e, LVL_SENT), defaults=False, chunk=chunk,
        )
        assert objects[O_NAME] == "", "the prepared name was left in place"
        assert vals == (), "the prepared vals were left in place"


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestAMemberStartsFromItsDeclaredDefault:
    """§7.4.1's third example, with members whose defaults are non-empty:

        seq[cmd]( seq[cfg]( [0:str] "zz" ) )  seq[cmd]( [0:u16] 5 )
        seq[cmd]( seq[cfg]( [2:u8] 7 ) )
            ->  cfg = { name = "abc", vals = [1, 2], lvl = 7 }
    """

    def test_a_string_member_starts_from_its_declared_default(self, engine, chunk):
        status, view, objects, _vals = _decode(
            engine, lambda e: (_cfg_name(e, "zz"), _u16(e, 5), _cfg_lvl(e, LVL_SENT)),
            defaults=True, chunk=chunk,
        )
        assert status == Status.COMPLETE
        assert view[W_WHICH] == OPT_CFG
        assert objects[O_NAME] == NAME_DEFAULT

    def test_an_array_member_starts_from_its_declared_default(self, engine, chunk):
        status, view, _objects, vals = _decode(
            engine, lambda e: (_cfg_name(e, "zz"), _u16(e, 5), _cfg_lvl(e, LVL_SENT)),
            defaults=True, chunk=chunk,
        )
        assert status == Status.COMPLETE
        assert view[W_WHICH] == OPT_CFG
        assert vals == VALS_DEFAULT

    def test_the_first_arrival_of_an_option_keeps_the_prepared_defaults(
        self, engine, chunk
    ):
        # Not only a *re*-selection: `which_at` is prepared with `default_id`, so
        # the first `cfg` frame is already "a different option arrived" and runs
        # the reset. Preparing the members at their declared defaults (§2) does
        # not help -- the reset overwrites them -- which is why there is no way
        # to read `name = "abc"` for a held `cfg` at all today.
        status, view, objects, vals = _decode(
            engine, lambda e: _cfg_lvl(e, LVL_SENT), defaults=True, chunk=chunk,
        )
        assert status == Status.COMPLETE
        assert view[W_WHICH] == OPT_CFG
        assert view[W_LVL] == LVL_SENT
        assert (objects[O_NAME], vals) == (NAME_DEFAULT, VALS_DEFAULT)


# --- several options, several tables: the reset payload must not cross over ---
#
# The accelerator flattens every row's reset list into two arrays and hands each
# row a slice of them (``_BEntry.reset_w_at`` / ``reset_o_at``). An off-by-one
# there writes another option's slots and nothing else notices, so the shapes
# below exist to make the slices provable from the outside: two one-of tables,
# each with two struct options, every member on its own slot and with its own
# default. Whichever option is selected, only its own members may move.

TWO = dict(
    #            which  iso-obj  arr-slot  arr-count  lvl
    outer_a=dict(which=0, obj=0, arr=10, cnt=20, lvl=30, iso="a-def", vals=(1, 2)),
    outer_b=dict(which=0, obj=1, arr=12, cnt=21, lvl=31, iso="b-def", vals=(3,)),
    inner_a=dict(which=1, obj=2, arr=14, cnt=22, lvl=32, iso="c-def", vals=(4, 5, 6)),
    inner_b=dict(which=1, obj=3, arr=16, cnt=23, lvl=33, iso="d-def", vals=(7,)),
)
OUTER_ID, INNER_ID = 40, 41
OPT_A, OPT_B = 0, 1
SUB_ISO, SUB_ARR, SUB_LVL, SUB_INNER = 0, 1, 2, 3


def _member(spec: dict, inner: Binding | None = None) -> Binding:
    m = Binding(closed=True)
    m.string(SUB_ISO, at=spec["obj"], maxlen=16, default=spec["iso"])
    m.unsigned_array(SUB_ARR, at=spec["arr"], cap=2 if len(spec["vals"]) < 3 else 3,
                     count_at=spec["cnt"], elem_max=0xFF, default=spec["vals"])
    m.unsigned(SUB_LVL, at=spec["lvl"], max_value=0xFF)
    if inner is not None:
        m.sequence(SUB_INNER, inner)
    return m


def _two_table() -> Binding:
    inner = Binding(closed=True, which_at=TWO["inner_a"]["which"], default_id=OPT_A)
    inner.sequence(OPT_A, _member(TWO["inner_a"]))
    inner.sequence(OPT_B, _member(TWO["inner_b"]))
    outer = Binding(closed=True, which_at=TWO["outer_a"]["which"], default_id=OPT_A)
    outer.sequence(OPT_A, _member(TWO["outer_a"], inner))
    outer.sequence(OPT_B, _member(TWO["outer_b"]))
    return Binding(closed=True).sequence(OUTER_ID, outer)


def _two_decode(engine, message, chunk: int | None = None):
    root = _two_table()
    words = bytearray(root.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    objects: list[object] = [None] * root.tree_objects_required
    # Nothing prepared: every value below must come from the table's own plan.
    data = encode(message)
    dec = engine(binding=root, words=words, objects=objects, **capped())
    step = chunk or len(data)
    for i in range(0, len(data), step):
        dec.feed(data[i:i + step])
    return view, objects


def _read(view, objects, name: str):
    spec = TWO[name]
    n = view[spec["cnt"]]
    return (objects[spec["obj"]],
            tuple(view[spec["arr"] + i] for i in range(n)),
            view[spec["lvl"]])


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestSeveralOptions:
    def test_selecting_option_b_leaves_option_a_untouched(self, engine, chunk):
        # B is not default_id, so its reset runs. A's slots were never prepared
        # and must stay zero/None -- if the slices crossed, A's defaults would
        # appear here.
        def m(e):
            e.write_sequence_begin_lazy(OUTER_ID)
            e.write_sequence_begin_lazy(OPT_B)
            e.write_unsigned(SUB_LVL, 9)
            e.write_sequence_end()
            e.write_sequence_end()

        view, objects = _two_decode(engine, m, chunk)
        assert _read(view, objects, "outer_b") == ("b-def", (3,), 9)
        assert _read(view, objects, "outer_a") == (None, (), 0), "A's slots moved"

    def test_a_nested_union_brings_its_own_default_option_along(self, engine, chunk):
        # Selecting outer A resets its whole subtree, which includes the inner
        # union's which slot AND the storage of the inner option that slot names
        # (default_id = A) -- but not the inner B option's.
        def m(e):
            e.write_sequence_begin_lazy(OUTER_ID)
            e.write_sequence_begin_lazy(OPT_B)          # first hold B ...
            e.write_unsigned(SUB_LVL, 9)
            e.write_sequence_end()
            e.write_sequence_begin_lazy(OPT_A)          # ... then switch to A
            e.write_unsigned(SUB_LVL, 4)
            e.write_sequence_end()
            e.write_sequence_end()

        view, objects = _two_decode(engine, m, chunk)
        assert _read(view, objects, "outer_a") == ("a-def", (1, 2), 4)
        assert view[TWO["inner_a"]["which"]] == OPT_A
        assert _read(view, objects, "inner_a") == ("c-def", (4, 5, 6), 0)
        assert _read(view, objects, "inner_b") == (None, (), 0), "inner B moved"

    def test_switching_back_and_forth_restores_each_option_in_turn(self, engine, chunk):
        def m(e):
            for opt, lvl in ((OPT_A, 1), (OPT_B, 2), (OPT_A, 3)):
                e.write_sequence_begin_lazy(OUTER_ID)
                e.write_sequence_begin_lazy(opt)
                e.write_unsigned(SUB_LVL, lvl)
                e.write_sequence_end()
                e.write_sequence_end()

        view, objects = _two_decode(engine, m, chunk)
        assert view[TWO["outer_a"]["which"]] == OPT_A
        assert _read(view, objects, "outer_a") == ("a-def", (1, 2), 3)
        # B is unreachable now, but its own last reset is what it holds.
        assert _read(view, objects, "outer_b") == ("b-def", (3,), 2)


# --- string_into / blob_into: the default is COPIED, not stored ---------------
#
# These rows hold the caller's own buffer, so §7.4.1's "starts from its own
# default" cannot be a store. The reset copies the prepared bytes in and the
# length slot says how much of the buffer is live -- and, unlike every other
# reset, this one can refuse what the caller offered (§6.3), exactly as an
# arriving payload's destination is refused.

IN_UNION = 30
IN_TIME, IN_META = 0, 1
IN_ISO, IN_BLOB, IN_LVL = 0, 1, 2
IW_WHICH, IW_TIME, IW_ILEN, IW_BLEN, IW_LVL = 0, 1, 2, 3, 4
IO_ISO, IO_BLOB = 0, 1
IN_ISO_DEFAULT = "1970-01-01"
IN_BLOB_DEFAULT = b"\xde\xad"


def _into_table(iso_default: str | None = IN_ISO_DEFAULT) -> Binding:
    member = Binding(closed=True)
    member.string_into(IN_ISO, at=IO_ISO, maxlen=16, count_at=IW_ILEN,
                       default=iso_default)
    member.blob_into(IN_BLOB, at=IO_BLOB, maxlen=8, count_at=IW_BLEN,
                     default=IN_BLOB_DEFAULT)
    member.unsigned(IN_LVL, at=IW_LVL, max_value=0xFF)
    option = Binding(closed=True, which_at=IW_WHICH, default_id=IN_TIME)
    option.unsigned(IN_TIME, at=IW_TIME, max_value=0xFFFF)
    option.sequence(IN_META, member)
    return Binding(closed=True).sequence(IN_UNION, option)


def _into_message(e) -> None:
    """meta(iso="2026-09-29"), then time=5, then meta(lvl=7) -- the third frame
    leaves both buffers to the reset."""
    e.write_sequence_begin_lazy(IN_UNION)
    e.write_sequence_begin_lazy(IN_META)
    e.write_string(IN_ISO, "2026-09-29")
    e.write_sequence_end()
    e.write_sequence_end()
    e.write_sequence_begin_lazy(IN_UNION)
    e.write_unsigned(IN_TIME, 5)
    e.write_sequence_end()
    e.write_sequence_begin_lazy(IN_UNION)
    e.write_sequence_begin_lazy(IN_META)
    e.write_unsigned(IN_LVL, 7)
    e.write_sequence_end()
    e.write_sequence_end()


def _into_decode(engine, table: Binding, iso_buf, blob_buf, chunk=None):
    words = bytearray(table.tree_words_required * 8)
    view = memoryview(words).cast("Q")
    view[IW_WHICH] = IN_TIME
    objects: list[object] = [None] * table.tree_objects_required
    objects[IO_ISO], objects[IO_BLOB] = iso_buf, blob_buf
    data = encode(_into_message)
    dec = engine(binding=table, words=words, objects=objects, **capped())
    step = chunk or len(data)
    status = None
    for i in range(0, len(data), step):
        status = dec.feed(data[i:i + step])
    return status, view


@pytest.mark.parametrize("chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
class TestIntoDefaults:
    def test_the_default_is_copied_into_the_callers_buffer(self, engine, chunk):
        iso, blob = bytearray(16), bytearray(8)
        status, view = _into_decode(engine, _into_table(), iso, blob, chunk)
        assert status == Status.COMPLETE
        assert view[IW_WHICH] == IN_META and view[IW_LVL] == 7
        n = view[IW_ILEN]
        assert bytes(iso[:n]) == IN_ISO_DEFAULT.encode(), bytes(iso[:n])
        assert n == len(IN_ISO_DEFAULT)
        m = view[IW_BLEN]
        assert bytes(blob[:m]) == IN_BLOB_DEFAULT and m == len(IN_BLOB_DEFAULT)

    def test_a_buffer_too_short_for_the_default_is_refused(self, engine, chunk):
        # §6.3 InvalidArgument: the message is well-formed, the storage is not.
        # The same verdict an arriving payload gets for the same buffer.
        with pytest.raises(SofaArgumentError, match="declared default"):
            _into_decode(engine, _into_table(), bytearray(4), bytearray(8), chunk)

    def test_a_missing_buffer_is_refused(self, engine, chunk):
        with pytest.raises(SofaArgumentError, match="declared default"):
            _into_decode(engine, _into_table(), None, bytearray(8), chunk)

    def test_a_read_only_buffer_is_refused(self, engine, chunk):
        with pytest.raises(SofaArgumentError, match="declared default"):
            _into_decode(engine, _into_table(), b"x" * 16, bytearray(8), chunk)

    def test_an_into_row_without_a_default_only_zeroes_the_length(self, engine, chunk):
        # No default declared: the reset zeroes the length slot and touches the
        # buffer not at all, exactly as before #169. What the buffer still holds
        # is the FIRST frame's payload -- unreadable, because the length says
        # zero, which is how an into row has always reported absence.
        iso = bytearray(b"stale-----------")
        status, view = _into_decode(engine, _into_table(iso_default=None),
                                    iso, bytearray(8), chunk)
        assert status == Status.COMPLETE
        assert view[IW_ILEN] == 0, "the length must not survive the switch"
        assert bytes(iso) == b"2026-09-29------", "the buffer was written into"


def test_an_into_default_longer_than_maxlen_is_refused():
    with pytest.raises(Exception, match="longer than the declared maxlen"):
        Binding().string_into(1, at=0, maxlen=4, default="abcde")
    with pytest.raises(Exception, match="longer than the declared maxlen"):
        Binding().blob_into(1, at=0, maxlen=2, default=b"abc")
