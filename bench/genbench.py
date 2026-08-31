#!/usr/bin/env python3
"""What a GENERATED handler costs: the shape generator#406 asks for.

Two drivers over bench/decode_shapes.py's 36-field message:

  gen_bounds   a visitor that declares schema bounds and nothing else --
               `on_schema_bound` only, which is what a generated handler needs
               once its hand-written bound chain moves into the hook.
  gen_full     the same plus `on_field`, which is what it needs while the
               receiver caps still live in generated code (generator#388).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decode_shapes import (  # noqa: E402
    ARRAY_AT,
    ARRAY_CAP,
    ARRAY_IDS,
    CAP_ARR,
    CAP_BLOB,
    CAP_STR,
    COUNT_AT,
    SCALAR_AT,
    build_msg,
    new_storage,
)

from sofab import Decoder, Visitor, WireType  # noqa: E402

MSG = build_msg()
_ARR = frozenset(ARRAY_IDS)
_ARR_WT = WireType.ARRAY_UNSIGNED


class Bounds(Visitor):
    """Declares the schema's count for every array, and writes into slots."""
    def __init__(self, wu):
        self._wu = wu
        self._dst = {f: wu[ARRAY_AT[f]:ARRAY_AT[f] + ARRAY_CAP] for f in ARRAY_IDS}

    def on_schema_bound(self, field_id, n, wtype, subtype):
        # The tag test §7.3 wants, in the hook rather than in an `on_field`
        # kept alive for it: an id the schema bounds under a wire type it never
        # declared is skipped, not bounded.
        return ARRAY_CAP if field_id in _ARR and wtype is _ARR_WT else -1

    def on_unsigned(self, field_id, value):
        self._wu[SCALAR_AT[field_id]] = value

    def on_array_begin(self, field_id, wtype, count):
        self._wu[COUNT_AT[field_id]] = count
        return (self._dst[field_id], None, None)


class Full(Bounds):
    """Bounds plus the cap arm generated code still carries."""
    def on_field(self, field):
        return None


def _driver(cls):
    def make():
        words, u = new_storage()
        v = cls(u)
        def body(data):
            Decoder(max_dyn_array_count=CAP_ARR,
                    max_dyn_string_len=CAP_STR, max_dyn_blob_len=CAP_BLOB,
                    visitor=v).feed(data)
        return body
    return make


DRIVERS = {"gen_bounds": _driver(Bounds), "gen_full": _driver(Full)}


def main(argv):
    body = DRIVERS[argv[1]]()
    for _ in range(int(argv[2])):
        body(MSG)
    return 0




# --- the hybrid: a table PLUS a fallback that wants on_field ------------------
#
# A generated object binds its whole schema and keeps a visitor for the ids a
# later schema revision may add. Every field here is mapped, so the fallback is
# never actually offered one -- but it overrides `on_field`, and that is what
# used to make the decoder build a Field for all 36 of them anyway.
from decode_shapes import BINDING  # noqa: E402


class _Fallback(Visitor):
    def on_field(self, field):
        return None                      # a forward-compat sink


def make_hybrid():
    words, u = new_storage()
    dec = Decoder(max_dyn_array_count=CAP_ARR,
                  max_dyn_string_len=CAP_STR, max_dyn_blob_len=CAP_BLOB,
                  binding=BINDING, words=words, visitor=_Fallback())
    def body(data):
        dec.reset()
        dec.feed(data)
    return body


DRIVERS["hybrid"] = make_hybrid


if __name__ == "__main__":
    sys.exit(main(sys.argv))
