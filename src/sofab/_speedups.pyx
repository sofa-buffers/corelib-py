# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, overflowcheck=False
# cython: binding=False, always_allow_keywords=False, embedsignature=False
# cython: optimize.use_switch=True, optimize.unpack_method_calls=True
"""Native (Cython) accelerator for the SofaBuffers wire format.

This module provides drop-in ``Encoder`` and ``Decoder`` classes that are
**byte-for-byte identical** to the pure-Python implementations in
``sofab.encoder`` / ``sofab.decoder`` but run the varint / buffer hot paths in
compiled C. ``sofab.__init__`` imports these when the extension is present and
silently falls back to the pure-Python classes otherwise, so the library still
runs everywhere CPython runs — this file is a pure speed layer, never a
requirement.

The design mirrors the pure-Python one exactly (same construction models, same
streaming/refill semantics, same errors) so the two are interchangeable and the
shared conformance vectors validate both.

Where the speed comes from
--------------------------

Being compiled is only the start; on a format this small the cost lives in the
per-field and per-element edges between Python and C. In rough order of what the
benchmark workloads showed:

* **Values reach C in one step.** A write's range check *is* its conversion —
  the converter fails on exactly the values the format rejects — and for an exact
  ``int`` the conversion reads CPython's digits directly (see ``__sofab_u64``).
  The previous spelling paid two Python rich-comparisons and then a conversion
  that walked the same digits a third time.
* **Varints move a word at a time.** The continuation bits are one bit per byte,
  so a whole varint fits in a 64-bit word: ``_varint_put`` / ``_varint_take``
  replace the ten-step byte chain with three mask/shift steps.
* **Attribute and call shapes the interpreter can specialize.** ``Field``'s five
  attributes are published as slot descriptors and the methods as plain method
  descriptors (hence ``binding=False`` / ``always_allow_keywords=False`` above),
  which is what lets CPython's inline caches turn ``field.id`` and ``dec.next()``
  into their specialized forms instead of the generic attribute path.
* **Nothing is copied that is only going to be read.** Array writes walk the
  caller's list in place, string writes take the str's own UTF-8 buffer, and
  fixlen reads are built straight off the decode buffer.

The two layout-dependent tricks (digit access, slot descriptors) verify their own
assumptions at import time and fall back to the portable path if they do not
hold — see ``__sofab_digits_selftest`` and ``__sofab_field_slot_attrs``. They can
therefore cost speed on an unexpected build, never correctness.
"""

cimport cython

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize, PyBytes_GET_SIZE
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_GET_SIZE
from cpython.exc cimport PyErr_Clear, PyErr_Occurred
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_CheckExact
from cpython.list cimport PyList_Append, PyList_GET_ITEM, PyList_GET_SIZE, PyList_New, PyList_SET_ITEM
from cpython.long cimport PyLong_CheckExact, PyLong_FromUnsignedLongLong, PyLong_FromLongLong
from cpython.ref cimport Py_INCREF, PyObject
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy

cdef extern from "stdint.h":
    # The widest each side of a declared element width can be, used when a field
    # declares only one of the two (see ``read_signed_array``).
    const int64_t INT64_MIN
    const int64_t INT64_MAX

cdef extern from *:
    # Widest-native-int converters, picked per platform.
    #
    # Both spellings are public CPython API and both range-check, but they are
    # not equally cheap: on CPython 3.12+ the ``LongLong`` forms are implemented
    # on top of the generic ``_PyLong_AsByteArray`` serializer, while the
    # ``Long`` forms are a three-iteration digit loop. Where a C ``long`` is
    # already 64 bits wide (every LP64 platform: Linux, macOS, the BSDs) the
    # narrower-looking spelling is therefore both equivalent *and* several times
    # faster on exactly the full-width values a varint format exists to carry.
    # Where it is not (LLP64, i.e. Windows), the ``LongLong`` form is the only
    # correct one and is used.
    """
    #include <stdint.h>

    #if ULONG_MAX >= 0xFFFFFFFFFFFFFFFFULL
      #define __sofab_as_u64(o) ((unsigned long long) PyLong_AsUnsignedLong(o))
      #define __sofab_as_i64(o) ((long long) PyLong_AsLong(o))
    #else
      #define __sofab_as_u64(o) PyLong_AsUnsignedLongLong(o)
      #define __sofab_as_i64(o) PyLong_AsLongLong(o)
    #endif

    /* --- int -> C integer, straight off the digits ------------------------
     *
     * Even the cheap converter above is an out-of-line call that re-derives
     * what the caller already knows (that the object is an exact int) before
     * walking the same two or three digits this does. A CPython int is a small
     * sign/size tag plus an array of fixed-width digits, and reading those
     * directly turns the conversion into a two-iteration shift-and-or with an
     * overflow test — which matters because it runs once per array element.
     *
     * The layout is CPython's internal one, so it is treated as an assumption
     * to be *proved*, never assumed: __sofab_digits_selftest() below round-trips
     * a set of values spanning every digit count and both failure modes
     * (negative, and above the 64-bit domain) through this code at import time,
     * and only a clean sweep sets __sofab_digits_ok. Otherwise — including on
     * any build where the layout is not reachable at all — every conversion
     * goes through the public API instead, at full speed of correctness.
     */
    #if !defined(Py_LIMITED_API) && defined(PyLong_SHIFT)
      #define __SOFAB_DIGITS 1
      #if PY_VERSION_HEX >= 0x030C0000
        #define __sofab_lv(x)      (&((PyLongObject *)(x))->long_value)
        #define __sofab_ndigits(x) ((Py_ssize_t)(__sofab_lv(x)->lv_tag >> 3))
        #define __sofab_isneg(x)   ((__sofab_lv(x)->lv_tag & 3) == 2)
        #define __sofab_digits(x)  (__sofab_lv(x)->ob_digit)
      #else
        #define __sofab_ndigits(x) (Py_SIZE(x) < 0 ? -Py_SIZE(x) : Py_SIZE(x))
        #define __sofab_isneg(x)   (Py_SIZE(x) < 0)
        #define __sofab_digits(x)  (((PyLongObject *)(x))->ob_digit)
      #endif
    #else
      #define __SOFAB_DIGITS 0
      #define __sofab_ndigits(x) 0
      #define __sofab_isneg(x)   1
      #define __sofab_digits(x)  ((const digit *) 0)
    #endif

    static int __sofab_digits_ok = 0;   /* set by the self-test at import time */

    /* Magnitude as a uint64_t. Returns 0 (leaving no exception set) when the
       value does not fit, which is exactly the format's unsigned domain. */
    static int __sofab_u64_digits(PyObject *x, uint64_t *out) {
        Py_ssize_t n = __sofab_ndigits(x);
        const digit *d = __sofab_digits(x);
        uint64_t v = 0;
        if (__sofab_isneg(x)) return 0;
        while (n-- > 0) {
            if (v > (~(uint64_t) 0 >> PyLong_SHIFT)) return 0;
            v = (v << PyLong_SHIFT) | (uint64_t) d[n];
        }
        *out = v;
        return 1;
    }

    static int __sofab_u64(PyObject *x, uint64_t *out) {
        unsigned long long v;
        if (__sofab_digits_ok) return __sofab_u64_digits(x, out);
        v = __sofab_as_u64(x);
        if (v == (unsigned long long) -1 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
        *out = (uint64_t) v;
        return 1;
    }

    static int __sofab_i64(PyObject *x, int64_t *out) {
        long long v;
        uint64_t mag;
        if (__sofab_digits_ok) {
            int neg = __sofab_isneg(x);
            Py_ssize_t n = __sofab_ndigits(x);
            const digit *d = __sofab_digits(x);
            mag = 0;
            while (n-- > 0) {
                if (mag > (~(uint64_t) 0 >> PyLong_SHIFT)) return 0;
                mag = (mag << PyLong_SHIFT) | (uint64_t) d[n];
            }
            /* Two's complement: -2**63 is representable, +2**63 is not. */
            if (neg) {
                if (mag > (uint64_t) 1 << 63) return 0;
                *out = (int64_t) (~mag + 1);
            } else {
                if (mag > (uint64_t) INT64_MAX) return 0;
                *out = (int64_t) mag;
            }
            return 1;
        }
        v = __sofab_as_i64(x);
        if (v == -1 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
        *out = (int64_t) v;
        return 1;
    }

    static int __sofab_digits_selftest(void) {
        static const unsigned long long probe[] = {
            0ULL, 1ULL, 127ULL, 128ULL, 255ULL, 65535ULL,
            ((unsigned long long) 1 << 30) - 1, (unsigned long long) 1 << 30,
            ((unsigned long long) 1 << 32) - 1, (unsigned long long) 1 << 32,
            (unsigned long long) 1 << 60, 0x9E3779B97F4A7C15ULL,
            ~0ULL - 1, ~0ULL
        };
        static const long long sprobe[] = {
            0, 1, -1, 127, -128, 65535, -65536,
            (long long) 1 << 40, -((long long) 1 << 40),
            (long long) 0x7FFFFFFFFFFFFFFFLL, (-(long long) 0x7FFFFFFFFFFFFFFFLL) - 1
        };
        size_t i;
    #if !__SOFAB_DIGITS
        return 0;
    #else
        __sofab_digits_ok = 1;      /* provisional: the probes below decide */
        for (i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
            PyObject *o = PyLong_FromUnsignedLongLong(probe[i]);
            uint64_t got = 0;
            int ok;
            if (o == NULL) { PyErr_Clear(); goto fail; }
            ok = __sofab_u64_digits(o, &got);
            Py_DECREF(o);
            if (!ok || got != (uint64_t) probe[i]) goto fail;
        }
        for (i = 0; i < sizeof(sprobe) / sizeof(sprobe[0]); i++) {
            PyObject *o = PyLong_FromLongLong(sprobe[i]);
            int64_t got = 0;
            int ok;
            if (o == NULL) { PyErr_Clear(); goto fail; }
            ok = __sofab_i64(o, &got);
            Py_DECREF(o);
            if (!ok || got != (int64_t) sprobe[i]) goto fail;
        }
        {   /* The edges of both domains: what must be accepted, and what must
               be *rejected* rather than wrapped. */
            static const struct { const char *text; int u_ok; int i_ok; } edge[] = {
                { "18446744073709551615",  1, 0 },   /*  2**64 - 1 */
                { "18446744073709551616",  0, 0 },   /*  2**64     */
                { "9223372036854775807",   1, 1 },   /*  2**63 - 1 */
                { "9223372036854775808",   1, 0 },   /*  2**63     */
                { "-9223372036854775808",  0, 1 },   /* -2**63     */
                { "-9223372036854775809",  0, 0 },   /* -2**63 - 1 */
                { "-1",                    0, 1 }
            };
            for (i = 0; i < sizeof(edge) / sizeof(edge[0]); i++) {
                PyObject *o = PyLong_FromString(edge[i].text, NULL, 10);
                uint64_t u; int64_t sv;
                int u_got, i_got;
                if (o == NULL) { PyErr_Clear(); goto fail; }
                u_got = __sofab_u64_digits(o, &u);
                i_got = __sofab_i64(o, &sv);
                Py_DECREF(o);
                if (u_got != edge[i].u_ok || i_got != edge[i].i_ok) goto fail;
            }
        }
        return 1;
    fail:
        __sofab_digits_ok = 0;
        return 0;
    #endif
    }
    """
    # Deliberately *unchecked* (``noexcept``): the ``cpython.long`` declarations
    # carry Cython's own ``except? -1``, which routes every call site through
    # Cython's generic conversion helper. Calling CPython directly lets the hot
    # path decide for itself what a failure means — an out-of-range value must
    # surface as SofaRangeError, not as the bare OverflowError left pending.
    #
    # They take a *borrowed* pointer so array element loops need no per-element
    # incref/decref pair: nothing in such a loop can run Python code, so the
    # list's own reference keeps every element alive.
    unsigned long long _AsU64 "__sofab_as_u64" (PyObject*) noexcept
    long long _AsI64 "__sofab_as_i64" (PyObject*) noexcept
    bint _IsLong "PyLong_CheckExact" (PyObject*) noexcept
    # Subclasses included: what __index__ hands back is an int, but only from
    # CPython 3.10 on is it guaranteed to be an *exact* one — before that a bool
    # or an IntEnum member comes back as itself. Both share PyLongObject's
    # layout, so every converter below reads them correctly.
    bint _IsLongLike "PyLong_Check" (PyObject*) noexcept
    bint _IsFloat "PyFloat_CheckExact" (PyObject*) noexcept
    double _AsDouble "PyFloat_AS_DOUBLE" (PyObject*) noexcept
    # Range-checked conversions that report failure by return value instead of a
    # pending OverflowError. See __sofab_u64 below.
    bint _ToU64 "__sofab_u64" (PyObject*, uint64_t*) noexcept
    bint _ToI64 "__sofab_i64" (PyObject*, int64_t*) noexcept
    bint __sofab_digits_selftest()

cdef extern from *:
    # --- Word-at-a-time varint codec -----------------------------------------
    #
    # A varint is a byte-serial format, and decoding or encoding it one byte at a
    # time is a chain of dependent shift/mask/branch steps — about eight
    # instructions per byte, ten bytes for a full-width 64-bit value. But the
    # continuation bits are just one bit in every byte, so a whole varint fits in
    # a single 64-bit word: load (or store) eight bytes at once and move the
    # 7-bit groups with three mask/shift steps, and the ten-step chain becomes a
    # dozen straight-line instructions with no data-dependent branching at all.
    #
    # The transform pair below is exact and total, not an approximation of the
    # byte loop: ``__sofab_gather7`` packs eight 7-bit groups down into the low
    # 56 bits and ``__sofab_spread7`` is its inverse. Both are pure integer
    # arithmetic, so the only platform assumption is that a byte sequence and a
    # 64-bit word agree on order — i.e. little-endian, which is what the format
    # is defined in anyway (§4). Where that cannot be established at compile
    # time, or the compiler offers no bit-scan builtin, the flag below stays 0
    # and the byte-serial paths are used unchanged.
    """
    #include <stdint.h>
    #include <string.h>

    #if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
        __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && \
        (defined(__GNUC__) || defined(__clang__))
      #define __SOFAB_SWAR 1
      #define __sofab_bitlen(v)  (64 - __builtin_clzll(v))   /* v != 0 */
      #define __sofab_ctz(v)     __builtin_ctzll(v)          /* v != 0 */
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
      #define __SOFAB_SWAR 1
      #include <intrin.h>
      static int __sofab_bitlen(unsigned __int64 v) {
          unsigned long i; _BitScanReverse64(&i, v); return (int) i + 1;
      }
      static int __sofab_ctz(unsigned __int64 v) {
          unsigned long i; _BitScanForward64(&i, v); return (int) i;
      }
    #else
      #define __SOFAB_SWAR 0
      #define __sofab_bitlen(v)  0
      #define __sofab_ctz(v)     0
    #endif

    #define __SOFAB_MSBS  0x8080808080808080ULL

    /* Eight 7-bit groups, one per byte, packed into the low 56 bits. */
    static uint64_t __sofab_gather7(uint64_t x) {
        x &= 0x7F7F7F7F7F7F7F7FULL;
        x = (x & 0x007F007F007F007FULL) | ((x & 0x7F007F007F007F00ULL) >> 1);
        x = (x & 0x00003FFF00003FFFULL) | ((x & 0x3FFF00003FFF0000ULL) >> 2);
        x = (x & 0x000000000FFFFFFFULL) | ((x & 0x0FFFFFFF00000000ULL) >> 4);
        return x;
    }

    /* The inverse: 56 payload bits back out into one 7-bit group per byte. */
    static uint64_t __sofab_spread7(uint64_t x) {
        x = (x & 0x000000000FFFFFFFULL) | ((x & 0x00FFFFFFF0000000ULL) << 4);
        x = (x & 0x00003FFF00003FFFULL) | ((x & 0x0FFFC0000FFFC000ULL) << 2);
        x = (x & 0x007F007F007F007FULL) | ((x & 0x3F803F803F803F80ULL) << 1);
        return x;
    }

    static uint64_t __sofab_load8(const unsigned char *p) {
        uint64_t w; memcpy(&w, p, 8); return w;
    }

    static void __sofab_store8(unsigned char *p, uint64_t w) {
        memcpy(p, &w, 8);
    }
    """
    int __SOFAB_SWAR
    uint64_t __SOFAB_MSBS
    uint64_t __sofab_gather7(uint64_t) noexcept nogil
    uint64_t __sofab_spread7(uint64_t) noexcept nogil
    uint64_t __sofab_load8(const unsigned char*) noexcept nogil
    void __sofab_store8(unsigned char*, uint64_t) noexcept nogil
    int __sofab_bitlen(uint64_t) noexcept nogil
    int __sofab_ctz(uint64_t) noexcept nogil

cdef extern from *:
    # --- Field attributes as *slot* descriptors ------------------------------
    #
    # Reading ``field.id`` / ``field.type`` is the most repeated operation any
    # consumer of the decoder performs — generated code touches both on every
    # field — and it is the one part of the hot path that runs in the caller's
    # bytecode, where the interpreter's inline caches decide the cost. CPython
    # specializes an attribute that is a *member* descriptor (LOAD_ATTR_SLOT: a
    # guarded load straight out of the object) but has no specialization for the
    # *getset* descriptors Cython emits for ``cdef readonly`` attributes, which
    # fall all the way back to the generic PyObject_GenericGetAttr path. Measured
    # on CPython 3.12 that is about a 2x difference per attribute read.
    #
    # So the five attributes are re-published as member descriptors after the
    # type exists. The offsets are not guessed blindly: the layout Cython emits
    # for a ``cdef class`` whose only attributes are five ``object`` slots is the
    # object head followed by those five pointers, and the installer *verifies*
    # exactly that against a probe instance — both the total size and that every
    # computed offset reads back the object the probe was constructed with —
    # before publishing anything. If the layout is ever anything else the
    # descriptors are simply not installed and the original getsets stay in
    # place, so this can lose the optimization but cannot misread memory.
    # Descriptors are read-only, matching ``cdef readonly``.
    """
    #if !defined(Py_LIMITED_API)
    #include <structmember.h>
    #ifndef Py_T_OBJECT_EX      /* spelling before CPython 3.12 */
      #define Py_T_OBJECT_EX T_OBJECT_EX
      #define Py_READONLY READONLY
    #endif

    static PyMemberDef __sofab_field_members[5];
    static const char *const __sofab_field_names[5] = {
        "id", "type", "size", "count", "subtype"
    };

    static int __sofab_field_slot_attrs(PyObject *type_obj, PyObject *probe) {
        PyTypeObject *tp = (PyTypeObject *) type_obj;
        const Py_ssize_t base = (Py_ssize_t) sizeof(PyObject);
        const Py_ssize_t step = (Py_ssize_t) sizeof(PyObject *);
        char *mem = (char *) probe;
        Py_ssize_t i;

        if (tp->tp_basicsize != base + 5 * step) return 0;
        if (!Py_IS_TYPE(probe, tp)) return 0;
        for (i = 0; i < 5; i++) {
            PyObject *slot = *(PyObject **) (mem + base + i * step);
            PyObject *want = PyObject_GetAttrString(probe, __sofab_field_names[i]);
            int same;
            if (want == NULL) { PyErr_Clear(); return 0; }
            same = (slot == want);
            Py_DECREF(want);
            if (!same) return 0;
        }
        for (i = 0; i < 5; i++) {
            PyObject *descr;
            __sofab_field_members[i].name = (char *) __sofab_field_names[i];
            __sofab_field_members[i].type = Py_T_OBJECT_EX;
            __sofab_field_members[i].offset = base + i * step;
            __sofab_field_members[i].flags = Py_READONLY;
            __sofab_field_members[i].doc = NULL;
            descr = PyDescr_NewMember(tp, &__sofab_field_members[i]);
            if (descr == NULL) { PyErr_Clear(); return 0; }
            if (PyDict_SetItemString(tp->tp_dict, __sofab_field_names[i], descr) < 0) {
                Py_DECREF(descr);
                PyErr_Clear();
                return 0;
            }
            Py_DECREF(descr);
        }
        PyType_Modified(tp);
        return 1;
    }
    #else
    static int __sofab_field_slot_attrs(PyObject *type_obj, PyObject *probe) {
        (void) type_obj; (void) probe; return 0;
    }
    #endif
    """
    bint __sofab_field_slot_attrs(object type_obj, object probe)

cdef extern from "Python.h":
    # The __index__ protocol: returns the int an object considers itself to be,
    # losslessly, or raises TypeError. See _index_arg.
    object PyNumber_Index(object)
    # Decodes UTF-8 straight out of the decoder's buffer, so a string field
    # never materialises an intermediate ``bytes`` object.
    str PyUnicode_DecodeUTF8(const char*, Py_ssize_t, const char*)
    # The str's own UTF-8 form (cached on the object), so encoding a string does
    # not allocate a ``bytes`` either.
    const char* PyUnicode_AsUTF8AndSize(object, Py_ssize_t*) except NULL

# Wire-format constants, enums, the Field descriptor and the error classes all
# live in the shared pure-Python ``types`` module — reuse them verbatim so the
# native path raises the *same* exception types and yields the *same* Field /
# enum objects the pure path does.
from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    MAX_DEPTH,
    MIN_OUTPUT_BUFFER,
    SIGNED_MAX,
    SIGNED_MIN,
    UNSIGNED_MAX,
    FixlenSubtype,
    SofaBufferError,
    SofaDecodeError,
    SofaError,
    SofaIncompleteError,
    SofaLimitError,
    SofaRangeError,
    SofaStateError,
    WireType,
)


# --- Field descriptor --------------------------------------------------------
#
# A ``cdef`` mirror of ``sofab.types.Field`` — same public attributes
# (``id``/``type``/``size``/``count``/``subtype``) but allocated at the C level,
# which is dramatically cheaper than the pure-Python ``@dataclass`` on the
# per-field decode hot path. Attribute reads are plain C-struct slot reads.
#
# One Field is built per decoded field and dropped again as soon as the caller
# moves on, so allocation is a per-field cost: the freelist recycles the last 64
# instances instead of round-tripping through the object allocator
# (tp_alloc/PyObject_GC_Del) every time. Sized well past the fields a decode loop
# holds at once, so the recycle hits on the steady state.
@cython.freelist(64)
cdef class Field:
    """Describes the field the decoder is currently positioned on.

    Byte-for-byte compatible attribute surface with :class:`sofab.types.Field`.
    """
    cdef readonly object id
    cdef readonly object type
    cdef readonly object size
    cdef readonly object count
    cdef readonly object subtype

    def __init__(self, id, type, size=0, count=0, subtype=None):
        self.id = id
        self.type = type
        self.size = size
        self.count = count
        self.subtype = subtype

    def __repr__(self):
        return "Field(id=%r, type=%r, size=%r, count=%r, subtype=%r)" % (
            self.id, self.type, self.size, self.count, self.subtype)

    def __eq__(self, other):
        return (isinstance(other, Field)
                and self.id == other.id and self.type == other.type
                and self.size == other.size and self.count == other.count
                and self.subtype == other.subtype)


# Whether the digit-level int conversion passed its import-time self-test (see
# __sofab_digits_selftest). ``False`` means every conversion goes through the
# public CPython converters instead — slower, identical results.
INT_DIGITS_FAST = __sofab_digits_selftest()


# Republish the five attributes as slot descriptors (see __sofab_field_slot_attrs
# above). The probe is built from five distinct objects so a layout that differs
# in *order* is caught as surely as one that differs in size. Exposed for the
# test suite; ``False`` simply means the getset descriptors are still in use.
FIELD_SLOT_ATTRS = __sofab_field_slot_attrs(
    Field, Field(object(), object(), object(), object(), object()))


cdef object _ZERO = 0
cdef object _NONE = None

cdef inline Field _mkfield(object fid, object ftype, object size, object count, object subtype):
    cdef Field f = Field.__new__(Field)
    f.id = fid
    f.type = ftype
    f.size = size
    f.count = count
    f.subtype = subtype
    return f

# --- C-level copies of the limits (hot-path checks avoid Python attr lookups) --
cdef uint64_t _UNSIGNED_MAX = <uint64_t>0xFFFFFFFFFFFFFFFFULL
cdef int64_t _SIGNED_MIN = <int64_t>(-0x8000000000000000LL)
cdef int64_t _SIGNED_MAX = <int64_t>0x7FFFFFFFFFFFFFFFLL
cdef uint64_t _ID_MAX = <uint64_t>0x7FFFFFFF
cdef uint64_t _ARRAY_MAX = <uint64_t>0x7FFFFFFF
cdef uint64_t _FIXLEN_MAX = <uint64_t>0x7FFFFFFF
cdef int _MAX_DEPTH = 255
# The smallest streaming output buffer this port accepts (S5.1). It is 1 because
# _put splits every atomic unit at any byte boundary; it binds only a buffer
# installed together with a flush sink. Mirrors types.MIN_OUTPUT_BUFFER.
cdef Py_ssize_t _MIN_OUTPUT_BUFFER = MIN_OUTPUT_BUFFER
# Size of the scratch buffer the convenience constructors install (S5.1
# "unbounded schema" shape). One allocation per encoder, made at construction and
# never resized -- the encoder drains it through its sink whenever it fills, so
# what an encode holds is this constant and not the message. Module-level (not
# only cdef) so it is readable as sofab._speedups._SCRATCH_SIZE, and it must
# equal sofab.encoder._SCRATCH_SIZE: both engines bound memory the same way.
_SCRATCH_SIZE = 1024
cdef Py_ssize_t _SCRATCH_SIZE_C = _SCRATCH_SIZE
# Largest value representable in a Py_ssize_t — a fixlen-array payload larger
# than this cannot be satisfied by any real buffer, so it is treated as a
# truncated (unsatisfiable) read rather than being cast to a negative size.
cdef uint64_t _SSIZE_MAX = <uint64_t>0x7FFFFFFFFFFFFFFFULL

# Wire types (low 3 bits of a field header).
cdef int _WT_UNSIGNED = 0
cdef int _WT_SIGNED = 1
cdef int _WT_FIXLEN = 2
cdef int _WT_ARRAY_UNSIGNED = 3
cdef int _WT_ARRAY_SIGNED = 4
cdef int _WT_ARRAY_FIXLEN = 5
cdef int _WT_SEQUENCE_START = 6
cdef int _WT_SEQUENCE_END = 7

# Fixlen subtypes (low 3 bits of a fixlen length word).
cdef int _ST_FP32 = 0
cdef int _ST_FP64 = 1
cdef int _ST_STRING = 2
cdef int _ST_BLOB = 3

# Pre-fetched enum members, indexed by their integer value, so the decoder can
# hand back the exact same singletons the pure path uses without paying the
# IntEnum coercion on every field.
cdef tuple _WT = tuple(WireType)
cdef tuple _ST = tuple(FixlenSubtype)

# Pending-value kinds (mirror the pure decoder's _SCALAR/_FIXLEN/_VARRAY/_FARRAY).
cdef int _PEND_NONE = 0
cdef int _PEND_SCALAR = 1
cdef int _PEND_FIXLEN = 2
cdef int _PEND_VARRAY = 3
cdef int _PEND_FARRAY = 4
# A pending value a receiver-side cap has rejected (§6.2.1). The kind it stands
# in for is kept in _pk_real and the rejection's message in _limit_msg, so the
# rejection can still be waived by schema_bounded() and a peek can read through
# it. Mirrors the pure decoder's _LIMIT wrapper tuple.
cdef int _PEND_LIMIT = 5


# --- ZigZag (identical math to sofab._varint) --------------------------------
cdef inline uint64_t _zigzag_encode(int64_t v) noexcept nogil:
    return (<uint64_t>v << 1) ^ <uint64_t>(v >> 63)

cdef inline int64_t _zigzag_decode(uint64_t u) noexcept nogil:
    return <int64_t>(u >> 1) ^ -<int64_t>(u & 1)


# --- varint codec ------------------------------------------------------------
#
# Both directions have the same shape: a one-byte fast path (the common case for
# ids, counts and small values), then a whole-word path for everything longer,
# then the byte-serial fallback for platforms the word path cannot serve. All
# three produce identical bytes and identical values — the word path is a
# reformulation of the byte loop, not a different encoding.

cdef inline int _varint_put(unsigned char* out, uint64_t value) noexcept nogil:
    """Encode ``value`` at ``out`` (which must have room for 10 bytes) and return
    the number of bytes written."""
    cdef uint64_t rest
    cdef unsigned char* q
    cdef int n
    if value < 0x80:                       # ids, counts, and small values
        out[0] = <unsigned char>value
        return 1
    if __SOFAB_SWAR:
        if value < (<uint64_t>1 << 56):
            # Up to eight groups: one spread, one 8-byte store. The continuation
            # bits are set on every byte but the last in a single mask.
            n = (__sofab_bitlen(value) - 1) // 7 + 1
            __sofab_store8(out, __sofab_spread7(value)
                                | (__SOFAB_MSBS & (((<uint64_t>1) << (8 * (n - 1))) - 1)))
            return n
        # 2**56 and above: the low 56 bits fill all eight bytes, one or two more
        # bytes carry the rest.
        __sofab_store8(out, __sofab_spread7(value & <uint64_t>0x00FFFFFFFFFFFFFF)
                            | __SOFAB_MSBS)
        rest = value >> 56
        if rest < 0x80:
            out[8] = <unsigned char>rest
            return 9
        out[8] = <unsigned char>((rest & 0x7F) | 0x80)
        out[9] = <unsigned char>(rest >> 7)
        return 10
    q = out
    while value >= 0x80:
        q[0] = <unsigned char>(value | 0x80)
        q += 1
        value >>= 7
    q[0] = <unsigned char>value
    return <int>(q + 1 - out)


cdef struct _Varint:
    uint64_t value
    int used            # bytes consumed, or -1 for an overlong (>64-bit) varint


cdef inline _Varint _varint_take(const unsigned char* p) noexcept nogil:
    """Decode the varint at ``p``, returning its value and its byte length.

    The caller must have proved at least 10 readable bytes are present, which is
    the longest a varint can be — so this never has to test for the end of the
    buffer, and a value that runs off it is impossible rather than checked for.
    Reporting the overlong case through ``used`` rather than an exception keeps
    the cursor in a register and this function ``nogil``.
    """
    cdef _Varint r
    cdef Py_ssize_t at = 0
    cdef unsigned char b = p[at]
    cdef uint64_t w, msbs, value
    cdef int shift, n
    if b < 0x80:                           # ids, counts, and small values
        r.value = b
        r.used = 1
        return r
    if __SOFAB_SWAR:
        w = __sofab_load8(p + at)
        msbs = (~w) & __SOFAB_MSBS
        if msbs:
            # A byte without its continuation bit set ends the value; its
            # position gives the length, and one gather yields the payload.
            n = (__sofab_ctz(msbs) >> 3) + 1
            r.value = __sofab_gather7(w) & ((((<uint64_t>1) << (7 * n)) - 1))
            r.used = n
            return r
        # Eight continuation bytes: 56 bits so far, one or two bytes to go.
        value = __sofab_gather7(w)
        b = p[at + 8]
        if b < 0x80:
            r.value = value | ((<uint64_t>b) << 56)
            r.used = 9
            return r
        value |= (<uint64_t>(b & 0x7F)) << 56
        b = p[at + 9]
        # Tenth byte: only bit 63 is still free, so anything above 0x01 either
        # carries a payload bit past bit 63 or continues into an eleventh byte —
        # both are the overlong (>64-bit) INVALID case (§4.1/§6.3, issue #43).
        if b > 0x01:
            r.used = -1
            return r
        r.value = value | ((<uint64_t>b) << 63)
        r.used = 10
        return r
    value = b & 0x7F
    shift = 7
    at += 1
    while shift < 63:
        b = p[at]
        at += 1
        value |= (<uint64_t>(b & 0x7F)) << shift
        if b < 0x80:
            r.value = value
            r.used = <int>at
            return r
        shift += 7
    b = p[at]
    if b > 0x01:
        r.used = -1
        return r
    r.value = value | ((<uint64_t>b) << 63)
    r.used = <int>(at + 1)
    return r


# --- Python int -> C int, with the format's range rule ------------------------
#
# Every value the encoder writes arrives as a Python object and has to end up in
# a C register, range-checked. The obvious spelling — compare against the
# UNSIGNED_MAX / SIGNED_* module constants, then cast — costs two Python
# rich-comparisons per value, and the cast itself then re-walks the int: for a
# full-width 64-bit value (three 30-bit digits) Cython's generic converter has no
# digit-level fast path at all and falls back to a *third* pass through
# PyObject_RichCompareBool + PyLong_AsUnsignedLong. That was the single largest
# cost in the array-encode workload.
#
# These helpers collapse all of it into one CPython call for the overwhelmingly
# common case (an exact ``int``): the converter is *itself* the range check — it
# fails exactly on the values the format rejects — so out-of-range is read off
# the pending OverflowError instead of being predicted by comparisons. Anything
# that is not an exact int (bool, enum, numpy scalar, float, __index__ provider)
# takes the original comparison path, so accepted-value semantics are unchanged.
# Which value is out of range, as a plain int: the message text is only ever
# needed on the error path, and a ``str`` argument would cost an incref/decref
# pair on every element of every array instead.
cdef int _WHAT_U = 0
cdef int _WHAT_S = 1
cdef int _WHAT_UA = 2
cdef int _WHAT_SA = 3
cdef int _WHAT_ID = 4
cdef tuple _WHATS = ("unsigned value", "signed value",
                     "unsigned array value", "signed array value", "id")

cdef inline uint64_t _u64_elem(PyObject* p, int what) except? 0xDEAD:
    cdef uint64_t r
    if _IsLong(p) and _ToU64(p, &r):
        return r
    return _u64_other(<object>p, what)

cdef inline int64_t _i64_elem(PyObject* p, int what) except? -0xDEAD:
    cdef int64_t r
    if _IsLong(p) and _ToI64(p, &r):
        return r
    return _i64_other(<object>p, what)

cdef inline uint64_t _u64_arg(object value, int what=_WHAT_U) except? 0xDEAD:
    return _u64_elem(<PyObject*>value, what)

cdef inline int64_t _i64_arg(object value, int what=_WHAT_S) except? -0xDEAD:
    return _i64_elem(<PyObject*>value, what)

cdef object _index_arg(object value, int what):
    """The int ``value`` losslessly is, or SofaRangeError.

    Integer fields accept whatever Python itself accepts where an integer is
    required: an object implementing ``__index__`` (int, bool, IntEnum, NumPy
    integers). ``float`` deliberately does not implement it — ``3.7`` cannot
    become an integer without discarding information — so it is refused rather
    than truncated, which would change the value the caller asked to send in a
    way the receiver could never detect. Mirrors sofab.encoder._as_int exactly;
    the two engines must accept and reject the same objects.
    """
    try:
        return PyNumber_Index(value)
    except TypeError:
        raise SofaRangeError("%s must be an integer, not %s"
                             % (_WHATS[what], type(value).__name__)) from None

cdef uint64_t _u64_other(object value, int what) except? 0xDEAD:
    # Cold: an exact int the converter rejected (outside the 64-bit domain), or
    # something that is not an exact int at all.
    cdef object idx = _index_arg(value, what)
    cdef uint64_t out
    if _IsLongLike(<PyObject*>idx) and _ToU64(<PyObject*>idx, &out):
        return out
    raise SofaRangeError("%s %d out of range" % (_WHATS[what], idx))

cdef int64_t _i64_other(object value, int what) except? -0xDEAD:
    cdef object idx = _index_arg(value, what)
    cdef int64_t out
    if _IsLongLike(<PyObject*>idx) and _ToI64(<PyObject*>idx, &out):
        return out
    raise SofaRangeError("%s %d out of range" % (_WHATS[what], idx))

cdef inline uint64_t _id_arg(object field_id) except? 0xDEAD:
    # Field ids are 0..ID_MAX (2**31-1), so the *value* range is narrower than
    # what the converter itself rejects; the explicit bound stays, but on C ints.
    cdef int64_t r
    cdef object idx
    if _IsLong(<PyObject*>field_id):
        if _ToI64(<PyObject*>field_id, &r):
            if r < 0 or r > <int64_t>_ID_MAX:
                raise SofaRangeError("id %d out of range 0..%d" % (field_id, _ID_MAX))
            return <uint64_t>r
        raise SofaRangeError("id %d out of range 0..%d" % (field_id, _ID_MAX))
    idx = _index_arg(field_id, _WHAT_ID)
    if _IsLongLike(<PyObject*>idx) and _ToI64(<PyObject*>idx, &r) and 0 <= r <= <int64_t>_ID_MAX:
        return <uint64_t>r
    raise SofaRangeError("id %d out of range 0..%d" % (idx, _ID_MAX))


# --- array-input plumbing -----------------------------------------------------
#
# An array write is handed any iterable. Materialising it with ``list(values)``
# is what the element loop needs — random access and a fixed length — but it is
# also a full copy, and the overwhelmingly common input already *is* a list. So
# take the caller's list as-is and pay the copy only for anything else.
cdef inline list _as_list(object values):
    if type(values) is list:
        return <list>values
    return list(values)

cdef inline PyObject* _elem(list seq, Py_ssize_t i) except NULL:
    # Element access with the length re-read every time. Converting an element
    # can run arbitrary Python (a non-int with ``__index__``), which could shrink
    # the very list being walked now that it is no longer a private copy; the
    # count is already on the wire at this point, so a short read has to be an
    # error rather than a silently truncated array.
    if i >= PyList_GET_SIZE(seq):
        raise SofaStateError("array shrank while it was being encoded")
    return PyList_GET_ITEM(seq, i)

cdef inline list _as_float_list(object values):
    # A float array is packed straight into the output buffer, so every element
    # must already be a float before the first byte is written: a ``__float__``
    # running mid-pack could re-enter this encoder and reallocate the very buffer
    # being written into. The check is a scan, not a copy — a list that is
    # already all floats (the normal case) is used as-is, and anything else falls
    # back to materialising the converted list up front, as before.
    cdef list seq = _as_list(values)
    cdef Py_ssize_t i, n = PyList_GET_SIZE(seq)
    for i in range(n):
        if not _IsFloat(PyList_GET_ITEM(seq, i)):
            return [float(x) for x in seq]
    return seq


# =============================================================================
# Encoder
# =============================================================================

cdef class Encoder:
    """Native encoder — see :class:`sofab.encoder.Encoder` for the full contract.

    One buffer-ownership model, byte-identical to the pure-Python encoder: the
    encoder writes into a **fixed** buffer and drains it through a flush sink,
    and never grows a buffer (CORELIB_PLAN S5.1).

    * ``Encoder.over_buffer(buffer, offset, flush)`` — the primitive: writes into
      a caller-owned ``bytearray``, draining through ``flush`` when it fills.
    * ``Encoder(writer=None, sticky=False)`` — the same over a scratch buffer of
      ``_SCRATCH_SIZE`` bytes installed with a sink, which forwards to
      ``writer.write`` or, with no writer, appends into the result
      :meth:`getvalue` hands back.
    """

    # output buffer (always installed: the convenience constructors install a
    # scratch buffer, over_buffer the caller's)
    cdef object _fixed_obj          # the bytearray being written into (keeps it alive)
    # The convenience constructors' scratch: one fixed block, allocated at
    # construction and never resized (S5.1 -- an output buffer is never grown).
    # Raw C memory rather than a bytearray because nothing outside the encoder
    # ever sees it; a caller's buffer arrives through buffer_set as _fixed_obj.
    cdef unsigned char* _scratch
    cdef unsigned char* _fixed_ptr
    cdef size_t _fixed_cap
    cdef size_t _cursor
    # Installation counter: bumped by every buffer_set, so _drain can tell whether
    # the sink took the buffer (installed a replacement, whose offset is the new
    # cursor) or merely copied it and returned (resume at 0) -- CORELIB_PLAN S5.1.
    cdef uint64_t _installs
    cdef object _flush_sink
    # shared
    cdef object _writer
    # The in-memory model's growing *result* -- the message getvalue() returns,
    # not a buffer the encoder writes into. A list of the drained chunks, joined
    # by getvalue(); None until the first drain, so a message that fits in the
    # scratch never needs one, and None throughout for the writer/over_buffer
    # forms, which retain nothing.
    cdef bint _in_memory
    cdef list _result
    cdef bint _sticky
    cdef object _error
    cdef int _depth
    # Ids of the innermost open sequences whose header has not been written yet
    # (MESSAGE_SPEC S2 lazy framing). Always a contiguous suffix of the open
    # sequences, so write_sequence_end simply pops the last entry.
    #
    # A heap block that is NULL until the first hold-back and doubles on demand
    # (CORELIB_PLAN S6: an implementation that can allocate MUST hold back to the
    # full MAX_DEPTH, so there is no fixed window and no eager-framing fallback;
    # only a heap-free profile may bound the run). Growing on demand also keeps an
    # encoder that never opens a sequence from paying for the run at all -- the
    # fixed 255-entry array this replaces cost every stream ~1 KiB it never used.
    cdef uint32_t* _pending
    cdef int _npending
    cdef int _pcap

    def __cinit__(self):
        self._fixed_obj = None
        self._scratch = NULL
        self._fixed_ptr = NULL
        self._fixed_cap = 0
        self._cursor = 0
        self._installs = 0
        self._flush_sink = None
        self._writer = None
        self._in_memory = False
        self._result = None
        self._sticky = False
        self._error = None
        self._depth = 0
        self._pending = NULL
        self._npending = 0
        self._pcap = 0

    def __init__(self, writer=None, *, bint sticky=False):
        # S5.1 "unbounded schema" shape: a fixed scratch buffer installed *with*
        # a sink. The corelib grows no output buffer -- with a writer nothing is
        # retained at all, and with none the sink appends into the result
        # getvalue() hands back.
        self._writer = writer
        self._sticky = sticky
        # With no writer the sink is the result the encoder hands back -- the
        # "growing result" S5.1 names for the unbounded shape, which is the
        # message and not a buffer written into. _drain allocates it.
        self._in_memory = writer is None
        # The one allocation, made here and never resized. Installed exactly as
        # buffer_set would: the checks it makes are constants here (offset 0 into
        # a buffer of _SCRATCH_SIZE >= MIN_OUTPUT_BUFFER).
        if self._scratch != NULL:        # __init__ called twice: no leak
            free(self._scratch)
        self._scratch = <unsigned char*>malloc(<size_t>_SCRATCH_SIZE_C)
        if self._scratch == NULL:
            raise MemoryError()
        self._fixed_ptr = self._scratch
        self._fixed_cap = <size_t>_SCRATCH_SIZE_C
        self._cursor = 0
        self._installs = 1

    cdef inline bint _has_sink(self):
        # Whether a flush can occur -- i.e. whether the installed buffer is a
        # sink-installed one, which is what MIN_OUTPUT_BUFFER binds (S5.1). All
        # three shapes of sink count: the caller's callback, the writer, and the
        # in-memory result.
        return self._in_memory or self._writer is not None or self._flush_sink is not None

    def __dealloc__(self):
        if self._scratch != NULL:
            free(self._scratch)
            self._scratch = NULL
        if self._pending != NULL:
            free(self._pending)
            self._pending = NULL
            self._pcap = 0

    @classmethod
    def over_buffer(cls, bytearray buffer, int offset=0, flush=None, *, bint sticky=False):
        cdef Encoder self = cls.__new__(cls)
        self._writer = None
        self._result = None
        self._flush_sink = flush
        self._sticky = sticky
        self.buffer_set(buffer, offset)
        return self

    def buffer_set(self, bytearray buffer, int offset=0):
        cdef Py_ssize_t size = PyByteArray_GET_SIZE(buffer)
        if not (0 <= <Py_ssize_t>offset <= size):
            raise SofaRangeError("offset must be within the buffer")
        # MIN_OUTPUT_BUFFER (S5.1) binds a buffer installed *with* a flush sink,
        # here and at every mid-stream set, so an unusable buffer is refused where
        # it is handed over rather than partway through a message. Without a sink
        # no flush can occur and no minimum applies -- the buffer holds the message
        # or reports buffer-full -- which is what keeps a caller sizing from a
        # generated MAX_SIZE exact, down to a zero-byte remainder.
        if size - <Py_ssize_t>offset < _MIN_OUTPUT_BUFFER and self._has_sink():
            raise SofaRangeError(
                "a buffer installed with a flush sink needs at least "
                "MIN_OUTPUT_BUFFER=%d usable byte(s), got %d"
                % (_MIN_OUTPUT_BUFFER, size - <Py_ssize_t>offset))
        self._fixed_obj = buffer
        self._fixed_ptr = <unsigned char*>PyByteArray_AS_STRING(buffer)
        self._fixed_cap = <size_t>size
        self._cursor = <size_t>offset
        # The offset belongs to this installation, not to the buffer (S5.1): it is
        # consumed once, and re-installing is what re-arms it for the next packet.
        self._installs += 1

    # --- error / output plumbing --------------------------------------------

    @property
    def error(self):
        return self._error

    cdef int _put(self, const unsigned char* data, size_t n) except -1:
        # Write n raw bytes into the output buffer, draining through the sink
        # whenever it fills. A run longer than the whole buffer is split across
        # drains: that is what makes MIN_OUTPUT_BUFFER == 1 hold (S5.1).
        cdef size_t pos = 0
        cdef size_t take
        if self._in_memory and n >= self._fixed_cap:
            # A divisible run at least as long as the whole buffer, in the model
            # whose sink is the result: hand it over as its own chunk instead of
            # copying it through the buffer a bufferful at a time. Draining first
            # is what keeps the wire order.
            if self._cursor:
                self._drain()
            if self._result is None:
                self._result = [PyBytes_FromStringAndSize(<char*>data, <Py_ssize_t>n)]
            else:
                PyList_Append(self._result,
                              PyBytes_FromStringAndSize(<char*>data, <Py_ssize_t>n))
            return 0
        while pos < n:
            if self._cursor >= self._fixed_cap:
                self._drain()
                if self._cursor >= self._fixed_cap:
                    raise SofaBufferError("encoder buffer full")
            take = self._fixed_cap - self._cursor
            if take > n - pos:
                take = n - pos
            memcpy(self._fixed_ptr + self._cursor, data + pos, take)
            self._cursor += take
            pos += take
        return 0

    cdef int _drain(self) except -1:
        # A sink that returns without installing a buffer *copied* it: the active
        # buffer stays active and encoding resumes at 0. A sink that *took* it must
        # install a replacement before returning, and that installation's offset is
        # the cursor -- resetting to 0 here would drop the header room it just
        # reserved and overwrite it with payload (CORELIB_PLAN S5.1).
        cdef uint64_t installs
        cdef bytes snapshot
        if self._in_memory:
            # The sink is the result the encoder hands back: the drained bytes are
            # kept as a chunk for getvalue() to join, which costs one copy here and
            # none there (b"".join of a single chunk returns it unchanged). The
            # buffer is never taken, so the cursor resumes at 0.
            snapshot = PyBytes_FromStringAndSize(
                <char*>self._fixed_ptr, <Py_ssize_t>self._cursor)
            if self._result is None:
                self._result = [snapshot]
            else:
                PyList_Append(self._result, snapshot)
            self._cursor = 0
            return 0
        if self._writer is None and self._flush_sink is None:
            raise SofaBufferError("encoder buffer full")
        installs = self._installs
        snapshot = PyBytes_FromStringAndSize(<char*>self._fixed_ptr, <Py_ssize_t>self._cursor)
        if self._writer is not None:
            self._writer.write(snapshot)
        else:
            self._flush_sink(snapshot)
        if self._installs == installs:
            self._cursor = 0
        return 0

    cdef inline int _emit_varint(self, uint64_t value) except -1:
        # The hot path. With a whole varint's room left the value is encoded
        # straight into the output buffer; on its last bytes it goes through a
        # stack scratch and the chunk-aware _put, which splits it across the
        # drain. Ten bytes is all _varint_put ever needs, so the room check
        # happens once per value rather than once per byte.
        cdef unsigned char scratch[10]
        if self._cursor + 10 <= self._fixed_cap:
            self._cursor += <size_t>_varint_put(self._fixed_ptr + self._cursor, value)
            return 0
        else:
            self._put(scratch, <size_t>_varint_put(scratch, value))
            return 0

    cdef inline int _header_c(self, uint64_t field_id, int wtype) except -1:
        # ``_header`` with the id already validated and in a C register — used by
        # every write path, which converts the id exactly once (see _id_arg).
        if self._npending:
            self._commit_pending()
        self._emit_varint((field_id << 3) | <uint64_t>wtype)
        return 0

    cdef inline int _header(self, object field_id, int wtype) except -1:
        # The single choke point every field write passes through, and therefore
        # where a held-back sequence run is committed: the field about to be
        # written is content, which proves every enclosing sequence differs from
        # its declared default and must be framed after all. Only genuine field
        # writes reach here -- a sequence is opened by write_sequence_begin_lazy
        # (which writes nothing) and closed by write_sequence_end /
        # write_sequence_end_keep (which emit the bare 0x07 themselves) -- so no
        # gate on wtype is needed and no writer can bypass the commit.
        self._header_c(_id_arg(field_id), wtype)
        return 0

    cdef int _commit_pending(self) except -1:
        # Emit the held-back sequence headers, outermost first. Cold: it runs at
        # most once per non-default sequence, never per field. The run is
        # detached before the first byte goes out, so a flush sink that re-enters
        # the encoder cannot observe a half-committed run (it starts a fresh run
        # of its own; the block below is then handed back or freed).
        cdef uint32_t* run = self._pending
        cdef int n = self._npending
        cdef int cap = self._pcap
        cdef int i
        self._pending = NULL
        self._npending = 0
        self._pcap = 0
        try:
            for i in range(n):
                self._emit_varint((<uint64_t>run[i] << 3) | <uint64_t>_WT_SEQUENCE_START)
        finally:
            if self._pending == NULL:
                # Nothing re-entered: keep the block for the next hold-back.
                self._pending = run
                self._pcap = cap
            else:
                free(run)
        return 0

    cdef int _pending_push(self, uint32_t field_id) except -1:
        # Append one id to the pending run, growing the block on demand. NULL
        # until the first hold-back, so an encoder that never opens a sequence
        # never allocates it; capacity doubles from 8 and is implicitly bounded
        # by _MAX_DEPTH (the run is a subset of the open sequences).
        cdef int newcap
        cdef uint32_t* grown
        if self._npending >= self._pcap:
            newcap = self._pcap * 2 if self._pcap else 8
            grown = <uint32_t*>realloc(self._pending, <size_t>newcap * sizeof(uint32_t))
            if grown == NULL:
                raise MemoryError()
            self._pending = grown
            self._pcap = newcap
        self._pending[self._npending] = field_id
        self._npending += 1
        return 0

    cdef inline bint _begin(self):
        return not (self._sticky and self._error is not None)

    cdef inline int _fail(self, exc) except -1:
        if self._sticky:
            if self._error is None:
                self._error = exc
        else:
            raise exc
        return 0

    def bytes_used(self):
        # Bytes standing in the output buffer: written since it was installed and
        # not yet drained. The buffer is fixed, so this never exceeds its size --
        # it is not the length of the message.
        return <object>self._cursor

    def flush(self):
        cdef size_t used = self._cursor
        if used and self._has_sink():
            self._drain()
        return <object>used

    def getvalue(self):
        # Only the in-memory model retains the message: with a writer the bytes
        # have been handed over, and over a caller's buffer they are in it, so
        # returning an undrained tail would be partial output dressed up as a
        # whole message (S5.1).
        if not self._in_memory:
            raise SofaStateError("getvalue() is only valid for the in-memory model")
        if self._result is None:        # never drained: the message is the prefix
            return PyBytes_FromStringAndSize(
                <char*>self._fixed_ptr, <Py_ssize_t>self._cursor)
        if not self._cursor:            # fully drained (the usual case, after flush())
            return b"".join(self._result)
        return b"".join(self._result + [PyBytes_FromStringAndSize(
            <char*>self._fixed_ptr, <Py_ssize_t>self._cursor)])

    # --- scalars ------------------------------------------------------------

    def write_unsigned(self, object field_id, object value):
        if not self._begin():
            return
        cdef uint64_t fid
        cdef uint64_t uv
        try:
            uv = _u64_arg(value)          # value range first, as the pure engine does
            fid = _id_arg(field_id)
            self._header_c(fid, _WT_UNSIGNED)
            self._emit_varint(uv)
        except SofaError as exc:
            self._fail(exc)

    def write_signed(self, object field_id, object value):
        if not self._begin():
            return
        cdef uint64_t fid
        cdef int64_t sv
        try:
            sv = _i64_arg(value)          # value range first, as the pure engine does
            fid = _id_arg(field_id)
            self._header_c(fid, _WT_SIGNED)
            self._emit_varint(_zigzag_encode(sv))
        except SofaError as exc:
            self._fail(exc)

    def write_bool(self, object field_id, object value):
        # A boolean is an unsigned 0/1 on the wire (§4.4). Written here rather
        # than by delegating to write_unsigned: the delegation was a full Python
        # method call per boolean, and the value needs no range check at all.
        if not self._begin():
            return
        try:
            self._header_c(_id_arg(field_id), _WT_UNSIGNED)
            self._emit_varint(1 if value else 0)
        except SofaError as exc:
            self._fail(exc)

    def write_float32(self, object field_id, double value):
        cdef unsigned char buf[4]
        _pack_f32(value, buf)
        self._write_fixlen_raw(field_id, buf, 4, _ST_FP32)

    def write_float64(self, object field_id, double value):
        cdef unsigned char buf[8]
        _pack_f64(value, buf)
        self._write_fixlen_raw(field_id, buf, 8, _ST_FP64)

    def write_string(self, object field_id, str text):
        # Strict UTF-8: no errors= argument, so a lone/unpaired surrogate raises
        # UnicodeEncodeError, which we map to SofaRangeError — the encode-side
        # InvalidArgument outcome (CORELIB_PLAN §6.4 / MESSAGE_SPEC §8). Python
        # str is a Unicode type, hence always strict; SOFAB_STRICT_UTF8 is a
        # no-op and omitted. Mirrors the pure-Python Encoder.write_string.
        #
        # The payload comes from the str's own (cached) UTF-8 form rather than a
        # fresh ``text.encode("utf-8")`` bytes object: the bytes were only ever a
        # vehicle for a pointer and a length, and this is the same encoder, same
        # strictness, one allocation fewer per string field.
        if not self._begin():
            return
        cdef const char* utf8
        cdef Py_ssize_t n
        try:
            utf8 = PyUnicode_AsUTF8AndSize(text, &n)
        except UnicodeEncodeError as exc:
            self._fail(SofaRangeError("string field is not valid UTF-8: %s" % exc))
            return
        self._write_fixlen_raw(field_id, <const unsigned char*>utf8, <size_t>n, _ST_STRING)

    def write_bytes(self, object field_id, object data):
        if not self._begin():
            return
        # Screen the blob on its declared length before the copy (§6.2
        # FIXLEN_MAX): a payload that is about to be refused is not worth
        # duplicating first. _write_fixlen_bytes re-checks the materialised
        # length, which is what actually reaches the wire. Mirrors the
        # pure-Python Encoder.write_bytes.
        cdef Py_ssize_t n = len(data)
        if n > <Py_ssize_t>_FIXLEN_MAX:
            self._fail(SofaRangeError(
                "fixlen payload of %d bytes exceeds FIXLEN_MAX=%d" % (n, _FIXLEN_MAX)))
            return
        cdef bytes b = bytes(data)
        self._write_fixlen_bytes(field_id, b, _ST_BLOB)

    cdef int _write_fixlen_raw(self, object field_id, const unsigned char* data,
                               size_t n, int subtype) except -1:
        if not self._begin():
            return 0
        try:
            # §6.2: FIXLEN_MAX is a format-wide ceiling — a longer payload could
            # only be framed by a fixlen word (§4.6) every conformant decoder
            # rejects, so it is InvalidArgument (§6.3) here rather than an
            # unreadable message reported as success (§5.1). Before the header,
            # so a refused field leaves nothing behind.
            if n > <size_t>_FIXLEN_MAX:
                raise SofaRangeError(
                    "fixlen payload of %d bytes exceeds FIXLEN_MAX=%d" % (n, _FIXLEN_MAX))
            self._header(field_id, _WT_FIXLEN)
            self._emit_varint((<uint64_t>n << 3) | <uint64_t>subtype)
            self._put(data, n)
        except SofaError as exc:
            self._fail(exc)
        return 0

    cdef int _write_fixlen_bytes(self, object field_id, bytes data, int subtype) except -1:
        if not self._begin():
            return 0
        cdef Py_ssize_t n = PyBytes_GET_SIZE(data)
        try:
            if n > <Py_ssize_t>_FIXLEN_MAX:  # §6.2 — see _write_fixlen_raw
                raise SofaRangeError(
                    "fixlen payload of %d bytes exceeds FIXLEN_MAX=%d" % (n, _FIXLEN_MAX))
            self._header(field_id, _WT_FIXLEN)
            self._emit_varint((<uint64_t>n << 3) | <uint64_t>subtype)
            self._put(<const unsigned char*>PyBytes_AS_STRING(data), <size_t>n)
        except SofaError as exc:
            self._fail(exc)
        return 0

    # --- arrays -------------------------------------------------------------

    def write_unsigned_array(self, object field_id, values):
        if not self._begin():
            return
        cdef list seq
        cdef Py_ssize_t i, count
        try:
            seq = _as_list(values)
            count = PyList_GET_SIZE(seq)
            self._array_header(field_id, _WT_ARRAY_UNSIGNED, count)
            for i in range(count):
                self._emit_varint(_u64_elem(_elem(seq, i), _WHAT_UA))
        except SofaError as exc:
            self._fail(exc)

    def write_signed_array(self, object field_id, values):
        if not self._begin():
            return
        cdef list seq
        cdef Py_ssize_t i, count
        try:
            seq = _as_list(values)
            count = PyList_GET_SIZE(seq)
            self._array_header(field_id, _WT_ARRAY_SIGNED, count)
            for i in range(count):
                self._emit_varint(_zigzag_encode(_i64_elem(_elem(seq, i), _WHAT_SA)))
        except SofaError as exc:
            self._fail(exc)

    def write_float32_array(self, object field_id, values):
        self._write_float_array(field_id, values, _ST_FP32, 4)

    def write_float64_array(self, object field_id, values):
        self._write_float_array(field_id, values, _ST_FP64, 8)

    cdef int _write_float_array(self, object field_id, values, int subtype,
                                int elem_size) except -1:
        if not self._begin():
            return 0
        cdef list seq
        cdef Py_ssize_t count, i
        cdef unsigned char* region
        cdef double d
        try:
            seq = _as_float_list(values)
            count = PyList_GET_SIZE(seq)
            self._array_header(field_id, _WT_ARRAY_FIXLEN, count)
            # §4.8: the fixlen_word is ALWAYS emitted (even for an empty array),
            # then the packed payload (zero bytes when empty).
            self._emit_varint((<uint64_t>elem_size << 3) | <uint64_t>subtype)
            if count == 0:
                return 0
            # Pack the whole payload straight into the output buffer when it
            # fits in the room left; otherwise element by element, which drains
            # as it goes (and may cross into a buffer the sink installs).
            if self._cursor + <size_t>(count * elem_size) <= self._fixed_cap:
                region = self._fixed_ptr + self._cursor
                if elem_size == 4:
                    for i in range(count):
                        _pack_f32(_AsDouble(PyList_GET_ITEM(seq, i)), region + i * 4)
                else:
                    for i in range(count):
                        _pack_f64(_AsDouble(PyList_GET_ITEM(seq, i)), region + i * 8)
                self._cursor += <size_t>(count * elem_size)
            else:
                if elem_size == 4:
                    for i in range(count):
                        self._put_f32(_AsDouble(PyList_GET_ITEM(seq, i)))
                else:
                    for i in range(count):
                        self._put_f64(_AsDouble(PyList_GET_ITEM(seq, i)))
        except SofaError as exc:
            self._fail(exc)
        return 0

    cdef inline int _put_f32(self, double value) except -1:
        cdef unsigned char buf[4]
        _pack_f32(value, buf)
        self._put(buf, 4)
        return 0

    cdef inline int _put_f64(self, double value) except -1:
        cdef unsigned char buf[8]
        _pack_f64(value, buf)
        self._put(buf, 8)
        return 0

    cdef int _array_header(self, object field_id, int wtype, Py_ssize_t count) except -1:
        if count < 0 or count > <Py_ssize_t>_ARRAY_MAX:
            raise SofaRangeError("array count %d out of range 0..%d" % (count, _ARRAY_MAX))
        self._header(field_id, wtype)
        self._emit_varint(<uint64_t>count)
        return 0

    # --- sequences ----------------------------------------------------------

    # The module disables keyword arguments so that the zero-argument methods —
    # the ones the decode loop calls per field — can use CPython's METH_NOARGS
    # calling convention, which the interpreter specializes. That trade is only
    # free where there is no keyword to pass, so the two methods that take one
    # argument keep accepting it by name, exactly as the pure engine does.
    @cython.always_allow_keywords(True)
    def write_sequence_begin_lazy(self, object field_id):
        # Open a sequence and hold its header back until it turns out to have
        # content (MESSAGE_SPEC S2). See sofab.encoder.Encoder for the contract.
        if not self._begin():
            return
        try:
            if self._depth >= _MAX_DEPTH:
                raise SofaRangeError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
            # This is the one write that does not go through _header — the id is
            # held back rather than emitted — so it applies the same rule itself.
            self._pending_push(<uint32_t>_id_arg(field_id))
            self._depth += 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end(self):
        # Close the innermost sequence, dropping it entirely (header and end
        # marker) if it received no content.
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            if self._npending:
                # The innermost open sequence is the last held-back one (the
                # pending run is a suffix), so dropping it is a plain pop.
                self._npending -= 1
                self._depth -= 1
                return
            self._emit_varint(<uint64_t>_WT_SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end_keep(self):
        # Close the innermost sequence, keeping its frame even when contentless:
        # behaves like a write, so it commits the whole pending run first.
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            if self._npending:
                self._commit_pending()
            self._emit_varint(<uint64_t>_WT_SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)


# --- float pack/unpack (always little-endian, endian-independent) ------------

cdef inline void _pack_f32(double value, unsigned char* out) noexcept nogil:
    cdef float f
    cdef uint32_t bits
    cdef uint64_t dbits
    # A hardware fp64->fp32 narrowing quiets a signaling NaN (sets the mantissa
    # is-quiet bit), which §4.6 forbids: every float payload, NaN included, must
    # round-trip bit-for-bit. For a NaN, narrow the raw bits by hand so the sign,
    # payload, and signaling bit survive; non-NaN values narrow exactly via cast.
    if value != value:  # NaN — only a NaN is unequal to itself
        memcpy(&dbits, &value, 8)
        # Top 23 mantissa bits of the fp64 (>> 29) become the fp32 mantissa,
        # keeping sign (bit 63->31) and the is-quiet bit (bit 51->22).
        bits = (<uint32_t>(dbits >> 63) << 31) | <uint32_t>0x7F800000 | \
               <uint32_t>((dbits >> 29) & <uint64_t>0x007FFFFF)
        if (bits & <uint32_t>0x007FFFFF) == 0:
            # Payload lived only in the dropped low bits; keep it a (quiet) NaN
            # rather than letting it collapse to inf.
            bits |= <uint32_t>0x00400000
    else:
        f = <float>value
        memcpy(&bits, &f, 4)
    out[0] = <unsigned char>(bits & 0xFF)
    out[1] = <unsigned char>((bits >> 8) & 0xFF)
    out[2] = <unsigned char>((bits >> 16) & 0xFF)
    out[3] = <unsigned char>((bits >> 24) & 0xFF)

cdef inline void _pack_f64(double value, unsigned char* out) noexcept nogil:
    cdef uint64_t bits
    memcpy(&bits, &value, 8)
    cdef int i
    for i in range(8):
        out[i] = <unsigned char>((bits >> (8 * i)) & 0xFF)

cdef inline double _unpack_f32(const unsigned char* p) noexcept nogil:
    cdef uint32_t bits = (<uint32_t>p[0]) | (<uint32_t>p[1] << 8) | \
                         (<uint32_t>p[2] << 16) | (<uint32_t>p[3] << 24)
    cdef float f
    cdef uint64_t dbits
    cdef double d
    # A hardware fp32->fp64 widening quiets a signaling NaN; §4.6 forbids any
    # normalization. For a NaN, widen the raw bits by hand: the 23-bit fp32
    # mantissa (top bit = is-quiet) maps to the top 23 bits of the fp64 mantissa
    # (<< 29), so the signaling bit and payload survive the trip through a
    # Python float. Non-NaN values widen exactly via the cast below.
    if (bits & <uint32_t>0x7F800000) == <uint32_t>0x7F800000 and \
       (bits & <uint32_t>0x007FFFFF):
        dbits = (<uint64_t>(bits >> 31) << 63) | (<uint64_t>0x7FF << 52) | \
                (<uint64_t>(bits & <uint32_t>0x007FFFFF) << 29)
        memcpy(&d, &dbits, 8)
        return d
    memcpy(&f, &bits, 4)
    return <double>f

cdef inline double _unpack_f64(const unsigned char* p) noexcept nogil:
    cdef uint64_t bits = 0
    cdef int i
    for i in range(8):
        bits |= (<uint64_t>p[i]) << (8 * i)
    cdef double d
    memcpy(&d, &bits, 8)
    return d


# =============================================================================
# Decoder
# =============================================================================

cdef class Decoder:
    """Native pull decoder — see :class:`sofab.decoder.Decoder` for the contract.

    Reads from any object exposing ``read(n) -> bytes``. Incoming bytes are held
    in one contiguous buffer and parsed by advancing a C cursor with direct
    pointer indexing; it refills transparently from the reader when it runs off
    the end mid-item, so it serves both a fully-buffered message and a reader
    that dribbles one byte at a time.
    """

    cdef object _read
    cdef int _chunk
    # Receiver-configured decode limits (None = no limit); kept as Python objects
    # so the comparison stays exact for a caller-supplied int of any magnitude.
    cdef object _max_array_count
    cdef object _max_string_len
    cdef object _max_blob_len
    cdef bytes _buf                 # owns the bytes the pointer indexes into
    # Owns a fixlen payload that had to be assembled across refills, for as long
    # as a pointer into it can still be in use (see _take_fixlen_ptr).
    cdef bytes _spill
    cdef const unsigned char* _p
    cdef Py_ssize_t _n
    cdef Py_ssize_t _pos
    cdef int _depth
    cdef object _cur
    # Wire type of ``_cur`` as a plain int (-1 before the first field / at EOF),
    # so the internal dispatch in skip()/drive() compares C ints instead of
    # rich-comparing WireType members.
    cdef int _cur_wtype
    # pending unconsumed value
    cdef int _pk                    # pending kind
    # A parked receiver-cap rejection (§6.2.1): the kind _PEND_LIMIT stands in
    # for, and the message the consume path raises. See _park_limit.
    cdef int _pk_real
    cdef object _limit_msg
    cdef int _pend_wtype
    cdef int _pend_subtype
    cdef uint64_t _pend_count
    cdef uint64_t _pend_size
    # Resume transaction (§5.2): _keep is the buffer offset the call in flight
    # started at, _floor is -1 or the compaction floor a multi-field walk pins,
    # _keep_cur_wtype the current-field wire type as of _keep. See _arm/_suspend.
    cdef Py_ssize_t _keep
    cdef Py_ssize_t _floor
    cdef int _keep_cur_wtype

    def __cinit__(self, reader, *, int chunk_size=65536,
                  max_array_count=None, max_string_len=None, max_blob_len=None):
        self._read = reader.read
        self._chunk = chunk_size
        self._max_array_count = max_array_count
        self._max_string_len = max_string_len
        self._max_blob_len = max_blob_len
        self._buf = b""
        self._p = <const unsigned char*>PyBytes_AS_STRING(self._buf)
        self._n = 0
        self._pos = 0
        self._depth = 0
        self._cur = None
        self._cur_wtype = -1
        self._spill = None
        self._pk = _PEND_NONE
        self._pk_real = _PEND_NONE
        self._limit_msg = None
        self._keep = 0
        self._floor = -1
        self._keep_cur_wtype = -1

    cdef inline void _rebind(self, bytes newbuf):
        self._buf = newbuf
        self._p = <const unsigned char*>PyBytes_AS_STRING(newbuf)
        self._n = PyBytes_GET_SIZE(newbuf)

    # --- resume transactions (CORELIB_PLAN §5.2) ----------------------------
    #
    # Mirrors Decoder._suspend in the pure engine — see the long comment there.
    # Running out of bytes mid-construct is INCOMPLETE, a first-class outcome the
    # caller answers with more bytes, so every public call is all-or-nothing: on
    # the suspension path the cursor goes back to where the call started and the
    # bytes already parsed stay buffered, so re-issuing the call re-parses the
    # construct from its first byte. The pending value / depth / current field
    # are committed only after the construct's last byte is in hand, which is why
    # the rewind is a cursor (plus the current wire type, which _next_field
    # publishes before the varints that trail a fixlen/array header). INVALID is
    # terminal and is deliberately not rewound.

    cdef inline void _arm(self):
        self._keep = self._pos
        self._keep_cur_wtype = self._cur_wtype

    cdef object _suspend(self, msg):
        # _keep tracks buffer compaction in _need, so it names the call's start
        # byte in the current buffer.
        self._pos = self._keep
        self._cur_wtype = self._keep_cur_wtype
        return SofaIncompleteError(msg)

    # --- byte sourcing ------------------------------------------------------

    cdef bint _need(self, Py_ssize_t n) except -1:
        # Ensure at least n bytes available at _pos, refilling from the reader.
        # Everything the reader hands over is kept: on failure the bytes read so
        # far stay buffered, which is what makes a suspension non-destructive.
        cdef bytes data
        cdef bytes buf
        cdef bytearray acc
        cdef Py_ssize_t pos = self._pos
        cdef Py_ssize_t base
        cdef Py_ssize_t want
        cdef Py_ssize_t req
        if self._n - pos >= n:
            return True
        # Compaction may only drop what can never be read again. With a resume
        # transaction open that floor is the transaction's start, not the
        # cursor: the bytes in between belong to the construct being parsed and
        # a suspension has to be able to replay them (§5.2).
        base = self._keep if self._floor < 0 else self._floor
        buf = self._buf
        if base:
            buf = buf[base:]
            pos -= base
            self._keep -= base
            if self._floor > 0:
                self._floor -= base
        want = pos + n
        if PyBytes_GET_SIZE(buf) < want:
            req = want - PyBytes_GET_SIZE(buf)
            if req < <Py_ssize_t>self._chunk:
                req = <Py_ssize_t>self._chunk
            data = self._read(req)
            if not data:
                self._rebind(buf)
                self._pos = pos
                return False
            buf = buf + data if PyBytes_GET_SIZE(buf) else data
            if PyBytes_GET_SIZE(buf) < want:
                # More than one read needed: accumulate in a bytearray from here,
                # since repeated bytes+bytes would make a chunk-fed large payload
                # quadratic. (One read is the common case and stays a plain
                # concatenation — often not even that, on the first fill.)
                acc = bytearray(buf)
                while <Py_ssize_t>len(acc) < want:
                    req = want - <Py_ssize_t>len(acc)
                    if req < <Py_ssize_t>self._chunk:
                        req = <Py_ssize_t>self._chunk
                    data = self._read(req)
                    if not data:
                        self._rebind(bytes(acc))
                        self._pos = pos
                        return False
                    acc += data
                buf = bytes(acc)
        self._rebind(buf)
        self._pos = pos
        return True

    cdef inline uint64_t _varint(self) except? 0xDEAD:
        # Fast path: a varint is at most 10 bytes, so with that many buffered the
        # whole value is known to be present and the per-byte "is there another
        # byte?" test (and the refill machinery behind it) disappears. Only the
        # tail of the buffer — where a value may genuinely straddle a refill —
        # needs the careful loop.
        cdef _Varint v
        if self._n - self._pos >= 10:
            v = _varint_take(self._p + self._pos)
            if v.used < 0:
                raise SofaDecodeError("overlong varint")
            self._pos += v.used
            return v.value
        return self._varint_refill()

    cdef uint64_t _varint_refill(self) except? 0xDEAD:
        cdef Py_ssize_t pos = self._pos
        cdef const unsigned char* p = self._p
        cdef Py_ssize_t n = self._n
        cdef unsigned char b
        cdef uint64_t result
        cdef int shift
        cdef int room
        if pos >= n:
            if not self._need(1):
                raise self._suspend("truncated varint")
            p = self._p
            pos = self._pos
            n = self._n
        b = p[pos]
        pos += 1
        if b < 0x80:
            self._pos = pos
            return b
        result = b & 0x7F
        shift = 7
        while True:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise self._suspend("truncated varint")
                p = self._p
                pos = self._pos
                n = self._n
            b = p[pos]
            pos += 1
            # Reject an overlong (>64-bit) varint before OR-ing: if this byte's
            # 7 payload bits would spill past bit 63 they would be truncated by
            # the uint64_t and must instead be INVALID (§4.1/§6.3, issue #43).
            # ``room`` (bits left below 64) is always >= 1 here, so the shift is
            # well-defined C (a `>> (64 - shift)` with shift 7 is UB for int).
            room = 64 - shift
            if room < 7 and (b & 0x7F) >> room:
                raise SofaDecodeError("overlong varint")
            result |= (<uint64_t>(b & 0x7F)) << shift
            if b < 0x80:
                self._pos = pos
                return result
            shift += 7
            if shift >= 64:
                raise SofaDecodeError("overlong varint")

    cdef bytes _read_exact(self, Py_ssize_t n):
        # The slow path accumulates inside the buffer (via _need) rather than in
        # a local, so a payload that stops halfway is still buffered when the
        # truncation is reported and the next attempt continues from it (§5.2).
        cdef Py_ssize_t pos = self._pos
        cdef bytes out
        if pos + n <= self._n:
            out = self._buf[pos:pos + n]
            self._pos = pos + n
            return out
        if not self._need(n):
            raise self._suspend("truncated payload")
        pos = self._pos
        out = self._buf[pos:pos + n]
        self._pos = pos + n
        return out

    cdef list _read_varints(
        self, Py_ssize_t count, bint zigzag, bint bounded, int64_t lo, int64_t hi
    ):
        # Decode ``count`` consecutive varints into a list. ``zigzag`` folds the
        # signed-array transform into this same pass, so a signed array is one
        # walk producing one list rather than a list of raw values rebuilt into a
        # second list of decoded ones.
        #
        # ``bounded``/``lo``/``hi`` carry the field's declared element width
        # (§7.1). It is checked AT the element, before the value is boxed: two
        # typed integer compares. The pure engine checks at the same point (see
        # Decoder._read_varints) — §7.1 requires the two to agree on which
        # messages are valid, so neither may make the verdict depend on whether
        # the array happened to complete (issue #67). Checking there also makes
        # §5.2's precedence fall out of the order: the INVALID is raised before
        # a truncation behind the bad element is ever reached (generator#267,
        # Crucible F-0043).
        #
        # The result is pre-sized, but never on the strength of the wire count
        # alone: ``count`` is attacker-controlled and capped only at ARRAY_MAX
        # (2^31), so PyList_New(count) would let a tiny hostile message claiming
        # count = 2^31 demand ~16 GB of NULL slots before a single element byte
        # is read (amplification DoS, issue #31). Each element occupies at least
        # one wire byte, so the bytes actually *buffered* are an upper bound on
        # the elements actually present — pre-size to that and the allocation
        # stays proportional to data really received, while the fully-buffered
        # common case still pre-sizes exactly once. Anything beyond it (a
        # chunk-fed reader delivering more as we go) falls back to appending.
        cdef Py_ssize_t pos = self._pos
        cdef const unsigned char* p = self._p
        cdef Py_ssize_t n = self._n
        cdef Py_ssize_t prealloc = n - pos
        cdef Py_ssize_t i = 0
        cdef _Varint v
        cdef uint64_t result
        cdef int64_t signed_val
        cdef object item
        if prealloc > count:
            prealloc = count
        cdef list out = PyList_New(prealloc)
        while i < count:
            if n - pos >= 10:
                # Whole element guaranteed buffered: no per-byte bounds test.
                v = _varint_take(p + pos)
                if v.used < 0:
                    raise SofaDecodeError("overlong varint")
                pos += v.used
                result = v.value
            else:
                # Near the end of the buffer an element may straddle a refill.
                self._pos = pos
                result = self._varint_refill()
                p = self._p
                pos = self._pos
                n = self._n
            if zigzag:
                signed_val = _zigzag_decode(result)
                if bounded and (signed_val < lo or signed_val > hi):
                    raise SofaDecodeError("array element outside declared width")
                item = PyLong_FromLongLong(signed_val)
            else:
                # hi is a NARROWED unsigned maximum (u32 at the widest), so it is
                # never negative and the unsigned compare is exact.
                if bounded and result > <uint64_t>hi:
                    raise SofaDecodeError("array element outside declared width")
                item = PyLong_FromUnsignedLongLong(result)
            if i < prealloc:
                Py_INCREF(item)
                PyList_SET_ITEM(out, i, item)
            else:
                PyList_Append(out, item)
            i += 1
        self._pos = pos
        return out

    cdef int _skip_varints(self, Py_ssize_t count) except -1:
        cdef Py_ssize_t pos = self._pos
        cdef const unsigned char* p = self._p
        cdef Py_ssize_t n = self._n
        cdef Py_ssize_t i = 0
        while i < count:
            if pos < n and p[pos] < 0x80:
                pos += 1
                i += 1
                continue
            self._pos = pos
            self._varint()
            p = self._p
            pos = self._pos
            n = self._n
            i += 1
        self._pos = pos
        return 0

    # --- field iteration ----------------------------------------------------

    @property
    def field(self):
        return self._cur

    def next(self):
        return self._next_field()

    cdef object _next_field(self):
        # The body of ``next()``, callable from C: ``skip()`` and ``drive()``
        # both iterate fields, and going through the Python method wrapper for
        # every one of them costs a full attribute lookup and call frame.
        cdef uint64_t header
        cdef int wtype
        cdef object field_id
        cdef uint64_t fid
        cdef uint64_t length_header, length, count, elem_header, elem_size
        cdef int subtype

        self._arm()   # opens this field's resume transaction (§5.2)

        if self._pk != _PEND_NONE:
            self._skip_pending()

        if self._pos >= self._n and not self._need(1):
            if self._depth != 0:
                raise self._suspend("truncated: unbalanced sequence")
            self._cur_wtype = -1
            return None

        header = self._varint()
        wtype = <int>(header & 0x07)
        fid = header >> 3
        self._cur_wtype = wtype

        # ID_MAX bounds every header's id (§6.2), the sequence end included even
        # though its id is discarded (§4.9); validate before the wire-type
        # dispatch so wire type 7 is not an exception to the ceiling.
        if fid > _ID_MAX:
            raise SofaDecodeError("id %d out of range" % PyLong_FromUnsignedLongLong(fid))

        if wtype == _WT_SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            self._cur = _mkfield(_ZERO, _WT[_WT_SEQUENCE_END], _ZERO, _ZERO, _NONE)
            return self._cur

        field_id = PyLong_FromUnsignedLongLong(fid)

        if wtype == _WT_SEQUENCE_START:
            if self._depth >= _MAX_DEPTH:
                raise SofaDecodeError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
            self._depth += 1
            self._cur = _mkfield(field_id, _WT[_WT_SEQUENCE_START], _ZERO, _ZERO, _NONE)
            return self._cur

        if wtype == _WT_UNSIGNED or wtype == _WT_SIGNED:
            self._cur = _mkfield(field_id, _WT[wtype], _ZERO, _ZERO, _NONE)
            self._pk = _PEND_SCALAR
            self._pend_wtype = wtype
            return self._cur

        if wtype == _WT_FIXLEN:
            length_header = self._varint()
            length = length_header >> 3
            subtype = <int>(length_header & 0x07)
            if subtype > _ST_BLOB:
                raise SofaDecodeError("invalid fixlen subtype %d" % subtype)
            if length > _FIXLEN_MAX:
                raise SofaDecodeError("fixlen length out of range")
            # A wrong-width fp field is malformed regardless of what bytes
            # follow, so raise this INVALID verdict at header time — before any
            # payload read — so it takes precedence over the INCOMPLETE a
            # truncated payload would otherwise raise (§7). Keeps this engine
            # byte-for-byte identical to the pure decoder. STRING/BLOB are
            # variable-length, so a truncated one is legitimately INCOMPLETE.
            if subtype == _ST_FP32 and length != 4:
                raise SofaDecodeError("fp32 fixlen length must be 4")
            if subtype == _ST_FP64 and length != 8:
                raise SofaDecodeError("fp64 fixlen length must be 8")
            self._cur = _mkfield(field_id, _WT[_WT_FIXLEN],
                                 PyLong_FromUnsignedLongLong(length), _ZERO, _ST[subtype])
            self._pk = _PEND_FIXLEN
            self._pend_subtype = subtype
            self._pend_size = length
            # Receiver-configured caps (policy, not malformation): the verdict on
            # an oversize string/blob is reached here, on the length word alone —
            # before its payload is read or buffered — and PARKED on the pending
            # value rather than raised, so the caller keeps the §6.2.1 window in
            # which it can declare the field schema-bounded and take the cap off
            # it. Every consume path raises it; see _park_limit / _pending_error.
            if subtype == _ST_STRING and self._max_string_len is not None \
                    and PyLong_FromUnsignedLongLong(length) > self._max_string_len:
                self._park_limit("string length %d exceeds max_string_len %s"
                                 % (PyLong_FromUnsignedLongLong(length), self._max_string_len))
            elif subtype == _ST_BLOB and self._max_blob_len is not None \
                    and PyLong_FromUnsignedLongLong(length) > self._max_blob_len:
                self._park_limit("blob length %d exceeds max_blob_len %s"
                                 % (PyLong_FromUnsignedLongLong(length), self._max_blob_len))
            return self._cur

        if wtype == _WT_ARRAY_UNSIGNED or wtype == _WT_ARRAY_SIGNED:
            count = self._varint()
            if count > _ARRAY_MAX:
                raise SofaDecodeError("array count %d out of range" % PyLong_FromUnsignedLongLong(count))
            self._cur = _mkfield(field_id, _WT[wtype], _ZERO,
                                 PyLong_FromUnsignedLongLong(count), _NONE)
            self._pk = _PEND_VARRAY
            self._pend_wtype = wtype
            self._pend_count = count
            # Parked, not raised — see the fixlen branch above (§6.2.1).
            if self._max_array_count is not None \
                    and PyLong_FromUnsignedLongLong(count) > self._max_array_count:
                self._park_limit("array count %d exceeds max_array_count %s"
                                 % (PyLong_FromUnsignedLongLong(count), self._max_array_count))
            return self._cur

        # wtype == _WT_ARRAY_FIXLEN
        count = self._varint()
        if count > _ARRAY_MAX:
            raise SofaDecodeError("array count %d out of range" % PyLong_FromUnsignedLongLong(count))
        # §4.8: a fixlen array ALWAYS carries its fixlen_word — read it
        # unconditionally to recover the true subtype/width.
        elem_header = self._varint()
        elem_size = elem_header >> 3
        subtype = <int>(elem_header & 0x07)
        if subtype > _ST_FP64:
            raise SofaDecodeError("invalid fixlen-array subtype %d" % subtype)
        # §4.8/§5.2: a fixlen array carries fp32 (element size 4) or fp64
        # (element size 8) — any other width is malformed. Raise this INVALID
        # verdict at header time, before any payload read, so it takes
        # precedence over the INCOMPLETE a truncated payload would raise (§7).
        # Mirrors the eager element-width check on the scalar fixlen path above.
        # subtype is already narrowed to fp32/fp64, so these exact-width checks
        # bound elem_size completely — no separate FIXLEN_MAX check is needed.
        if subtype == _ST_FP32 and elem_size != 4:
            raise SofaDecodeError("fp32 fixlen-array element size must be 4")
        if subtype == _ST_FP64 and elem_size != 8:
            raise SofaDecodeError("fp64 fixlen-array element size must be 8")
        self._cur = _mkfield(field_id, _WT[_WT_ARRAY_FIXLEN],
                             PyLong_FromUnsignedLongLong(elem_size),
                             PyLong_FromUnsignedLongLong(count), _ST[subtype])
        self._pk = _PEND_FARRAY
        self._pend_subtype = subtype
        self._pend_count = count
        self._pend_size = elem_size
        # Parked, not raised — see the fixlen branch above (§6.2.1).
        if self._max_array_count is not None \
                and PyLong_FromUnsignedLongLong(count) > self._max_array_count:
            self._park_limit("array count %d exceeds max_array_count %s"
                             % (PyLong_FromUnsignedLongLong(count), self._max_array_count))
        return self._cur

    # --- receiver caps vs. schema bounds (§6.2.1) ---------------------------

    cdef inline int _park_limit(self, msg) except -1:
        # Park a receiver-cap rejection on the pending value instead of raising
        # it. The verdict is already final and was reached at the count/length
        # header, before anything was read or allocated; parking it only moves
        # the RAISE to the call that would consume the field, which is what
        # leaves the caller room to declare the field schema-bounded first.
        self._pk_real = self._pk
        self._pk = _PEND_LIMIT
        self._limit_msg = msg
        return 0

    cdef object _pending_error(self, msg):
        # The exception a mismatched pending kind deserves: the parked cap
        # rejection when one is what stands in the way, the caller's state error
        # otherwise. Reached only once a read has already found the kind wrong,
        # so the ordinary path never pays for it.
        if self._pk == _PEND_LIMIT:
            return SofaLimitError(self._limit_msg)
        return SofaStateError(msg)

    def schema_bounded(self):
        # Declare the current field schema-bounded, so the receiver-side caps do
        # not apply to it (§6.2.1). See sofab.decoder.Decoder.schema_bounded for
        # the contract — the declaration covers this field only, is a no-op on a
        # field no cap has rejected, and is a promise that the caller enforces
        # the schema bound itself (as INVALID, MESSAGE_SPEC §7.1).
        if self._pk == _PEND_LIMIT:
            self._pk = self._pk_real
            self._limit_msg = None

    # --- skipping -----------------------------------------------------------

    cdef Py_ssize_t _farray_nbytes(self, uint64_t count, uint64_t elem_size) except -1:
        # On-wire payload size of a fixlen array = count * elem_size. Both are
        # attacker-controlled; the product can overflow uint64 or exceed
        # Py_ssize_t, and casting a wrapped/oversized value straight to a signed
        # size is undefined and can drive the cursor negative. Any size that
        # cannot fit a real buffer is unsatisfiable, so surface it as a truncated
        # payload — the same rejection the pure path reaches when _read_exact
        # runs the reader dry.
        cdef uint64_t total = count * elem_size
        if elem_size != 0 and total // elem_size != count:
            raise self._suspend("truncated payload")
        if total > _SSIZE_MAX:
            raise self._suspend("truncated payload")
        return <Py_ssize_t>total

    cdef bytes _read_farray_payload(self, uint64_t count, uint64_t elem_size, uint64_t width):
        # Read a fixlen array's on-wire payload and verify its element width
        # matches the subtype (4 for fp32, 8 for fp64) before any fixed-width
        # unpack. The returned buffer is guaranteed to be exactly count*width
        # bytes, so an unpack loop reading width bytes per element stays in
        # bounds. A width mismatch is a malformed fixlen_word -> SofaDecodeError
        # (an empty array, count == 0, carries no payload and so cannot mismatch,
        # matching the pure path).
        cdef bytes data = self._read_exact(self._farray_nbytes(count, elem_size))
        if <uint64_t>PyBytes_GET_SIZE(data) != count * width:
            raise SofaDecodeError("fixlen-array element width does not match its subtype")
        return data

    cdef int _skip_pending(self) except -1:
        cdef int kind = self._pk
        if kind == _PEND_SCALAR:
            self._varint()
        elif kind == _PEND_FIXLEN:
            self._read_exact(<Py_ssize_t>self._pend_size)
        elif kind == _PEND_VARRAY:
            self._skip_varints(<Py_ssize_t>self._pend_count)
        elif kind == _PEND_FARRAY:
            self._read_exact(self._farray_nbytes(self._pend_count, self._pend_size))
        else:  # _PEND_LIMIT — a skip still buffers the payload, so the cap binds it
            raise SofaLimitError(self._limit_msg)
        # Cleared only now: had the value run out mid-skip, the field has to stay
        # pending so the retry skips it again from its first byte (§5.2).
        self._pk = _PEND_NONE
        return 0

    def skip(self):
        self._arm()
        self._skip()

    cdef int _skip(self) except -1:
        cdef int target
        cdef int depth, cur_wtype, pk, pend_wtype, pend_subtype
        cdef uint64_t pend_count, pend_size
        cdef object cur
        if self._cur_wtype == _WT_SEQUENCE_START:
            # Walking a whole sequence spans many fields, so unlike every other
            # call this one moves the field state — and lets _next_field re-arm
            # _keep — before it can suspend. _floor pins the refill path's
            # compaction at the first byte *inside* the sequence for the
            # duration, and the field state is put back here, so a re-issued
            # skip() replays the whole sequence (§5.2).
            self._floor = self._pos
            depth = self._depth
            cur = self._cur
            cur_wtype = self._cur_wtype
            pk = self._pk
            pend_wtype = self._pend_wtype
            pend_subtype = self._pend_subtype
            pend_count = self._pend_count
            pend_size = self._pend_size
            target = depth - 1
            try:
                while self._depth > target:
                    if self._next_field() is None:
                        raise self._suspend("truncated sequence")
            except SofaIncompleteError:
                self._pos = self._floor
                self._keep = self._floor
                self._depth = depth
                self._cur = cur
                self._cur_wtype = cur_wtype
                self._pk = pk
                self._pend_wtype = pend_wtype
                self._pend_subtype = pend_subtype
                self._pend_count = pend_count
                self._pend_size = pend_size
                raise
            finally:
                self._floor = -1
            return 0
        if self._pk != _PEND_NONE:
            self._skip_pending()
        return 0

    # --- scalar reads -------------------------------------------------------

    cdef uint64_t _take_scalar(self, int wtype) except? 0xDEAD:
        if self._pk != _PEND_SCALAR or self._pend_wtype != wtype:
            raise SofaStateError("no matching scalar value for the current field")
        self._arm()
        cdef uint64_t value = self._varint()
        self._pk = _PEND_NONE   # committed only once the value is in hand (§5.2)
        return value

    def unsigned(self):
        return PyLong_FromUnsignedLongLong(self._take_scalar(_WT_UNSIGNED))

    def signed(self):
        return PyLong_FromLongLong(_zigzag_decode(self._take_scalar(_WT_SIGNED)))

    def bool(self):
        return self._take_scalar(_WT_UNSIGNED) != 0

    cdef bytes _take_fixlen(self, int subtype):
        cdef bytes data
        if self._pk != _PEND_FIXLEN:
            raise self._pending_error("current field is not a fixlen value")
        if self._pend_subtype != subtype:
            raise SofaStateError("fixlen subtype does not match the requested read")
        self._arm()
        data = self._read_exact(<Py_ssize_t>self._pend_size)
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return data

    cdef const unsigned char* _take_fixlen_ptr(self, int subtype, Py_ssize_t n) except NULL:
        # Fixlen payload as a pointer instead of a ``bytes``. When the payload is
        # already buffered — the ordinary case — the value can be built straight
        # off the buffer, so the copy the intermediate ``bytes`` object used to
        # cost disappears. When it is not (a chunk-fed reader mid-payload), fall
        # back to _read_exact and park its object in ``_spill``, which owns it
        # for as long as the caller can still hold the pointer.
        if self._pk != _PEND_FIXLEN:
            raise self._pending_error("current field is not a fixlen value")
        if self._pend_subtype != subtype:
            raise SofaStateError("fixlen subtype does not match the requested read")
        cdef const unsigned char* p
        if self._n - self._pos >= n:
            self._pk = _PEND_NONE
            self._spill = None
            p = self._p + self._pos
            self._pos += n
            return p
        # Payload not (yet) buffered: this can suspend, so it runs inside a
        # resume transaction (§5.2).
        self._arm()
        self._spill = self._read_exact(n)
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return <const unsigned char*>PyBytes_AS_STRING(self._spill)

    def float32(self):
        # Width is settled at header time (§4.6/§7), so _pend_size is 4 here.
        return _unpack_f32(self._take_fixlen_ptr(_ST_FP32, 4))

    def float64(self):
        return _unpack_f64(self._take_fixlen_ptr(_ST_FP64, 8))

    def fixlen_len(self):
        # Peek the current fixlen field's payload byte length (from its length
        # header) without consuming it — a following string()/bytes()/float* read
        # still takes the same field. Lets a caller bound a string/blob against its
        # schema maxlen on the exact wire byte length, before allocation and without
        # re-encoding a decoded str. Mirrors Decoder.fixlen_len in the pure engine.
        if self._pk != _PEND_FIXLEN:
            # A parked receiver cap (§6.2.1) keeps the pending fixlen intact, and
            # this peek reads and allocates nothing — so it answers through the
            # parked rejection. That is what lets generated code decide the
            # SCHEMA bound (INVALID, §7.1) whether or not schema_bounded() has
            # been called yet. Mirrors Decoder.fixlen_len in the pure engine.
            if self._pk == _PEND_LIMIT and self._pk_real == _PEND_FIXLEN:
                return self._pend_size
            raise SofaStateError("current field is not a fixlen value")
        return self._pend_size

    def string(self):
        cdef Py_ssize_t n
        cdef const unsigned char* p
        if self._pk != _PEND_FIXLEN:
            raise self._pending_error("current field is not a fixlen value")
        n = <Py_ssize_t>self._pend_size      # bounded by FIXLEN_MAX in next()
        p = self._take_fixlen_ptr(_ST_STRING, n)
        try:
            # Decoded straight off the buffer: strict UTF-8 (no ``errors=``), so
            # an invalid payload raises rather than being replaced (§6.4).
            return PyUnicode_DecodeUTF8(<const char*>p, n, NULL)
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def bytes(self):
        cdef Py_ssize_t n
        cdef const unsigned char* p
        if self._pk != _PEND_FIXLEN:
            raise self._pending_error("current field is not a fixlen value")
        n = <Py_ssize_t>self._pend_size
        p = self._take_fixlen_ptr(_ST_BLOB, n)
        if self._spill is not None:
            return self._spill      # already an exact-size bytes; hand it over
        return PyBytes_FromStringAndSize(<const char*>p, n)

    # --- array reads --------------------------------------------------------

    cdef uint64_t _take_varray(self, int wtype) except? 0xDEAD:
        # Validates the pending array and returns its count. The pending value is
        # cleared by the caller only once the payload has actually been decoded,
        # so a suspension leaves the array re-readable from element one (§5.2).
        if self._pk != _PEND_VARRAY or self._pend_wtype != wtype:
            raise self._pending_error("current field is not a matching varint array")
        return self._pend_count

    def read_unsigned_array(self, elem_max=None):
        # elem_max is the field's declared element width; see the pure engine's
        # Decoder.read_unsigned_array for what it buys (§7.1/§5.2, #267).
        cdef uint64_t count = self._take_varray(_WT_ARRAY_UNSIGNED)
        cdef list out
        self._arm()
        if elem_max is None:
            out = self._read_varints(<Py_ssize_t>count, False, False, 0, 0)
        else:
            out = self._read_varints(
                <Py_ssize_t>count, False, True, 0, <int64_t>elem_max
            )
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return out

    def read_signed_array(self, elem_min=None, elem_max=None):
        # The two halves of the declared width are independent: either may be
        # given on its own, in which case it bounds its own side and the other
        # stays at the widest an i64 element can be — passing one alone must not
        # fault on the missing half (issue #67).
        cdef uint64_t count = self._take_varray(_WT_ARRAY_SIGNED)
        cdef list out
        cdef int64_t lo = INT64_MIN
        cdef int64_t hi = INT64_MAX
        self._arm()
        if elem_min is None and elem_max is None:
            out = self._read_varints(<Py_ssize_t>count, True, False, 0, 0)
        else:
            if elem_min is not None:
                lo = <int64_t>elem_min
            if elem_max is not None:
                hi = <int64_t>elem_max
            out = self._read_varints(<Py_ssize_t>count, True, True, lo, hi)
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return out

    cdef _take_farray(self, int subtype):
        # Like _take_varray, the pending value is cleared by the caller only
        # after the payload has been read (§5.2).
        if self._pk != _PEND_FARRAY:
            raise self._pending_error("current field is not a fixlen array")
        if self._pend_subtype != subtype:
            raise SofaStateError("fixlen-array subtype does not match the requested read")
        return self._pend_count, self._pend_size

    cdef bytes _take_farray_payload(self, int subtype, uint64_t width):
        # Validate the pending fixlen array, then read its payload inside a
        # resume transaction: an array whose payload has not fully arrived stays
        # pending and re-readable from its first payload byte (§5.2).
        cdef uint64_t count, elem_size
        cdef bytes data
        count, elem_size = self._take_farray(subtype)
        self._arm()
        data = self._read_farray_payload(count, elem_size, width)
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return data

    def read_float32_array(self):
        # Consume the payload the fixlen_word claims (count * elem_size bytes),
        # then require it to be exactly count*4 — i.e. the element width must be
        # 4 for an fp32 array. Without this an elem_size != 4 (e.g. 0) leaves the
        # buffer shorter than the count*4 bytes the fixed-width unpack loop reads,
        # a heap over-read (SIGSEGV under boundscheck=False). The pure path is
        # implicitly guarded by struct.unpack demanding an exact-size buffer.
        cdef bytes data = self._take_farray_payload(_ST_FP32, 4)
        cdef uint64_t count = <uint64_t>PyBytes_GET_SIZE(data) // 4
        cdef const unsigned char* p = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef list out = PyList_New(<Py_ssize_t>count)
        cdef Py_ssize_t i
        cdef object item
        for i in range(<Py_ssize_t>count):
            item = float(_unpack_f32(p + i * 4))
            Py_INCREF(item)
            PyList_SET_ITEM(out, i, item)
        return out

    def read_float64_array(self):
        # See read_float32_array: the element width must be 8 for an fp64 array,
        # or the count*8-byte unpack loop over-reads a shorter buffer.
        cdef bytes data = self._take_farray_payload(_ST_FP64, 8)
        cdef uint64_t count = <uint64_t>PyBytes_GET_SIZE(data) // 8
        cdef const unsigned char* p = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef list out = PyList_New(<Py_ssize_t>count)
        cdef Py_ssize_t i
        cdef object item
        for i in range(<Py_ssize_t>count):
            item = float(_unpack_f64(p + i * 8))
            Py_INCREF(item)
            PyList_SET_ITEM(out, i, item)
        return out

    # --- visitor driver -----------------------------------------------------

    @cython.always_allow_keywords(True)
    def drive(self, visitor):
        # Dispatch on the C wire type rather than rich-comparing the WireType
        # member against up to eight candidates per field.
        cdef object f
        cdef int t
        cdef object st
        while True:
            f = self._next_field()
            if f is None:
                break
            t = self._cur_wtype
            if t == _WT_SEQUENCE_END:
                visitor.on_sequence_end()
            elif t == _WT_SEQUENCE_START:
                if visitor.on_sequence_begin(f.id) is False:
                    self.skip()
            elif visitor.on_field(f) is False:
                self.skip()
            elif t == _WT_UNSIGNED:
                visitor.on_unsigned(f.id, self.unsigned())
            elif t == _WT_SIGNED:
                visitor.on_signed(f.id, self.signed())
            elif t == _WT_FIXLEN:
                st = f.subtype
                if st == _ST[_ST_FP32]:
                    visitor.on_float32(f.id, self.float32())
                elif st == _ST[_ST_FP64]:
                    visitor.on_float64(f.id, self.float64())
                elif st == _ST[_ST_STRING]:
                    visitor.on_string(f.id, self.string())
                else:
                    visitor.on_bytes(f.id, self.bytes())
            elif t == _WT_ARRAY_UNSIGNED:
                visitor.on_unsigned_array(f.id, self.read_unsigned_array())
            elif t == _WT_ARRAY_SIGNED:
                visitor.on_signed_array(f.id, self.read_signed_array())
            else:  # ARRAY_FIXLEN
                if f.subtype == _ST[_ST_FP32]:
                    visitor.on_float32_array(f.id, self.read_float32_array())
                else:
                    visitor.on_float64_array(f.id, self.read_float64_array())


# Marker so callers / tests can assert which implementation is active.
IMPL = "native"
