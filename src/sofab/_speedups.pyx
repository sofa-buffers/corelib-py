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
  which is what lets CPython's inline caches turn ``field.id`` and
  ``dec.feed(chunk)`` into their specialized forms instead of the generic
  attribute path.
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
from cpython.bytearray cimport (PyByteArray_AS_STRING, PyByteArray_GET_SIZE,
                                PyByteArray_FromStringAndSize)
from cpython.list cimport PyList_Append, PyList_GET_ITEM, PyList_GET_SIZE, PyList_New
from cpython.buffer cimport (PyObject_GetBuffer, PyBuffer_Release,
                             PyBUF_WRITABLE, PyBUF_SIMPLE)
cdef extern from "Python.h":
    # One object over the memory, with no Py_buffer round trip and no slice of a
    # parent view -- the flush path builds one of these per drain.
    object PyMemoryView_FromMemory(char* mem, Py_ssize_t size, int flags)
    int PyBUF_READ
from cpython.long cimport PyLong_FromUnsignedLongLong, PyLong_FromLongLong
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)
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

    /* How many digits a 64-bit magnitude can occupy, and how many bits the most
     * significant of them may carry. With PyLong_SHIFT == 30 that is 3 digits
     * whose top one holds 4 bits; with 15 it is 5 digits, top one 4 bits. Both
     * are compile-time constants, which is what lets the width test run ONCE per
     * value instead of once per digit. */
    #define __SOFAB_NDIG ((Py_ssize_t) ((64 + PyLong_SHIFT - 1) / PyLong_SHIFT))
    #define __SOFAB_TOPBITS (64 - (__SOFAB_NDIG - 1) * PyLong_SHIFT)

    /* |x| as a uint64_t, ignoring its sign. Returns 0 (leaving no exception set)
       when the magnitude does not fit in 64 bits. */
    static int __sofab_mag_digits(PyObject *x, uint64_t *out) {
        Py_ssize_t n = __sofab_ndigits(x);
        const digit *d = __sofab_digits(x);
        uint64_t v = 0;
        /* Only the most significant digit can push the value past 64 bits, so
           one test on the digit count settles the whole value and the
           accumulate loop below carries no branch at all. */
        if (n >= __SOFAB_NDIG) {
            if (n > __SOFAB_NDIG) return 0;
            if (((uint64_t) d[n - 1]) >> __SOFAB_TOPBITS) return 0;
        }
        while (n-- > 0) v = (v << PyLong_SHIFT) | (uint64_t) d[n];
        *out = v;
        return 1;
    }

    /* Magnitude as a uint64_t. Returns 0 (leaving no exception set) when the
       value does not fit, which is exactly the format's unsigned domain. */
    static int __sofab_u64_digits(PyObject *x, uint64_t *out) {
        if (__sofab_isneg(x)) return 0;
        return __sofab_mag_digits(x, out);
    }

    static int __sofab_u64(PyObject *x, uint64_t *out) {
        unsigned long long v;
        if (__sofab_digits_ok) return __sofab_u64_digits(x, out);
        v = __sofab_as_u64(x);
        if (v == (unsigned long long) -1 && PyErr_Occurred()) { PyErr_Clear(); return 0; }
        *out = (uint64_t) v;
        return 1;
    }

    /* A field id, when it is one that needs no range test at all. A CPython
       digit is 30 bits wide (15 on a narrow build), so a non-negative int of at
       most one digit is always inside 0..ID_MAX — which is every id a schema
       realistically uses. Anything else returns 0 and takes the general path. */
    static int __sofab_small_id(PyObject *x, uint64_t *out) {
        if (!__sofab_digits_ok || __sofab_isneg(x)) return 0;
        switch (__sofab_ndigits(x)) {
            case 0:  *out = 0; return 1;
            case 1:  *out = (uint64_t) __sofab_digits(x)[0]; return 1;
            default: return 0;
        }
    }

    static int __sofab_i64(PyObject *x, int64_t *out) {
        long long v;
        uint64_t mag;
        if (__sofab_digits_ok) {
            int neg = __sofab_isneg(x);
            if (!__sofab_mag_digits(x, &mag)) return 0;
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
    # surface as SofaArgumentError, not as the bare OverflowError left pending.
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
    # An id small enough that being an id is already proof it is in range.
    bint _ToSmallId "__sofab_small_id" (PyObject*, uint64_t*) noexcept
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
    # --- element builders that stay at the C level ---------------------------
    #
    # The array decode loops build one Python object per element and put it
    # straight into the result list. Spelled with the ordinary ``object``-typed
    # declarations, every element additionally pays Cython's bookkeeping for the
    # temporary that holds it: an incref for the borrowed ``PyList_SET_ITEM``
    # signature and a decref when the temporary is reassigned on the next
    # iteration. These spellings hand the fresh reference to the list directly —
    # ``PyList_SET_ITEM`` *steals* it — so the round trip disappears and the
    # element costs exactly one allocation.
    PyObject* _NewU64 "PyLong_FromUnsignedLongLong" (uint64_t) except NULL
    PyObject* _NewI64 "PyLong_FromLongLong" (int64_t) except NULL
    PyObject* _NewF64 "PyFloat_FromDouble" (double) except NULL
    void _SetItemSteal "PyList_SET_ITEM" (object, Py_ssize_t, PyObject*)
    # Same steal, but for a slot that already holds an item: releases the old
    # reference instead of leaking it. The array reads above fill a freshly
    # allocated list and use the unchecked spelling; the push driver stores
    # into a caller-supplied list and needs this one.
    int _SetItemStealOwned "PyList_SetItem" (object, Py_ssize_t, PyObject*) except -1
    int _AppendBorrowed "PyList_Append" (object, PyObject*) except -1
    void _DecRef "Py_DECREF" (PyObject*)

# Wire-format constants, enums, the Field descriptor and the error classes all
# live in the shared pure-Python ``types`` module — reuse them verbatim so the
# native path raises the *same* exception types and yields the *same* Field /
# enum objects the pure path does.
from .types import (
    ARRAY_MAX,
    DEFAULT_REASSEMBLY,
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
    SofaArgumentError,
    Status,
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
cdef enum: _MAX_DEPTH_C = 255   # compile-time twin, for sizing fixed state
cdef int _MAX_DEPTH = _MAX_DEPTH_C
# The smallest streaming output buffer this port accepts (S5.1). It is 1 because
# _put splits every atomic unit at any byte boundary; it binds only a buffer
# installed together with a flush sink. Mirrors types.MIN_OUTPUT_BUFFER.
cdef Py_ssize_t _MIN_OUTPUT_BUFFER = MIN_OUTPUT_BUFFER
# Reassembly space a decoder takes when the caller names no size. Mirrors
# sofab.types.DEFAULT_REASSEMBLY; both engines bound memory the same way.
cdef Py_ssize_t _DEFAULT_REASSEMBLY = DEFAULT_REASSEMBLY

cdef inline object _fresh_bytearray(Py_ssize_t n):
    # A bytearray of n bytes whose contents are not zeroed. Only the span
    # _reassemble has actually written is ever read back, and zeroing costs one
    # store per byte on a buffer a one-shot decode never touches at all --
    # measurably, ~208 ns of a ~450 ns decoder construction at the default size.
    return PyByteArray_FromStringAndSize(NULL, n)
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
cdef tuple _STATUS = tuple(Status)

# The base class's no-op control hooks, to tell an overridden one from the
# default without a getattr per decoder.
from .visitor import Visitor as _Visitor
cdef object _BASE_ON_FIELD = _Visitor.on_field
cdef object _BASE_ON_SEQUENCE_BEGIN = _Visitor.on_sequence_begin
cdef object _BASE_ON_ARRAY_BEGIN = _Visitor.on_array_begin
cdef object _BASE_ON_BLOB_BEGIN = _Visitor.on_blob_begin
cdef object _BASE_ON_STRING_BEGIN = _Visitor.on_string_begin
cdef object _BASE_ON_FARRAY_BEGIN = _Visitor.on_float_array_begin
cdef object _BASE_ON_F32_BITS = _Visitor.on_float32_bits
cdef object _BASE_ON_F32_ARRAY_BITS = _Visitor.on_float32_array_bits
cdef object _BASE_ON_SCHEMA_BOUND = _Visitor.on_schema_bound
cdef object _BASE_DESTINATIONS = _Visitor.destinations
# S5.3.1: one decode surface. A `binding` is compiled to a Visitor by this
# function -- the same one the pure engine calls -- so neither engine carries a
# decode path of its own for it.


cdef int _decode_utf8_checked(const unsigned char* p, Py_ssize_t size) except -1:
    # Is this valid UTF-8? CORELIB_PLAN S6.4.3's `utf8_valid` primitive, for the
    # one caller that needs the answer without the value: a string read into a
    # destination the caller supplied (S6.6.3) still has to be validated
    # (S6.7.2), and decoding it to find out would build the very str the
    # destination exists to avoid.
    #
    # A byte walk rather than a decode, so a megabyte payload costs no object at
    # all. The bounds below are the whole of RFC 3629 as CPython applies it:
    #
    #   * a lead below 0xC2 or above 0xF4 is never valid -- that covers the
    #     overlong two-byte forms (0xC0/0xC1) and everything past U+10FFFF;
    #   * a lead of 0xE0 needs a continuation of at least 0xA0 (overlong
    #     three-byte forms), and 0xED at most 0x9F (the UTF-16 surrogates);
    #   * a lead of 0xF0 needs at least 0x90 (overlong four-byte forms), and
    #     0xF4 at most 0x8F (past U+10FFFF);
    #   * every other continuation byte is 0x80..0xBF.
    #
    # A sequence running past `size` is malformed, not truncated: `size` is the
    # payload's own declared length, so no more bytes are coming for it.
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k
    cdef int extra
    cdef unsigned char b, lo, hi
    while i < size:
        b = p[i]
        if b < 0x80:
            i += 1
            continue
        if b < 0xC2 or b > 0xF4:
            raise SofaDecodeError("invalid UTF-8 in string field")
        if b < 0xE0:
            extra = 1
            lo = 0x80
            hi = 0xBF
        elif b < 0xF0:
            extra = 2
            lo = 0xA0 if b == 0xE0 else 0x80
            hi = 0x9F if b == 0xED else 0xBF
        else:
            extra = 3
            lo = 0x90 if b == 0xF0 else 0x80
            hi = 0x8F if b == 0xF4 else 0xBF
        if i + extra >= size:
            raise SofaDecodeError("invalid UTF-8 in string field")
        b = p[i + 1]
        if b < lo or b > hi:
            raise SofaDecodeError("invalid UTF-8 in string field")
        for k in range(2, extra + 1):
            b = p[i + k]
            if b < 0x80 or b > 0xBF:
                raise SofaDecodeError("invalid UTF-8 in string field")
        i += extra + 1
    return 0


cdef object _ARRAY_MAX_OBJ = _ARRAY_MAX
cdef object _FIXLEN_MAX_OBJ = _FIXLEN_MAX
cdef object _INT64_MAX_OBJ = INT64_MAX
cdef tuple _ST = tuple(FixlenSubtype)
cdef object _FP32_OBJ = FixlenSubtype.FP32
cdef object _FP64_OBJ = FixlenSubtype.FP64


# Pending-value kinds (mirror the pure decoder's _SCALAR/_FIXLEN/_VARRAY/_FARRAY).
cdef int _PEND_NONE = 0
cdef int _PEND_SCALAR = 1
cdef int _PEND_FIXLEN = 2
cdef int _PEND_VARRAY = 3
cdef int _PEND_FARRAY = 4
# A pending value a receiver-side cap has rejected (§6.2.1). The kind it stands
# in for is kept in _pk_real and the rejection's message in _limit_msg, so the
# rejection can still be waived by a binding entry's declared bound, and a
# route that needs the count or the subtype can read through
# it.
#
# Parked rather than raised because the verdict is not final at the header: it
# depends on where the value is headed, which only the field's route knows
# (#128). A path that would put the value in storage of the DECODER's own —
# sized by the wire — raises it; a path that skips the field, or that fills a
# destination the handler supplied, unparks it and walks on. Mirrors the pure
# decoder's _LIMIT wrapper tuple.
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
    """The int ``value`` losslessly is, or SofaArgumentError.

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
        raise SofaArgumentError("%s must be an integer, not %s"
                             % (_WHATS[what], type(value).__name__)) from None

cdef uint64_t _u64_other(object value, int what) except? 0xDEAD:
    # Cold: an exact int the converter rejected (outside the 64-bit domain), or
    # something that is not an exact int at all.
    cdef object idx = _index_arg(value, what)
    cdef uint64_t out
    if _IsLongLike(<PyObject*>idx) and _ToU64(<PyObject*>idx, &out):
        return out
    raise SofaArgumentError("%s %d out of range" % (_WHATS[what], idx))

cdef int64_t _i64_other(object value, int what) except? -0xDEAD:
    cdef object idx = _index_arg(value, what)
    cdef int64_t out
    if _IsLongLike(<PyObject*>idx) and _ToI64(<PyObject*>idx, &out):
        return out
    raise SofaArgumentError("%s %d out of range" % (_WHATS[what], idx))

cdef inline uint64_t _id_arg(object field_id) except? 0xDEAD:
    # Field ids are 0..ID_MAX (2**31-1), so the *value* range is narrower than
    # what the converter itself rejects; the explicit bound stays, but on C ints.
    # An ordinary id is a small non-negative int, and one CPython digit already
    # proves that range — so the usual case reads a single digit and returns,
    # with no 64-bit conversion and no comparison at all. This runs once per
    # field write, which is the most repeated conversion the encoder performs.
    cdef int64_t r
    cdef object idx
    cdef uint64_t small
    if _IsLong(<PyObject*>field_id):
        if _ToSmallId(<PyObject*>field_id, &small):
            return small
        if _ToI64(<PyObject*>field_id, &r):
            if r < 0 or r > <int64_t>_ID_MAX:
                raise SofaArgumentError("id %d out of range 0..%d" % (field_id, _ID_MAX))
            return <uint64_t>r
        raise SofaArgumentError("id %d out of range 0..%d" % (field_id, _ID_MAX))
    idx = _index_arg(field_id, _WHAT_ID)
    if _IsLongLike(<PyObject*>idx) and _ToI64(<PyObject*>idx, &r) and 0 <= r <= <int64_t>_ID_MAX:
        return <uint64_t>r
    raise SofaArgumentError("id %d out of range 0..%d" % (idx, _ID_MAX))


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
        raise SofaArgumentError("array shrank while it was being encoded")
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
    # One memoryview per installation, so a flush costs a slice of it rather than
    # a fresh view plus a slice. The blob-streaming row flushes ~977 times.
    cdef object _fixed_view
    # The full-buffer flush slice, kept for the installation -- see the pure
    # engine and §6.6.2 ("keeps one and reuses it ... not a conformance
    # question"). A streaming encode drains a full buffer over and over.
    cdef object _flush_view
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
    # A heap block of _MAX_DEPTH entries, allocated in __cinit__ and never
    # resized. CORELIB_PLAN S6.6 fixes both halves: an implementation that can
    # allocate MUST hold back to the full MAX_DEPTH (S6.0.1, so there is no fixed
    # window and no eager-framing fallback), and bounded working state "MUST be
    # sized to its full extent when the codec is constructed" -- naming this
    # exact shape as the counter-example, "a pending run that doubles as nesting
    # deepens allocates on a write path, and that is what this section forbids".
    # It cost ~1 KiB per stream when it grew on demand too; it now costs it once,
    # where a source-level audit sees it.
    cdef uint32_t* _pending
    cdef int _npending
    cdef int _pcap

    def __cinit__(self):
        self._fixed_obj = None
        self._fixed_view = None
        self._flush_view = None
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
        # Sized here, not on the first hold-back: __cinit__ runs for every
        # construction shape, over_buffer's cls.__new__ included, and it is the
        # only place S6.6 lets an allocation happen at all.
        self._pending = <uint32_t*>malloc(<size_t>_MAX_DEPTH_C * sizeof(uint32_t))
        if self._pending == NULL:
            raise MemoryError()
        self._npending = 0
        self._pcap = _MAX_DEPTH_C

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
            raise SofaArgumentError("offset must be within the buffer")
        # MIN_OUTPUT_BUFFER (S5.1) binds a buffer installed *with* a flush sink,
        # here and at every mid-stream set, so an unusable buffer is refused where
        # it is handed over rather than partway through a message. Without a sink
        # no flush can occur and no minimum applies -- the buffer holds the message
        # or reports buffer-full -- which is what keeps a caller sizing from a
        # generated MAX_SIZE exact, down to a zero-byte remainder.
        if size - <Py_ssize_t>offset < _MIN_OUTPUT_BUFFER and self._has_sink():
            raise SofaArgumentError(
                "a buffer installed with a flush sink needs at least "
                "MIN_OUTPUT_BUFFER=%d usable byte(s), got %d"
                % (_MIN_OUTPUT_BUFFER, size - <Py_ssize_t>offset))
        self._fixed_obj = buffer
        # Dropped, not released: a sink that took the previous buffer holds this
        # very view (§5.1.5). See the pure engine.
        self._flush_view = None
        self._fixed_view = memoryview(buffer)
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
        # §5.1.6: the sink is handed the installed buffer itself, never a copy of
        # it and never any other memory -- see the pure engine for why a copy is
        # both "other memory" and the thing that makes §5.1.5's take-the-buffer
        # half unreachable.
        cdef Py_ssize_t used = <Py_ssize_t>self._cursor
        # A caller-installed buffer keeps its object behind the view, so a sink
        # can see whose buffer it holds and the bytearray stays alive while it
        # does. The scratch shape has no such object -- one memoryview straight
        # over the memory then, which is also the cheaper of the two and the one
        # the streaming rows take.
        cdef bint full = used == <Py_ssize_t>self._fixed_cap
        cdef bint kept = False
        if self._fixed_view is not None:
            if full:
                # The full-buffer slice is made once per installation and handed
                # out again -- see the pure engine.
                if self._flush_view is None:
                    self._flush_view = self._fixed_view[0:used]
                view = self._flush_view
                kept = True
            else:
                view = self._fixed_view[0:used]
        else:
            # The scratch shape has no object behind its memory, so its view is
            # built each time and is never the kept one.
            view = PyMemoryView_FromMemory(<char*>self._fixed_ptr, used, PyBUF_READ)
        if self._writer is not None:
            self._writer.write(view)
        else:
            self._flush_sink(view)
        if self._installs == installs:
            # The sink copied, so the buffer is ours again. The short slice goes;
            # the kept one stays for the next flush. Either way _fixed_view
            # already pinned the caller's bytearray for the whole installation,
            # so nothing here decides whether it can be resized.
            if not kept:
                view.release()
            self._cursor = 0
        else:
            # The sink took the buffer, so the view it holds is its own now.
            self._flush_view = None
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
        # most once per non-default sequence, never per field. The run is copied
        # to the stack and the encoder's own run emptied before the first byte
        # goes out, so a flush sink that re-enters the encoder starts a fresh run
        # of its own and cannot observe a half-committed one. (The block itself
        # is never detached -- it is construction-time state, not a per-commit
        # allocation.)
        cdef uint32_t run[_MAX_DEPTH_C]
        cdef int n = self._npending
        cdef int i
        for i in range(n):
            run[i] = self._pending[i]
        self._npending = 0
        for i in range(n):
            self._emit_varint((<uint64_t>run[i] << 3) | <uint64_t>_WT_SEQUENCE_START)
        return 0

    cdef int _pending_push(self, uint32_t field_id) except -1:
        # Append one id to the pending run. The block holds _MAX_DEPTH entries
        # and the run is a subset of the open sequences, whose count
        # write_sequence_begin_lazy has already bounded by _MAX_DEPTH -- so this
        # never grows, which is what S6.6 requires of it. The guard is the
        # invariant written down, not a path a caller can reach.
        if self._npending >= self._pcap:
            raise SofaArgumentError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
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
            raise SofaArgumentError("getvalue() is only valid for the in-memory model")
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

    @cython.always_allow_keywords(True)
    def write_float32_bits(self, object field_id, object bits):
        # S6.5: an fp32 written from its raw wire bits, verbatim -- no float is
        # constructed, so nothing can quiet a signaling NaN on the way out.
        # Mirrors sofab.encoder.Encoder.write_float32_bits.
        cdef unsigned char buf[4]
        cdef uint64_t raw
        cdef object value
        if not self._begin():
            return
        try:
            value = bits if type(bits) is int else _index_arg(bits, _WHAT_U)
            if value < 0 or value > 0xFFFFFFFF:
                raise SofaArgumentError(
                    "fp32 bits %s out of range 0..4294967295" % value)
            raw = <uint64_t>value
        except SofaError as exc:
            self._fail(exc)
            return
        buf[0] = <unsigned char>(raw & 0xFF)
        buf[1] = <unsigned char>((raw >> 8) & 0xFF)
        buf[2] = <unsigned char>((raw >> 16) & 0xFF)
        buf[3] = <unsigned char>((raw >> 24) & 0xFF)
        self._write_fixlen_raw(field_id, buf, 4, _ST_FP32)

    def write_float64(self, object field_id, double value):
        cdef unsigned char buf[8]
        _pack_f64(value, buf)
        self._write_fixlen_raw(field_id, buf, 8, _ST_FP64)

    def write_string(self, object field_id, str text):
        # Strict UTF-8: no errors= argument, so a lone/unpaired surrogate raises
        # UnicodeEncodeError, which we map to SofaArgumentError — the encode-side
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
            self._fail(SofaArgumentError("string field is not valid UTF-8: %s" % exc))
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
            self._fail(SofaArgumentError(
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
                raise SofaArgumentError(
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
                raise SofaArgumentError(
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

    @cython.always_allow_keywords(True)
    def write_float32_array_bits(self, object field_id, object payload, count=None):
        # The array half of write_float32_bits (S6.5): the payload goes out
        # exactly as it came off the wire. Mirrors the pure engine.
        cdef Py_buffer view
        cdef Py_ssize_t n, have
        if not self._begin():
            return
        try:
            PyObject_GetBuffer(payload, &view, PyBUF_SIMPLE)
            try:
                n = view.len
                if n % 4:
                    raise SofaArgumentError(
                        "an fp32 array payload of %d bytes is not a whole "
                        "number of 4-byte elements" % n)
                have = n // 4
                if count is not None and <Py_ssize_t>count != have:
                    raise SofaArgumentError(
                        "count=%s does not match the %d elements %d payload "
                        "bytes carry" % (count, have, n))
                self._array_header(field_id, _WT_ARRAY_FIXLEN, have)
                # S4.8: the fixlen_word is always present, empty array included.
                self._emit_varint((<uint64_t>4 << 3) | <uint64_t>_ST_FP32)
                if n:
                    self._put(<const unsigned char*>view.buf, <size_t>n)
            finally:
                PyBuffer_Release(&view)
        except SofaError as exc:
            self._fail(exc)

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
            raise SofaArgumentError("array count %d out of range 0..%d" % (count, _ARRAY_MAX))
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
                raise SofaArgumentError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
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
                raise SofaArgumentError("sequence_end without matching begin")
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
                raise SofaArgumentError("sequence_end without matching begin")
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

# What the push driver was doing when the bytes ran out, so the next feed()
# resumes *that* call rather than walking on and auto-skipping the value the
# caller is still owed (§5.2).
cdef int _R_NONE = 0
cdef int _R_SKIP = 1
cdef int _R_VISIT = 2

#: Highest field id still worth a direct-index lookup array. Above it the table
#: falls back to a linear scan — schemas that sparse are rare, and 4096 ids cost
#: 16 KB of index at most.

# Binding entry kinds, mirroring sofab.binding's K_* -- a destination map's
# vocabulary, not a decode path's.
cdef int _K_UNSIGNED = 0
cdef int _K_SIGNED = 1
cdef int _K_FLOAT32 = 2
cdef int _K_FLOAT64 = 3
cdef int _K_STRING = 4
cdef int _K_BYTES = 5
cdef int _K_ARRAY_UNSIGNED = 6
cdef int _K_ARRAY_SIGNED = 7
cdef int _K_ARRAY_FLOAT32 = 8
cdef int _K_ARRAY_FLOAT64 = 9
cdef int _K_SEQUENCE = 10


#: Highest field id still worth a direct-index lookup array. Above it the table
#: falls back to a linear scan — schemas that sparse are rare, and 4096 ids cost
#: 16 KB of index at most.
cdef uint64_t _IDX_MAX_ID = 4096


cdef struct _BEntry:
    uint64_t field_id
    int kind
    int wt                  # wire type this binding accepts
    int st                  # fixlen subtype, or -1 when the kind has none
    Py_ssize_t at           # slot in words[] (or objects[] for string/blob)
    Py_ssize_t cap          # array capacity / declared fixlen maxlen, 0 = none
    Py_ssize_t count_at     # slot to write arrival into, or -1
    int child               # table index of a sequence's child, or -1
    bint elem_bounded       # the schema declares the array's element width
    int64_t elem_lo
    uint64_t elem_hi


cdef struct _BTable:
    Py_ssize_t first        # index of this table's first entry in _bent
    Py_ssize_t n
    int* idx                # id -> local entry index, or NULL for linear scan
    uint64_t idx_max


@cython.final
cdef class _Compiled:
    """The compiled form of a :class:`sofab.Binding`, owned by the Binding.

    Compiling walks the table in Python — a few hundred attribute reads for a
    real schema — which is more than a whole decode costs. A Binding is
    build-once (see ``Binding.freeze``), so the result can be cached on it and
    every Decoder built from that table reuses it. Held by the Binding *and* by
    each Decoder, so it outlives either.
    """

    cdef _BEntry* bent
    cdef _BTable* btab
    cdef Py_ssize_t nent
    cdef int ntab
    cdef Py_ssize_t words_required
    cdef Py_ssize_t objects_required

    def __cinit__(self):
        self.bent = NULL
        self.btab = NULL
        self.nent = 0
        self.ntab = 0
        self.words_required = 0
        self.objects_required = 0

    def __dealloc__(self):
        cdef int i
        if self.btab != NULL:
            for i in range(self.ntab):
                if self.btab[i].idx != NULL:
                    free(self.btab[i].idx)
            free(self.btab)
            self.btab = NULL
        if self.bent != NULL:
            free(self.bent)
            self.bent = NULL

    cdef int build(self, object binding) except -1:
        # Flatten the Binding tree into one entry array plus one table per
        # sequence scope, and give each table a direct id->entry index where the
        # ids are dense enough for one to be worth the memory.
        cdef list tables = binding.freeze()
        cdef dict seen = {}
        cdef object b, e
        cdef Py_ssize_t total = 0, k = 0, first, i, ntab
        cdef int ti
        cdef uint64_t maxid, fid
        cdef int* idx

        ntab = len(tables)
        for i in range(ntab):
            seen[id(tables[i])] = i
            total += len(tables[i]._entries)
        self.words_required = <Py_ssize_t>binding.tree_words_required
        self.objects_required = <Py_ssize_t>binding.tree_objects_required

        self.btab = <_BTable*>malloc(ntab * sizeof(_BTable))
        if self.btab == NULL:
            raise MemoryError()
        self.ntab = <int>ntab
        for ti in range(self.ntab):
            self.btab[ti].idx = NULL
            self.btab[ti].first = 0
            self.btab[ti].n = 0
            self.btab[ti].idx_max = 0
        # malloc(0) may legally return NULL, which would read as failure.
        self.bent = <_BEntry*>malloc((total if total else 1) * sizeof(_BEntry))
        if self.bent == NULL:
            raise MemoryError()
        self.nent = total

        for ti in range(self.ntab):
            b = tables[ti]
            first = k
            maxid = 0
            for e in b._entries:
                fid = <uint64_t>e.field_id
                self.bent[k].field_id = fid
                self.bent[k].kind = <int>e.kind
                self.bent[k].wt = <int>e.wt
                self.bent[k].st = -1 if e.st is None else <int>e.st
                self.bent[k].at = <Py_ssize_t>e.at
                self.bent[k].cap = <Py_ssize_t>e.cap
                self.bent[k].count_at = <Py_ssize_t>e.count_at
                self.bent[k].child = <int>seen[id(e.child)] if e.child is not None else -1
                self.bent[k].elem_bounded = <bint>e.elem_bounded
                self.bent[k].elem_lo = <int64_t>e.elem_lo
                self.bent[k].elem_hi = <uint64_t>e.elem_hi
                if fid > maxid:
                    maxid = fid
                k += 1
            self.btab[ti].first = first
            self.btab[ti].n = k - first
            self.btab[ti].idx_max = maxid
            if k > first and maxid <= _IDX_MAX_ID:
                idx = <int*>malloc(<Py_ssize_t>(maxid + 1) * sizeof(int))
                if idx == NULL:
                    raise MemoryError()
                for i in range(<Py_ssize_t>(maxid + 1)):
                    idx[i] = -1
                for i in range(k - first):
                    idx[self.bent[first + i].field_id] = <int>i
                self.btab[ti].idx = idx
        return 0


cdef _Compiled _compiled_for(object binding):
    """The Binding's compiled table, built on first use and cached on it."""
    cdef object cached = binding._compiled
    cdef _Compiled c
    if type(cached) is _Compiled:
        return <_Compiled>cached
    c = _Compiled.__new__(_Compiled)
    c.build(binding)
    binding._compiled = c
    return c

cdef class Decoder:
    """Native push decoder — see :class:`sofab.decoder.Decoder` for the contract.

    Bytes go in through ``feed`` and fields come out at a ``visitor`` or a
    ``binding``. Incoming bytes are held in one contiguous buffer and parsed by
    advancing a C cursor with direct pointer indexing; a construct that runs off
    the end suspends and resumes from its first byte on the next ``feed``, so the
    same path serves a whole message and a reader that dribbles one byte at a
    time.
    """

    # Receiver-configured decode limits; kept as Python objects
    # so the comparison stays exact for a caller-supplied int of any magnitude.
    # Whether any receiver cap is configured at all. All three default to off,
    # and without this the header walk touches a Python attribute per string,
    # blob and array field only to find None there.
    cdef bint _capped
    cdef object _max_dyn_array_count
    cdef object _max_dyn_string_len
    cdef object _max_dyn_blob_len
    # Owns the storage the pointer indexes into. ``bytes`` while a chunk is
    # being consumed whole (adopted, never copied); a ``bytearray`` while a
    # construct is being accumulated across chunks (appended to, never
    # rebuilt). See feed().
    cdef object _buf
    # The caller's reassembly buffer, and the span of it holding a construct
    # that spans a chunk boundary. See the pure engine for what it is for.
    # A latched receiver-limit rejection -- see the pure engine.
    cdef object _limit
    cdef object _rbuf
    cdef Py_ssize_t _rstart
    cdef Py_ssize_t _rend
    # Owns a fixlen payload that had to be assembled across refills, for as long
    # as a pointer into it can still be in use (see _take_fixlen_ptr).
    cdef bytes _spill
    cdef const unsigned char* _p
    cdef Py_ssize_t _n
    cdef Py_ssize_t _pos
    cdef int _depth
    # Field id of the header _next_wire last parsed, unboxed.
    cdef uint64_t _cur_id
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
    # started at, and _keep_cur_wtype the current-field wire type as of _keep.
    # See _arm/_suspend.
    cdef Py_ssize_t _keep
    cdef int _keep_cur_wtype

    # --- push mode (§5.2) ---------------------------------------------------
    cdef object _visitor
    # Visitors suspended by a descent -- see the pure engine.
    # Whether the visitor overrides the two control hooks. Both default to a
    # no-op on the base class, and calling one that was never overridden costs a
    # Python call per field for nothing. ``_wants_field`` also decides whether a
    # Field object is built at all: it is the only thing that receives one.
    cdef bint _wants_field
    cdef bint _wants_bound
    cdef bint _make_field
    cdef bint _wants_seq_begin
    cdef bint _wants_array_begin
    cdef bint _wants_blob_begin
    cdef bint _wants_string_begin
    cdef bint _wants_farray_begin
    cdef bint _wants_f32_bits
    cdef bint _wants_f32_array_bits
    # The handlers a descent suspended: sized in __cinit__ from MAX_DEPTH and
    # never grown, which is what S6.6 asks of the codec's bounded working state.
    #
    # One heap block rather than an inline array: an inline array is part of the
    # object, and tp_alloc zeroes the whole object -- 2 KiB of memset per
    # decoder, on a path that builds one per message. A single malloc costs a
    # fraction of that and is freed in __dealloc__. (A Python list is dearer
    # still: `[None] * 255` alone was ~565 ns of a ~990 ns construction.)
    cdef void* _stackmem
    # Each occupied slot holds one strong reference, dropped when it is popped,
    # when the decoder is reset, and in __dealloc__.
    cdef PyObject** _vstack         # _MAX_DEPTH borrowed-then-owned pointers
    cdef int _vsp
    cdef int _status
    cdef object _err
    cdef int _resume_kind
    cdef bint _running
    # --- the destination map (counterproposal) -------------------------------
    # Where a bound field's value goes, and what the schema declares for it.
    # It answers those two questions and nothing else: no rule below reads it
    # except the store, so there is one implementation of every rule (S5.3.1).
    cdef object _objects
    cdef _Compiled _tables
    cdef _BEntry* _bent
    cdef _BTable* _btab
    cdef uint64_t* _words
    cdef Py_buffer* _wview
    cdef Py_ssize_t _nwords
    cdef int _tab
    cdef int* _tstack
    cdef int _tsp
    cdef Py_ssize_t _resume_entry

    def __cinit__(self, *, binding=None, visitor=None, words=None, objects=None,
                  max_dyn_array_count=_ARRAY_MAX_OBJ,
                  max_dyn_string_len=_FIXLEN_MAX_OBJ,
                  max_dyn_blob_len=_FIXLEN_MAX_OBJ,
                  reassembly=None):
        self._stackmem = NULL
        self._vstack = NULL
        self._vsp = 0
        self._stackmem = malloc(<size_t>_MAX_DEPTH_C * sizeof(PyObject*))
        if self._stackmem == NULL:
            raise MemoryError()
        self._vstack = <PyObject**>self._stackmem
        self._status = <int>Status.COMPLETE
        self._err = None
        self._limit = None
        self._resume_kind = _R_NONE
        self._running = False
        self._rstart = 0
        self._rend = 0
        # One reassembly shape, never grown (S6.6.2) -- see the pure engine.
        if reassembly is None:
            self._rbuf = _fresh_bytearray(_DEFAULT_REASSEMBLY)
        elif type(reassembly) is bytearray:
            self._rbuf = reassembly
        elif type(reassembly) is int:
            if reassembly < 16:
                raise SofaArgumentError(
                    "reassembly=%d is too small; 16 bytes is the least that "
                    "can hold a construct spanning a chunk" % reassembly)
            self._rbuf = _fresh_bytearray(<Py_ssize_t>reassembly)
        else:
            raise SofaArgumentError(
                "reassembly must be a bytearray, a byte count, or omitted")

        if binding is None and visitor is None:
            raise SofaArgumentError("a decoder needs a field handler (binding / visitor)")
        # S5.3.1: one decode surface, and the table is reached *through* it. A
        # handler declares its destinations once, from Visitor.destinations();
        # `binding=` is the constructor shorthand for a handler that declares
        # exactly that and nothing else. Either way there is one handler object,
        # one walk and one set of rules -- the map only says *where* a value
        # goes, never how it is decoded.
        if (binding is None and visitor is not None
                and type(visitor).destinations is not _BASE_DESTINATIONS):
            # Asked once, and only of a handler that overrides it -- see the
            # pure engine.
            declared = visitor.destinations()
            if declared is not None:
                binding, words, objects = declared
        self._tables = None
        self._bent = NULL
        self._btab = NULL
        self._words = NULL
        self._wview = NULL
        self._nwords = 0
        self._tab = -1
        self._tstack = NULL
        self._tsp = 0
        self._resume_entry = -1
        self._objects = objects
        if binding is not None:
            self._tables = _compiled_for(binding)
            self._bent = self._tables.bent
            self._btab = self._tables.btab
            self._bind_words(words, objects)
            self._tab = 0
            # S6.6: sized to its full extent HERE, never on a feed path.
            # _next_wire refuses a message nesting past MAX_DEPTH before any of
            # these slots is written, so the slots are the ceiling.
            self._tstack = <int*>malloc((_MAX_DEPTH + 1) * sizeof(int))
            if self._tstack == NULL:
                raise MemoryError()
        self._visitor = visitor
        self._wants_field = False
        self._wants_bound = False
        self._make_field = False
        self._wants_seq_begin = False
        self._wants_array_begin = False
        self._wants_blob_begin = False
        if visitor is not None:
            self._bind_visitor(visitor)
        # §6.2.1: no unset state and no unlimited mode. None is refused rather
        # than read as "no limit"; the defaults are the format ceilings above
        # which the value is already INVALID, which is the widest a limit can be
        # while still being one. See the pure engine for the full note.
        # Written out rather than looped: a decoder is constructed per message on
        # the one-shot path, and a Python-level loop over three tuples is a
        # measurable share of that.
        # The identity test is not a shortcut past the check: a default IS the
        # ceiling object, and the ceiling is what the check would accept.
        if max_dyn_array_count is not _ARRAY_MAX_OBJ:
            self._check_limit("max_dyn_array_count", max_dyn_array_count, _ARRAY_MAX_OBJ)
        if max_dyn_string_len is not _FIXLEN_MAX_OBJ:
            self._check_limit("max_dyn_string_len", max_dyn_string_len, _FIXLEN_MAX_OBJ)
        if max_dyn_blob_len is not _FIXLEN_MAX_OBJ:
            self._check_limit("max_dyn_blob_len", max_dyn_blob_len, _FIXLEN_MAX_OBJ)
        self._capped = (max_dyn_array_count < _ARRAY_MAX_OBJ
                        or max_dyn_string_len < _FIXLEN_MAX_OBJ
                        or max_dyn_blob_len < _FIXLEN_MAX_OBJ)
        self._max_dyn_array_count = max_dyn_array_count
        self._max_dyn_string_len = max_dyn_string_len
        self._max_dyn_blob_len = max_dyn_blob_len
        self._buf = b""
        self._p = <const unsigned char*>PyBytes_AS_STRING(self._buf)
        self._n = 0
        self._pos = 0
        self._depth = 0
        self._cur_wtype = -1
        self._spill = None
        self._pk = _PEND_NONE
        self._pk_real = _PEND_NONE
        self._limit_msg = None
        self._keep = 0
        self._keep_cur_wtype = -1

    def __dealloc__(self):
        # _vstack points into _stackmem; only the references it holds have to be
        # dropped before the block goes.
        cdef int i
        if self._stackmem != NULL:
            for i in range(self._vsp):
                Py_XDECREF(self._vstack[i])
            self._vsp = 0
            free(self._stackmem)
            self._stackmem = NULL
            self._vstack = NULL
        # _bent / _btab belong to the cached _Compiled, not to this decoder.
        if self._tstack != NULL:
            free(self._tstack)
            self._tstack = NULL
        if self._wview != NULL:
            PyBuffer_Release(self._wview)
            free(self._wview)
            self._wview = NULL

    cdef inline int _check_limit(self, str name, object value, object ceiling) except -1:
        # §6.2.1: no unset state, no unlimited mode, and a domain of 0..ceiling.
        if value is None:
            raise SofaArgumentError("%s has no unset state (§6.2.1)" % name)
        if value < 0 or value > ceiling:
            raise SofaArgumentError("%s=%s is outside 0..%s" % (name, value, ceiling))
        return 0

    cdef int _bind_words(self, object words, object objects) except -1:
        cdef Py_ssize_t need
        if words is None:
            raise SofaArgumentError("a binding needs a words buffer")
        self._wview = <Py_buffer*>malloc(sizeof(Py_buffer))
        if self._wview == NULL:
            raise MemoryError()
        try:
            PyObject_GetBuffer(words, self._wview, PyBUF_WRITABLE | PyBUF_SIMPLE)
        except (BufferError, TypeError) as exc:
            free(self._wview)
            self._wview = NULL
            # The pure engine reaches the same verdict through memoryview and
            # reports it as §6.3 InvalidArgument. §5.3 requires the accelerator
            # to be invisible, so it must not surface a different exception type.
            raise SofaArgumentError(
                "the words buffer must be a writable, contiguous buffer") from exc
        if self._wview.len % 8:
            raise SofaArgumentError("the words buffer must be a multiple of 8 bytes")
        self._words = <uint64_t*>self._wview.buf
        self._nwords = self._wview.len // 8
        need = self._tables.words_required
        if self._nwords < need:
            raise SofaArgumentError("words buffer holds %d slots, the binding needs %d"
                                 % (self._nwords, need))
        need = self._tables.objects_required
        if objects is None:
            if need:
                raise SofaArgumentError("a binding with string/blob fields needs objects")
        elif <Py_ssize_t>len(objects) < need:
            raise SofaArgumentError("objects holds %d entries, the binding needs %d"
                                 % (len(objects), need))
        return 0


    cdef inline int _push_table(self, int tab) except -1:
        # The stack was sized at construction (S6.6). Nothing is allocated here,
        # on a feed path, whatever the message nests to -- _next_wire has
        # already refused anything past MAX_DEPTH.
        self._tstack[self._tsp] = self._tab
        self._tsp += 1
        self._tab = tab
        return 0


    cdef inline Py_ssize_t _lookup(self, uint64_t fid) noexcept:
        cdef _BTable* t = &self._btab[self._tab]
        cdef Py_ssize_t i
        cdef int local
        if t.idx != NULL:
            if fid > t.idx_max:
                return -1
            local = t.idx[fid]
            return t.first + local if local >= 0 else -1
        for i in range(t.n):
            if self._bent[t.first + i].field_id == fid:
                return t.first + i
        return -1


    cdef int _take_object(self, int kind, Py_ssize_t at) except -1:
        cdef Py_ssize_t n = <Py_ssize_t>self._pend_size
        cdef const unsigned char* p = self._take_fixlen_ptr(n)
        cdef object value
        if kind == _K_STRING:
            try:
                value = PyUnicode_DecodeUTF8(<const char*>p, n, NULL)
            except UnicodeDecodeError as exc:
                raise SofaDecodeError("invalid UTF-8 in string field") from exc
        elif self._spill is not None:
            value = self._spill      # already an exact-size bytes; hand it over
        else:
            value = PyBytes_FromStringAndSize(<const char*>p, n)
        # The slot may already hold a value from a previous message, so the old
        # reference has to go rather than leak.
        Py_INCREF(value)
        _SetItemStealOwned(self._objects, at, <PyObject*>value)
        return 0


    cdef int _fill_varints(self, uint64_t* dst, Py_ssize_t count, bint zigzag,
                           _BEntry* e) except -1:
        # The binding's destination. Both callers meet in _fill_varints_at; this
        # one just unpacks the declared width off the compiled entry.
        return self._fill_varints_at(<void*>dst, 8, count, zigzag, e.elem_bounded,
                                     e.elem_lo, e.elem_hi)


    cdef int _bind_visitor(self, object visitor) except -1:
        # Take the hook flags off `visitor`'s type. Computed once per handler;
        # a descent into a child is the only thing that changes the answer.
        self._visitor = visitor
        self._wants_field = type(visitor).on_field is not _BASE_ON_FIELD
        self._wants_bound = (
            type(visitor).on_schema_bound is not _BASE_ON_SCHEMA_BOUND)
        # A Field is built for the ONE hook that takes one. Every other hook --
        # on_schema_bound included -- takes integers, so declaring a schema
        # bound costs no object per field.
        self._make_field = self._wants_field
        self._wants_seq_begin = (
            type(visitor).on_sequence_begin is not _BASE_ON_SEQUENCE_BEGIN)
        self._wants_array_begin = (
            type(visitor).on_array_begin is not _BASE_ON_ARRAY_BEGIN)
        self._wants_blob_begin = (
            type(visitor).on_blob_begin is not _BASE_ON_BLOB_BEGIN)
        self._wants_string_begin = (
            type(visitor).on_string_begin is not _BASE_ON_STRING_BEGIN)
        self._wants_farray_begin = (
            type(visitor).on_float_array_begin is not _BASE_ON_FARRAY_BEGIN)
        # S6.5's raw fp32 channel, opt-in by override -- see the pure engine.
        self._wants_f32_bits = (
            type(visitor).on_float32_bits is not _BASE_ON_F32_BITS)
        self._wants_f32_array_bits = (
            type(visitor).on_float32_array_bits is not _BASE_ON_F32_ARRAY_BITS)
        return 0

    cdef inline void _rebind(self, object newbuf):
        self._buf = newbuf
        if type(newbuf) is bytes:
            self._p = <const unsigned char*>PyBytes_AS_STRING(newbuf)
            self._n = PyBytes_GET_SIZE(newbuf)
        else:
            self._p = <const unsigned char*>PyByteArray_AS_STRING(newbuf)
            self._n = PyByteArray_GET_SIZE(newbuf)

    # --- resume transactions (CORELIB_PLAN §5.2) ----------------------------
    #
    # Mirrors Decoder._suspend in the pure engine — see the long comment there.
    # Running out of bytes mid-construct is INCOMPLETE, a first-class outcome the
    # caller answers with more bytes, so every public call is all-or-nothing: on
    # the suspension path the cursor goes back to where the call started and the
    # bytes already parsed stay buffered, so re-issuing the call re-parses the
    # construct from its first byte. The pending value / depth / current field
    # are committed only after the construct's last byte is in hand, which is why
    # the rewind is a cursor (plus the current wire type, which _next_wire
    # publishes before the varints that trail a fixlen/array header). INVALID is
    # terminal and is deliberately not rewound.

    cdef inline void _arm(self):
        self._keep = self._pos
        self._keep_cur_wtype = self._cur_wtype

    cdef object _suspend(self, msg):
        # _keep tracks buffer compaction in feed(), so it names the call's start
        # byte in the current buffer.
        self._pos = self._keep
        self._cur_wtype = self._keep_cur_wtype
        return SofaIncompleteError(msg)

    # --- byte sourcing ------------------------------------------------------

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
            raise self._suspend("truncated varint")
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
                raise self._suspend("truncated varint")
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
        # A payload that stops halfway stays buffered when the truncation is
        # reported, so the next attempt continues from it (§5.2).
        cdef Py_ssize_t pos = self._pos
        cdef bytes out
        if pos + n > self._n:
            raise self._suspend("truncated payload")
        # Built from the pointer rather than sliced out of ``_buf``: one copy,
        # and always a real ``bytes`` even while the buffer is a bytearray.
        out = PyBytes_FromStringAndSize(<const char*>self._p + pos, n)
        self._pos = pos + n
        return out

    cdef int _skip_exact(self, Py_ssize_t n) except -1:
        # Consume n bytes without building the object holding them: §5.2 makes a
        # skip pure consumption, so it must not pay for a copy of a payload it
        # discards. Same sourcing and same suspension as _read_exact, mirroring
        # Decoder._skip_exact in the pure engine — the bytes stay buffered on the
        # slow path (the resume contract replays them), only the copy is gone.
        cdef Py_ssize_t pos = self._pos
        if pos + n > self._n:
            raise self._suspend("truncated payload")
        self._pos = pos + n
        return 0

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
        cdef PyObject* item
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
                item = _NewI64(signed_val)
            else:
                # hi is a NARROWED unsigned maximum (u32 at the widest), so it is
                # never negative and the unsigned compare is exact.
                if bounded and result > <uint64_t>hi:
                    raise SofaDecodeError("array element outside declared width")
                item = _NewU64(result)
            if i < prealloc:
                _SetItemSteal(out, i, item)      # steals the new reference
            else:
                # Beyond the pre-sized prefix: append takes its own reference, so
                # ours has to go — including when the append itself fails.
                try:
                    _AppendBorrowed(out, item)
                finally:
                    _DecRef(item)
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

    cdef int _next_wire(self) except -2:
        # Parse one field header and publish its identity; returns the wire type,
        # or -1 at clean EOF. ``skip()``, ``drive()`` and the push driver all
        # iterate fields, and going through the Python method wrapper for every
        # one of them costs a full attribute lookup and call frame.
        #
        # No Field is built here, and that is what the push driver buys its
        # speed with: building one means an allocation plus up to five boxed
        # ints, measurably more than parsing the header, and this walk cannot
        # yet know whether anyone will be offered the field -- one the handler's
        # destination map names never reaches on_field. Nothing is boxed that
        # the wire does not force: the id, the fixlen length and the array count
        # stay in C registers unless a configured receiver cap has to compare
        # against them. _build_field makes one where it is actually needed.
        cdef uint64_t header
        cdef int wtype
        cdef object field_id
        cdef uint64_t fid
        cdef uint64_t length_header, length, count, elem_header, elem_size
        cdef int subtype
        cdef object boxed, cap

        self._arm()   # opens this field's resume transaction (§5.2)

        if self._pk != _PEND_NONE:
            self._skip_pending()
            # The auto-skip committed, so it must not be replayed: re-open the
            # transaction *after* it. Without this, a suspension later in this
            # same call rewinds to before the skipped value and the retry reads
            # those bytes as a new field. See the pure engine for the long note.
            self._arm()

        if self._pos >= self._n:
            if self._depth != 0:
                raise self._suspend("truncated: unbalanced sequence")
            self._cur_wtype = -1
            # ``_cur`` deliberately keeps the last field: `field` is documented
            # as "the most recently returned Field", and the pure engine holds it
            # past EOF too. §5.3 wants the accelerator invisible, and that
            # includes this.
            return -1

        header = self._varint()
        wtype = <int>(header & 0x07)
        fid = header >> 3
        self._cur_wtype = wtype
        self._cur_id = fid

        # ID_MAX bounds every header's id (§6.2), the sequence end included even
        # though its id is discarded (§4.9); validate before the wire-type
        # dispatch so wire type 7 is not an exception to the ceiling.
        if fid > _ID_MAX:
            raise SofaDecodeError("id %d out of range" % PyLong_FromUnsignedLongLong(fid))

        if wtype == _WT_FIXLEN:
            length_header = self._varint()
            length = length_header >> 3
            subtype = <int>(length_header & 0x07)
            # Split on the subtype family, one comparison instead of four, and
            # the same split the pure engine makes. STRING/BLOB are
            # variable-length, so only the format-wide ceiling binds them and a
            # truncated one is legitimately INCOMPLETE. FP32/FP64 carry one
            # fixed width each; a wrong one is malformed whatever follows, so
            # that INVALID is raised here at header time, before any payload
            # read, taking precedence over the INCOMPLETE a truncated payload
            # would otherwise raise (§7).
            if subtype >= _ST_STRING:
                if subtype > _ST_BLOB:
                    raise SofaDecodeError("invalid fixlen subtype %d" % subtype)
                if length > _FIXLEN_MAX:
                    raise SofaDecodeError("fixlen length out of range")
            elif subtype == _ST_FP32:
                if length != 4:
                    raise SofaDecodeError("fp32 fixlen length must be 4")
            elif length != 8:
                raise SofaDecodeError("fp64 fixlen length must be 8")
            self._pk = _PEND_FIXLEN
            self._pend_subtype = subtype
            self._pend_size = length
            # Receiver-configured caps (policy, not malformation): the verdict on
            # an oversize string/blob is reached here, on the length word alone —
            # before its payload is read or buffered — and PARKED on the pending
            # value rather than raised, so the caller keeps the §6.2.1 window in
            # which it can declare the field schema-bounded and take the cap off
            # it. A consume into the decoder's own storage raises it; a skip or a
            # handler-supplied destination unparks it (#128). See _park_limit.
            cap = _NONE
            if self._capped:
                if subtype == _ST_STRING:
                    cap = self._max_dyn_string_len
                elif subtype == _ST_BLOB:
                    cap = self._max_dyn_blob_len
            if cap is not None:
                # Boxed only for the cap: the value a configured limit is
                # compared against and named in its message. No Field is built
                # here -- see _build_field.
                boxed = PyLong_FromUnsignedLongLong(length)
                if boxed > cap:
                    if subtype == _ST_STRING:
                        self._park_limit("string length %d exceeds max_dyn_string_len %s"
                                         % (boxed, cap))
                    else:
                        self._park_limit("blob length %d exceeds max_dyn_blob_len %s"
                                         % (boxed, cap))
            return wtype

        if wtype < _WT_FIXLEN:  # UNSIGNED (0) or SIGNED (1)
            self._pk = _PEND_SCALAR
            self._pend_wtype = wtype
            return wtype

        if wtype == _WT_SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            return wtype

        if wtype == _WT_SEQUENCE_START:
            if self._depth >= _MAX_DEPTH:
                raise SofaDecodeError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
            self._depth += 1
            return wtype

        if wtype == _WT_ARRAY_UNSIGNED or wtype == _WT_ARRAY_SIGNED:
            count = self._varint()
            if count > _ARRAY_MAX:
                raise SofaDecodeError("array count %d out of range" % PyLong_FromUnsignedLongLong(count))
            self._pk = _PEND_VARRAY
            self._pend_wtype = wtype
            self._pend_count = count
            cap = self._max_dyn_array_count if self._capped else _NONE
            if cap is not _NONE:
                boxed = PyLong_FromUnsignedLongLong(count)   # see the fixlen branch
                # Parked, not raised — see the fixlen branch above (§6.2.1).
                if boxed > cap:
                    self._park_limit("array count %d exceeds max_dyn_array_count %s" % (boxed, cap))
            return wtype

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
        self._pk = _PEND_FARRAY
        self._pend_subtype = subtype
        self._pend_count = count
        self._pend_size = elem_size
        cap = self._max_dyn_array_count if self._capped else _NONE
        if cap is not _NONE:
            boxed = PyLong_FromUnsignedLongLong(count)      # see the fixlen branch
            # Parked, not raised — see the fixlen branch above (§6.2.1).
            if boxed > cap:
                self._park_limit("array count %d exceeds max_dyn_array_count %s" % (boxed, cap))
        return wtype

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

    cdef object _mismatch(self):
        # The answer a typed read owes a pending value whose wire tag
        # contradicts it: None — MESSAGE_SPEC §7.3, not an error. The value
        # stays pending, so the next header walk (or an explicit skip) discards it
        # exactly like an unknown id, nothing is written to the caller, and the
        # decode stays COMPLETE. Two conditions reach here that are not that:
        # no pending value at all is a caller mistake (§6.3 InvalidArgument),
        # and a parked receiver cap (§6.2.1) is the field this decoder is about
        # to materialize into storage of its own — the allocation the cap
        # exists to refuse. Both raise. A §7.3 skip never arrives here: the
        # driver settles the tag itself and leaves the value pending for
        # _skip_pending, which drops the cap (#128). Reached only once a read
        # has found the kind wrong, so the ordinary path never pays for it.
        if self._pk == _PEND_NONE:
            raise SofaArgumentError("no value pending for the current field")
        if self._pk == _PEND_LIMIT:
            raise SofaLimitError(self._limit_msg)
        return None

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

    cdef int _skip_pending(self) except -1:
        cdef int kind = self._pk
        if kind == _PEND_LIMIT:
            # A skipped field is not capped (#128). §6.2.1 enforces a receiver
            # limit "at the count/length header — before the allocation it is
            # meant to prevent", and a skip makes no allocation for it to
            # prevent: the payload is walked, never materialized, which is what
            # §6.7.2's skip row means by "neither materializes nor validates".
            # So the parked verdict is dropped and the real value walked like
            # any other — including a §7.3 tag mismatch, whose payload "was
            # never this field's value" and cannot be measured against a bound
            # meant for one.
            #
            # Unparked before the walk rather than after it: should the payload
            # run out mid-skip the field stays pending, and the retry then
            # re-enters with the cap already gone instead of clearing it twice.
            kind = self._pk = self._pk_real
            self._limit_msg = None
        if kind == _PEND_SCALAR:
            self._varint()
        elif kind == _PEND_FIXLEN:
            self._skip_exact(<Py_ssize_t>self._pend_size)
        elif kind == _PEND_VARRAY:
            self._skip_varints(<Py_ssize_t>self._pend_count)
        else:  # _PEND_FARRAY
            self._skip_exact(self._farray_nbytes(self._pend_count, self._pend_size))
        # Cleared only now: had the value run out mid-skip, the field has to stay
        # pending so the retry skips it again from its first byte (§5.2).
        self._pk = _PEND_NONE
        return 0


    cdef int _skip(self) except -1:
        cdef int target
        cdef int depth, cur_wtype, pk, pend_wtype, pend_subtype
        cdef uint64_t pend_count, pend_size
        cdef object cur
        if self._cur_wtype == _WT_SEQUENCE_START:
            # Walking a whole sequence spans many fields, so unlike every other
            # call this one moves the field state — and lets _next_wire re-arm
            # _keep — before it can suspend. The field state is put back here,
            # so a re-issued skip replays the whole sequence (§5.2).
            floor = self._pos
            depth = self._depth
            cur_wtype = self._cur_wtype
            pk = self._pk
            pend_wtype = self._pend_wtype
            pend_subtype = self._pend_subtype
            pend_count = self._pend_count
            pend_size = self._pend_size
            target = depth - 1
            try:
                while self._depth > target:
                    if self._next_wire() < 0:
                        raise self._suspend("truncated sequence")
            except SofaIncompleteError:
                self._pos = floor
                self._keep = floor
                self._depth = depth
                self._cur_wtype = cur_wtype
                self._pk = pk
                self._pend_wtype = pend_wtype
                self._pend_subtype = pend_subtype
                self._pend_count = pend_count
                self._pend_size = pend_size
                raise
            return 0
        if self._pk != _PEND_NONE:
            self._skip_pending()
        return 0

    # --- typed reads and §7.3 -----------------------------------------------
    #
    # Each read tests the whole tag of the pending value (wire type plus, for
    # the fixlen kinds, the subtype) against the type it declares, in one
    # combined test on the fast path, and hands every cold outcome to
    # _mismatch. Mirrors Decoder._take_* in the pure engine — see the long note
    # there.

    cdef uint64_t _take_scalar(self) except? 0xDEAD:
        # The tag is already matched by the caller (§7.3), so this only consumes.
        self._arm()
        cdef uint64_t value = self._varint()
        self._pk = _PEND_NONE   # committed only once the value is in hand (§5.2)
        return value

    def _unsigned(self):
        if self._pk != _PEND_SCALAR or self._pend_wtype != _WT_UNSIGNED:
            return self._mismatch()
        return PyLong_FromUnsignedLongLong(self._take_scalar())

    def _signed(self):
        if self._pk != _PEND_SCALAR or self._pend_wtype != _WT_SIGNED:
            return self._mismatch()
        return PyLong_FromLongLong(_zigzag_decode(self._take_scalar()))

    def _bool(self):
        if self._pk != _PEND_SCALAR or self._pend_wtype != _WT_UNSIGNED:
            return self._mismatch()
        return self._take_scalar() != 0

    cdef const unsigned char* _take_fixlen_ptr(self, Py_ssize_t n) except NULL:
        # Fixlen payload as a pointer instead of a ``bytes``. When the payload is
        # already buffered — the ordinary case — the value can be built straight
        # off the buffer, so the copy the intermediate ``bytes`` object used to
        # cost disappears. When it is not (a chunk-fed reader mid-payload), fall
        # back to _read_exact and park its object in ``_spill``, which owns it
        # for as long as the caller can still hold the pointer. The whole tag
        # is already matched by the caller (§7.3), so this only consumes.
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

    def _float32(self):
        # Width is settled at header time (§4.6/§7), so _pend_size is 4 here.
        if self._pk != _PEND_FIXLEN or self._pend_subtype != _ST_FP32:
            return self._mismatch()
        return _unpack_f32(self._take_fixlen_ptr(4))

    def _float32_bits(self):
        # The raw little-endian wire bits of the pending fp32, as an int -- no
        # float on the way, so nothing can quiet a signaling NaN (S6.5).
        cdef const unsigned char* p
        cdef uint32_t bits
        if self._pk != _PEND_FIXLEN or self._pend_subtype != _ST_FP32:
            return self._mismatch()
        p = self._take_fixlen_ptr(4)
        bits = (<uint32_t>p[0] | (<uint32_t>p[1] << 8)
                | (<uint32_t>p[2] << 16) | (<uint32_t>p[3] << 24))
        return PyLong_FromUnsignedLongLong(<uint64_t>bits)

    def _float64(self):
        if self._pk != _PEND_FIXLEN or self._pend_subtype != _ST_FP64:
            return self._mismatch()
        return _unpack_f64(self._take_fixlen_ptr(8))

    def _fixlen_len(self):
        # Peek the current fixlen field's payload byte length (from its length
        # header) without consuming it — a following string()/bytes()/float* read
        # still takes the same field. Lets a caller bound a string/blob against its
        # schema maxlen on the exact wire byte length, before allocation and without
        # re-encoding a decoded str. Mirrors Decoder.fixlen_len in the pure engine.
        if self._pk != _PEND_FIXLEN:
            # A parked receiver cap (§6.2.1) keeps the pending fixlen intact, and
            # this peek reads and allocates nothing — so it answers through the
            # parked rejection. That is what lets generated code decide the
            # SCHEMA bound (INVALID, §7.1) whether or not a binding entry has
            # been called yet. Mirrors Decoder.fixlen_len in the pure engine.
            # ... and, for the same reason, it does not re-raise the cap the
            # way a consuming read does.
            if self._pk == _PEND_LIMIT and self._pk_real == _PEND_FIXLEN:
                return self._pend_size
            if self._pk == _PEND_NONE:
                raise SofaArgumentError("no value pending for the current field")
            return None  # §7.3: not a fixlen field, so it has no fixlen length
        return self._pend_size

    def _string(self):
        cdef Py_ssize_t n
        cdef const unsigned char* p
        if self._pk != _PEND_FIXLEN or self._pend_subtype != _ST_STRING:
            return self._mismatch()
        n = <Py_ssize_t>self._pend_size      # bounded by FIXLEN_MAX at the header
        p = self._take_fixlen_ptr(n)
        try:
            # Decoded straight off the buffer: strict UTF-8 (no ``errors=``), so
            # an invalid payload raises rather than being replaced (§6.4).
            return PyUnicode_DecodeUTF8(<const char*>p, n, NULL)
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def _bytes(self):
        cdef Py_ssize_t n
        cdef const unsigned char* p
        if self._pk != _PEND_FIXLEN or self._pend_subtype != _ST_BLOB:
            return self._mismatch()
        n = <Py_ssize_t>self._pend_size
        p = self._take_fixlen_ptr(n)
        if self._spill is not None:
            return self._spill      # already an exact-size bytes; hand it over
        return PyBytes_FromStringAndSize(<const char*>p, n)

    # --- array reads --------------------------------------------------------

    def _read_unsigned_array(self, elem_max=None):
        # elem_max is the field's declared element width; see the pure engine's
        # Decoder.read_unsigned_array for what it buys (§7.1/§5.2, #267). The
        # pending value is cleared only once the payload has actually been
        # decoded, so a suspension leaves the array re-readable from element
        # one (§5.2).
        cdef uint64_t count
        cdef list out
        if self._pk != _PEND_VARRAY or self._pend_wtype != _WT_ARRAY_UNSIGNED:
            return self._mismatch()
        count = self._pend_count
        self._arm()
        if elem_max is None:
            out = self._read_varints(<Py_ssize_t>count, False, False, 0, 0)
        else:
            out = self._read_varints(
                <Py_ssize_t>count, False, True, 0, <int64_t>elem_max
            )
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return out

    def _read_signed_array(self, elem_min=None, elem_max=None):
        # The two halves of the declared width are independent: either may be
        # given on its own, in which case it bounds its own side and the other
        # stays at the widest an i64 element can be — passing one alone must not
        # fault on the missing half (issue #67).
        cdef uint64_t count
        cdef list out
        cdef int64_t lo = INT64_MIN
        cdef int64_t hi = INT64_MAX
        if self._pk != _PEND_VARRAY or self._pend_wtype != _WT_ARRAY_SIGNED:
            return self._mismatch()
        count = self._pend_count
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

    cdef bytes _take_farray_payload(self):
        # Read the pending fixlen array's payload inside a resume transaction: an
        # array whose payload has not fully arrived stays pending and re-readable
        # from its first payload byte (§5.2). Like the varint arrays, the pending
        # value is cleared only once the payload is in hand — the count and
        # element width stay in the decoder's C fields rather than being handed
        # back through a tuple nobody outlives. The whole tag is already matched
        # by the caller (§7.3).
        cdef bytes data
        self._arm()
        data = self._read_exact(self._farray_nbytes(self._pend_count, self._pend_size))
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return data

    cdef const unsigned char* _take_farray_ptr(self) except NULL:
        # _take_fixlen_ptr for a fixlen-array payload: the bytes in place where
        # they are already buffered, and _read_exact parked in _spill where they
        # are not. The whole tag is matched by the caller (S7.3).
        cdef Py_ssize_t n = self._farray_nbytes(self._pend_count,
                                                self._pend_size)
        cdef const unsigned char* p
        if self._n - self._pos >= n:
            self._pk = _PEND_NONE
            self._spill = None
            p = self._p + self._pos
            self._pos += n
            return p
        self._arm()
        self._spill = self._read_exact(n)
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return <const unsigned char*>PyBytes_AS_STRING(self._spill)

    cdef object _read_farray(self, int subtype, Py_ssize_t width):
        # The element width is settled at the fixlen_word — §4.8 fixes
        # it to 4 for fp32 and 8 for fp64, and §5.2 wants that INVALID verdict
        # before any payload read — so a pending array always matches ``width``
        # and the payload read below yields exactly count*width bytes or raises.
        # Re-checking the width here could only restate a decision no input can
        # reach (issue #75). The unpack loop stays in bounds regardless: it
        # derives its element count from the buffer it actually holds, never
        # from the wire.
        #
        # §4.8: a fixlen array always carries its fixlen_word, so the subtype is
        # known even for a zero-count array — check it like any other read.
        if self._pk != _PEND_FARRAY or self._pend_subtype != subtype:
            return self._mismatch()
        cdef bytes data = self._take_farray_payload()
        cdef const unsigned char* p = <const unsigned char*>PyBytes_AS_STRING(data)
        cdef Py_ssize_t count = PyBytes_GET_SIZE(data) // width
        cdef list out = PyList_New(count)
        cdef Py_ssize_t i
        if width == 4:
            for i in range(count):
                _SetItemSteal(out, i, _NewF64(_unpack_f32(p + i * 4)))
        else:
            for i in range(count):
                _SetItemSteal(out, i, _NewF64(_unpack_f64(p + i * 8)))
        return out

    cdef int _visit_farray_bits(self, object visitor, object fid) except -1:
        # Hand an fp32 array's payload over as raw wire bytes (S6.5's array
        # half). Nothing is decoded and nothing sized by the wire is allocated:
        # the payload is passed through the callback, which is S6.7's second
        # route -- the bytes are the caller's own input and stop being valid
        # when the callback returns. The view is *released* on the way out, so
        # that is enforced rather than merely documented.
        cdef Py_ssize_t count = <Py_ssize_t>self._pend_count
        cdef Py_ssize_t nbytes
        cdef object view
        cdef bytes data
        if self._pk == _PEND_LIMIT:
            # Nothing of this decoder's is at stake here, but the handler has
            # not been asked and cannot refuse -- unlike on_blob_begin, which is
            # offered the length first. The wire still chose how much has to be
            # contiguous before the callback can see it, so the configured
            # ceiling governs the route the way it always did (S6.2.1). The pure
            # engine does the same.
            raise SofaLimitError(self._limit_msg)
        nbytes = self._farray_nbytes(self._pend_count, self._pend_size)
        if self._n - self._pos >= nbytes:
            # Already buffered, which is the ordinary case: the view is made
            # straight over the fed bytes, so an array of any length costs one
            # handle and no copy.
            view = PyMemoryView_FromMemory(<char*>(self._p + self._pos),
                                           nbytes, PyBUF_READ)
            self._pos += nbytes
            self._pk = _PEND_NONE
        else:
            # Mid-payload on a chunk-fed reader: the pieces have to be joined
            # before the handler can see them whole, which is what
            # _take_farray_payload does (and it suspends if they have not all
            # arrived).
            data = self._take_farray_payload()
            view = memoryview(data).toreadonly()
        try:
            visitor.on_float32_array_bits(fid, count, view)
        finally:
            view.release()
        return 0

    def _read_float32_array(self):
        return self._read_farray(_ST_FP32, 4)

    def _read_float64_array(self):
        return self._read_farray(_ST_FP64, 8)

    # --- push-feed driver (CORELIB_PLAN §5.2) -------------------------------
    #
    # The native half of sofab.decoder.Decoder's push mode — same API and the
    # same outcomes. The header parses in C, and a destination a handler returns
    # from one of the five *begin* hooks is filled by a C loop over the fed
    # buffer, so an array costs no Python object per element.

    @property
    def error(self):
        return self._err

    @property
    def status(self):
        return _STATUS[self._status]

    @cython.always_allow_keywords(True)
    def feed(self, data):
        """Consume ``data`` and report the outcome for the bytes so far (§5.2).

        See :meth:`sofab.decoder.Decoder.feed` for the contract; this is the
        same call with the loop in C.
        """
        cdef object buf
        if self._running:
            raise SofaArgumentError("feed() is not re-entrant")
        if self._limit is not None:
            raise self._limit
        if self._status == <int>Status.INVALID:
            return Status.INVALID
        # Two shapes, because they want opposite things.
        #
        # Nothing carried — the whole previous feed was consumed, which is the
        # one-shot case and the steady state of a chunked one — and the chunk is
        # ADOPTED: ``bytes`` is immutable, so there is nothing to copy. (§6's
        # chunk-lifetime rule is what bytes(data) answers for every other input.)
        #
        # Something carried, i.e. a construct that could not complete: the buffer
        # becomes a bytearray and the chunk is APPENDED. Such a construct leaves
        # _pos at 0 (the suspension rewinds it), so every following feed only
        # extends, at amortised O(len(chunk)). Rebuilding ``carry + chunk``
        # instead would copy the whole carry per chunk — a 1 MB blob fed in
        # 4 KiB pieces costs ~122 MB of copying that way.
        # Sets _pos itself -- see the pure engine.
        self._reassemble(data)
        self._keep = self._pos
        self._running = True
        try:
            if self._drive_push():
                self._status = <int>Status.INCOMPLETE
                return Status.INCOMPLETE
        except SofaIncompleteError:
            self._status = <int>Status.INCOMPLETE
            return Status.INCOMPLETE
        except SofaLimitError as exc:
            # §6.3: a terminal, receiver-local policy rejection -- see the pure
            # engine. Raised by this decoder's own check or by a handler deciding
            # on an index the codec surfaced (§6.2.1).
            self._limit = exc
            self._err = exc
            raise
        except SofaDecodeError as exc:
            self._err = exc
            self._status = <int>Status.INVALID
            return Status.INVALID
        finally:
            self._running = False
            self._retain()
        self._status = <int>Status.COMPLETE
        return Status.COMPLETE

    cdef int _reassemble(self, object data) except -1:
        # Put ``data`` where the walk can reach it, using only the caller's
        # reassembly buffer (§6.6). Mirrors Decoder._reassemble.
        cdef Py_ssize_t held = self._rend - self._rstart
        cdef Py_ssize_t n
        cdef object r
        if not held:
            self._rebind(data if type(data) is bytes else bytes(data))
            self._rstart = 0
            self._rend = 0
            self._pos = 0
            return 0
        r = self._rbuf
        n = len(data)
        if self._rend + n > len(r):
            if self._rstart:
                r[:held] = r[self._rstart:self._rend]
                self._rstart = 0
                self._rend = held
            if held + n > len(r):
                raise SofaArgumentError(
                    "reassembly buffer holds %d bytes; the construct spanning "
                    "this chunk needs %d" % (len(r), held + n))
        r[self._rend:self._rend + n] = data
        self._rend += n
        self._rebind(r)
        # _rebind takes the whole bytearray's length; only _rend of it is data.
        self._n = self._rend
        self._pos = self._rstart
        return 0

    cdef int _retain(self) except -1:
        # Keep what this feed did not consume and let the chunk go, so §6's
        # chunk-lifetime promise holds. Mirrors Decoder._retain.
        cdef object r = self._rbuf
        cdef Py_ssize_t carry
        if self._status == <int>Status.INVALID or self._limit is not None:
            # Terminal (S5.2.3, S6.3): nothing resumes, so nothing is kept --
            # see the pure engine.
            self._rstart = 0
            self._rend = 0
            self._rebind(b"")
            self._pos = 0
            return 0
        carry = self._n - self._pos
        if not carry:
            self._rstart = 0
            self._rend = 0
            self._rebind(b"")
            self._pos = 0
            return 0
        if self._buf is r:
            self._rstart = self._pos
            return 0
        if carry > len(r):
            raise SofaArgumentError(
                "reassembly buffer holds %d bytes; the construct spanning "
                "this chunk needs %d" % (len(r), carry))
        r[:carry] = self._buf[self._pos:self._n]
        self._rstart = 0
        self._rend = carry
        self._rebind(r)
        self._n = carry          # see _reassemble
        self._pos = 0
        return 0

    def reset(self):
        """Forget the stream and start a new message, keeping the compiled
        binding and its destinations. See the pure engine for the contract."""
        self._rebind(b"")
        self._pos = 0
        self._rstart = 0
        self._rend = 0
        self._depth = 0
        self._cur_wtype = -1
        self._spill = None
        self._pk = _PEND_NONE
        self._pk_real = _PEND_NONE
        self._limit_msg = None
        self._keep = 0
        self._keep_cur_wtype = -1
        self._status = <int>Status.COMPLETE
        self._err = None
        self._limit = None
        if self._vsp:
            # A descent left mid-message: the handler the caller gave us is the
            # one at the bottom of the stack. The slots are the decoder's own
            # storage (S6.6), so reset drops their references and rewinds the
            # index rather than releasing anything.
            self._bind_visitor(<object>self._vstack[0])
            while self._vsp > 0:
                self._vsp -= 1
                Py_XDECREF(self._vstack[self._vsp])
                self._vstack[self._vsp] = NULL
        self._resume_kind = _R_NONE
        self._resume_entry = -1
        self._tab = 0 if self._tables is not None else -1
        self._tsp = 0

    cdef inline bint _value_ready(self) noexcept:
        # Are all of the pending value's bytes buffered? Asked once per feed, on
        # the resume path only, and only for the two kinds whose length is known
        # from the header. The first attempt at a value still just tries and
        # catches; what this removes is the retry raising again on every later
        # chunk — a 1 MB blob fed in 4 KiB pieces suspends 244 times.
        cdef uint64_t want
        if self._pk == _PEND_FIXLEN:
            return <uint64_t>(self._n - self._pos) >= self._pend_size
        if self._pk == _PEND_FARRAY:
            if self._pend_size and self._pend_count > _SSIZE_MAX / self._pend_size:
                return False   # unsatisfiable; the read path reports it
            want = self._pend_count * self._pend_size
            return <uint64_t>(self._n - self._pos) >= want
        return True

    cdef int _drive_push(self) except -1:
        # Returns 1 when it stopped short of a value whose bytes have not all
        # arrived — the cheap half of INCOMPLETE, with no exception raised.
        # Every other suspension still comes through SofaIncompleteError.
        cdef int t, rk
        cdef Py_ssize_t ei
        cdef _BEntry* e
        cdef object visitor = self._visitor
        cdef bint make_field = self._make_field
        # Hoisted: a decoder with no destination map must not pay a test per
        # field for one, which is what keeps the pure-visitor path at its old
        # cost. Recomputed only where _tab changes -- a descent and its close.
        cdef bint mapped = self._tab >= 0
        cdef bint has_visitor = visitor is not None
        cdef object answer
        cdef object f

        rk = self._resume_kind
        if rk != _R_NONE:
            # A value read ran out of bytes last time. Finish *it* — walking on
            # to the next header would auto-skip the value the caller is owed.
            if rk != _R_SKIP and not self._value_ready():
                return 1
            self._resume_kind = _R_NONE
            try:
                if rk == _R_VISIT:
                    if self._resume_entry >= 0:
                        self._mapped_field(self._resume_entry)
                    else:
                        self._visit_value(visitor)
                else:
                    self._skip()
            except SofaIncompleteError:
                self._resume_kind = rk
                raise

        while True:
            t = self._next_wire()
            if t < 0:
                return 0

            if t == _WT_SEQUENCE_END:
                if self._tsp > 0:
                    self._tsp -= 1
                    self._tab = self._tstack[self._tsp]
                    mapped = self._tab >= 0
                # The end belongs to whoever was handling the scope, so a child
                # hears its own scope close before it is popped.
                if visitor is not None:
                    visitor.on_sequence_end()
                if self._vsp:
                    self._vsp -= 1
                    visitor = <object>self._vstack[self._vsp]
                    Py_XDECREF(self._vstack[self._vsp])
                    self._vstack[self._vsp] = NULL
                    self._bind_visitor(visitor)
                    make_field = self._make_field
                continue

            # --- the destination map, consulted ONCE per field ---------------
            # It answers two questions, both of them the CALLER's: where does
            # this value go, and what does the schema declare for it. Neither is
            # a decode rule, and no rule below branches on the answer -- the
            # walk, the cap, the bound, the §7.3 test, the UTF-8 check and the
            # element width are the same code for a mapped field and an
            # unmapped one. That is what makes this one surface and not two.
            ei = self._lookup(self._cur_id) if mapped else -1
            if ei >= 0:
                e = &self._bent[ei]
                if e.wt != t or (e.st >= 0 and e.st != self._pend_subtype):
                    # §7.3: the wire tag contradicts what the schema declared
                    # for this id. Treated exactly like an unknown id.
                    ei = -1
                    if t != _WT_SEQUENCE_START:
                        continue

            if t == _WT_SEQUENCE_START:
                if ei >= 0:
                    e = &self._bent[ei]
                    self._push_table(e.child)
                    mapped = self._tab >= 0
                    if e.count_at >= 0:
                        self._words[e.count_at] += 1
                    continue
                if not has_visitor:
                    try:
                        self._skip()
                    except SofaIncompleteError:
                        self._resume_kind = _R_SKIP
                        raise
                    continue
                answer = None
                if self._wants_seq_begin:
                    answer = visitor.on_sequence_begin(
                        PyLong_FromUnsignedLongLong(self._cur_id))
                if answer is not False:
                    # §4.9 opens a fresh id scope, so the enclosing table must
                    # not match inside it. A decoder that has no map at all
                    # keeps no table stack either.
                    if self._tables is not None:
                        self._push_table(-1)
                        mapped = False
                    if isinstance(answer, _Visitor):
                        # The handler named someone else for this sub-tree.
                        Py_INCREF(visitor)
                        self._vstack[self._vsp] = <PyObject*>visitor
                        self._vsp += 1
                        visitor = answer
                        self._bind_visitor(answer)
                        make_field = self._make_field
                    continue
                try:
                    self._skip()
                except SofaIncompleteError:
                    self._resume_kind = _R_SKIP
                    raise
                continue

            if ei >= 0:
                # One call, so the hot loop keeps the shape it has for every
                # other field: the mapped field's bound and store live in
                # _mapped_field, not here.
                try:
                    self._mapped_field(ei)
                except SofaIncompleteError:
                    self._resume_kind = _R_VISIT
                    self._resume_entry = ei
                    raise
                continue
            if make_field and visitor.on_field(self._build_field(t)) is False:
                # Skipped: the pending value stays pending and the next header
                # discards it, which suspends in exactly the same place an
                # explicit skip would and costs one call less. Nothing is
                # materialized and nothing is validated (S6.7.2), so neither the
                # cap nor a schema bound is answered for it.
                continue
            if self._wants_bound:
                # Independent of make_field now: this hook takes integers, so a
                # handler that declares schema bounds and nothing else builds no
                # Field at all.
                self._schema_bound(
                    visitor, PyLong_FromUnsignedLongLong(self._cur_id))
            if not has_visitor:
                # Nobody wants it: the pending value stays pending and the next
                # header discards it, which suspends in exactly the same place
                # an explicit skip would and costs one less call.
                continue
            try:
                self._visit_value(visitor)
            except SofaIncompleteError:
                self._resume_kind = _R_VISIT
                self._resume_entry = -1
                raise

    cdef int _schema_bound(self, object visitor, object fid) except -1:
        # The hook half: ask the handler, then hand the answer to the one site
        # that owns the rule. The map path calls _settle_bound directly. Both
        # arguments are integers, so overriding the hook costs no object per
        # field -- which is the whole reason it does not take a Field.
        cdef int pk = self._pk
        cdef int real = self._pk_real if pk == _PEND_LIMIT else pk
        cdef object n
        if real == _PEND_FIXLEN:
            if self._pend_subtype < _ST_STRING:
                return 0   # an fp32/fp64 payload: a fixed width, nothing to bound
            n = PyLong_FromUnsignedLongLong(self._pend_size)
        elif real == _PEND_VARRAY or real == _PEND_FARRAY:
            n = PyLong_FromUnsignedLongLong(self._pend_count)
        else:
            return 0
        return self._settle_bound(<Py_ssize_t>visitor.on_schema_bound(fid, n))

    cdef int _settle_bound(self, Py_ssize_t bound) except -1:
        # THE site. What the schema declares does two things (S6.2.1,
        # MESSAGE_SPEC S7.1): above it is INVALID, and the receiver-side cap
        # stops applying. Both happen here, at the count/length header, before a
        # payload byte is read -- whether the number came from a table or from
        # on_schema_bound.
        cdef int pk = self._pk
        cdef int real = self._pk_real if pk == _PEND_LIMIT else pk
        cdef uint64_t n
        cdef str what
        if bound < 0:
            return 0
        # Only a count- or length-bearing field can reach here with a bound --
        # see the pure engine's note.
        if real == _PEND_FIXLEN:
            n = self._pend_size
            what = "fixlen length"
        else:
            n = self._pend_count
            what = "array count"
        if n > <uint64_t>bound:
            raise SofaDecodeError(
                "%s %d exceeds the %d the schema declares"
                % (what, PyLong_FromUnsignedLongLong(n), bound))
        if pk == _PEND_LIMIT:
            # Spent: the schema bounds this field, so the cap never governed it.
            self._pk = self._pk_real
            self._limit_msg = None
        return 0

    cdef int _fill_varints_at(self, void* dst, int itemsize, Py_ssize_t count,
                              bint zigzag, bint bounded, int64_t lo,
                              uint64_t hi) except -1:
        # _read_varints without the list: the elements go straight from the fed
        # buffer into the caller's slots, so an array costs no allocation and no
        # Python object per element. Reached from a binding (above) and from a
        # visitor that handed back a destination in on_array_begin (§6.6.3).
        cdef Py_ssize_t pos, n, i = 0
        cdef const unsigned char* p
        cdef _Varint v
        cdef uint64_t result
        cdef int64_t sv
        self._arm()
        pos = self._pos
        p = self._p
        n = self._n
        while i < count:
            if n - pos >= 10:
                # Whole element guaranteed buffered: no per-byte bounds test.
                v = _varint_take(p + pos)
                if v.used < 0:
                    raise SofaDecodeError("overlong varint")
                pos += v.used
                result = v.value
            else:
                # Near the end of the buffer an element may straddle the chunk.
                self._pos = pos
                result = self._varint_refill()
                p = self._p
                pos = self._pos
                n = self._n
            if zigzag:
                sv = _zigzag_decode(result)
                # Checked AT the element, before it is stored, so the INVALID is
                # reached before a truncation behind the bad element (§7.1/§5.2).
                if bounded and (sv < lo or sv > <int64_t>hi):
                    raise SofaDecodeError("array element outside declared width")
                # The declared width was matched against the destination's item
                # size at the header, so the narrowing here cannot lose a bit.
                if itemsize == 8:
                    (<int64_t*>dst)[i] = sv
                elif itemsize == 4:
                    (<int32_t*>dst)[i] = <int32_t>sv
                elif itemsize == 2:
                    (<int16_t*>dst)[i] = <int16_t>sv
                else:
                    (<int8_t*>dst)[i] = <int8_t>sv
            else:
                if bounded and result > hi:
                    raise SofaDecodeError("array element outside declared width")
                if itemsize == 8:
                    (<uint64_t*>dst)[i] = result
                elif itemsize == 4:
                    (<uint32_t*>dst)[i] = <uint32_t>result
                elif itemsize == 2:
                    (<uint16_t*>dst)[i] = <uint16_t>result
                else:
                    (<uint8_t*>dst)[i] = <uint8_t>result
            i += 1
        self._pos = pos
        self._pk = _PEND_NONE   # committed only once the payload is in hand
        return 0

    cdef int _fill_farray(self, double* dst, Py_ssize_t count, int width) except -1:
        cdef Py_ssize_t nbytes = self._farray_nbytes(self._pend_count, self._pend_size)
        cdef const unsigned char* p
        cdef Py_ssize_t i
        self._arm()
        if self._n - self._pos < nbytes:
            raise self._suspend("truncated payload")
        p = self._p + self._pos
        if width == 4:
            for i in range(count):
                dst[i] = _unpack_f32(p + i * 4)
        else:
            for i in range(count):
                dst[i] = _unpack_f64(p + i * 8)
        self._pos += nbytes
        self._pk = _PEND_NONE
        return 0

    cdef bint _width_fits(self, int itemsize, bint zigzag, bint bounded,
                          int64_t lo, uint64_t hi):
        # Does the width the handler declared fit an itemsize-byte slot?
        cdef int bits = itemsize * 8
        if not bounded:
            return False
        if zigzag:
            return lo >= -(<int64_t>1 << (bits - 1)) and <int64_t>hi < (<int64_t>1 << (bits - 1))
        return hi < (<uint64_t>1 << bits) if bits < 64 else True

    cdef int _take_blob_into(self, object dst, Py_ssize_t size) except -1:
        # Copy a blob's payload into the caller's buffer (§6.6.3) -- no bytes
        # built on the way, which is the point: the only size a codec could
        # build one from is the wire's.
        cdef Py_buffer view
        cdef const unsigned char* p
        try:
            PyObject_GetBuffer(dst, &view, PyBUF_WRITABLE | PyBUF_SIMPLE)
        except (BufferError, TypeError) as exc:
            raise SofaArgumentError(
                "on_blob_begin returned a destination that is not a writable, "
                "contiguous buffer") from exc
        try:
            if view.itemsize != 1:
                raise SofaArgumentError(
                    "on_blob_begin's destination must hold single bytes")
            if view.len < size:
                raise SofaArgumentError(
                    "on_blob_begin returned %d bytes for a blob of %d"
                    % (view.len, size))
            p = self._take_fixlen_ptr(size)
            memcpy(view.buf, <const void*>p, size)
            self._spill = None
        finally:
            PyBuffer_Release(&view)
        return 0

    cdef int _take_string_into(self, object dst, Py_ssize_t size) except -1:
        # S6.6.3's destination route for a string. No str is built on the way,
        # but the bytes are still validated: S6.7.2 makes a field the handler
        # *reads* both materialized and validated. Mirrors the pure engine.
        cdef Py_buffer view
        cdef const unsigned char* p
        cdef Py_ssize_t used = 0
        try:
            PyObject_GetBuffer(dst, &view, PyBUF_WRITABLE | PyBUF_SIMPLE)
        except (BufferError, TypeError) as exc:
            raise SofaArgumentError(
                "on_string_begin returned a destination that is not a "
                "writable, contiguous buffer") from exc
        try:
            if view.itemsize != 1:
                raise SofaArgumentError(
                    "on_string_begin's destination must hold single bytes")
            if view.len < size:
                raise SofaArgumentError(
                    "on_string_begin returned %d bytes for a string of %d"
                    % (view.len, size))
            p = self._take_fixlen_ptr(size)
            # Validate before the destination is touched, so a caller never
            # sees a half-written buffer behind an INVALID verdict -- and by
            # walking the bytes, so the str the destination exists to avoid is
            # never built.
            _decode_utf8_checked(p, size)
            memcpy(view.buf, <const void*>p, size)
            self._spill = None
        finally:
            PyBuffer_Release(&view)
        return 0

    cdef int _visit_farray_into(self, object visitor, object fid,
                                Py_ssize_t width) except -1:
        # Offer a fixlen array's elements a destination (S6.6.3). Returns 0 when
        # the handler wants the list instead. Mirrors the pure engine.
        cdef object dst
        cdef Py_ssize_t count = <Py_ssize_t>self._pend_count
        cdef Py_buffer view
        cdef const unsigned char* p
        cdef double* out
        cdef Py_ssize_t i
        if not self._wants_farray_begin:
            return 0
        dst = visitor.on_float_array_begin(
            fid, _FP32_OBJ if width == 4 else _FP64_OBJ, count)
        if dst is None:
            return 0
        if self._pk == _PEND_LIMIT:
            # Spent: the destination is the handler's own storage, so there is
            # no allocation of this decoder's left for the cap to prevent.
            self._pk = self._pk_real
            self._limit_msg = None
        try:
            PyObject_GetBuffer(dst, &view, PyBUF_WRITABLE | PyBUF_SIMPLE)
        except (BufferError, TypeError) as exc:
            raise SofaArgumentError(
                "on_float_array_begin returned a destination that is not a "
                "writable, contiguous buffer") from exc
        try:
            if view.itemsize != 8:
                raise SofaArgumentError(
                    "on_float_array_begin's destination must hold 8-byte "
                    "items; a Python float is a double, and so is every value "
                    "written here")
            if view.len < count * 8:
                raise SofaArgumentError(
                    "on_float_array_begin returned %d slots for an array of %d"
                    % (view.len // 8, count))
            p = self._take_farray_ptr()
            out = <double*>view.buf
            if width == 4:
                for i in range(count):
                    out[i] = _unpack_f32(p + i * 4)
            else:
                for i in range(count):
                    out[i] = _unpack_f64(p + i * 8)
            self._spill = None
        finally:
            PyBuffer_Release(&view)
        return 1

    cdef int _visit_varints(self, object visitor, object fid, int wtype,
                            bint zigzag) except -1:
        # Deliver an integer array by whichever route on_array_begin asked for
        # (§6.6.3). Two things can only be settled here, at the header: the
        # element width the schema declares, and where the elements are to go.
        # Both are gone by the time the typed hook holds the list.
        cdef object spec, dst = None, lo = None, hi = None
        cdef Py_ssize_t count = self._pend_count
        cdef Py_buffer view
        cdef int isz
        cdef bint bounded
        cdef int64_t clo = INT64_MIN
        cdef uint64_t chi = <uint64_t>0xFFFFFFFFFFFFFFFF
        cdef bint capped = self._pk == _PEND_LIMIT
        if self._wants_array_begin:
            # Asked before a parked receiver cap is answered (#128) — see the
            # pure engine for why. In short: §6.2.1's limit stops the sender
            # dictating the receiver's allocation, and a handler that hands back
            # a buffer has chosen the size itself.
            spec = visitor.on_array_begin(fid, _WT[wtype], count)
            if spec is not None:
                dst, lo, hi = spec
        # A declared width that admits every value of its domain is not a
        # constraint, and the list route's entry points take a narrowed maximum
        # (they were written for the binding, which never declares a wider one).
        # Dropping it here keeps the two engines on the same verdict -- the pure
        # one compares and never fires -- instead of overflowing the cast.
        if hi is not None and hi > _INT64_MAX_OBJ:
            hi = None
        if dst is None:
            # Nowhere to put them but a list of this decoder's own, sized by the
            # wire. That is the allocation §6.2.1 exists to prevent, and the
            # count header — here, before an element is read — is where it says
            # so. The list reads below reach the same verdict through _mismatch;
            # raised here so the message names the count rather than the kind.
            if capped:
                raise SofaLimitError(self._limit_msg)
            if zigzag:
                visitor.on_signed_array(fid, self._read_signed_array(lo, hi))
            else:
                visitor.on_unsigned_array(fid, self._read_unsigned_array(hi))
            return 0
        if capped:
            # Spent: unpark, or the fill below would find the wrapper kind where
            # it looks for the array.
            self._pk = self._pk_real
            self._limit_msg = None
        bounded = lo is not None or hi is not None
        if lo is not None:
            clo = lo
        if hi is not None:
            chi = hi
        # The buffer is the caller's: a short one is refused, never grown
        # (§6.6), and the verdict is reached here, at the count header, before
        # an element is read.
        try:
            PyObject_GetBuffer(dst, &view, PyBUF_WRITABLE | PyBUF_SIMPLE)
        except (BufferError, TypeError) as exc:
            raise SofaArgumentError(
                "on_array_begin returned a destination that is not a writable, "
                "contiguous buffer") from exc
        try:
            isz = view.itemsize
            if isz != 1 and isz != 2 and isz != 4 and isz != 8:
                raise SofaArgumentError(
                    "on_array_begin's destination holds %d-byte items; 1, 2, 4 "
                    "or 8 are supported" % isz)
            # A destination narrower than 8 bytes is only safe if the declared
            # width says every element fits it. Checked once, here, so the fill
            # loop can narrow without a second test per element -- and refused
            # rather than silently truncated.
            if isz != 8 and not self._width_fits(isz, zigzag, bounded, clo, chi):
                raise SofaArgumentError(
                    "on_array_begin declared a width that does not fit its "
                    "%d-byte destination" % isz)
            if view.len < count * isz:
                raise SofaArgumentError(
                    "on_array_begin returned %d slots for an array of %d"
                    % (view.len // isz, count))
            self._fill_varints_at(view.buf, isz, count, zigzag,
                                  bounded, clo, chi)
        finally:
            PyBuffer_Release(&view)
        return 0

    cdef object _build_field(self, int t):
        # The Field on_field is handed. Built HERE, not in the header walk, and
        # only for a field that is actually going to be offered: a field the
        # handler's destination map names never reaches on_field, so one built
        # for it was an object the caller could not observe. Everything it
        # carries is already on the decoder, so nothing is re-parsed for it.
        cdef int pk = self._pk
        cdef int real = self._pk_real if pk == _PEND_LIMIT else pk
        cdef object fid = PyLong_FromUnsignedLongLong(self._cur_id)
        if real == _PEND_SCALAR:
            return _mkfield(fid, _WT[t], _ZERO, _ZERO, _NONE)
        if real == _PEND_FIXLEN:
            return _mkfield(fid, _WT[_WT_FIXLEN],
                            PyLong_FromUnsignedLongLong(self._pend_size), _ZERO,
                            _ST[self._pend_subtype])
        if real == _PEND_VARRAY:
            return _mkfield(fid, _WT[t], _ZERO,
                            PyLong_FromUnsignedLongLong(self._pend_count), _NONE)
        return _mkfield(fid, _WT[_WT_ARRAY_FIXLEN],
                        PyLong_FromUnsignedLongLong(self._pend_size),
                        PyLong_FromUnsignedLongLong(self._pend_count),
                        _ST[self._pend_subtype])

    cdef int _mapped_field(self, Py_ssize_t ei) except -1:
        # A field the handler's declared destination map names.
        #
        # NOTHING IS DECIDED HERE that is not decided for every other field.
        # The schema bound comes off the map instead of off a hook, but it is
        # the same number reaching the same rule (_settle_bound) -- which is why
        # a mapped field and a hooked one cannot disagree about it, the
        # divergence A2-0147 measured. What follows is the assignment a typed
        # hook would otherwise have made: words[at] = value instead of
        # visitor.on_unsigned(id, value). No parse, no check and no verdict of
        # its own, which is the whole difference from the _take_bound this
        # replaces.
        #
        # Consumes nothing on the suspension path, like every other read, so the
        # retry redoes the whole value -- including refilling a partly written
        # array from element zero (S5.2).
        cdef int t = self._cur_wtype
        cdef int st = self._pend_subtype
        cdef _BEntry* e = &self._bent[ei]
        cdef uint64_t got
        if e.kind >= _K_ARRAY_UNSIGNED and e.kind != _K_SEQUENCE:
            # A declared array's destination IS its schema bound.
            self._settle_bound(e.cap)
        elif e.kind == _K_STRING or e.kind == _K_BYTES:
            self._settle_bound(e.cap if e.cap else -1)
        # --- the store, and nothing else ---------------------------------
        # Every rule has already run above, in the code an unmapped field
        # runs too. What is left is the assignment a hook would otherwise
        # have made: ``words[at] = value`` instead of
        # ``visitor.on_unsigned(id, value)``. No parse, no check and no
        # verdict lives down here, which is the whole difference from the
        # ``_take_bound`` this replaces.
        got = 1
        if t == _WT_UNSIGNED:
            self._words[e.at] = self._take_scalar()
        elif t == _WT_SIGNED:
            (<int64_t*>self._words)[e.at] = _zigzag_decode(self._take_scalar())
        elif t == _WT_FIXLEN:
            if st == _ST_FP32:
                (<double*>self._words)[e.at] = _unpack_f32(self._take_fixlen_ptr(4))
            elif st == _ST_FP64:
                (<double*>self._words)[e.at] = _unpack_f64(self._take_fixlen_ptr(8))
            else:
                if self._pk == _PEND_LIMIT:
                    # The schema left this field unbounded, so the cap still
                    # governs it and has already rejected it. The hook path
                    # reaches the same verdict through _mismatch; the store
                    # goes straight to the payload, so it is raised here.
                    raise SofaLimitError(self._limit_msg)
                self._take_object(e.kind, e.at)
        elif t == _WT_ARRAY_UNSIGNED:
            got = self._pend_count
            self._fill_varints(self._words + e.at, <Py_ssize_t>got, False, e)
        elif t == _WT_ARRAY_SIGNED:
            got = self._pend_count
            self._fill_varints(self._words + e.at, <Py_ssize_t>got, True, e)
        elif st == _ST_FP32:
            got = self._pend_count
            self._fill_farray((<double*>self._words) + e.at, <Py_ssize_t>got, 4)
        else:
            got = self._pend_count
            self._fill_farray((<double*>self._words) + e.at, <Py_ssize_t>got, 8)
        if e.count_at >= 0:
            self._words[e.count_at] = got
        return 0

    cdef int _visit_value(self, object visitor) except -1:
        # The value half of the walk. Every read is chosen by the field's own
        # wire type, so none can hit the §7.3 mismatch path. The id and subtype
        # come off the decoder's own state rather than a Field: the typed hooks
        # take an id, not a Field, so unless on_field is overridden there is no
        # reason to build one.
        cdef int t = self._cur_wtype
        cdef int st = self._pend_subtype
        cdef object fid = PyLong_FromUnsignedLongLong(self._cur_id)
        cdef object dst
        if t == _WT_UNSIGNED:
            visitor.on_unsigned(fid, self._unsigned())
        elif t == _WT_SIGNED:
            visitor.on_signed(fid, self._signed())
        elif t == _WT_FIXLEN:
            if st == _ST_FP32:
                if self._wants_f32_bits:
                    # S6.5: a bit-exact consumer takes the wire bits, never the
                    # widened value.
                    visitor.on_float32_bits(fid, self._float32_bits())
                else:
                    visitor.on_float32(fid, self._float32())
            elif st == _ST_FP64:
                visitor.on_float64(fid, self._float64())
            elif st == _ST_STRING:
                # Same bargain as on_blob_begin below (#128): the handler is
                # told the announced byte length before a byte is copied, and a
                # buffer it hands back is one it sized itself.
                if self._wants_string_begin:
                    dst = visitor.on_string_begin(fid, self._pend_size)
                    if dst is not None:
                        if self._pk == _PEND_LIMIT:
                            self._pk = self._pk_real
                            self._limit_msg = None
                        self._take_string_into(dst, <Py_ssize_t>self._pend_size)
                        return 0
                visitor.on_string(fid, self._string())
            else:
                # Offered before a parked receiver cap is answered (#128).
                # §6.2.1's limit is there to stop the sender dictating the
                # receiver's allocation, and a handler that hands back a buffer
                # has already chosen the size itself — there is no allocation of
                # this decoder's left for the cap to prevent. It is told the
                # announced length and may refuse it; that call is the
                # handler's. With no destination back, the ``bytes`` this
                # decoder would build is the allocation §6.2.1 is about, and
                # _bytes() reaches the parked verdict through _mismatch.
                if self._wants_blob_begin:
                    dst = visitor.on_blob_begin(fid, self._pend_size)
                    if dst is not None:
                        if self._pk == _PEND_LIMIT:
                            # Spent: unpark, or the copy below would find the
                            # wrapper kind where it looks for the payload.
                            self._pk = self._pk_real
                            self._limit_msg = None
                        self._take_blob_into(dst, <Py_ssize_t>self._pend_size)
                        return 0
                visitor.on_bytes(fid, self._bytes())
        elif t == _WT_ARRAY_UNSIGNED:
            self._visit_varints(visitor, fid, t, False)
        elif t == _WT_ARRAY_SIGNED:
            self._visit_varints(visitor, fid, t, True)
        elif st == _ST_FP32:
            if self._wants_f32_array_bits:
                self._visit_farray_bits(visitor, fid)
            elif not self._visit_farray_into(visitor, fid, 4):
                visitor.on_float32_array(fid, self._read_float32_array())
        elif not self._visit_farray_into(visitor, fid, 8):
            visitor.on_float64_array(fid, self._read_float64_array())
        return 0


# Marker so callers / tests can assert which implementation is active.
IMPL = "native"
