# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
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
"""

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize, PyBytes_GET_SIZE
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_GET_SIZE
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.long cimport PyLong_FromUnsignedLongLong, PyLong_FromLongLong
from cpython.ref cimport Py_INCREF
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy

# Wire-format constants, enums, the Field descriptor and the error classes all
# live in the shared pure-Python ``types`` module — reuse them verbatim so the
# native path raises the *same* exception types and yields the *same* Field /
# enum objects the pure path does.
from .types import (
    ARRAY_MAX,
    FIXLEN_MAX,
    ID_MAX,
    MAX_DEPTH,
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


# --- ZigZag (identical math to sofab._varint) --------------------------------
cdef inline uint64_t _zigzag_encode(int64_t v) noexcept nogil:
    return (<uint64_t>v << 1) ^ <uint64_t>(v >> 63)

cdef inline int64_t _zigzag_decode(uint64_t u) noexcept nogil:
    return <int64_t>(u >> 1) ^ -<int64_t>(u & 1)


# =============================================================================
# Encoder
# =============================================================================

cdef class Encoder:
    """Native encoder — see :class:`sofab.encoder.Encoder` for the full contract.

    Two construction models, byte-identical to the pure-Python encoder:

    * ``Encoder(writer=None, sticky=False)`` — Go-style growable buffer; bytes
      accumulate in an internal C buffer drained to ``writer`` on :meth:`flush`
      (or read back via :meth:`getvalue`).
    * ``Encoder.over_buffer(buffer, offset, flush)`` — Rust/C-style; writes into
      a caller-owned ``bytearray``, draining through ``flush`` when it fills.
    """

    # growable (in-memory) mode
    cdef unsigned char* _out
    cdef size_t _len
    cdef size_t _cap
    # fixed-buffer mode
    cdef object _fixed_obj          # the caller's bytearray (keeps it alive)
    cdef unsigned char* _fixed_ptr
    cdef size_t _fixed_cap
    cdef size_t _cursor
    cdef object _flush_sink
    cdef bint _is_fixed
    # shared
    cdef object _writer
    cdef bint _sticky
    cdef object _error
    cdef int _depth

    def __cinit__(self):
        self._out = NULL
        self._len = 0
        self._cap = 0
        self._fixed_obj = None
        self._fixed_ptr = NULL
        self._fixed_cap = 0
        self._cursor = 0
        self._flush_sink = None
        self._is_fixed = False
        self._writer = None
        self._sticky = False
        self._error = None
        self._depth = 0

    def __init__(self, writer=None, *, bint sticky=False):
        self._writer = writer
        self._sticky = sticky

    def __dealloc__(self):
        if self._out != NULL:
            free(self._out)
            self._out = NULL

    @classmethod
    def over_buffer(cls, bytearray buffer, int offset=0, flush=None, *, bint sticky=False):
        cdef Encoder self = cls.__new__(cls)
        self._writer = None
        self._flush_sink = flush
        self._sticky = sticky
        self.buffer_set(buffer, offset)
        return self

    def buffer_set(self, bytearray buffer, int offset=0):
        if not (0 <= offset < PyByteArray_GET_SIZE(buffer)):
            raise SofaRangeError("offset must be within the buffer")
        self._fixed_obj = buffer
        self._fixed_ptr = <unsigned char*>PyByteArray_AS_STRING(buffer)
        self._fixed_cap = <size_t>PyByteArray_GET_SIZE(buffer)
        self._cursor = <size_t>offset
        self._is_fixed = True

    # --- error / output plumbing --------------------------------------------

    @property
    def error(self):
        return self._error

    cdef inline int _ensure(self, size_t extra) except -1:
        # grow-mode capacity guarantee
        cdef size_t need = self._len + extra
        cdef size_t newcap
        if need <= self._cap:
            return 0
        newcap = self._cap * 2 if self._cap else 64
        while newcap < need:
            newcap *= 2
        self._out = <unsigned char*>realloc(self._out, newcap)
        if self._out == NULL:
            raise MemoryError()
        self._cap = newcap
        return 0

    cdef int _put(self, const unsigned char* data, size_t n) except -1:
        # Append n raw bytes, honouring whichever output mode is active.
        cdef size_t pos = 0
        cdef size_t take
        if not self._is_fixed:
            self._ensure(n)
            memcpy(self._out + self._len, data, n)
            self._len += n
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
        if self._flush_sink is None:
            raise SofaBufferError("encoder buffer full")
        self._flush_sink(PyBytes_FromStringAndSize(<char*>self._fixed_ptr, <Py_ssize_t>self._cursor))
        self._cursor = 0
        return 0

    cdef inline int _emit_varint(self, uint64_t value) except -1:
        # The hot path. Grow-mode writes straight into the C buffer; fixed-mode
        # encodes to a small stack scratch and hands it to the chunk-aware _put.
        cdef unsigned char scratch[10]
        cdef int i = 0
        cdef unsigned char b
        if not self._is_fixed:
            self._ensure(10)
            while True:
                b = <unsigned char>(value & 0x7F)
                value >>= 7
                if value:
                    self._out[self._len] = b | 0x80
                    self._len += 1
                else:
                    self._out[self._len] = b
                    self._len += 1
                    return 0
        else:
            while True:
                b = <unsigned char>(value & 0x7F)
                value >>= 7
                if value:
                    scratch[i] = b | 0x80
                    i += 1
                else:
                    scratch[i] = b
                    i += 1
                    break
            self._put(scratch, <size_t>i)
            return 0

    cdef inline int _header(self, object field_id, int wtype) except -1:
        if field_id < 0 or field_id > ID_MAX:
            raise SofaRangeError("id %d out of range 0..%d" % (field_id, _ID_MAX))
        self._emit_varint((<uint64_t>field_id << 3) | <uint64_t>wtype)
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
        return <object>self._cursor if self._is_fixed else <object>self._len

    def flush(self):
        cdef size_t used
        if self._is_fixed:
            used = self._cursor
            if self._flush_sink is not None and used:
                self._drain()
            return <object>used
        used = self._len
        if self._writer is not None and used:
            self._writer.write(PyBytes_FromStringAndSize(<char*>self._out, <Py_ssize_t>self._len))
            self._len = 0
        return <object>used

    def getvalue(self):
        if self._is_fixed:
            raise SofaStateError("getvalue() is only valid for the in-memory model")
        return PyBytes_FromStringAndSize(<char*>self._out, <Py_ssize_t>self._len)

    # --- scalars ------------------------------------------------------------

    def write_unsigned(self, object field_id, object value):
        if not self._begin():
            return
        try:
            if value < 0 or value > UNSIGNED_MAX:
                raise SofaRangeError("unsigned value %d out of range" % value)
            self._header(field_id, _WT_UNSIGNED)
            self._emit_varint(<uint64_t>value)
        except SofaError as exc:
            self._fail(exc)

    def write_signed(self, object field_id, object value):
        if not self._begin():
            return
        try:
            if value < SIGNED_MIN or value > SIGNED_MAX:
                raise SofaRangeError("signed value %d out of range" % value)
            self._header(field_id, _WT_SIGNED)
            self._emit_varint(_zigzag_encode(<int64_t>value))
        except SofaError as exc:
            self._fail(exc)

    def write_bool(self, object field_id, object value):
        self.write_unsigned(field_id, 1 if value else 0)

    def write_float32(self, object field_id, double value):
        cdef unsigned char buf[4]
        _pack_f32(value, buf)
        self._write_fixlen_raw(field_id, buf, 4, _ST_FP32)

    def write_float64(self, object field_id, double value):
        cdef unsigned char buf[8]
        _pack_f64(value, buf)
        self._write_fixlen_raw(field_id, buf, 8, _ST_FP64)

    def write_string(self, object field_id, str text):
        cdef bytes data = text.encode("utf-8")
        self._write_fixlen_bytes(field_id, data, _ST_STRING)

    def write_bytes(self, object field_id, object data):
        cdef bytes b = bytes(data)
        self._write_fixlen_bytes(field_id, b, _ST_BLOB)

    cdef int _write_fixlen_raw(self, object field_id, const unsigned char* data,
                               size_t n, int subtype) except -1:
        if not self._begin():
            return 0
        try:
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
        cdef object v
        cdef list seq
        try:
            seq = list(values)
            self._array_header(field_id, _WT_ARRAY_UNSIGNED, len(seq))
            # Convert straight to uint64 in C; an out-of-range element makes the
            # cast raise OverflowError, which we surface as SofaRangeError below.
            # This spares the hot loop two Python rich-comparisons per element.
            try:
                for v in seq:
                    self._emit_varint(<uint64_t>v)
            except OverflowError:
                raise SofaRangeError("unsigned array value out of range")
        except SofaError as exc:
            self._fail(exc)

    def write_signed_array(self, object field_id, values):
        if not self._begin():
            return
        cdef object v
        cdef list seq
        try:
            seq = list(values)
            self._array_header(field_id, _WT_ARRAY_SIGNED, len(seq))
            try:
                for v in seq:
                    self._emit_varint(_zigzag_encode(<int64_t>v))
            except OverflowError:
                raise SofaRangeError("signed array value out of range")
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
            seq = [float(x) for x in values]
            count = len(seq)
            self._array_header(field_id, _WT_ARRAY_FIXLEN, count)
            # §4.8: the fixlen_word is ALWAYS emitted (even for an empty array),
            # then the packed payload (zero bytes when empty).
            self._emit_varint((<uint64_t>elem_size << 3) | <uint64_t>subtype)
            if count == 0:
                return 0
            # Pack directly into the output for the common grow-mode path.
            if not self._is_fixed:
                self._ensure(<size_t>(count * elem_size))
                region = self._out + self._len
                if elem_size == 4:
                    for i in range(count):
                        _pack_f32(<double>seq[i], region + i * 4)
                else:
                    for i in range(count):
                        _pack_f64(<double>seq[i], region + i * 8)
                self._len += <size_t>(count * elem_size)
            else:
                if elem_size == 4:
                    for i in range(count):
                        d = <double>seq[i]
                        self._put_f32(d)
                else:
                    for i in range(count):
                        d = <double>seq[i]
                        self._put_f64(d)
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

    def write_sequence_begin(self, object field_id):
        if not self._begin():
            return
        try:
            if self._depth >= _MAX_DEPTH:
                raise SofaRangeError("nesting exceeds MAX_DEPTH=%d" % _MAX_DEPTH)
            self._header(field_id, _WT_SEQUENCE_START)
            self._depth += 1
        except SofaError as exc:
            self._fail(exc)

    def write_sequence_end(self):
        if not self._begin():
            return
        try:
            if self._depth <= 0:
                raise SofaStateError("sequence_end without matching begin")
            self._emit_varint(<uint64_t>_WT_SEQUENCE_END)
            self._depth -= 1
        except SofaError as exc:
            self._fail(exc)


# --- float pack/unpack (always little-endian, endian-independent) ------------

cdef inline void _pack_f32(double value, unsigned char* out) noexcept nogil:
    cdef float f = <float>value
    cdef uint32_t bits
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
    cdef const unsigned char* _p
    cdef Py_ssize_t _n
    cdef Py_ssize_t _pos
    cdef int _depth
    cdef object _cur
    # pending unconsumed value
    cdef int _pk                    # pending kind
    cdef int _pend_wtype
    cdef int _pend_subtype
    cdef uint64_t _pend_count
    cdef uint64_t _pend_size

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
        self._pk = _PEND_NONE

    cdef inline void _rebind(self, bytes newbuf):
        self._buf = newbuf
        self._p = <const unsigned char*>PyBytes_AS_STRING(newbuf)
        self._n = PyBytes_GET_SIZE(newbuf)

    # --- byte sourcing ------------------------------------------------------

    cdef bint _need(self, Py_ssize_t n) except -1:
        # Ensure at least n bytes available at _pos, refilling from the reader.
        cdef bytes data
        cdef bytes buf
        if self._n - self._pos >= n:
            return True
        if self._pos:
            buf = self._buf[self._pos:]
        else:
            buf = self._buf
        self._pos = 0
        while PyBytes_GET_SIZE(buf) < n:
            data = self._read(self._chunk)
            if not data:
                self._rebind(buf)
                self._pos = 0
                return False
            buf = buf + data if PyBytes_GET_SIZE(buf) else data
        self._rebind(buf)
        self._pos = 0
        return True

    cdef uint64_t _varint(self) except? 0xDEAD:
        cdef Py_ssize_t pos = self._pos
        cdef const unsigned char* p = self._p
        cdef Py_ssize_t n = self._n
        cdef unsigned char b
        cdef uint64_t result
        cdef int shift
        cdef int room
        if pos >= n:
            if not self._need(1):
                raise SofaIncompleteError("truncated varint")
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
                    raise SofaIncompleteError("truncated varint")
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
        cdef Py_ssize_t pos = self._pos
        cdef bytes out
        cdef bytearray acc
        cdef bytes data
        if pos + n <= self._n:
            out = self._buf[pos:pos + n]
            self._pos = pos + n
            return out
        acc = bytearray(self._buf[pos:])
        self._rebind(b"")
        self._pos = 0
        cdef int want
        while <Py_ssize_t>len(acc) < n:
            want = self._chunk
            if want < n - <Py_ssize_t>len(acc):
                want = <int>(n - <Py_ssize_t>len(acc))
            data = self._read(want)
            if not data:
                raise SofaIncompleteError("truncated payload")
            acc += data
        if <Py_ssize_t>len(acc) > n:
            self._rebind(bytes(acc[n:]))
            self._pos = 0
        return bytes(acc[:n])

    cdef list _read_varints(self, Py_ssize_t count):
        # Build the result incrementally rather than pre-sizing to the wire count.
        # ``count`` is attacker-controlled and capped only at ARRAY_MAX (2^31), so
        # PyList_New(count) would try to allocate ~16 GB of NULL slots for a tiny
        # hostile message claiming count = 2^31 before a single element byte is
        # read (amplification DoS, issue #31). Appending grows the list only as
        # elements are actually decoded, so a truncated oversize claim runs the
        # reader dry and raises SofaIncompleteError promptly. (When a caller sets
        # max_array_count the count is already bounded in next(); this keeps the
        # unconfigured default safe too.)
        cdef list out = []
        cdef Py_ssize_t i = 0
        cdef Py_ssize_t pos = self._pos
        cdef const unsigned char* p = self._p
        cdef Py_ssize_t n = self._n
        cdef unsigned char b
        cdef uint64_t result
        cdef int shift
        cdef int room
        while i < count:
            if pos >= n:
                self._pos = pos
                if not self._need(1):
                    raise SofaIncompleteError("truncated varint")
                p = self._p
                pos = self._pos
                n = self._n
            b = p[pos]
            pos += 1
            if b < 0x80:
                out.append(PyLong_FromUnsignedLongLong(b))
                i += 1
                continue
            result = b & 0x7F
            shift = 7
            while True:
                if pos >= n:
                    self._pos = pos
                    if not self._need(1):
                        raise SofaIncompleteError("truncated varint")
                    p = self._p
                    pos = self._pos
                    n = self._n
                b = p[pos]
                pos += 1
                # Reject an overlong (>64-bit) varint before OR-ing (see
                # ``_varint`` above; §4.1/§6.3, issue #43).
                room = 64 - shift
                if room < 7 and (b & 0x7F) >> room:
                    raise SofaDecodeError("overlong varint")
                result |= (<uint64_t>(b & 0x7F)) << shift
                if b < 0x80:
                    break
                shift += 7
                if shift >= 64:
                    raise SofaDecodeError("overlong varint")
            out.append(PyLong_FromUnsignedLongLong(result))
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
        cdef uint64_t header
        cdef int wtype
        cdef object field_id
        cdef uint64_t fid
        cdef uint64_t length_header, length, count, elem_header, elem_size
        cdef int subtype

        if self._pk != _PEND_NONE:
            self._skip_pending()

        if not self._need(1):
            if self._depth != 0:
                raise SofaIncompleteError("truncated: unbalanced sequence")
            return None

        header = self._varint()
        wtype = <int>(header & 0x07)
        fid = header >> 3

        if wtype == _WT_SEQUENCE_END:
            if self._depth <= 0:
                raise SofaDecodeError("unbalanced sequence end")
            self._depth -= 1
            self._cur = _mkfield(_ZERO, _WT[_WT_SEQUENCE_END], _ZERO, _ZERO, _NONE)
            return self._cur

        if fid > _ID_MAX:
            raise SofaDecodeError("id %d out of range" % PyLong_FromUnsignedLongLong(fid))
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
            # Receiver-configured limits (policy, not malformation): reject an
            # oversize string/blob here — before its payload is read or buffered.
            if subtype == _ST_STRING and self._max_string_len is not None \
                    and PyLong_FromUnsignedLongLong(length) > self._max_string_len:
                raise SofaLimitError("string length %d exceeds max_string_len %s"
                                     % (PyLong_FromUnsignedLongLong(length), self._max_string_len))
            if subtype == _ST_BLOB and self._max_blob_len is not None \
                    and PyLong_FromUnsignedLongLong(length) > self._max_blob_len:
                raise SofaLimitError("blob length %d exceeds max_blob_len %s"
                                     % (PyLong_FromUnsignedLongLong(length), self._max_blob_len))
            self._cur = _mkfield(field_id, _WT[_WT_FIXLEN],
                                 PyLong_FromUnsignedLongLong(length), _ZERO, _ST[subtype])
            self._pk = _PEND_FIXLEN
            self._pend_subtype = subtype
            self._pend_size = length
            return self._cur

        if wtype == _WT_ARRAY_UNSIGNED or wtype == _WT_ARRAY_SIGNED:
            count = self._varint()
            if count > _ARRAY_MAX:
                raise SofaDecodeError("array count %d out of range" % PyLong_FromUnsignedLongLong(count))
            if self._max_array_count is not None \
                    and PyLong_FromUnsignedLongLong(count) > self._max_array_count:
                raise SofaLimitError("array count %d exceeds max_array_count %s"
                                     % (PyLong_FromUnsignedLongLong(count), self._max_array_count))
            self._cur = _mkfield(field_id, _WT[wtype], _ZERO,
                                 PyLong_FromUnsignedLongLong(count), _NONE)
            self._pk = _PEND_VARRAY
            self._pend_wtype = wtype
            self._pend_count = count
            return self._cur

        # wtype == _WT_ARRAY_FIXLEN
        count = self._varint()
        if count > _ARRAY_MAX:
            raise SofaDecodeError("array count %d out of range" % PyLong_FromUnsignedLongLong(count))
        if self._max_array_count is not None \
                and PyLong_FromUnsignedLongLong(count) > self._max_array_count:
            raise SofaLimitError("array count %d exceeds max_array_count %s"
                                 % (PyLong_FromUnsignedLongLong(count), self._max_array_count))
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
        return self._cur

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
            raise SofaIncompleteError("truncated payload")
        if total > _SSIZE_MAX:
            raise SofaIncompleteError("truncated payload")
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
        self._pk = _PEND_NONE
        if kind == _PEND_SCALAR:
            self._varint()
        elif kind == _PEND_FIXLEN:
            self._read_exact(<Py_ssize_t>self._pend_size)
        elif kind == _PEND_VARRAY:
            self._skip_varints(<Py_ssize_t>self._pend_count)
        else:  # _PEND_FARRAY
            self._read_exact(self._farray_nbytes(self._pend_count, self._pend_size))
        return 0

    def skip(self):
        cdef int target
        if self._cur is not None and self._cur.type == _WT[_WT_SEQUENCE_START]:
            target = self._depth - 1
            while self._depth > target:
                if self.next() is None:
                    raise SofaIncompleteError("truncated sequence")
            return
        if self._pk != _PEND_NONE:
            self._skip_pending()

    # --- scalar reads -------------------------------------------------------

    cdef uint64_t _take_scalar(self, int wtype) except? 0xDEAD:
        if self._pk != _PEND_SCALAR or self._pend_wtype != wtype:
            raise SofaStateError("no matching scalar value for the current field")
        self._pk = _PEND_NONE
        return self._varint()

    def unsigned(self):
        return PyLong_FromUnsignedLongLong(self._take_scalar(_WT_UNSIGNED))

    def signed(self):
        return PyLong_FromLongLong(_zigzag_decode(self._take_scalar(_WT_SIGNED)))

    def bool(self):
        return self._take_scalar(_WT_UNSIGNED) != 0

    cdef bytes _take_fixlen(self, int subtype):
        if self._pk != _PEND_FIXLEN:
            raise SofaStateError("current field is not a fixlen value")
        if self._pend_subtype != subtype:
            raise SofaStateError("fixlen subtype does not match the requested read")
        self._pk = _PEND_NONE
        return self._read_exact(<Py_ssize_t>self._pend_size)

    def float32(self):
        cdef bytes data = self._take_fixlen(_ST_FP32)
        if PyBytes_GET_SIZE(data) != 4:
            raise SofaDecodeError("fp32 payload must be 4 bytes")
        return _unpack_f32(<const unsigned char*>PyBytes_AS_STRING(data))

    def float64(self):
        cdef bytes data = self._take_fixlen(_ST_FP64)
        if PyBytes_GET_SIZE(data) != 8:
            raise SofaDecodeError("fp64 payload must be 8 bytes")
        return _unpack_f64(<const unsigned char*>PyBytes_AS_STRING(data))

    def string(self):
        cdef bytes raw = self._take_fixlen(_ST_STRING)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SofaDecodeError("invalid UTF-8 in string field") from exc

    def bytes(self):
        return self._take_fixlen(_ST_BLOB)

    # --- array reads --------------------------------------------------------

    cdef uint64_t _take_varray(self, int wtype) except? 0xDEAD:
        if self._pk != _PEND_VARRAY or self._pend_wtype != wtype:
            raise SofaStateError("current field is not a matching varint array")
        self._pk = _PEND_NONE
        return self._pend_count

    def read_unsigned_array(self):
        cdef uint64_t count = self._take_varray(_WT_ARRAY_UNSIGNED)
        return self._read_varints(<Py_ssize_t>count)

    def read_signed_array(self):
        cdef uint64_t count = self._take_varray(_WT_ARRAY_SIGNED)
        cdef list raw = self._read_varints(<Py_ssize_t>count)
        cdef Py_ssize_t i
        cdef list out = PyList_New(<Py_ssize_t>count)
        cdef object item
        for i in range(<Py_ssize_t>count):
            item = PyLong_FromLongLong(_zigzag_decode(<uint64_t><object>raw[i]))
            Py_INCREF(item)
            PyList_SET_ITEM(out, i, item)
        return out

    cdef _take_farray(self, int subtype):
        if self._pk != _PEND_FARRAY:
            raise SofaStateError("current field is not a fixlen array")
        if self._pend_subtype != subtype:
            raise SofaStateError("fixlen-array subtype does not match the requested read")
        self._pk = _PEND_NONE
        return self._pend_count, self._pend_size

    def read_float32_array(self):
        cdef uint64_t count, elem_size
        count, elem_size = self._take_farray(_ST_FP32)
        # Consume the payload the fixlen_word claims (count * elem_size bytes),
        # then require it to be exactly count*4 — i.e. the element width must be
        # 4 for an fp32 array. Without this an elem_size != 4 (e.g. 0) leaves the
        # buffer shorter than the count*4 bytes the fixed-width unpack loop reads,
        # a heap over-read (SIGSEGV under boundscheck=False). The pure path is
        # implicitly guarded by struct.unpack demanding an exact-size buffer.
        cdef bytes data = self._read_farray_payload(count, elem_size, 4)
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
        cdef uint64_t count, elem_size
        count, elem_size = self._take_farray(_ST_FP64)
        # See read_float32_array: the element width must be 8 for an fp64 array,
        # or the count*8-byte unpack loop over-reads a shorter buffer.
        cdef bytes data = self._read_farray_payload(count, elem_size, 8)
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

    def drive(self, visitor):
        cdef object f
        cdef object t
        cdef object st
        while True:
            f = self.next()
            if f is None:
                break
            t = f.type
            if t == _WT[_WT_SEQUENCE_END]:
                visitor.on_sequence_end()
            elif t == _WT[_WT_SEQUENCE_START]:
                if visitor.on_sequence_begin(f.id) is False:
                    self.skip()
            elif visitor.on_field(f) is False:
                self.skip()
            elif t == _WT[_WT_UNSIGNED]:
                visitor.on_unsigned(f.id, self.unsigned())
            elif t == _WT[_WT_SIGNED]:
                visitor.on_signed(f.id, self.signed())
            elif t == _WT[_WT_FIXLEN]:
                st = f.subtype
                if st == _ST[_ST_FP32]:
                    visitor.on_float32(f.id, self.float32())
                elif st == _ST[_ST_FP64]:
                    visitor.on_float64(f.id, self.float64())
                elif st == _ST[_ST_STRING]:
                    visitor.on_string(f.id, self.string())
                else:
                    visitor.on_bytes(f.id, self.bytes())
            elif t == _WT[_WT_ARRAY_UNSIGNED]:
                visitor.on_unsigned_array(f.id, self.read_unsigned_array())
            elif t == _WT[_WT_ARRAY_SIGNED]:
                visitor.on_signed_array(f.id, self.read_signed_array())
            else:  # ARRAY_FIXLEN
                if f.subtype == _ST[_ST_FP32]:
                    visitor.on_float32_array(f.id, self.read_float32_array())
                else:
                    visitor.on_float64_array(f.id, self.read_float64_array())


# Marker so callers / tests can assert which implementation is active.
IMPL = "native"
