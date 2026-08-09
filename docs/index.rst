SofaBuffers — Python API
========================

Streaming, dependency-free runtime for the SofaBuffers binary wire format,
byte-for-byte compatible with the C/Rust/Go/Java/C# core libraries. The hot path
ships as an optional compiled accelerator (``sofab._speedups``), loaded
automatically when present, with a pure-Python fallback used when it is not; the
active engine is reported by ``sofab.IMPL``.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   modules
