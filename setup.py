"""Optional native-acceleration build for the SofaBuffers Python runtime.

The package is a *pure-Python* library that ships an optional Cython extension
(``sofab._speedups``) implementing the encoder/decoder hot paths in compiled C.
The extension is strictly a speed layer: if Cython or a C compiler is missing,
or the compile fails on the target platform, the build falls back to a working
pure-Python install (``sofab.__init__`` selects the pure classes at import
time). This is what lets the library "run everywhere Python runs" while still
being fast where a compiler is available.

Metadata lives in ``pyproject.toml`` (PEP 621); this file only adds the
optional C extension and the tolerant build command.
"""

from __future__ import annotations

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

PYX = os.path.join("src", "sofab", "_speedups.pyx")
CSRC = os.path.join("src", "sofab", "_speedups.c")

# Allow an explicit opt-out (e.g. to build a pure-Python-only wheel/sdist).
_DISABLE = os.environ.get("SOFAB_DISABLE_NATIVE") == "1"


def _make_extension() -> list[Extension]:
    if _DISABLE:
        return []
    source = None
    try:
        from Cython.Build import cythonize  # noqa: F401

        source = PYX
    except ImportError:
        # No Cython at build time: fall back to a committed, pre-generated C
        # file if one is present (so sdists build without Cython).
        if os.path.exists(CSRC):
            source = CSRC
        else:
            return []

    ext = Extension(
        "sofab._speedups",
        sources=[source],
        optional=True,  # never fail the whole build if this ext won't compile
    )
    if source == PYX:
        from Cython.Build import cythonize

        return cythonize(
            [ext],
            compiler_directives={"language_level": "3"},
            annotate=False,
        )
    return [ext]


class BuildFailed(Exception):
    pass


class tolerant_build_ext(build_ext):
    """Turn a failed native build into a pure-Python install instead of an error."""

    def run(self) -> None:
        try:
            super().run()
        except (PlatformError, FileNotFoundError):
            self._warn()

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except (CCompilerError, ExecError, PlatformError, FileNotFoundError, OSError):
            self._warn()

    @staticmethod
    def _warn() -> None:
        sys.stderr.write(
            "sofabuffers: native extension failed to build; "
            "falling back to the pure-Python implementation.\n"
        )


setup(
    ext_modules=_make_extension(),
    cmdclass={"build_ext": tolerant_build_ext},
)
