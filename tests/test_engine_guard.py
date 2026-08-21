"""The suite must know which engine it just exercised (CORELIB_PLAN §12.1).

The native accelerator is *optional by design*: ``setup.py`` marks the extension
``optional=True`` so a missing compiler or a broken compile installs the working
pure-Python build with a warning instead of failing the install. Every
native-gated test in this suite is therefore behind ``pytest.importorskip`` or an
``ENGINES`` parametrisation — a lost accelerator makes those tests *disappear*
rather than fail, so a compile break, a Cython incompatibility on a new Python,
or a ``setup.py`` mistake would leave a green run with the accelerator (and the
native↔pure parity tests that pin the two engines to the same bytes) silently
missing.

Nothing in the suite could tell the two situations apart. This module closes
that hole from both ends:

* ``SOFAB_REQUIRE_ENGINE=native|python`` states which engine a run is *supposed*
  to exercise; the run fails if ``sofab.IMPL`` says otherwise. CI sets it on
  every ``pytest`` invocation, so an unbuilt accelerator turns the leg red
  instead of quietly halving its coverage.
* the workflow itself is checked, so no future edit can drop the declaration,
  drop one of the two engines, or drop the no-compiler leg that proves the
  fallback the README promises.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import sofab

REQUIRE_ENV = "SOFAB_REQUIRE_ENGINE"
_ENGINES = ("native", "python")

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


# --- the run-time guard ------------------------------------------------------


def test_active_engine_is_the_required_one() -> None:
    """``sofab.IMPL`` must match ``SOFAB_REQUIRE_ENGINE`` when that is set."""
    required = os.environ.get(REQUIRE_ENV)
    if required is None:
        pytest.skip(f"{REQUIRE_ENV} not set: this run does not pin an engine")
    assert required in _ENGINES, f"{REQUIRE_ENV}={required!r}, expected one of {_ENGINES}"
    assert sofab.IMPL == required, (
        f"{REQUIRE_ENV}={required!r} but sofab.IMPL == {sofab.IMPL!r}. "
        "The native accelerator did not build (setup.py keeps the extension optional, "
        "so a failed compile falls back to pure Python) or SOFAB_PUREPYTHON is set — "
        "either way this run is not testing the engine it claims to."
    )
    if required == "native":
        # Not just `IMPL`: the module the native-gated suites import must load,
        # otherwise they would still skip themselves out of the run.
        assert importlib.util.find_spec("sofab._speedups") is not None
        from sofab import _speedups

        assert sofab.Encoder is _speedups.Encoder
        assert sofab.Decoder is _speedups.Decoder


def test_the_guard_fails_when_the_required_engine_is_missing() -> None:
    """The guard must actually bite — a passing no-op would restore the hole.

    Runs this file's guard in a subprocess that forces the pure-Python engine
    while claiming to require the native one, which is exactly the shape of a
    silently-unbuilt accelerator.
    """
    if importlib.util.find_spec("sofab._speedups") is None:
        pytest.skip("native extension not built: 'native' cannot be the required engine here")
    env = dict(os.environ, SOFAB_PUREPYTHON="1", **{REQUIRE_ENV: "native"})
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{__file__}::test_active_engine_is_the_required_one",
        ],
        env=env,
        capture_output=True,
        text=True,
        # The child is a pytest run, and it writes in *its* locale encoding —
        # cp1252 on Windows, where an em dash is 0x97. Pinning utf-8 here would
        # make the reader raise UnicodeDecodeError, leaving stdout None; pinning
        # nothing risks the same on a stricter locale. Decode with the platform
        # default, but never let a stray byte turn an assertion into a crash:
        # this test only looks for an ASCII test name in the output.
        errors="replace",
    )
    assert proc.returncode != 0, f"guard passed with the wrong engine active:\n{proc.stdout}"
    # ...and for the right reason.
    assert "test_active_engine_is_the_required_one" in proc.stdout
    assert "sofab.IMPL" in proc.stdout


# --- the workflow that has to set it ----------------------------------------


def _steps() -> list[str]:
    """The workflow's steps, one text block each (``- name:`` starts a step)."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    blocks = re.split(r"^\s*-\s+(?=name:|uses:)", text, flags=re.MULTILINE)
    return blocks[1:]


# A step *runs* the suite when a command word is ``pytest`` (or ``python -m
# pytest``) — as opposed to merely naming it as a package to install.
_RUNS_PYTEST = re.compile(
    r"^\s*run:\s*(?:.*(?:\|\||&&|;)\s*)?(?:python[\d.]*\s+-m\s+)?pytest\b",
    flags=re.MULTILINE,
)


def _pytest_steps() -> list[str]:
    return [s for s in _steps() if _RUNS_PYTEST.search(s)]


def test_workflow_is_present() -> None:
    assert _WORKFLOW.exists(), f"{_WORKFLOW} is missing: CI cannot be checked"


def test_every_workflow_pytest_run_pins_its_engine() -> None:
    """No CI leg may run the suite without declaring the engine it exercises."""
    steps = _pytest_steps()
    assert steps, "no step in ci.yml runs pytest"
    unpinned = [s.splitlines()[0].strip() for s in steps if REQUIRE_ENV not in s]
    assert not unpinned, (
        f"ci.yml runs pytest without {REQUIRE_ENV} in: {unpinned}. "
        "An unbuilt accelerator would skip the native tests and keep the leg green."
    )


def test_workflow_exercises_both_engines() -> None:
    """Neither engine may vanish from CI: one leg pins each."""
    required = {
        m.group(1)
        for s in _pytest_steps()
        for m in re.finditer(rf"{REQUIRE_ENV}:\s*['\"]?(\w+)", s)
    }
    assert set(_ENGINES) <= required, f"ci.yml pins only {sorted(required)}; want both {_ENGINES}"


def test_workflow_keeps_a_no_compiler_leg() -> None:
    """The README promises a pure-Python install; CI must build one and test it."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "SOFAB_DISABLE_NATIVE" in text, (
        "no ci.yml leg installs with SOFAB_DISABLE_NATIVE=1, so the compiler-less "
        "fallback install is never built or tested"
    )


def test_workflow_asserts_the_engine_right_after_install() -> None:
    """A dedicated assertion step names the failure before any test output.

    ``pip install -e .`` prints the fallback warning and exits 0; asserting
    ``sofab.IMPL`` immediately after the install makes the missing accelerator
    the first thing the log shows, rather than a subtle test failure later.
    """
    asserts = [s for s in _steps() if "sofab.IMPL" in s and re.search(r"^\s*run:", s, re.MULTILINE)]
    engines = {m.group(1) for s in asserts for m in re.finditer(r"IMPL\s*==\s*['\"](\w+)['\"]", s)}
    assert set(_ENGINES) <= engines, (
        f"ci.yml asserts sofab.IMPL for {sorted(engines)}; both {_ENGINES} legs must assert "
        "the engine their install produced"
    )
