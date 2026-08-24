"""``sofab.__version__`` must be the version the distribution actually ships.

``__version__`` is public (it is in ``sofab.__all__``), so callers log it and
compatibility-check against it. CORELIB_PLAN §9 requires every version number to
match the code as it stands today, which means there must be exactly *one*
source of truth for it: the literal in ``sofab/__init__.py``, from which
``pyproject.toml`` derives the distribution version. These tests fail if a
second, hand-maintained version literal reappears in ``pyproject.toml`` and
drifts, or if an install ends up carrying a different version than the import.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import sofab

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project_table() -> dict:
    """The ``[project]`` table of ``pyproject.toml``.

    ``tomllib`` is stdlib only from 3.11 and this package supports 3.9+, so on
    older interpreters the (very small, fully controlled) file is parsed with a
    line scanner instead of pulling in a third-party TOML reader.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.9 / 3.10 only
        table: dict = {}
        in_project = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_project = stripped == "[project]"
                continue
            if not in_project or "=" not in stripped or stripped.startswith("#"):
                continue
            key, _, value = stripped.partition("=")
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                table[key.strip()] = value[1:-1]
            elif value.startswith("["):
                table[key.strip()] = re.findall(r'"([^"]*)"', value)
        return table
    return dict(tomllib.loads(text)["project"])


def test_version_is_pep440_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", sofab.__version__), sofab.__version__


def test_pyproject_does_not_carry_a_second_version_literal() -> None:
    """The distribution version is derived, not typed out a second time.

    A static ``version = "…"`` in ``pyproject.toml`` is exactly how
    ``__version__`` went stale: it was bumped for each release while the literal
    in ``__init__.py`` was not. If a static version is ever reintroduced it has
    to agree with the module.
    """
    project = _project_table()
    static = project.get("version")
    if static is None:
        assert "version" in project.get("dynamic", []), (
            "pyproject.toml declares neither a static nor a dynamic version"
        )
    else:
        assert static == sofab.__version__, (
            f"pyproject.toml says {static!r} but sofab.__version__ is {sofab.__version__!r}"
        )


def test_installed_distribution_matches_the_module() -> None:
    """An installed ``sofa-buffers-corelib`` reports the imported version.

    Skipped when running straight from the source tree (``PYTHONPATH=src``,
    nothing installed); CI installs the package, so this runs there.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    try:
        installed = dist_version("sofa-buffers-corelib")
    except PackageNotFoundError:
        pytest.skip("sofa-buffers-corelib is not installed; running from the source tree")
    assert installed == sofab.__version__, (
        f"installed distribution reports {installed!r} but sofab.__version__ is "
        f"{sofab.__version__!r} — reinstall (`pip install -e .`) if the version was "
        f"just bumped and the metadata on sys.path is a stale build artifact"
    )


def test_version_is_exported() -> None:
    assert "__version__" in sofab.__all__


def test_the_old_error_name_is_still_importable():
    """``SofaRangeError`` was renamed to :class:`sofab.SofaArgumentError`, after
    the ``InvalidArgument`` code it carries (CORELIB_PLAN §6.3). The old name is
    the same class, not a subclass, so ``except`` and ``isinstance`` on either
    name catch what the other raises."""
    import sofab

    assert sofab.SofaRangeError is sofab.SofaArgumentError
    assert "SofaRangeError" in sofab.__all__
    assert "SofaArgumentError" in sofab.__all__
