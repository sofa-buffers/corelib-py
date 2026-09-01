"""Suite-wide pytest hooks.

One job: make a run **say what the shared conformance suite covered**. The
vectors in ``assets/test_vectors.json`` are copied verbatim from
``corelib-c-cpp`` (CORELIB_PLAN §7.1/§8) and each one is replayed through
several scenarios, so "the tests passed" on its own does not distinguish a full
run from one that quietly ran a smaller file or gated half the matrix out on
``requires``. The C reference runner prints its vector and check counts for the
same reason — it is how the 583 → 1033 growth of the skip matrix was visible at
all — and this is that line for the Python port.

Nothing here asserts; the assertions live in ``tests/test_conformance_vectors.py``
(``test_suite_metadata`` pins the expected sizes). This only reports.
"""

from __future__ import annotations

#: The module whose items are counted as conformance-vector checks.
_VECTOR_MODULE = "test_conformance_vectors.py"


class _Tally:
    """Per-run counts, filled by ``pytest_runtest_logreport``."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.gated = 0  # skipped by `requires` (a capability this port lacks)
        self.other_skips = 0  # skipped for any other stated reason
        self.by_scenario: dict[str, int] = {}
        self.vectors: set[str] = set()
        self.skip_vectors: set[str] = set()


_tally = _Tally()


_names: frozenset | None = None


def _vector_names() -> frozenset:
    global _names
    if _names is None:
        from vectors import VECTORS  # local: keeps conftest cheap for runs without it

        _names = frozenset(v["name"] for v in VECTORS)
    return _names


def _split_nodeid(nodeid: str) -> tuple[str, str | None]:
    """``(scenario, vector name)`` for one conformance item.

    Ids look like ``test_vector_decode[unsigned_0]`` and, where a second
    parameter is stacked on, ``test_vector_chunked_encode[unsigned_0-3]``; no
    vector name contains ``-``, so the trailing parameters peel off one at a
    time. An item that names no vector (``test_suite_metadata``) reports
    ``None``.
    """
    func, _, tail = nodeid.partition("::")[2].partition("[")
    scenario = func.removeprefix("test_vector_").removeprefix("test_")
    param = tail[:-1] if tail.endswith("]") else ""
    names = _vector_names()
    while param:
        if param in names:
            return scenario, param
        param = param.rpartition("-")[0]
    return scenario, None


def _skip_reason(report) -> str:
    # A skip's longrepr is (path, lineno, "Skipped: <reason>").
    longrepr = getattr(report, "longrepr", None)
    return longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else ""


def pytest_runtest_logreport(report) -> None:
    if _VECTOR_MODULE not in report.nodeid:
        return
    if report.when not in ("setup", "call"):
        return
    if report.when == "setup" and not report.skipped:
        return

    scenario, vector = _split_nodeid(report.nodeid)
    if report.skipped:
        # `requires` gating is reported apart from every other skip: a
        # feature-reduced build is expected to gate part of the matrix out, and
        # that is exactly the number a reader of the log needs to see.
        if "requires unsupported capabilities" in _skip_reason(report):
            _tally.gated += 1
        else:
            _tally.other_skips += 1
        return

    if report.passed:
        _tally.passed += 1
    elif report.failed:
        _tally.failed += 1
    _tally.by_scenario[scenario] = _tally.by_scenario.get(scenario, 0) + 1
    if vector is not None:
        _tally.vectors.add(vector)
        if "skip_ids" in scenario:
            _tally.skip_vectors.add(vector)


def pytest_terminal_summary(terminalreporter) -> None:
    executed = _tally.passed + _tally.failed
    if not executed and not (_tally.gated or _tally.other_skips):
        return  # this run did not collect the conformance suite

    from vectors import VECTOR_DOC, VECTORS, VECTORS_PATH

    groups: dict[str, int] = {}
    for vec in VECTORS:
        groups[vec.get("group", "-")] = groups.get(vec.get("group", "-"), 0) + 1
    with_skip_ids = sum(1 for v in VECTORS if v.get("skip_ids"))

    write = terminalreporter.write_line
    terminalreporter.write_sep("=", "shared conformance vectors")
    write(
        f"{VECTORS_PATH.name}: {len(VECTORS)} vectors "
        f"({groups.get('skip/matrix', 0)} skip/matrix, {groups.get('skip', 0)} skip), "
        f"{with_skip_ids} carry skip_ids; "
        f"{len(VECTOR_DOC.get('invalid_utf8', ()))} invalid_utf8, "
        f"{len(VECTOR_DOC.get('sequence_growth', ()))} sequence_growth cases"
    )
    write(
        f"vectors exercised: {len(_tally.vectors)}/{len(VECTORS)}"
        f" ({len(_tally.skip_vectors)} through the skip scenarios)"
    )
    write(
        f"checks executed: {executed} ({_tally.passed} passed, {_tally.failed} failed), "
        f"{_tally.gated} gated out by `requires`, {_tally.other_skips} skipped otherwise"
    )
    if _tally.by_scenario:
        per = ", ".join(f"{k} {v}" for k, v in sorted(_tally.by_scenario.items()))
        write(f"  by scenario: {per}")
