# Publishing to PyPI (maintainers)

Maintainer runbook for the `sofa-buffers-corelib` distribution. End-user docs are
in [`README.md`](./README.md); this file is about getting a release onto PyPI.

## What a release consists of

- **One sdist.** The from-source path: pip compiles the accelerator from
  `_speedups.pyx` (Cython is declared in `build-system.requires`), and
  `setup.py` degrades to a working pure-Python install if there is no compiler.
- **42 wheels** — CPython 3.9–3.14 × {manylinux, musllinux} × {x86_64, aarch64},
  macOS {x86_64, arm64}, Windows AMD64 — each with the accelerator already
  compiled in, so a user needs no toolchain at all.

Selection, the per-wheel test command and the accelerator assertion live in
`[tool.cibuildwheel]` in `pyproject.toml`, so a local `cibuildwheel` run builds
exactly what CI builds. `cibuildwheel --print-build-identifiers` lists the set.

**Why every wheel is tested for `sofab.IMPL == "native"`:** `setup.py` is
deliberately tolerant — the extension is `optional=True` and a failed compile
degrades to pure Python. That is right for a from-source install and wrong for a
wheel: it would produce a *platform* wheel with no accelerator, which installs
like the real thing and is simply slow. The assertion turns that silent
degradation into a failed build.

## Releasing

The tag is compared **verbatim** against `sofab.__version__`, so the literal is
bumped first and the tag follows — the opposite of the generator repo, where the
tag is injected into a placeholder:

```sh
# 1. bump src/sofab/__init__.py: __version__ = "0.11.0"   (via PR, as usual)
# 2. then tag the merged commit
git tag v0.11.0
git push origin v0.11.0
```

`release.yml` gates on that equality before building anything, then builds the
sdist and the wheels and uploads via **trusted publishing (OIDC, no token)** with
automatic [PEP 740 attestations](https://docs.pypi.org/attestations/).

Pre-releases are spelled the **PEP 440** way, because the tag has to equal a
Python version literal: `v0.11.0rc1`, never `v0.11.0-rc1`.

## Trusted publishing prerequisites (pypi.org)

Auth is OIDC — **no API token**. The project needs a Trusted Publisher:

| Field | Value |
|---|---|
| Provider | GitHub Actions |
| Owner | `sofa-buffers` |
| Repository | `corelib-py` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

- The **workflow filename and the environment name are matched claims.** Renaming
  `release.yml`, or changing the job's `environment:`, breaks publishing with
  `invalid-publisher`.
- The repo needs a GitHub environment named `pypi` (Settings → Environments). It
  may stay unprotected, or carry a required reviewer as a gate before uploads.

### The first-ever release

PyPI OIDC **can** create a project, so there is no manual bootstrap upload. Before
the first tag, file a **pending publisher** with the fields above plus the project
name `sofa-buffers-corelib` — at the **organization** level
([`pypi.org/manage/organization/sofa-buffers/publishing/`](https://pypi.org/manage/organization/sofa-buffers/publishing/)),
not from a personal account: a pending publisher filed personally makes *that
account* the owner of the project on first upload. It converts into a normal
publisher by itself once the first upload succeeds.

## What the release workflow checks

Three layers, each answering a question the one before it cannot:

1. **The artifacts** — the sdist is installed into a clean venv and must report
   `IMPL == "native"` (a runner has a compiler) and pass the full suite from a
   directory with no `./src` to import by accident. Listing files proves
   `MANIFEST.in`; only an install proves the sdist *builds*. Each wheel is tested
   the same way by cibuildwheel, on the interpreter it was built for.
2. **The upload** — after publishing, `pip install sofa-buffers-corelib==<version>`
   from PyPI on Linux, macOS and Windows × Python 3.9 and 3.14 (the boundaries
   `requires-python` promises), asserting version and `IMPL == "native"`.
3. **The set** — a per-platform job cannot see whether the *other* platforms'
   wheels were uploaded. `verify-file-set` diffs what PyPI serves against exactly
   what the run built, and checks PEP 740 provenance on every file. It compares
   sets rather than counting to a hard-coded number, so adding a Python version to
   the matrix does not need a second edit.

Manually, after any release:

```sh
pip install sofa-buffers-corelib==<version>
python -c "import sofab; print(sofab.__version__, sofab.IMPL)"
```

`IMPL` must print `native` on any platform that has a wheel — that is the whole
point of shipping them.

## Re-running a failed publish

No re-tag needed; fix the cause and re-run the failed jobs:

```sh
gh run rerun <run-id> --repo sofa-buffers/corelib-py --failed
```

A PyPI version is immutable: if a *wrong* file was uploaded it cannot be
replaced, only yanked and superseded by a new version.
