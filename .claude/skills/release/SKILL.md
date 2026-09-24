---
name: release
description: Cut a release of sofa-buffers-corelib (corelib-py) — bump the single version literal via PR, tag the merged commit, and watch release.yml publish to PyPI. Use when the user asks to release, cut a version, bump the version, tag a release, or publish to PyPI.
---

# Release corelib-py

Publishes `sofa-buffers-corelib` to PyPI. The long-form reasoning lives in
[`PUBLISHING.md`](../../../PUBLISHING.md); this is the procedure.

## The one thing that makes this repo different

**The tag is compared verbatim against the version literal** — the literal is
bumped *first*, the tag follows. (The generator repo is the opposite: there the
tag is injected into a placeholder. Do not carry that habit over.)

There is exactly **one** place a version lives:

- `src/sofab/__init__.py` → `__version__ = "X.Y.Z"`

`pyproject.toml` declares `dynamic = ["version"]` and reads it from there
(`[tool.setuptools.dynamic] version = { attr = "sofab.__version__" }`). Nothing
else in the tree carries a version — not the README, not `docs/conf.py`, and
there is no CHANGELOG. Confirm before bumping:

```sh
grep -rn "$(grep -oP '^__version__\s*=\s*"\K[^"]+' src/sofab/__init__.py)" \
  --include='*.py' --include='*.toml' --include='*.md' --include='*.yml' . \
  | grep -v '\.venv\|/build/\|uv\.lock\|\.claude/'
```

Only `src/sofab/__init__.py` should come back. If a second hit appears, someone
added a version literal — bump it too, or better, wire it to `__version__`.

## Tag spelling

**A release tag always begins with a lowercase `v`: `v1.2.3`.** Not `1.2.3`,
not `V1.2.3`. The prefix is load-bearing in both directions — `release.yml` and
`version-consistency.yml` trigger on `tags: ['v*']` (a case-sensitive glob) and
then strip exactly that one character with `${GITHUB_REF_NAME#v}` to get the
version. A tag spelled any other way is the worst kind of mistake here: nothing
fails, because nothing runs at all. No build, no publish, no error — just a tag
sitting in the repo and no release.

After the `v`, the version is spelled the **PEP 440** way, because the tag has
to equal a Python version literal verbatim:

| Want | Tag |
|---|---|
| Release | `v1.2.3` |
| Release candidate | `v1.2.3rc1` — **never** `v1.2.3-rc1` |
| Alpha / beta | `v1.2.3a1`, `v1.2.3b1` |

`release.yml` checks the part after the `v` against
`^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$` and refuses anything else.

Existing tags: `v0.9.0`, `v0.10.0`, `v0.10.1`, `v0.11.0`.

## Procedure

### 1. Preconditions

```sh
git checkout main && git pull -p
git status --short                      # must be clean
gh run list --branch main --limit 5     # CI green on the commit you will tag
```

Never tag a commit whose CI is red or still running. A PyPI version is
immutable: a bad upload can only be yanked, never replaced.

### 2. Bump the literal — via PR, not on main

```sh
git checkout -b release/vX.Y.Z origin/main
sed -i 's/^__version__ = ".*"/__version__ = "X.Y.Z"/' src/sofab/__init__.py
git commit -am "chore(release): X.Y.Z"
gh pr create --fill
```

Wait for green CI, then merge. **The repo allows rebase merges only** —
`allow_squash_merge` and `allow_merge_commit` are both off, so `--squash`
fails with `Squash merges are not allowed on this repository`:

```sh
gh pr merge <n> --rebase --delete-branch
```

### 2b. A rebase merge does not start CI on `main`

Observed on v0.11.0: after `--rebase`, `main` carried a new commit and GitHub
created **no** workflow run for it — `ci.yml` has no `paths` filter, the push
event simply did not fire. The PR's green run belongs to the pre-rebase SHA, so
the commit about to be tagged has no CI of its own. Check, and start one if it
is missing:

```sh
gh api "repos/sofa-buffers/corelib-py/actions/runs?head_sha=$(git rev-parse HEAD)" --jq .total_count
gh workflow run ci.yml   --ref main    # if that printed 0
gh workflow run docs.yml --ref main    # docs deploy on push to main, same gap
```

The rebased commit usually has the same tree *and* the same parent as the
tested PR head (`git rev-parse <pr-sha>^{tree} HEAD^{tree}` — compare them), so
this is a paperwork gap rather than a risk. Run it anyway: the `coverage` job
publishes the badge only from `main`, and the skill's own rule is that the
commit being tagged is green.

### 3. Tag the merged commit

```sh
git checkout main && git pull -p
grep -n '^__version__' src/sofab/__init__.py   # must read X.Y.Z
git tag -a vX.Y.Z -m "vX.Y.Z"      # lowercase v, PEP 440 after it
git push origin vX.Y.Z
```

Annotated (`-a`) matches the most recent tags, `v0.10.1` and `v0.11.0`.

### 4. Watch the publish

The tag fires two workflows:

- **`version-consistency.yml`** — tag vs. the literal vs. what an editable
  install reports through `importlib.metadata`.
- **`release.yml`** — the real one: `check-version` → `sdist` + `wheels`
  (5 runners, CPython 3.9–3.14 × platforms) → `publish` (OIDC trusted
  publishing, no token) → `verify-published` + `verify-file-set`.

```sh
gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

Wheels take ~15–25 min per runner; the whole run is roughly 30–40 min.

### 5. Confirm by hand

Use a throwaway venv, so the project's `.venv` is not repointed at a released
build:

```sh
python3 -m venv /tmp/v-check
/tmp/v-check/bin/python -m pip install --no-cache-dir "sofa-buffers-corelib==X.Y.Z"
/tmp/v-check/bin/python -c "import sofab; print(sofab.__version__, sofab.IMPL)"
```

`IMPL` must print `native` on any platform that has a wheel — that is the point
of shipping them. `python` means the wheel is a dud.

### 6. GitHub Release (optional)

`release.yml` does **not** create one. Only `v0.10.0` has a GitHub Release, so
this is not established practice — ask before doing it.

```sh
gh release create vX.Y.Z --generate-notes
```

## When it fails

- **After a failed publish, do not re-tag.** Fix the cause and re-run:
  `gh run rerun <run-id> --repo sofa-buffers/corelib-py --failed`
- **`invalid-publisher` on upload** — the OIDC identity is the matched triple
  repo + workflow *filename* `release.yml` + environment `pypi`. Renaming or
  moving `release.yml`, or changing the job's `environment:`, breaks publishing.
  Never rename that file as part of a release.
- **The tag is pushed and no workflow starts** — almost always the `v` prefix:
  `tags: ['v*']` matched nothing. Check `gh run list --limit 5`; if it shows no
  run for the tag, delete it and re-tag with the right spelling:
  `git tag -d <tag> && git push origin :refs/tags/<tag>`
- **`sofab.__version__ != tag`** — the literal was not merged before tagging.
  Delete the tag locally and remotely, merge the bump, tag again.
- **sdist or wheel builds without the accelerator** — `setup.py` is deliberately
  tolerant (`optional=True`) and degrades to pure Python. The build asserts
  `IMPL == "native"` to turn that silent degradation into a failure; treat it as
  a real build break, not a flake.
- **`verify-published` cannot install yet** — the job already retries 20× at 15 s
  for CDN propagation. A failure after that is real.
- **`tests/test_version.py` fails locally right after the bump** — only the
  editable install's *metadata* is stale, still built at the old version. The
  assertion says so. `pip install -e .` and re-run; CI installs fresh and never
  sees this.

## Not part of a release

- **Docs** deploy from `docs.yml` on every push to `main`, not on a tag.
- **`assets/test_vectors.json`** is synced by `shared-vectors.yml` (daily and on
  PRs that touch it) — refresh it in its own PR, never inside a release PR.
