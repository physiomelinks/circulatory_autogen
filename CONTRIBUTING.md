# Contributing

The repository is `circulatory_autogen`; the package it builds and publishes is
[`libcuflynx`](https://pypi.org/project/libcuflynx/).

- How to install, configure and run the project: the
  [tutorial](https://physiomelinks.github.io/circulatory_autogen/) (source under `tutorial/docs/`).
- Conventions, gotchas and the layout of the code: `CLAUDE.md`.
- Tests: `./run_pytest.sh` (see `CLAUDE.md` for the MPI/marker details). Every feature and
  bugfix needs a test; a bugfix should include one that fails before the fix.
- Branches: development happens on `devel`, pull requests target `master` of
  `physiomelinks/circulatory_autogen`.

## Making a release

Releases are cut from a **tag**, and everything after the tag is automated by
`.github/workflows/release.yml`. Nobody uploads by hand, and there is no PyPI API token stored
anywhere: the workflow authenticates to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), which mints a
single-use credential for one workflow run.

### Before tagging

1. **Bump `version` in `pyproject.toml`** to the version being released. The tag and this
   number must agree: tag `v0.4.0` ⟺ `version = "0.4.0"`. The workflow checks this before it
   builds and fails the release if they differ — because a mismatch cannot be repaired after
   the fact (see the re-upload rule below).
2. **Move the `CHANGELOG.md` entries** from *Unreleased* into a section for the version.
3. **Write the release notes.** They are what a user upgrading actually reads, so they must
   carry the items in the checklist below, not just a commit list.
4. **Build and check locally**:

    ```
    python -m build          # produces dist/*.whl and dist/*.tar.gz
    twine check dist/*
    ```

5. **Install the built wheel into a clean venv** — ideally on a machine with no MPI toolchain,
   which is the install path most first-time users take — and run a small model end to end
   (generation, a short calibration, a plot) from a directory that is *not* the checkout:

    ```
    python -m venv /tmp/v
    /tmp/v/bin/pip install dist/libcuflynx-*.whl
    cd /tmp && /tmp/v/bin/python -c "from libcuflynx.param_id.paramID import CVS0DParamID"
    /tmp/v/bin/cuflynx-generate --help
    ```

    Nothing may be written into `site-packages` during that run (`tests/test_no_writes_into_package.py`
    covers this in CI, but the clean-venv run is the honest check).

6. If you want a rehearsal of the upload itself, publish to
   [TestPyPI](https://test.pypi.org/) first and install from there. TestPyPI needs its own
   trusted publisher (or its own token) — it is a separate index with separate accounts.

### Tagging

```
git tag -a v0.4.0 -m "libcuflynx 0.4.0"
git push upstream v0.4.0
```

The tag must point at a commit on `master` of `physiomelinks/circulatory_autogen` — that is the
repository the PyPI trusted publisher is bound to. A tag pushed to a fork builds nothing that
can publish.

### What CI does with the tag

`.github/workflows/release.yml` runs on any `v*` tag:

- **`build`** — checks the tag against `pyproject.toml`, runs `python -m build` (wheel +
  sdist), runs `twine check`, and uploads the `dist/` directory as a workflow artifact.
- **`publish`** — downloads that artifact and publishes it with
  `pypa/gh-action-pypi-publish`, in the GitHub environment named **`pypi`**, with
  `permissions: id-token: write`.
- **`github-release`** — creates the entry on the
  [releases page](https://github.com/physiomelinks/circulatory_autogen/releases), taking its
  notes from the `CHANGELOG.md` section for the version. Gated on `publish`, so a release
  never announces a version PyPI rejected. Notes only: PyPI is the distribution channel, and
  a wheel attached here would be a second copy that can drift from it.

  This job was added late. **v0.4.1, v0.5.0 and v0.5.1 are on PyPI with no entry on the
  releases page** because the workflow used to stop at `publish`, and v0.4.0's and v0.6.0's
  entries were written by hand. If you want the history complete, they can be backfilled with
  `gh release create vX.Y.Z --notes-file <section of CHANGELOG.md>`.

A `workflow_dispatch` run builds and checks but does **not** publish, so the build can be
exercised without cutting a release.

**Do not rename the workflow file or the environment.** The PyPI trusted publisher for this
project is registered against the four-tuple *owner* `physiomelinks`, *repository*
`circulatory_autogen`, *workflow* `release.yml`, *environment* `pypi`. If any of those stops
matching, the build still succeeds and the upload is rejected.

**One-time setup, done by the project owner on their own PyPI account** (not by CI, and not
scriptable from here): claim the `libcuflynx` project name, add a *pending publisher* with
exactly those four values, and create the `pypi` environment in the GitHub repository settings
(optionally with required reviewers, which turns the publish step into an approval gate).

### A published version can never be replaced

PyPI accepts a given `name-version` file exactly once. Deleting a release does **not** free the
version — the filename stays burned, and re-uploading the same version, even a corrected build
of it, is rejected. If something is wrong after publishing, **bump to a new version** and
release again. This is why the tag/version check and the local build happen before the tag is
pushed, not after.

### Release-notes checklist

The notes for a release must state:

- [ ] **The deprecation shims and when they go.** The flat import names (`import parsers`,
      `from param_id.paramID import ...`) still work in 0.4.0 and emit a `DeprecationWarning`;
      they were **removed in 0.6.0**. Migrate by prefixing the import with `libcuflynx.`.
- [ ] **The `funcs_user` migration** (issue #433). The built-in cost / operation / modifier
      functions now live in the package (`libcuflynx.funcs.*`). Anyone who added their own by
      editing `funcs_user/cost_funcs_user.py`, `operation_funcs_user.py` or
      `modifier_funcs_user.py` in place will find them silently unregistered; the functions must
      move into a file named by `cost_funcs_external_path` / `operation_funcs_external_path` /
      `modifier_funcs_external_path` in `user_inputs.yaml`. Full text in `CHANGELOG.md`.
- [ ] **Any change to the dependency extras**, with the on-disk cost — the size table in
      `README.md` is the place readers check before installing.
- [ ] **The paired CUFLynx release**, if the GUI needs one. CUFLynx reads CA's discoverable
      schemas (`SOLVER_SCHEMA`, `PARAM_ID_METHODS`, `ANALYSIS_OPTIONS`, `gradient_sources()`,
      `cost_func_metadata()`), so a namespace or schema change has to land on both sides
      together — the 0.4.0 namespace move pairs with CUFLynx PR #263.
- [ ] **The minimum Python** and any extra that narrows it (`[emulation]`, and therefore
      `[all]`, needs Python >=3.10,<3.13).
