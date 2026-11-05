# Developer documentation for tutorial

## Local preview before a PR

For developers adapting the tutorial locally, check that the formatting renders correctly:

1. Install the documentation dependencies (Material for MkDocs + mkdocstrings).
   From the repo root:

    ```
    pip install -e ".[docs]"
    ```

    (The `docs` extra is declared in `pyproject.toml`. The API reference is built
    with the `mkdocstrings` plugin, so this is needed for the full site — plain
    `pip install mkdocs-material` only renders the hand-written pages.)

2. Navigate to the `/tutorial` directory and start the live preview server:

    ```
    mkdocs serve
    ```

    The site will be accessible at http://localhost:8000/.
    The server automatically rebuilds on save so you can preview changes as you write.

3. Before opening a PR, run a strict build to catch broken links and unresolved
   API references:

    ```
    mkdocs build --strict
    ```

## Documenting the API

The **API Reference** section is generated automatically from the docstrings in
`src/libcuflynx/` via `mkdocstrings` (`paths: [../src]` in `mkdocs.yml`, so every
identifier is spelled from the top of the package: `libcuflynx.<subpackage>.<module>`). There is a single source of truth — the docstrings —
so keep them up to date when you change a public function or class.

- Use **Google-style** docstrings (`Args:` / `Returns:` / `Raises:`), matching
  the existing style in `src/libcuflynx/utilities/obs_data_helpers.py`.
- The reference pages live under `tutorial/docs/api/` and are mostly
  `::: libcuflynx.module.path.Symbol` directives plus a short intro. The
  `libcuflynx.` prefix is required — an unprefixed identifier is not resolvable
  and `mkdocs build --strict` fails on it. To document a new
  public class/function, add a `:::` line on the appropriate page (and a nav
  entry in `mkdocs.yml` if you add a new page).
- `mkdocstrings` reads the source statically (it does not import it), so heavy
  optional dependencies like OpenCOR/libCellML do not need to be installed to
  build the docs.

## Expected outcome

You should be able to browse the tutorial locally and verify links, images, and formatting.

Note: The site is deployed automatically when changes to `tutorial/**` or `src/**`
are merged into the master branch (so docstring updates also republish the API
reference).
