"""Every filesystem path the tutorial notebook names must exist (#463).

`tutorial/interactive/generation_and_calibration.ipynb` had no test at all -- the harness in
`test_interactive_tutorial_notebooks.py` only knows `generation_and_calibration_test.ipynb` and
`image_to_hemodynamics_model.ipynb`. So when `f101a1b` moved `src/solver1d` and `src/coupler`
into the `libcuflynx` namespace, Section C kept pointing at the old layout and nothing noticed
for a release.

This is a static check, not an execution: Section C needs a C++/PETSc toolchain and cannot run
in CI. It costs nothing and is exactly what would have caught the breakage.
"""
import ast
import json
import os

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_NOTEBOOKS = [
    os.path.join(_ROOT, 'tutorial', 'interactive', 'generation_and_calibration.ipynb'),
]

#: Spellings that moved in the libcuflynx namespace migration. A notebook naming one of these is
#: pointing at a layout that has not existed since 0.4.0.
_STALE_PREFIXES = ('src/coupler', 'src/solver1d', 'src/param_id', 'src/generators',
                   'src/parsers', 'src/utilities', 'src/solver_wrappers', 'src/scripts')


def _code(notebook_path):
    with open(notebook_path, encoding='utf-8') as fh:
        doc = json.load(fh)
    for idx, cell in enumerate(doc.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        lines = [ln for ln in (cell.get('source') or [])
                 if not ln.lstrip().startswith(('%', '!'))]
        yield idx, ''.join(lines)


def _joined_repo_paths(tree):
    """Repo-relative paths built as ``os.path.join(CA_root, ...)`` or ``CA_root / "..."``."""
    found = []

    def parts_of(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        return None

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'join' and node.args
                and isinstance(node.args[0], ast.Name) and node.args[0].id == 'CA_root'):
            rest = [parts_of(a) for a in node.args[1:]]
            if all(r is not None for r in rest):
                found.append(os.path.join(*[p for r in rest for p in r]))
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Name) and node.left.id == 'CA_root'):
            rest = parts_of(node.right)
            if rest:
                found.append(rest[0])
    return found


@pytest.mark.unit
@pytest.mark.parametrize('notebook', _NOTEBOOKS, ids=os.path.basename)
def test_the_notebook_names_no_pre_libcuflynx_source_path(notebook):
    offenders = []
    for idx, source in _code(notebook):
        for stale in _STALE_PREFIXES:
            if stale in source:
                offenders.append(f'cell {idx}: {stale!r}')
    assert not offenders, (
        f'{os.path.relpath(notebook, _ROOT)} points at the pre-libcuflynx layout '
        f'({"; ".join(offenders)}). src/<pkg> moved to src/libcuflynx/<pkg> in 0.4.0.')


@pytest.mark.unit
@pytest.mark.parametrize('notebook', _NOTEBOOKS, ids=os.path.basename)
def test_every_repo_path_the_notebook_builds_exists(notebook):
    missing = []
    for idx, source in _code(notebook):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue          # a cell that is not valid standalone python is not this test's job
        for rel in _joined_repo_paths(tree):
            if not os.path.exists(os.path.join(_ROOT, rel)):
                missing.append(f'cell {idx}: {rel!r}')
    assert not missing, (
        f'{os.path.relpath(notebook, _ROOT)} builds paths under the repo root that do not '
        f'exist: {"; ".join(missing)}')


@pytest.mark.unit
@pytest.mark.parametrize('notebook', _NOTEBOOKS, ids=os.path.basename)
def test_the_notebook_does_not_hardcode_an_interpreter_or_a_home_directory(notebook):
    """`/opt/OpenCOR/python` and `/home/<someone>/...` are one machine's layout, and the
    OpenCOR interpreter is deprecated besides."""
    offenders = []
    for idx, source in _code(notebook):
        for bad in ('/opt/OpenCOR', '/home/', '/hpc/'):
            if bad in source:
                offenders.append(f'cell {idx}: {bad!r}')
    assert not offenders, (
        f'{os.path.relpath(notebook, _ROOT)} hardcodes a machine-specific path '
        f'({"; ".join(offenders)}). Use sys.executable for the interpreter and derive data '
        f'directories from CA_root or the installed package.')


@pytest.mark.unit
@pytest.mark.parametrize('notebook', _NOTEBOOKS, ids=os.path.basename)
def test_the_notebook_writes_its_outputs_outside_resources(notebook):
    """Section C used to create a top-level `files_1d/` and write generated CSVs back into
    `resources/`, i.e. it edited the checkout it was run from."""
    offenders = []
    for idx, source in _code(notebook):
        if 'CA_root / "files_1d"' in source or "CA_root / 'files_1d'" in source:
            offenders.append(f'cell {idx}: creates a top-level files_1d/')
        if 'folder_hyb=None' in source:
            offenders.append(f'cell {idx}: convert_0d_to_1d writes back into resources/')
    assert not offenders, '; '.join(offenders)
