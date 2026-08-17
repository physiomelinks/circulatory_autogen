"""The CellML module library must be reachable as *package data*, not as a repo path.

A wheel that carries only ``.py`` files imports fine and then fails at the first
``generate_with_new_architecture()`` call, because the generator has no modules to read
(issue #432). Locating the library through ``importlib.resources`` is what makes it resolve
identically from a checkout, an editable install and an installed (or zipped) wheel -- so
these tests go through the same accessors the generator does, and read the bytes.
"""
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

from libcuflynx.utilities.package_resources import (
    builtin_module_file,
    builtin_modules_dir,
    builtin_modules_traversable,
    generator_template,
    package_data_dir,
    package_data_file,
)

# Present since the module library existed; the generator reads all three on every run.
_ALWAYS_SHIPPED = ('base_script.cellml', 'units.cellml', 'BG_modules.cellml')

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.unit
@pytest.mark.parametrize('filename', _ALWAYS_SHIPPED)
def test_builtin_module_file_is_readable(filename):
    """Resolve a built-in module file through importlib.resources and read it."""
    resource = builtin_module_file(filename)
    assert resource.is_file(), f'{filename} is missing from the shipped module library'
    text = resource.read_text(encoding='utf-8')
    assert '<model' in text, f'{filename} does not look like CellML'


@pytest.mark.unit
def test_builtin_module_config_parses():
    """The built-in module *config* ships too -- the vessel/BC lookup is built from it."""
    entries = json.loads(builtin_module_file('BG_modules_config.json').read_text(encoding='utf-8'))
    assert entries, 'BG_modules_config.json is empty'
    assert {'vessel_type', 'BC_type'} <= set(entries[0])


@pytest.mark.unit
def test_module_library_holds_both_halves():
    """Every ``*_modules.cellml`` the generator globs, and the configs beside them."""
    names = {entry.name for entry in builtin_modules_traversable().iterdir()}
    cellml = {n for n in names if n.endswith('modules.cellml')}
    configs = {n for n in names if n.endswith('config.json')}
    assert len(cellml) >= 15, sorted(cellml)
    assert len(configs) >= 15, sorted(configs)


@pytest.mark.unit
def test_builtin_modules_dir_is_a_real_directory():
    """The consumers that ``os.listdir`` the library need a real path, not a Traversable."""
    directory = builtin_modules_dir()
    assert os.path.isdir(directory), directory
    for filename in _ALWAYS_SHIPPED:
        path = os.path.join(directory, filename)
        assert os.path.isfile(path), path
        with open(path) as rf:
            assert rf.read(1)
    # Cached: repeated calls must not re-extract, or callers would hold stale paths.
    assert builtin_modules_dir() is directory


@pytest.mark.unit
def test_generator_uses_the_packaged_library():
    """The generator's own lookup must land in the package, not in a sibling checkout."""
    from libcuflynx.generators import CVSCellMLGenerator

    assert os.path.isdir(CVSCellMLGenerator.solver_make_files_dir)
    assert os.path.isfile(os.path.join(CVSCellMLGenerator.solver_make_files_dir, 'Makefile'))


@pytest.mark.unit
def test_other_shipped_data_is_readable():
    """The rest of the non-Python payload the wheel now carries."""
    assert generator_template('main0dTemplate.cpp').read_text(encoding='utf-8')
    assert generator_template('cppGeneratorTemplateFunctions.cpp').read_text(encoding='utf-8')
    example = package_data_file('libcuflynx.scripts', 'example_data',
                                'example_data_for_conversion.csv')
    assert example.is_file()
    assert example.read_text(encoding='utf-8').splitlines()[0]
    make_files = package_data_dir('libcuflynx.solver1d', 'Make_files')
    assert {'Makefile', 'runCVODE.bash'} <= set(os.listdir(make_files))


# ---------------------------------------------------------------------------
# The wheel itself
# ---------------------------------------------------------------------------
# Everything above resolves package data against *this checkout*, where the files exist
# whether or not `pyproject.toml` says to ship them. Delete the whole
# [tool.setuptools.package-data] block and every assertion above still passes, while
# `pip install libcuflynx` gets a wheel with zero .cellml in it -- which is issue #432
# exactly, unfixed and green. So one test has to look inside a built artefact.


def _hook(destination, hook, cwd):
    """Run one setuptools PEP 517 hook in a subprocess and return the artefact it wrote.

    Calls the hook directly rather than running ``python -m build``: build creates an
    isolated venv and downloads the build requirements over the network, and none of that
    changes what ends up inside the archive. This is the same hook build would invoke,
    using the interpreter already running the suite.
    """
    program = (
        "import sys\n"
        "from setuptools import build_meta\n"
        "sys.stdout.write(getattr(build_meta, sys.argv[1])(sys.argv[2]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, hook, str(destination)],
        cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=900,
    )
    assert result.returncode == 0, f'{hook} failed:\n{result.stderr}'
    artefact = pathlib.Path(destination) / result.stdout.strip().splitlines()[-1]
    assert artefact.is_file(), artefact
    return artefact


def _build_wheel(destination):
    """Build an sdist, unpack it, and build the wheel from *that*.

    The indirection is the point, and it is what ``python -m build`` does by default. A
    wheel built in the checkout reuses whatever ``build/lib`` a previous build left behind,
    so a package-data entry deleted from pyproject.toml would still appear in the archive
    and this test would pass on a stale directory. Building from a freshly unpacked sdist
    has no history to inherit -- and it checks the sdist carries the data files too, which
    is the other half of "what gets published".
    """
    sdist = _hook(destination, 'build_sdist', _REPO_ROOT)
    unpacked = pathlib.Path(destination) / 'unpacked'
    with tarfile.open(sdist) as archive:
        archive.extractall(str(unpacked))
    roots = [p for p in unpacked.iterdir() if p.is_dir()]
    assert len(roots) == 1, roots
    return _hook(destination, 'build_wheel', roots[0])


@pytest.mark.slow
def test_the_built_wheel_carries_the_data_files_and_not_the_dead_code(tmp_path):
    """The payload `pip install libcuflynx` actually receives."""
    with zipfile.ZipFile(_build_wheel(tmp_path)) as archive:
        names = set(archive.namelist())

    # The CellML module library: without it the generator has nothing to assemble from.
    assert 'libcuflynx/generators/resources/units.cellml' in names
    assert 'libcuflynx/generators/resources/BG_modules.cellml' in names
    assert 'libcuflynx/generators/resources/BG_modules_config.json' in names
    # The C++ templates CVSCppGenerator reads for model_type: cpp.
    assert 'libcuflynx/generators/main0dTemplate.cpp' in names
    # The build/run scripts copied next to each generated 1D model (#157).
    assert 'libcuflynx/solver1d/Make_files/Makefile' in names
    # Example input for example_format_obs_data_json_file(), which is itself shipped.
    assert any(n.startswith('libcuflynx/scripts/example_data/') and n.endswith('.csv')
               for n in names), sorted(n for n in names if 'example_data' in n)

    # ...and the dead code that packages.find excludes stays excluded.
    obsolete = sorted(n for n in names if n.startswith('libcuflynx/obsolete/'))
    assert obsolete == [], obsolete
