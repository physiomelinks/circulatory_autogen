"""The CellML module library must be reachable as *package data*, not as a repo path.

A wheel that carries only ``.py`` files imports fine and then fails at the first
``generate_with_new_architecture()`` call, because the generator has no modules to read
(issue #432). Locating the library through ``importlib.resources`` is what makes it resolve
identically from a checkout, an editable install and an installed (or zipped) wheel -- so
these tests go through the same accessors the generator does, and read the bytes.
"""
import json
import os

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
