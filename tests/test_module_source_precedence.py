"""A (vessel_type, BC_type) that several module sources define comes from the most specific one
(utilities/module_library.py): external_modules_dir, then module_library_dirs in order, then
module_config_user, then the built-in modules. The others' config entries and CellML are left
out, a ModuleShadowWarning lists the competition, and a type defined twice within one source is
an error.

The copies here are of the built-in Lotka_Volterra module, with a marker comment inside the
component so the generated model shows whose CellML it got.
"""
import json
import os
import shutil
import warnings

import pytest

from libcuflynx.utilities.module_library import ModuleShadowWarning, ModuleSources
from libcuflynx.utilities.package_resources import builtin_modules_dir

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')
KEY = ('Lotka_Volterra', 'nn')
BUILTIN_CONFIG = os.path.join(builtin_modules_dir(), 'Lotka_Volterra_module_config.json')
BUILTIN_CELLML = os.path.join(builtin_modules_dir(), 'Lotka_Volterra_modules.cellml')


def _copy(directory, marker, config_name='Lotka_Volterra_modules_config.json'):
    '''A copy of the built-in Lotka_Volterra module in ``directory``, marked ``marker``.'''
    os.makedirs(directory, exist_ok=True)
    with open(BUILTIN_CELLML) as f:
        cellml = f.read()
    opening = '<component name="Lotka_Volterra">'
    assert opening in cellml
    with open(os.path.join(directory, 'Lotka_Volterra_modules.cellml'), 'w') as f:
        f.write(cellml.replace(opening, f'{opening}\n        <!-- {marker} -->', 1))
    shutil.copy(BUILTIN_CONFIG, os.path.join(directory, config_name))
    return os.path.join(directory, config_name)


def _sources(**inputs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sources = ModuleSources(inputs)
    shadows = [w.message for w in caught if isinstance(w.message, ModuleShadowWarning)]
    return sources, shadows


@pytest.mark.unit
def test_a_library_s_copy_of_a_built_in_module_wins_with_a_warning(tmp_path):
    library = str(tmp_path / 'library')
    config = _copy(os.path.join(library, 'lv'), 'library')
    sources, shadows = _sources(module_library_dirs=[library])
    (shadow,) = shadows
    assert shadow.shadowed[KEY] == (config, [BUILTIN_CONFIG])
    assert 'Lotka_Volterra/nn' in str(shadow) and 'module_library_dirs[0]' in str(shadow)
    assert sources.component_registry()[KEY]['config_path'] == config
    assert (BUILTIN_CONFIG, KEY) in sources.excluded_entries
    # the built-in component is left out of the model; the library's is used
    assert not sources.uses_component(BUILTIN_CELLML, 'Lotka_Volterra')
    assert sources.uses_component(os.path.join(library, 'lv', 'Lotka_Volterra_modules.cellml'),
                                  'Lotka_Volterra')


@pytest.mark.unit
def test_without_competition_nothing_is_left_out_or_warned_about(tmp_path):
    sources, shadows = _sources()
    assert not shadows and not sources.excluded_entries and not sources.excluded_components
    library = str(tmp_path / 'library')
    _copy(os.path.join(library, 'lv'), 'library')
    sources, shadows = _sources(module_library_dirs=[library], use_builtin_modules=False)
    assert not shadows and not sources.excluded_entries


@pytest.mark.unit
def test_the_first_listed_library_wins_and_external_modules_beat_libraries(tmp_path):
    first, second = str(tmp_path / 'first'), str(tmp_path / 'second')
    first_config = _copy(os.path.join(first, 'lv'), 'first')
    second_config = _copy(os.path.join(second, 'lv'), 'second')
    sources, (shadow,) = _sources(module_library_dirs=[first, second], use_builtin_modules=False)
    assert shadow.shadowed[KEY] == (first_config, [second_config])

    external = str(tmp_path / 'external')
    external_config = _copy(external, 'external')
    sources, (shadow,) = _sources(module_library_dirs=[first, second],
                                  external_modules_dir=external)
    winner, losers = shadow.shadowed[KEY]
    assert winner == external_config
    assert set(losers) == {first_config, second_config, BUILTIN_CONFIG}


@pytest.mark.unit
def test_a_type_defined_twice_within_one_source_is_an_error(tmp_path):
    library = str(tmp_path / 'library')
    _copy(os.path.join(library, 'a'), 'a')
    _copy(os.path.join(library, 'b'), 'b')
    with pytest.raises(ValueError, match=r"defined 2 times in module_library_dirs\[0\]"):
        _sources(module_library_dirs=[library], use_builtin_modules=False)


@pytest.mark.unit
def test_a_component_another_kept_type_still_needs_is_not_left_out(tmp_path):
    # the library's one file defines Lotka_Volterra/nn and /vv on one component; external
    # modules win nn, but the library's vv still needs that component
    library = str(tmp_path / 'library')
    config = _copy(os.path.join(library, 'lv'), 'library')
    with open(config) as f:
        entry = json.load(f)[0]
    with open(config, 'w') as f:
        json.dump([entry, dict(entry, BC_type='vv')], f)
    library_cellml = os.path.join(library, 'lv', 'Lotka_Volterra_modules.cellml')
    external = str(tmp_path / 'external')
    _copy(external, 'external')
    sources, (shadow,) = _sources(module_library_dirs=[library], external_modules_dir=external,
                                  use_builtin_modules=False)
    assert list(shadow.shadowed) == [KEY]
    assert sources.component_registry()[('Lotka_Volterra', 'vv')]['config_path'] == config
    assert sources.uses_component(library_cellml, 'Lotka_Volterra')

    # once nothing in the library uses it any more, it is left out
    with open(config, 'w') as f:
        json.dump([entry], f)
    sources, _ = _sources(module_library_dirs=[library], external_modules_dir=external,
                          use_builtin_modules=False)
    assert not sources.uses_component(library_cellml, 'Lotka_Volterra')


@pytest.mark.integration
def test_generation_uses_the_winning_source_s_module_once(tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    prefix = 'Lotka_Volterra'
    resources = tmp_path / 'resources'
    os.makedirs(resources)
    for suffix in ('_module_array.csv', '_parameters.csv'):
        shutil.copy(os.path.join(RESOURCES_DIR, prefix + suffix), resources)
    library = str(tmp_path / 'library')
    _copy(os.path.join(library, 'lv'), 'from the library')

    def generate(name, **extra):
        out = tmp_path / name
        config = dict({'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
                       'model_type': 'cellml', 'solver': 'CVODE_myokit',
                       'resources_dir': str(resources), 'generated_models_dir': str(out),
                       'DEBUG': False}, **extra)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert generate_with_new_architecture(False, config), name
        with open(out / prefix / f'{prefix}_modules.cellml') as f:
            return f.read(), [w for w in caught if isinstance(w.message, ModuleShadowWarning)]

    builtin_only, shadows = generate('builtin')
    assert not shadows
    with_library, shadows = generate('library', module_library_dirs=[library])
    # one warning (load_model's; the generator does not repeat it), the library's component,
    # written once
    assert len(shadows) == 1
    assert 'from the library' in with_library
    assert with_library.count('<component name="Lotka_Volterra">') == 1
    assert with_library.replace('\n        <!-- from the library -->', '') == builtin_only
