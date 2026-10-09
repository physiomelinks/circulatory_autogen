"""
Tests for ``module_library_dirs`` / ``use_builtin_modules``: generating models from an
external, per-module-directory library (e.g. circulatory-autogen-modules) instead of, or as
well as, the built-in modules.
"""
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.module_library import ModuleShadowWarning, ModuleSources, collect_units
from libcuflynx.utilities.package_resources import builtin_modules_dir

DECAY_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="{module_type}">
        <variable name="t" public_interface="in" units="second"/>
        <variable name="x_init" public_interface="in" units="dimensionless"/>
        <variable name="k" public_interface="in" units="lib_test_per_second"/>
        <variable initial_value="x_init" name="x" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
                <apply><minus/><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
            </apply>
        </math>
    </component>
</model>
"""

DECAY_CONFIG = """\
[
  {{
    "vessel_type": "{vessel_type}",
    "BC_type": "nn",
    "module_format": "cellml",
    "module_file": "{name}_modules.cellml",
    "module_type": "{module_type}",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["x_init", "dimensionless", "access", "constant"],
      ["k", "lib_test_per_second", "access", "constant"],
      ["x", "dimensionless", "access", "variable"]
    ]
  }}
]
"""

UNITS_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="Units" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <units name="lib_test_per_second">
        <unit exponent="{exponent}" units="second"/>
    </units>
</model>
"""


def _write_module(library_dir, name, vessel_type, module_type, units_exponent='-1'):
    module_dir = os.path.join(library_dir, name)
    os.makedirs(module_dir, exist_ok=True)
    with open(os.path.join(module_dir, f'{name}_modules.cellml'), 'w') as f:
        f.write(DECAY_CELLML.format(module_type=module_type))
    with open(os.path.join(module_dir, f'{name}_modules_config.json'), 'w') as f:
        f.write(DECAY_CONFIG.format(vessel_type=vessel_type, name=name, module_type=module_type))
    with open(os.path.join(module_dir, f'{name}_units.cellml'), 'w') as f:
        f.write(UNITS_CELLML.format(exponent=units_exponent))
    # Non-config JSON next to a module must not be read as a module config.
    with open(os.path.join(module_dir, f'{name}_obs_data.json'), 'w') as f:
        f.write('{"data_items": []}')
    return module_dir


def _write_resources(resources_dir, prefix, vessel_type, k=0.5, x_init=2.0):
    os.makedirs(resources_dir, exist_ok=True)
    with open(os.path.join(resources_dir, f'{prefix}_module_array.csv'), 'w') as f:
        f.write(textwrap.dedent(f"""\
            name,BC_type,vessel_type,inp_vessels,out_vessels
            decay,nn,{vessel_type},,
            """))
    with open(os.path.join(resources_dir, f'{prefix}_parameters.csv'), 'w') as f:
        f.write(textwrap.dedent(f"""\
            variable_name,units,value,data_reference
            x_init_decay,dimensionless,{x_init},test
            k_decay,lib_test_per_second,{k},test
            """))


def _config(tmp_path, prefix, library_dir, use_builtin_modules):
    config = {
        'file_prefix': prefix,
        'input_param_file': f'{prefix}_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': str(tmp_path / 'resources'),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'module_library_dirs': [library_dir],
        'DEBUG': False,
    }
    if use_builtin_modules is not None:
        config['use_builtin_modules'] = use_builtin_modules
    return config


@pytest.mark.integration
def test_module_library_dir_only(tmp_path):
    """With built-ins off, a model is generated from the library alone, and its units come
    from the module's own units file."""
    library_dir = str(tmp_path / 'library')
    _write_module(library_dir, 'decay', 'decay', 'decay_type')
    _write_resources(str(tmp_path / 'resources'), 'lib_decay', 'decay')

    config = _config(tmp_path, 'lib_decay', library_dir, use_builtin_modules=False)
    assert generate_with_new_architecture(False, config)

    generated_dir = tmp_path / 'generated_models' / 'lib_decay'
    modules_text = (generated_dir / 'lib_decay_modules.cellml').read_text()
    assert '<component name="decay_type"' in modules_text
    # No built-in component made it into the model.
    assert 'Lotka_Volterra' not in modules_text
    units_text = (generated_dir / 'lib_decay_units.cellml').read_text()
    assert 'lib_test_per_second' in units_text
    assert 'J_per_m3' not in units_text  # built-in units.cellml was not used

    # The library model runs and matches x = x_init * exp(-k t).
    helper = get_simulation_helper(model_path=str(generated_dir / 'lib_decay.cellml'),
                                   solver='CVODE_myokit', model_type='cellml',
                                   dt=0.01, sim_time=2.0, pre_time=0.0)
    helper.run()
    x = np.asarray(helper.get_results([['decay/x']], flatten=True)).ravel()
    t = np.asarray(helper.get_time())
    np.testing.assert_allclose(x, 2.0 * np.exp(-0.5 * t), rtol=1e-4)


@pytest.mark.integration
def test_module_library_redefines_builtin_when_builtins_off(tmp_path):
    """A library may redefine a built-in (vessel_type, BC_type) once built-ins are off."""
    library_dir = str(tmp_path / 'library')
    _write_module(library_dir, 'Lotka_Volterra', 'Lotka_Volterra', 'Lotka_Volterra_lib')
    _write_resources(str(tmp_path / 'resources'), 'lib_lv', 'Lotka_Volterra')

    config = _config(tmp_path, 'lib_lv', library_dir, use_builtin_modules=False)
    assert generate_with_new_architecture(False, config)


@pytest.mark.integration
def test_module_library_redefines_builtin_with_a_warning_when_builtins_on(tmp_path):
    """With built-ins on, a library's redefinition of a built-in pair wins over it (the more
    specific source), with a ModuleShadowWarning: it used to stop generation as a duplicate.
    See also tests/test_module_source_precedence.py."""
    library_dir = str(tmp_path / 'library')
    _write_module(library_dir, 'Lotka_Volterra', 'Lotka_Volterra', 'Lotka_Volterra_lib')
    _write_resources(str(tmp_path / 'resources'), 'lib_lv_dup', 'Lotka_Volterra')

    config = _config(tmp_path, 'lib_lv_dup', library_dir, use_builtin_modules=None)
    with pytest.warns(ModuleShadowWarning, match='Lotka_Volterra/nn'):
        assert generate_with_new_architecture(False, config)
    modules_text = (tmp_path / 'generated_models' / 'lib_lv_dup'
                    / 'lib_lv_dup_modules.cellml').read_text()
    assert '<component name="Lotka_Volterra_lib"' in modules_text
    assert '<component name="Lotka_Volterra">' not in modules_text


def test_module_sources_default_is_builtin_library():
    """Without the new keys, sources are exactly the built-in (and user) modules."""
    sources = ModuleSources({'external_modules_dir': None})
    builtin = builtin_modules_dir()
    assert os.path.join(builtin, 'units.cellml') in sources.units_files
    assert os.path.join(builtin, 'heart_modules.cellml') in sources.cellml_files
    assert any(p.startswith(builtin) and p.endswith('.json') for p in sources.config_files)
    assert sources.base_script == os.path.join(builtin, 'base_script.cellml')


def test_module_sources_library_recursive_and_builtins_off(tmp_path):
    library_dir = str(tmp_path / 'library')
    _write_module(library_dir, 'decay', 'decay', 'decay_type')
    sources = ModuleSources({'external_modules_dir': None, 'use_builtin_modules': False,
                             'module_library_dirs': library_dir})
    assert [os.path.basename(p) for p in sources.cellml_files] == ['decay_modules.cellml']
    assert [os.path.basename(p) for p in sources.config_files] == ['decay_modules_config.json']
    assert [os.path.basename(p) for p in sources.units_files] == ['decay_units.cellml']
    assert os.path.isfile(sources.base_script)


def test_identical_units_in_several_files_written_once(tmp_path):
    library_dir = str(tmp_path / 'library')
    a = _write_module(library_dir, 'a', 'a', 'a_type')
    b = _write_module(library_dir, 'b', 'b', 'b_type')
    units = collect_units([os.path.join(a, 'a_units.cellml'), os.path.join(b, 'b_units.cellml')])
    assert [name for name, _ in units] == ['lib_test_per_second']


def test_conflicting_units_raise(tmp_path):
    library_dir = str(tmp_path / 'library')
    a = _write_module(library_dir, 'a', 'a', 'a_type', units_exponent='-1')
    b = _write_module(library_dir, 'b', 'b', 'b_type', units_exponent='-2')
    with pytest.raises(ValueError, match='lib_test_per_second'):
        collect_units([os.path.join(a, 'a_units.cellml'), os.path.join(b, 'b_units.cellml')])


def test_missing_library_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModuleSources({'external_modules_dir': None, 'module_library_dirs': [str(tmp_path / 'nope')]})
