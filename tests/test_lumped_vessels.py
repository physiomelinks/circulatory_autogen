'''
The test models on circulatory-autogen-modules' lumped vessels.

Every vessel version that splits into a compliance, a resistance and an inertance has a lumped
twin in the library (``<version>_lumped``, a supermodule of those submodules), and
``libcuflynx.utilities.lumped_migration`` moves a model onto them. Here each test model in
``resources/`` that has such vessels is generated both ways and simulated, and every output of
the original must be reproduced, under the same name, by the migrated model: the vessels'
outputs through their twins' ``outputs`` (``aortic_root/u`` is the lumped vessel's compliance
pressure), everything else unchanged.

These are the models the tests will run on once the lumped vessels replace the monolithic ones
(the follow-up that moves resources/ onto them and drops these comparisons).
'''
import contextlib
import io
import os
import re
import shutil

import numpy as np
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper
from libcuflynx.utilities.lumped_migration import Library, migrate_files

pytestmark = pytest.mark.integration

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
LIBRARY = os.environ['CUFLYNX_MODULE_LIBRARY'].split(os.pathsep)[0]
SIM_TIME, DT = 1.0, 0.01
SOLVER_INFO = {'rtol': 1e-10, 'atol': 1e-14}
TOL = 1e-6

# Every resources/ model with a vessel that has a lumped twin: None runs by default, 'slow' is a
# large model (over a minute), any other value is why it is skipped.
_NO_REFERENCE = 'the monolithic model does not generate on this library either ({}), so there is nothing to compare'
MODELS = {
    '3compartment': None,
    '3compartment_nonstiff': None,
    '3compartment_extra_ops': None,
    'neonatal': None,
    'test_fft': None,
    'behdad_test1': None,
    'FTU_wCVS': None,
    'aortic_bif_1d': None,
    'simple_physiological': 'slow',
    'physiological': 'slow',
    'control_parasymp': 'slow',
    'cvs_model_0d': 'slow',
    'cvs_model_with_arm_0d': 'slow',
    'generic_junction_test_closed_loop': 'slow',
    'generic_junction_test_open_loop': 'slow',
    'control_phys': 'the monolithic model is not reproducible to itself: the heart\'s atrial phase floor() switches '
                    'at a step-dependent time, so its own outputs differ by up to 23% between rtol 1e-8 and 1e-10',
    'control_phys_asd': 'as control_phys (the same reflexes)',
    'new_valve_p_est': _NO_REFERENCE.format('an invalid mapping'),
    'cerebral': _NO_REFERENCE.format('an invalid model'),
    'cerebral_elic': _NO_REFERENCE.format("terminal2/pp_controller's undeclared u_in"),
    'elic': _NO_REFERENCE.format('pressure_observer_terminal does not exist'),
    'FinalModel': _NO_REFERENCE.format('Delta_q_us_venous_lb has the wrong units in its parameters file'),
    'aortic_bif_hybrid_V1': _NO_REFERENCE.format('a node of three with no summing end'),
    'aortic_bif_hybrid_V2': _NO_REFERENCE.format('a node of three with no summing end'),
    'cvs_model_with_arm_hybrid': _NO_REFERENCE.format('a node of three with no summing end'),
}


def _params():
    out = []
    for model, mark in MODELS.items():
        marks = [pytest.mark.slow] if mark == 'slow' else ([pytest.mark.skip(reason=mark)] if mark else [])
        out.append(pytest.param(model, marks=marks, id=model))
    return out


@pytest.fixture(scope='module')
def library():
    return Library([LIBRARY])


def _generate(resources, prefix, out_dir):
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv', 'model_type': 'cellml',
              'solver': 'CVODE_myokit', 'resources_dir': str(resources), 'generated_models_dir': str(out_dir),
              'DEBUG': False, 'use_builtin_modules': False, 'module_library_dirs': [LIBRARY]}
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    return os.path.join(str(out_dir), prefix, f'{prefix}.cellml')


def _simulate(cellml_path):
    '''{qname: values} of every state and computed variable, and the set of states.'''
    import myokit
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_simulation_helper(model_path=cellml_path, solver='CVODE_myokit', model_type='cellml',
                                      dt=DT, sim_time=SIM_TIME, solver_info=SOLVER_INFO, pre_time=0.0).model
    names = [v.qname() for v in model.variables(deep=True) if v.is_state() or not v.is_constant()]
    sim = myokit.Simulation(model)
    sim.set_tolerance(SOLVER_INFO['atol'], SOLVER_INFO['rtol'])
    log = sim.run(SIM_TIME, log=names, log_interval=DT)
    return {n: np.asarray(log[n], dtype=float) for n in names}, {v.qname() for v in model.states()}


def compare(ref, ref_states, new, replaced, outputs):
    '''(rows, missing, unresolved): every output of the original against the migrated model. A
    replaced vessel's output is compared where its twin exposes it (``<vessel>_module.<var>`` ->
    ``<vessel>.<var>``); its other variables have no single counterpart. Each difference is
    relative to the output's own scale (largest magnitude, or range), so small flows are judged
    against themselves; a state the solver's atol cannot resolve (atol above 1% of its scale) is
    reported rather than trusted.'''
    exposed = {f'{o.split("/")[0]}_module.{o.split("/")[1]}': o.replace('/', '.') for o in outputs}
    prefixes = tuple(f'{v}_module.' for v in replaced)
    own_sums = tuple(f'multiport_sum_{v}_' for v in replaced)
    rows, missing, unresolved = [], [], []
    for name, r in ref.items():
        var = name.split('.')[-1]
        if var in ('t', 'time') or name.startswith(own_sums) or (name.startswith(prefixes) and name not in exposed):
            continue    # time; a replaced vessel's own port sums and unexposed variables (its submodules' now)
        target = exposed.get(name, name)
        if target not in new:
            missing.append(name)
            continue
        n = new[target]
        if name.startswith('heart') and re.fullmatch(r'[LB]_\w+', var):
            r, n = 1.0 / r, 1.0 / n     # a valve's inertance spikes near closure: compared as 1/x
        scale = max(float(np.max(np.abs(r))), float(np.ptp(r)), 1e-300)
        if name in ref_states and np.max(np.abs(r)) > 0 and SOLVER_INFO['atol'] > 1e-2 * scale \
                and r.shape == n.shape and np.max(np.abs(r - n)) > 10 * SOLVER_INFO['atol']:
            # a state the solver's atol does not resolve, that the two models also do not agree on
            # to within atol: noise in both, not a comparison
            unresolved.append(name)
        if r.shape != n.shape:
            rows.append((np.inf, name, target))
            continue
        d = np.abs(r - n)
        if re.fullmatch(r'mt|chi_[av]?floor(_final)?', var):
            d = np.minimum(d % 1.0, 1.0 - d % 1.0)     # a sawtooth phase wrapping one step earlier
        rows.append((float(np.max(d)) / scale, name, target))
    return sorted(rows, reverse=True), missing, unresolved


@pytest.mark.parametrize('model', _params())
def test_lumped_vessels_reproduce_the_model(tmp_path, library, model):
    mono = tmp_path / 'monolithic'
    mono.mkdir()
    module_array = os.path.join(RESOURCES, f'{model}_module_array.csv')
    parameters = os.path.join(RESOURCES, f'{model}_parameters.csv')
    shutil.copy(module_array, mono)
    shutil.copy(parameters, mono)
    result = migrate_files(library, module_array, parameters, out_dir=str(tmp_path / 'lumped'),
                           prefix=f'{model}_lumped', old_prefix=model)
    assert result['replaced'], f'{model} has no vessel with a lumped twin'
    ref, ref_states = _simulate(_generate(mono, model, tmp_path / 'generated'))
    new, _ = _simulate(_generate(tmp_path / 'lumped', f'{model}_lumped', tmp_path / 'generated'))
    rows, missing, unresolved = compare(ref, ref_states, new, result['replaced'], result['outputs'])
    bad = [f'{name} -> {target}: {d:.3g}' for d, name, target in rows if not d <= TOL]
    assert rows and not bad and not missing and not unresolved, (
        f'{model}: {len(bad)} of {len(rows)} outputs differ by more than {TOL:g}: {bad[:10]}; '
        f'missing in the lumped model: {missing[:10]}; states atol does not resolve: {unresolved[:10]}')


def test_shared_parameters_keep_their_names(library):
    '''Migrating 3compartment changes only what the lumped vessels compute: the terminal's
    q_init becomes its compliance's q_C_init (q_init - q_us); every other parameter, and every
    output obs_data names, keeps its name.'''
    module_array = os.path.join(RESOURCES, '3compartment_module_array.csv')
    parameters = os.path.join(RESOURCES, '3compartment_parameters.csv')
    records, rows = None, None
    from libcuflynx.utilities.lumped_migration import _read_parameters, _read_records, migrate
    records = _read_records(module_array)
    _, rows = _read_parameters(parameters)
    _, new_rows, outputs, notes = migrate(records, rows, library)
    before = {r['variable_name'] for r in rows}
    after = {r['variable_name'] for r in new_rows}
    assert before - after == {'q_init_systemic_T'}
    assert after - before == {'q_C_init_systemic_T_C'}
    with open(os.path.join(RESOURCES, '3compartment_obs_data.json')) as f:
        from libcuflynx.utilities.lumped_migration import unexposed_outputs
        assert unexposed_outputs(f.read(), records, library, outputs) == []
    assert not notes
