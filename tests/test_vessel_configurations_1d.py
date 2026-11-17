'''0D nodes of summing ports downstream of FV 1D vessels, run under the 0D-1D coupler.

aortic_bif_hybrid_V1 (a 1D aortic bifurcation) is extended downstream with 0D nodes, each owned
by the one compliant end alone on its side (node rule D), with every module from the module
library:

    inflow -> parent (1D) -> daughter_1 (1D) -> terminal_1 -----------------------\\
                          -> daughter_2 (1D) -> e (pp) -> s (vv) -> terminal_2 -----> venous (vp) -> sink
                                                                 -> terminal_3 ---/
    inflow_2 -> c (vp) --------------------------------------------------------------/

s's outlet owns a split downstream of a 1D link, and venous's inlet owns the merge of two
1D-fed branches and a 0D one. Each 1D-0D link stays one-to-one: a node of three or more ends
with a 1D end is not one rule D checks, and the coupler couples a 1D end to one 0D end.

The model is generated as C++ and run for 0.1 s under the coupler with the Python FV 1D solver
and CVODE. Each owner's volume must change by the integral of the flows its node sums, less its
outflow.
'''
import json
import shutil

import numpy as np
import pytest

from _module_library import modules_dir_or_skip
from test_cpp_template_generator import _generation_inputs, _load_output, _run_coupled, _short_coupled_run

pytestmark = [pytest.mark.integration, pytest.mark.slow]

PREFIX = 'nodes_1d'
# the trapezoidal integral of the saved flows (1 ms apart) against the volume the solver integrated
TRAPEZOID_TOL = 1e-3
RECORDS = [
    ('input_flow_aorticroot', 'inlet_flow', 'aorticbif', [], ['parent']),
    ('parent', 'FV1D_vessel', 'nn', ['input_flow_aorticroot'], ['daughter_1', 'daughter_2']),
    ('daughter_1', 'FV1D_vessel', 'nn', ['parent'], ['terminal_1']),
    ('daughter_2', 'FV1D_vessel', 'nn', ['parent'], ['e']),
    ('terminal_1', 'terminal', 'pp', ['daughter_1'], ['venous']),
    ('e', 'arterial_simple', 'pp', ['daughter_2'], ['s']),
    ('s', 'arterial_simple', 'vv', ['e'], ['terminal_2', 'terminal_3']),
    ('terminal_2', 'terminal', 'pp', ['s'], ['venous']),
    ('terminal_3', 'terminal', 'pp', ['s'], ['venous']),
    ('inflow_2', 'inlet_flow', 'constant', [], ['c']),
    ('c', 'arterial_simple', 'vp', ['inflow_2'], ['venous']),
    ('venous', 'venous', 'vp', ['terminal_1', 'terminal_2', 'terminal_3', 'c'], ['sink']),
    ('sink', 'outlet_pressure', 'constant', ['venous'], []),
    ('FV1D_solver', 'FV1D_solver', 'nn', [], []),
]


def leaf(units, value):
    return {'value': value, 'units': units, 'data_reference': 'test'}


def _write_model(resources):
    '''The extended model's module array and parameters, from aortic_bif_hybrid_V1's.'''
    lines = ['name,module_subtype,module_type,inp_instances,out_instances']
    lines += [f'{n},{st},{mt},{" ".join(i)},{" ".join(o)}' for n, mt, st, i, o in RECORDS]
    (resources / f'{PREFIX}_module_array.csv').write_text('\n'.join(lines) + '\n')
    params = json.loads((resources / 'aortic_bif_hybrid_V1_parameters.json').read_text())
    for name in ('terminal_1', 'terminal_2'):
        params[name].pop('u_out')           # drained into venous now
    params['terminal_3'] = json.loads(json.dumps(params['terminal_2']))
    vessel = {'R': leaf('Js_per_m6', 1e8), 'C': leaf('m6_per_J', 1e-9), 'I': leaf('Js2_per_m6', 1e5),
              'q_0': leaf('m3', 0.0), 'u_0': leaf('J_per_m3', 0.0), 'u_ext': leaf('J_per_m3', 0.0)}
    for name in ('e', 's', 'c'):
        params[name] = json.loads(json.dumps(vessel))
    # e's inlet pressure comes from the 1D solver at run time; like the terminals' it needs a start
    params['e']['u_in'] = leaf('J_per_m3', 10000.0)
    params['venous'] = {'R': leaf('Js_per_m6', 1e6), 'C': leaf('m6_per_J', 1e-6), 'I': leaf('Js2_per_m6', 1e4),
                        'u_ext': leaf('J_per_m3', 0.0), 'q_C_init': leaf('m3', 0.0), 'q_us_0': leaf('m3', 0.0),
                        'Delta_q_us': leaf('dimensionless', 0.0), 'Delta_C': leaf('dimensionless', 0.0)}
    params['inflow_2'] = {'v': leaf('m3_per_s', 1e-6)}
    params['sink'] = {'P': leaf('J_per_m3', 0.0)}
    (resources / f'{PREFIX}_parameters.json').write_text(json.dumps(params, indent=1))


def test_summing_nodes_downstream_of_1d_vessels(user_inputs_dir, resources_dir, tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    library = str(modules_dir_or_skip())
    cpp_dir = tmp_path / 'gen' / 'cpp'
    ini = tmp_path / '1d' / 'run000' / 'input.ini'
    inp = _generation_inputs(user_inputs_dir, resources_dir, tmp_path, 'aortic_bif_hybrid_V1', 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini),
                             dt=0.001, pre_time=0.0, sim_time=1.1, coupler_pipe_dir=str(tmp_path / 'pipes'),
                             module_library_dirs=[library], use_builtin_modules=False)
    resources = tmp_path / 'resources'
    _write_model(resources)
    inp.update(file_prefix=PREFIX, input_param_file=f'{PREFIX}_parameters.json')
    assert generate_with_new_architecture(False, inp)
    _short_coupled_run(cpp_dir, t0=0.1)
    log = _run_coupled(cpp_dir, tmp_path, timeout=480)
    assert 'integration weights' not in log

    out = tmp_path / 'simulation_outputs_cpp' / PREFIX
    cols, values = _load_output(out / 'sol0D_states.txt')
    assert values.shape[0] > 10 and np.all(np.isfinite(values))

    def col(name):
        assert name in cols, f'{name} not in the 0D states ({sorted(cols)})'
        return values[:, cols[name]]

    # each owner's volume changes by what its node sums in less what leaves it: the sums are
    # not written out, so their integral is compared with the owner's volume change
    # from 10 ms: the terminals' flows rise in the first millisecond, faster than the 1 ms output
    t = col('t')
    k = int(np.searchsorted(t, 0.01))
    t = t[k:]
    for volume, inflows, outflows in (
            ('venous/q_C_change', ['terminal_1/v_T', 'terminal_2/v_T', 'terminal_3/v_T', 'c/v'], ['venous/v']),
            ('s/q_C_d', ['s/v'], ['terminal_2/v', 'terminal_3/v'])):
        net = (sum(col(n) for n in inflows) - sum(col(n) for n in outflows))[k:]
        integral = np.concatenate(([0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(t))))
        change = col(volume)[k:] - col(volume)[k]
        error = np.max(np.abs(change - integral))
        assert error <= TRAPEZOID_TOL * np.max(np.abs(change)), \
            f'{volume}: volume change and integrated node flows differ by {error:.3g}'
    # the 1D-fed branches carry flow into the node
    assert np.max(np.abs(col('terminal_1/v_T'))) > 0 and np.max(np.abs(col('terminal_2/v_T'))) > 0
    sol1d = np.genfromtxt(ini.parent / 'res' / 'sol1D_daughter_2.txt')
    assert np.all(np.isfinite(sol1d))
