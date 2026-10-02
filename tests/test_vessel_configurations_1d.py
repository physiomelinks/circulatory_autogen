'''
0D nodes where several vessels meet, coupled to the FV 1D solver.

aortic_bif_hybrid_V1 (a 1D aortic bifurcation with a 0D terminal on each daughter) is extended
downstream of the 1D tree with a 0D node that an ordinary vessel owns through its summing port
(libcuflynx.generators.port_nodes):

    inflow -> parent (1D) -> daughter_1 (1D) -> terminal_1 --\\
                          -> daughter_2 (1D) -> terminal_2 ----> venous (vp) -> sink
    inflow_2 -> c (vp) ----------------------------------------/

venous's inlet owns the merge of both 1D-fed terminals and the 0D branch c. Each 1D-0D link stays
one-to-one, which is what the coupling supports; a node of three or more ends that includes a 1D
vessel is refused.

The model is generated as C++, built with CMake, and run under the coupler with the Python 1D
solver for one cardiac cycle, with CVODE (with RK4 even aortic_bif_hybrid_V1 itself diverges on
its first step). The 0D output must conserve flow at the node.

Not covered, because the coupling itself stops on them, with or without a node: a 0D vessel with a
compliant end at a 1D link (e.g. daughter_2 -> a vv vessel), a 0D vessel feeding a 1D inlet, and
a 0D split downstream of a 1D link (daughter_2 -> pp -> vv -> two terminals). The Python 1D solver
adds dt_stage/dt for every 0D right-hand-side evaluation in its step and stops when that passes 1
("Sum of integration weights > 1"); CVODE evaluates the right-hand side several times per step
in an implicit solve, so a 0D model that needs Newton iterations there overruns. See
test_split_downstream_of_a_1d_link.
'''
import json
import os
import shutil
import signal
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_cpp_template_generator import _cmake_build, _generation_inputs, _load_output

PREFIX = 'nodes_1d'
VESSEL_ARRAY = '''name,BC_type,vessel_type,inp_vessels,out_vessels
input_flow_aorticroot,nn_aorticbif,inlet_flow,,parent
parent,nn,FV1D_vessel,input_flow_aorticroot,daughter_1 daughter_2
daughter_1,nn,FV1D_vessel,parent,terminal_1
daughter_2,nn,FV1D_vessel,parent,terminal_2
terminal_1,pp,terminal,daughter_1,venous
terminal_2,pp,terminal,daughter_2,venous
input_flow_2,nn_constant,inlet_flow,,c
c,vp,arterial_simple,input_flow_2,venous
venous,vp,venous,terminal_1 terminal_2 c,sink
sink,nn_constant,outlet_pressure,venous,
'''
ZERO_D = {
    'c': {'R': 1e8, 'C': 1e-9, 'I': 1e5},
}
UNITS = {'R': 'Js_per_m6', 'C': 'm6_per_J', 'I': 'Js2_per_m6', 'q_0': 'm3', 'u_0': 'J_per_m3', 'u_ext': 'J_per_m3'}


def _resources(tmp_path, resources_dir, vessel_array):
    '''The extended model's vessel array and parameters, in a resources dir of their own.'''
    res = tmp_path / 'node_resources'
    res.mkdir()
    (res / f'{PREFIX}_vessel_array.csv').write_text(vessel_array)
    params = pd.read_csv(os.path.join(resources_dir, 'aortic_bif_hybrid_V1_parameters.csv'))
    params = params[~params['variable_name'].isin(['u_out_terminal_1', 'u_out_terminal_2'])].copy()
    # aortic_bif_hybrid_V1's terminal inertance (I_T_global 1e-6 against R_T ~ 3e9: a time constant
    # of ~1e-16 s) is harmless while a terminal drains into a fixed pressure; draining into a 0D vein
    # it makes the 0D model needlessly stiff. 1e6 gives ~3e-4 s.
    params.loc[params['variable_name'] == 'I_T_global', 'value'] = 1e6
    rows = []
    for name, values in ZERO_D.items():
        for var, value in dict(values, q_0=0.0, u_0=0.0, u_ext=0.0).items():
            rows.append((f'{var}_{name}', UNITS[var], value))
    rows += [('R_venous', 'Js_per_m6', 1e6), ('C_venous', 'm6_per_J', 1e-6), ('I_venous', 'Js2_per_m6', 1e4),
             ('u_ext_venous', 'J_per_m3', 0.0), ('q_C_init_venous', 'm3', 0.0), ('q_us_0_venous', 'm3', 0.0),
             ('Delta_q_us_venous', 'dimensionless', 0.0), ('Delta_C_venous', 'dimensionless', 0.0),
             ('P_sink', 'J_per_m3', 0.0), ('v_input_flow_2', 'm3_per_s', 1e-6)]
    extra = pd.DataFrame([{'variable_name': n, 'units': u, 'value': v, 'data_reference': 'test'} for n, u, v in rows])
    pd.concat([params, extra]).to_csv(res / f'{PREFIX}_parameters.csv', index=False)
    return res


@pytest.mark.integration
@pytest.mark.slow
def test_nodes_downstream_of_a_1d_tree(user_inputs_dir, resources_dir, tmp_path):
    solver = 'CVODE'
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    from libcuflynx.utilities.package_resources import package_data_file
    res = _resources(tmp_path, resources_dir, VESSEL_ARRAY)
    cpp_dir = tmp_path / 'gen' / 'cpp'
    ini = tmp_path / '1d' / 'run000' / 'input.ini'
    inp = _generation_inputs(user_inputs_dir, str(res), tmp_path, PREFIX, solver,
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(cpp_dir), cpp_1d_model_config_path=str(ini), dt=0.01)
    assert generate_with_new_architecture(False, inp)
    _cmake_build(cpp_dir, cpp_dir / 'build')
    coupler_src = os.path.join(os.path.dirname(__file__), '..', 'src', 'libcuflynx', 'coupler')
    if not os.path.isfile(os.path.join(coupler_src, 'coupler.cpp')):
        pytest.skip('coupler sources need a source checkout')
    _cmake_build(coupler_src, tmp_path / 'coupler_build')

    pipes = tmp_path / 'pipes'
    pipes.mkdir()
    config = {'inputFold': str(cpp_dir) + '/', 'networkName': PREFIX, 'ODEsolver': solver, 'T0': 1.1, 'nCC': 1,
              'tmp_pipe_path': str(pipes) + '/', 'initStatePath': 'None', 'python_path': sys.executable,
              'solver1d_path': str(package_data_file('libcuflynx.solver1d', 'main1D.py')),
              'solver0d_path': str(cpp_dir / 'build' / 'main0d'), 'initFile_sim1d_path': str(ini)}
    with open(cpp_dir / 'coupler_config.json', 'w') as f:
        json.dump(config, f)
    log_path = tmp_path / 'coupler.log'
    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen([str(tmp_path / 'coupler_build' / 'coupler'), str(cpp_dir / 'coupler_config.json')],
                                stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode = None
    log = log_path.read_text(errors='replace')
    tail = '\n'.join(log.splitlines()[-80:])
    assert returncode == 0 and 'Both processes terminated successfully with status codes : 0 0' in log, tail

    out = tmp_path / 'simulation_outputs_cpp' / PREFIX
    cs, states = _load_output(out / 'sol0D_states.txt')
    cv, variables = _load_output(out / 'sol0D_variables.txt')
    assert states.shape[0] > 50 and np.all(np.isfinite(states)) and np.all(np.isfinite(variables))

    def col(name):
        for names, values in ((cs, states), (cv, variables)):
            if name in names:
                return values[:, names[name]]
        raise AssertionError(f'{name} not in the 0D output: {sorted(cs) + sorted(cv)}')

    # the merge terminal_1, terminal_2, c -> venous: what venous holds changes by everything the
    # three bring in less what it passes on (the sum component itself is not in the output)
    t = col('t')
    inflow = col('terminal_1/v_T') + col('terminal_2/v_T') + col('c/v')
    stored = col('venous/q') - col('venous/q')[0]
    expected = np.concatenate([[0.0], np.cumsum(np.diff(t) * 0.5 * ((inflow - col('venous/v'))[1:] + (inflow - col('venous/v'))[:-1]))])
    assert np.max(np.abs(stored - expected)) <= 0.02 * np.max(np.abs(np.cumsum(np.diff(t) * inflow[1:])))
    # flow comes out of the 1D tree into the 0D node
    assert np.max(col('terminal_1/v_T')) > 1e-6 and np.max(col('terminal_2/v_T')) > 1e-6
    sol1d = np.genfromtxt(ini.parent / 'res' / 'sol1D_parent.txt')
    assert np.all(np.isfinite(sol1d))


@pytest.mark.integration
def test_a_1d_vessel_cannot_share_a_node_with_two_0d_modules(user_inputs_dir, resources_dir, tmp_path):
    '''The 1D coupling exchanges one flow and one pressure per 1D-0D link, so a 1D vessel meeting
    two 0D vessels at one node is refused, with the vessel named.'''
    from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
    # daughter_2 feeds terminal_2 and the venous vessel (whose inlet owns the node) directly
    array = VESSEL_ARRAY.replace('daughter_2,nn,FV1D_vessel,parent,terminal_2', 'daughter_2,nn,FV1D_vessel,parent,terminal_2 venous')
    array = array.replace('venous,vp,venous,terminal_1 terminal_2 c,sink', 'venous,vp,venous,terminal_1 terminal_2 c daughter_2,sink')
    res = _resources(tmp_path, resources_dir, array)
    inp = _generation_inputs(user_inputs_dir, str(res), tmp_path, PREFIX, 'CVODE',
                             couple_to_1d=True, create_main_0d=True, generate_1d=True, solver_1d_type='py',
                             cpp_generated_models_dir=str(tmp_path / 'gen' / 'cpp'),
                             cpp_1d_model_config_path=str(tmp_path / '1d' / 'run000' / 'input.ini'), dt=0.01)
    with pytest.raises(NotImplementedError, match='daughter_2 has no CellML'):
        generate_with_new_architecture(False, inp)


@pytest.mark.skip(reason='the 1D coupling stops on it, with or without the split: see the module docstring')
def test_split_downstream_of_a_1d_link():
    '''daughter_2 -> p2 (pp) -> a2 (vv) -> t2a, t2b, coupled like test_nodes_downstream_of_a_1d_tree,
    stops on the 1D solver's first step with "Sum of integration weights > 1" (1.03-1.07). So does
    daughter_2 -> a2 (vv) -> t2a with no split, and a 0D vessel feeding a 1D inlet. Turn this into a
    real test once the coupling counts each 0D step once rather than every right-hand-side
    evaluation in it.'''
