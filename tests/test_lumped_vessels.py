'''
The test models run on circulatory-autogen-modules' lumped vessels.

Every vessel version that splits into a compliance, a resistance and an inertance has a lumped
twin in the library (``<version>_lumped``, a supermodule of those submodules, keeping the
monolithic version's parameter and output names), and ``resources/`` uses the twins: the
models were moved with ``libcuflynx.utilities.lumped_migration`` after each was checked to
reproduce its monolithic original on every output (the comparison this file held before the
move). These tests keep it that way.
'''
import glob
import os

import pytest

from libcuflynx.utilities.lumped_migration import Library, _read_records, migrate

pytestmark = pytest.mark.unit

RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources')
LIBRARY = os.environ['CUFLYNX_MODULE_LIBRARY'].split(os.pathsep)[0]

_NO_REFERENCE = 'its monolithic version does not generate on this library ({}), so it was not checked'
# models that keep the monolithic vessels, and why
NOT_MOVED = {
    'aortic_bif_1d': '0D-1D: the 0D-1D split writes the 0D vessel array to a CSV, which cannot carry the '
                     'supermodules\' parameter names yet',
    'aortic_bif_1d_0d': 'the 0D half of aortic_bif_1d',
    'control_phys': 'not reproducible to itself: the heart\'s atrial phase floor() switches at a step-dependent '
                    'time, so it differs from itself by up to 23% between rtol 1e-8 and 1e-10',
    'control_phys_asd': 'as control_phys',
    'new_valve_p_est': _NO_REFERENCE.format('an invalid mapping'),
    'cerebral': _NO_REFERENCE.format('an invalid model'),
    'cerebral_elic': _NO_REFERENCE.format("terminal2/pp_controller's undeclared u_in"),
    'elic': _NO_REFERENCE.format('pressure_observer_terminal does not exist'),
    'FinalModel': _NO_REFERENCE.format('Delta_q_us_venous_lb has the wrong units in its parameters file'),
    'aortic_bif_hybrid_V1': _NO_REFERENCE.format('a node of three with no summing end'),
    'aortic_bif_hybrid_V2': _NO_REFERENCE.format('a node of three with no summing end'),
    'cvs_model_with_arm_hybrid': _NO_REFERENCE.format('a node of three with no summing end'),
    'generic_junction_test2_closed_loop': 'no parameters file',
    'generic_junction_test2_open_loop': 'no parameters file',
}


@pytest.fixture(scope='module')
def library():
    return Library([LIBRARY])


def _models():
    return sorted(os.path.basename(p)[:-len('_vessel_array.csv')]
                  for p in glob.glob(os.path.join(RESOURCES, '*_vessel_array.csv')))


@pytest.mark.parametrize('model', _models())
def test_test_models_use_the_lumped_vessels(library, model):
    records = _read_records(os.path.join(RESOURCES, f'{model}_vessel_array.csv'))
    monolithic = sorted({f"{r['vessel_type']}/{r['BC_type']}" for r in records
                         if (r['vessel_type'], r['BC_type']) in library.twins})
    if model in NOT_MOVED:
        pytest.skip(NOT_MOVED[model])
    assert not monolithic, (f'{model} still uses monolithic vessels with a lumped twin: {monolithic}. Move it with '
                            f'python -m libcuflynx.utilities.lumped_migration (and check it reproduces the original).')


def test_migrating_a_moved_model_changes_nothing(library):
    records = _read_records(os.path.join(RESOURCES, '3compartment_vessel_array.csv'))
    from libcuflynx.utilities.lumped_migration import _read_parameters
    _, rows = _read_parameters(os.path.join(RESOURCES, '3compartment_parameters.csv'))
    new_records, new_rows, _outputs, notes = migrate(records, rows, library)
    assert new_records == records and new_rows == rows and not notes
