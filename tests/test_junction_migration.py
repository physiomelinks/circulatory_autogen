"""libcuflynx.utilities.junction_migration and lumped_migration: the names they write are the
module library's."""
import json

import pytest

from libcuflynx.utilities.junction_migration import migrate_records


def _record(name, module_type, subtype, inp=(), out=(), **extra):
    return dict({'name': name, 'module_type': module_type, 'module_subtype': subtype,
                 'inp_instances': list(inp), 'out_instances': list(out)}, **extra)


@pytest.mark.unit
@pytest.mark.parametrize('junction, version, twin', [
    ('Min_junction_2', 'vp_noI', 'vp_noI_SI'),
    ('Nout_junction_2', 'pv_noI', 'pv_noI_SI'),
    ('MinNout_junction_2', 'vv_noI', 'vv_noI_SI'),
])
def test_an_si_junction_becomes_the_libraries_si_simple_vessel(junction, version, twin):
    records, renamed, _ = migrate_records([_record('j', junction, version)])
    assert (records[0]['module_type'], records[0]['module_subtype']) == ('simple', twin)
    assert renamed['j'] == ((junction, version), ('simple', twin))


@pytest.mark.unit
@pytest.mark.parametrize('two, one', [('constant_2', 'constant'), ('nn_constant_2', 'nn_constant')])
def test_a_two_inflow_source_into_a_junction_splits_into_two_one_inflow_sources(two, one):
    records = [_record('src', 'inlet_flow', two, out=['j']),
               _record('j', 'Min_junction', 'vp_noI', inp=['src'], out=['v']),
               _record('v', 'simple', 'pv_noI', inp=['j'])]
    out, renamed, parameters = migrate_records(records)
    by_name = {r['name']: r for r in out}
    assert [(by_name[n]['module_type'], by_name[n]['module_subtype']) for n in ('src_1', 'src_2')] \
        == [('inlet_flow', one)] * 2
    assert by_name['j']['inp_instances'] == ['src_1', 'src_2']
    assert renamed['src'] == (('inlet_flow', two), ('inlet_flow', f'{one} x2'))
    assert parameters == {'v_1_src': 'v_src_1', 'v_2_src': 'v_src_2'}


@pytest.mark.unit
def test_lumped_migration_writes_a_records_parameter_set_as_parameterisation(tmp_path):
    from libcuflynx.utilities.config_schemas import read_module_array_records
    from libcuflynx.utilities.lumped_migration import _write_records
    source = tmp_path / 'in_module_array.json'
    source.write_text(json.dumps([_record('a', 'simple', 'vv_noI', parameterisation='fit')]))
    out = tmp_path / 'out_module_array.json'
    _write_records(str(out), read_module_array_records(str(source)), str(source))
    written = json.loads(out.read_text())[0]
    assert written['parameterisation'] == 'fit' and 'instance' not in written
