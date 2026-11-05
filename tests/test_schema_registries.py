"""The shipped JSON Schemas are what every reader checks a file against on load
(libcuflynx.schemas.validation), and their enums are generated from libcuflynx's registries
(libcuflynx.schemas.registries)."""
import numpy as np
import pytest

from libcuflynx.schemas import OBS_DATA_SCHEMA, load_schema
from libcuflynx.schemas import registries
from libcuflynx.schemas.validation import MAX_LISTED, SchemaValidationError, validate
from libcuflynx.utilities.config_schemas import VARIABLE_KINDS, normalise_module_config


@pytest.mark.unit
def test_the_shipped_schemas_list_what_the_registries_define():
    assert registries.out_of_step() == [], (
        'a registry changed without its schema: run python -m libcuflynx.schemas.registries '
        '--write')


@pytest.mark.unit
def test_a_registry_change_is_reported_and_written(monkeypatch):
    real = registries._registries()
    key = next(k for k in real if k[1].endswith('/prefixItems/3/enum'))
    changed = dict(real)
    changed[key] = list(VARIABLE_KINDS) + ['state']
    monkeypatch.setattr(registries, '_registries', lambda: changed)
    assert registries.out_of_step() == [(key[0], key[1], list(VARIABLE_KINDS),
                                         list(VARIABLE_KINDS) + ['state'])]
    # what --write does, on a copy rather than the package's file
    document = load_schema(key[0])
    assert registries._apply(document, key[1], changed[key])
    parent, last = registries._parent(document, key[1])
    assert parent[last] == list(VARIABLE_KINDS) + ['state']
    assert not registries._apply(document, key[1], changed[key])


_PROTOCOL = {'pre_times': [0.0], 'sim_times': [[1.0]]}


def _item(**fields):
    return dict({'data_item_name': 'systolic', 'data_type': 'constant', 'unit': 'Pa',
                 'operands': ['aorta/u'], 'operation': 'max', 'value': 1.0, 'std': 0.1},
                **fields)


@pytest.mark.unit
def test_a_problem_is_named_by_where_it_is_and_what_the_field_is_for():
    doc = {'protocol_info': _PROTOCOL,
           'data_items': [_item(), _item(data_item_name='mean', value=[1.0, 2.0])]}
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(OBS_DATA_SCHEMA, doc, source='aorta_obs_data.json', what='obs_data file')
    message = str(excinfo.value)
    assert message.startswith('aorta_obs_data.json: 1 problem with the obs_data file')
    # the entry's own name after its index, and the rule it broke, in its own words
    assert "at data_items[1] ('mean').value: [1.0, 2.0] is not of type 'number'" in message
    assert "(A constant's value and std are single numbers.)" in message
    assert excinfo.value.problems == [("data_items[1] ('mean').value",
                                       "[1.0, 2.0] is not of type 'number', 'null'")]
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.unit
def test_a_document_built_in_python_is_checked_as_its_json_would_be():
    validate(OBS_DATA_SCHEMA, {'protocol_info': _PROTOCOL,
                               'data_items': [
        _item(data_type='series', operation=None, value=np.array([1.0, 2.0]),
              std=np.float64(0.1), obs_dt=np.float32(0.5), experiment_idx=np.int64(0))]})


@pytest.mark.unit
def test_every_problem_is_counted_and_the_first_ones_listed():
    doc = {'protocol_info': _PROTOCOL, 'data_items': [_item(data_item_name=f'i{n}', colour='red')
                          for n in range(MAX_LISTED + 3)]}
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(OBS_DATA_SCHEMA, doc)
    message = str(excinfo.value)
    assert f'{MAX_LISTED + 3} problems' in message and 'and 3 more' in message
    assert len(excinfo.value.problems) == MAX_LISTED + 3


def _component(**changes):
    entry = {'module_type': 'vessel', 'module_subtype': 'nn', 'component_file': 'm.cellml',
             'component_type': 'vessel_type', 'module_format': 'cellml',
             'entrance_ports': [{'port_type': 'flow_port', 'variables': ['q']}], 'exit_ports': [],
             'variables_and_units': [['q', 'm3_per_s', 'access', 'variable']]}
    entry.update(changes)
    return entry


@pytest.mark.unit
@pytest.mark.parametrize('entry, message', [
    # each of these passed the reader's own checks, and is now refused on load
    (_component(variables_and_units=[['q', 'm3_per_s', 'acess', 'variable']]),
     r"variables_and_units\[0\]\[2\]: 'acess' is not one of \['access', 'no_access'\]"),
    (_component(variables_and_units=[['q', 'm3_per_s', 'access', 'varible']]),
     r"variables_and_units\[0\]\[3\]: 'varible' is not one of"),
    (_component(entrance_ports=[{'port_type': 'flow_port'}]),
     r"entrance_ports\[0\]: 'variables' is a required property"),
])
def test_a_module_config_the_schema_refuses_is_refused_on_load(entry, message):
    normalise_module_config([_component()], source='good.json')
    with pytest.raises(SchemaValidationError, match=message):
        normalise_module_config([entry], source='vessel_modules_config.json')



@pytest.mark.unit
def test_a_document_that_passed_is_not_checked_again_but_an_edited_one_is(monkeypatch):
    from libcuflynx.schemas import validation
    doc = {'protocol_info': _PROTOCOL, 'data_items': [_item(data_item_name='cached')]}
    validate(OBS_DATA_SCHEMA, doc)
    calls = []
    monkeypatch.setattr(validation, 'problems',
                        lambda *args: calls.append(args) or [])
    validate(OBS_DATA_SCHEMA, doc)              # the same contents: remembered
    assert calls == []
    doc['data_items'][0]['value'] = 2.0         # edited: checked again
    validate(OBS_DATA_SCHEMA, doc)
    assert len(calls) == 1


@pytest.mark.unit
def test_without_jsonschema_files_still_load_with_a_warning(monkeypatch):
    """A CUFLynx app released before libcuflynx required jsonschema runs newer CA source with
    its own bundle: reading must still work there."""
    import builtins
    from libcuflynx.schemas import validation
    real_import = builtins.__import__

    def no_jsonschema(name, *args, **kwargs):
        if name == 'jsonschema' or name.startswith('jsonschema.'):
            # what CUFLynx v0.7.1's bundle raises: an optional format package's grammar file
            # was not bundled
            raise FileNotFoundError('rfc3987_syntax/syntax_rfc3987.lark')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_jsonschema)
    validation.jsonschema_available.cache_clear()
    try:
        with pytest.warns(UserWarning, match='does not import .FileNotFoundError'):
            validate(OBS_DATA_SCHEMA, {'protocol_info': _PROTOCOL,
                                       'data_items': [_item(colour='red')]})
    finally:
        validation.jsonschema_available.cache_clear()


@pytest.mark.unit
def test_a_numpy_value_is_not_taken_for_its_string_spelling():
    """np.int64(0) and "0" must not share a cache entry: the second is not a valid index."""
    validate(OBS_DATA_SCHEMA, {'protocol_info': _PROTOCOL,
                               'data_items': [_item(experiment_idx=np.int64(0))]})
    with pytest.raises(SchemaValidationError, match='experiment_idx'):
        validate(OBS_DATA_SCHEMA, {'protocol_info': _PROTOCOL,
                                   'data_items': [_item(experiment_idx='0')]})


@pytest.mark.unit
def test_problems_are_listed_in_document_order():
    doc = {'protocol_info': _PROTOCOL, 'data_items': [_item(data_item_name=f'i{n}', **({'colour': 'r'} if n in (2, 10) else {}))
                          for n in range(11)]}
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(OBS_DATA_SCHEMA, doc)
    assert [p for p, _ in excinfo.value.problems] == ["data_items[2] ('i2')", "data_items[10] ('i10')"]


@pytest.mark.unit
def test_a_value_of_the_wrong_type_is_one_problem():
    from libcuflynx.schemas import PARAMS_FOR_ID_SCHEMA
    with pytest.raises(SchemaValidationError) as excinfo:
        validate(PARAMS_FOR_ID_SCHEMA, {'params': [5]})
    assert excinfo.value.problems == [('params[0]', "5 is not of type 'object'")]


@pytest.mark.unit
def test_write_refuses_a_copy_outside_a_source_checkout(monkeypatch, tmp_path, capsys):
    from importlib.resources import files
    installed = tmp_path / 'site-packages' / 'libcuflynx' / 'schemas'
    installed.mkdir(parents=True)
    monkeypatch.setattr(registries, 'schema_file', lambda name: installed / name)
    monkeypatch.setattr(registries, 'write', lambda: pytest.fail('wrote an installed copy'))
    assert registries.main(['--write']) == 2
    assert 'not in a source checkout' in capsys.readouterr().out
    assert files  # importlib.resources is what schema_file uses in the package
