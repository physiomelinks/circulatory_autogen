"""The schemas' enums, generated from the Python registries that define them.

Each entry of :data:`REGISTRY_ENUMS` names a place in a shipped schema (a JSON Pointer) and
the registry whose values it lists, so a new prior type, step kind or setting is added in one
place, the registry, and written into the schema by::

    python -m libcuflynx.schemas.registries --write

``tests/test_schema_registries.py`` fails when a shipped schema is out of step.
"""

import argparse
import json
import os
import sys

from libcuflynx.schemas import (CALIBRATION_WORKFLOW_SCHEMA, MODULE_CONFIG_SCHEMA,
                                OBS_DATA_SCHEMA, PARAMS_FOR_ID_SCHEMA, VARIABLE_MAPPING_SCHEMA,
                                load_schema, schema_file)


def _registries():
    from libcuflynx.calibration_workflow.spec import ALLOWED_SETTINGS, PRIOR_KINDS, STEP_KINDS
    from libcuflynx.param_id.prediction_plots import (FEATURE_KINDS, KINDS, REDUCE_MODES,
                                                      TIME_COURSE_KINDS)
    from libcuflynx.parsers.PrimitiveParsers import PARAM_PRIOR_TYPES
    from libcuflynx.reporting.variable_mapping import KINDS as MAPPING_KINDS
    from libcuflynx.sensitivity_analysis.parameter_reduction import INDEX_TYPES
    from libcuflynx.sensitivity_analysis.sobol_ranking import CRITERIA
    from libcuflynx.utilities.config_schemas import VARIABLE_ACCESS, VARIABLE_KINDS
    from libcuflynx.utilities.obs_data_helpers import VALID_DATA_TYPES
    from libcuflynx.utilities.protocol_shapes import SHAPE_TYPES
    return {
        # (schema, JSON Pointer to the enum) -> values
        (OBS_DATA_SCHEMA, '/$defs/dataItem/properties/data_type/enum'): list(VALID_DATA_TYPES),
        (OBS_DATA_SCHEMA, '/$defs/shape/oneOf/1/properties/type/enum'): list(SHAPE_TYPES),
        (OBS_DATA_SCHEMA, '/$defs/prediction_plot/properties/kind/enum'): list(KINDS),
        (OBS_DATA_SCHEMA, '/$defs/prediction_plot/allOf/0/if/properties/kind/enum'):
            list(FEATURE_KINDS),
        (OBS_DATA_SCHEMA, '/$defs/prediction_plot/allOf/1/if/properties/kind/enum'):
            list(TIME_COURSE_KINDS),
        (OBS_DATA_SCHEMA, '/$defs/input_reference/properties/reduce/oneOf/0/enum'):
            list(REDUCE_MODES),
        (PARAMS_FOR_ID_SCHEMA, '/$defs/entryKeys/properties/prior/anyOf/0/enum'):
            list(PARAM_PRIOR_TYPES),
        (CALIBRATION_WORKFLOW_SCHEMA, '/$defs/step/properties/kind/enum'): list(STEP_KINDS),
        (CALIBRATION_WORKFLOW_SCHEMA, '/$defs/prior/oneOf/1/properties/kind/enum'):
            list(PRIOR_KINDS),
        (CALIBRATION_WORKFLOW_SCHEMA,
         '/$defs/step/properties/reduction/properties/index_type/enum'): list(INDEX_TYPES),
        (CALIBRATION_WORKFLOW_SCHEMA,
         '/$defs/step/properties/reduction/properties/criterion/enum'): list(CRITERIA),
        # the settings a workflow may set: the keys of its settings object
        (CALIBRATION_WORKFLOW_SCHEMA, '/$defs/settings/properties'): list(ALLOWED_SETTINGS),
        (VARIABLE_MAPPING_SCHEMA, '/$defs/row/properties/kind/anyOf/0/enum'): list(MAPPING_KINDS),
        (MODULE_CONFIG_SCHEMA,
         '/$defs/phlynxComponent/properties/variables_and_units/items/prefixItems/2/enum'):
            list(VARIABLE_ACCESS),
        (MODULE_CONFIG_SCHEMA,
         '/$defs/phlynxComponent/properties/variables_and_units/items/prefixItems/3/enum'):
            list(VARIABLE_KINDS),
    }


def _parent(document, pointer):
    *path, last = [p.replace('~1', '/').replace('~0', '~') for p in pointer.split('/')[1:]]
    node = document
    for part in path:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node, last


def _apply(document, pointer, values):
    """Set the enum (or, for a ``properties`` object, its keys) at ``pointer`` to ``values``.
    Returns whether anything changed."""
    parent, last = _parent(document, pointer)
    if last == 'properties':
        old = parent.get('properties', {})
        new = {key: old.get(key, {}) for key in values}
        changed = list(old) != list(new)
        parent['properties'] = new
        return changed
    changed = parent.get(last) != values
    parent[last] = list(values)
    return changed


def out_of_step():
    """``[(schema, pointer, values in the schema, values of the registry)]``."""
    found = []
    for (schema_name, pointer), values in _registries().items():
        document = load_schema(schema_name)
        parent, last = _parent(document, pointer)
        current = list(parent.get(last, {})) if last == 'properties' else parent.get(last)
        if current != values:
            found.append((schema_name, pointer, current, values))
    return found


def write():
    """Rewrite each schema's enums from its registries; the names of the files changed."""
    documents, changed = {}, set()
    for (schema_name, pointer), values in _registries().items():
        document = documents.setdefault(schema_name, load_schema(schema_name))
        if _apply(document, pointer, values):
            changed.add(schema_name)
    for schema_name in changed:
        path = schema_file(schema_name)
        with open(str(path), 'w', encoding='utf-8') as f:  # checked by main()
            f.write(json.dumps(documents[schema_name], indent=2, ensure_ascii=False) + '\n')
    return sorted(changed)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check, or with --write rewrite, the shipped '
                                                 'schemas\' enums from libcuflynx\'s registries.')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    if args.write:
        # the schemas of whichever libcuflynx is imported: refuse an installed copy, and say
        # which checkout was written, so a venv pointing elsewhere cannot go unnoticed
        here = os.path.dirname(os.path.abspath(str(schema_file(OBS_DATA_SCHEMA))))
        root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        if not os.path.isfile(os.path.join(root, 'pyproject.toml')):
            print(f'{here} is not in a source checkout; run from one with its src on '
                  f'PYTHONPATH.')
            return 2
        print(f'rewrote {", ".join(write()) or "nothing"} in {here}')
        return 0
    stale = out_of_step()
    for schema_name, pointer, current, values in stale:
        print(f'{schema_name} {pointer}: {current} but the registry has {values}')
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
