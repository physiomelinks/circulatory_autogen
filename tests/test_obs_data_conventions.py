"""``name_for_plotting`` names the output *before* the operation, not the feature after it.

The canonical example is ``resources/3compartment_obs_data.json``, where the ``mean`` and the
``max`` of ``aortic_root/v`` carry the *same* label, ``v_{AR}``::

    mean          of aortic_root/v  ->  'v_{AR}'
    max           of aortic_root/v  ->  'v_{AR}'
    max_minus_min of heart/q_lv     ->  'q_{lv}'

That repetition is deliberate, and the rest of the codebase is built on it: a label names the
series, so ``observable_base_label`` composes ``name (op operand)`` to identify a *feature*,
and ``_observable_label`` appends ``[exp e, sub s]`` when even that repeats. A label that
already spells the operation -- ``mean(T_p1)``, ``x_{max}`` -- says it twice in every
composed name, and titles a plot of a whole trace after one scalar drawn from it.

Seven shipped examples had drifted this way, all of them written after the convention was
established and none of them caught by anything: the three ``external_python`` examples used
``mean(T_p1)``, and four older files used the ``x_{max}`` subscript spelling. Hence this test
-- the convention is only discoverable by reading an example that happens to be right.
"""
import json
import os
import re

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), '..')

#: Directories that hold run artifacts rather than authored inputs. A run writes its resolved
#: obs_data back out, so these are full of copies whose labels were correct when written.
_ARTIFACT_DIRS = ('test_outputs', 'generated_models', 'param_id_output', 'benchmarks',
                  'venv', '.claude', '.git')

#: The reducing operations whose name showing up in a label is the mistake this catches.
#: Deliberately not every registered operation: an operation named after a physical quantity
#: (say a user's ``pressure``) would make a legitimate label look like a violation.
_REDUCERS = ('mean', 'min', 'max', 'max_minus_min', 'median', 'sum', 'std')


def _authored_obs_data_files():
    found = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _ARTIFACT_DIRS]
        for name in filenames:
            if 'obs_data' in name and name.endswith('.json'):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def _data_items(path):
    with open(path) as fh:
        doc = json.load(fh)
    items = doc if isinstance(doc, list) else doc.get('data_items')
    return [item for item in (items or []) if isinstance(item, dict)]


def _spells_its_own_operation(label, operation):
    """Whether ``label`` embeds ``operation`` in either spelling seen in the wild.

    ``mean(T_p1)`` -- functional application -- and ``x_{max}`` / ``x_max`` -- the operation
    as a subscript. Anything else (a variable that merely contains the letters, e.g. a
    quantity genuinely called ``mean_flow``) is left alone: the point is to catch a label
    that restates its data_item's own operation, not to police vocabulary.
    """
    if not label or not operation:
        return False
    op = re.escape(operation)
    return bool(
        re.match(rf'^\s*{op}\s*\(', label)                 # mean(T_p1)
        or re.search(rf'_\{{?{op}\}}?\s*(\(|$)', label)     # x_{max}, x_max, x_max (trace)
    )


@pytest.mark.unit
def test_the_sweep_actually_finds_the_examples():
    """Guard the guard: a bad root or filter would make every assertion below vacuous."""
    files = _authored_obs_data_files()
    assert len(files) > 10, files
    assert any(f.endswith('3compartment_obs_data.json') for f in files), files


@pytest.mark.unit
def test_the_canonical_example_still_shows_the_convention():
    """If 3compartment ever stops sharing one label across two operations, the convention has
    changed and the rest of this file is enforcing a rule nobody follows any more."""
    path = os.path.join(_ROOT, 'resources', '3compartment_obs_data.json')
    labels_by_operand = {}
    for item in _data_items(path):
        operand = (item.get('operands') or [None])[0]
        labels_by_operand.setdefault(operand, set()).add(item.get('trace_name_for_plotting'))

    shared = {k: v for k, v in labels_by_operand.items() if len(v) == 1}
    assert shared, 'no operand keeps one label across its operations'
    assert labels_by_operand.get('aortic_root/v') == {'v_{AR}'}


@pytest.mark.unit
@pytest.mark.parametrize('path', _authored_obs_data_files(),
                         ids=lambda p: os.path.basename(p))
def test_no_label_restates_its_own_operation(path):
    # `item_name_for_plotting`, not the retired `name_for_plotting`. The shipped files were
    # migrated to the #466 vocabulary, so the old key was always absent and this assertion
    # could never fire whatever the files said. The *item* label is the one that would restate
    # an operation, since it defaults to "<trace> (<operation>)".
    #
    # No shipped file trips this today -- the derived default cannot, so only a hand-written
    # label could. It is a guard on what someone adds next, which is why
    # `test_the_operation_check_catches_a_label_that_restates_itself` exists: without it a
    # clean corpus and a broken check look identical, which is exactly how this rotted.
    offenders = [
        (item.get('item_name_for_plotting'), item.get('operation'),
         (item.get('operands') or [''])[0])
        for item in _data_items(path)
        if item.get('operation') in _REDUCERS
        and _spells_its_own_operation(item.get('item_name_for_plotting'), item.get('operation'))
    ]
    assert not offenders, (
        f'{os.path.relpath(path, _ROOT)}: item_name_for_plotting names the output *before* the '
        f'operation (see resources/3compartment_obs_data.json, where mean and max of '
        f'aortic_root/v are both "v_{{AR}}"). These restate their own operation: '
        + '; '.join(f'{label!r} (operation {op!r} on {operand})'
                    for label, op, operand in offenders))


@pytest.mark.unit
@pytest.mark.parametrize('label,operation,caught', [
    ('mean(T_p1)', 'mean', True),        # functional application
    ('v_{max}', 'max', True),            # latex subscript
    ('v_max', 'max', True),              # plain subscript
    ('v_{AR}', 'mean', False),           # names the trace, not the operation
    ('mean_flow', 'mean', False),        # a quantity that merely contains the word
    (None, 'mean', False),               # absent label -- the state that made this vacuous
])
def test_the_operation_check_catches_a_label_that_restates_itself(label, operation, caught):
    """The sweep above scans a corpus that currently has nothing to find.

    A check with no violations to catch and a check that cannot catch anything look the same
    from the outside, and this one spent a release in the second state while reading as the
    first. This asserts the predicate itself still works.
    """
    assert _spells_its_own_operation(label, operation) is caught
