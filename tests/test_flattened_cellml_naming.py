"""The flattened-CellML cache is named per input model, not per run.

Regression tests for the /tmp leak: `_prepare_cellml_for_myokit_libcellml` used a
`NamedTemporaryFile(delete=False)` per call and nothing ever deleted the results, so a
machine accumulated one ~100 kB flattened model per simulation indefinitely.
"""

import os
import sys

import pytest

from libcuflynx.solver_wrappers.myokit_helper import (
    flattened_cellml_path,
    write_flattened_cellml,
)


@pytest.mark.unit
def test_the_same_input_always_gets_the_same_path(tmp_path):
    model = tmp_path / 'heart.cellml'
    model.write_text('<model/>')

    assert flattened_cellml_path(str(model)) == flattened_cellml_path(str(model))


@pytest.mark.unit
def test_a_relative_and_absolute_path_agree(tmp_path, monkeypatch):
    model = tmp_path / 'heart.cellml'
    model.write_text('<model/>')

    monkeypatch.chdir(tmp_path)
    assert flattened_cellml_path('heart.cellml') == flattened_cellml_path(str(model))


@pytest.mark.unit
def test_the_name_carries_the_input_stem(tmp_path):
    model = tmp_path / 'simple_physiological.cellml'
    model.write_text('<model/>')

    assert os.path.basename(flattened_cellml_path(str(model))).startswith(
        'simple_physiological_'
    )


@pytest.mark.unit
def test_two_models_sharing_a_basename_do_not_collide(tmp_path):
    # The whole reason the name is not just the stem: overwriting here would mean
    # silently simulating the wrong model.
    first = tmp_path / 'a' / 'heart.cellml'
    second = tmp_path / 'b' / 'heart.cellml'
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text('<model/>')

    assert flattened_cellml_path(str(first)) != flattened_cellml_path(str(second))


@pytest.mark.unit
def test_rewriting_overwrites_rather_than_accumulating(tmp_path):
    target = tmp_path / 'cache' / 'heart_deadbeef_flat.cellml'

    for text in ('<model>one</model>', '<model>two</model>', '<model>three</model>'):
        write_flattened_cellml(str(target), text)

    assert target.read_text() == '<model>three</model>'
    # One file, and no staging files left behind by the atomic swap.
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


@pytest.mark.unit
def test_a_failed_write_leaves_no_staging_file(tmp_path):
    target = tmp_path / 'cache' / 'heart_deadbeef_flat.cellml'
    write_flattened_cellml(str(target), '<model>good</model>')

    class Unwritable:
        def __str__(self):
            raise RuntimeError('boom')

    with pytest.raises(Exception):
        write_flattened_cellml(str(target), Unwritable())

    # The previous complete file survives, and nothing partial is left beside it.
    assert target.read_text() == '<model>good</model>'
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]
