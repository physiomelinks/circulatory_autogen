"""CellML connection uniqueness in the generated model (issue #343).

CellML allows at most one ``<connection>`` per component pair (spec 2.15.4, CONNECTION_UNIQUE)
and at most one ``<map_variables>`` per variable pair inside it (2.16.3, MAP_VARIABLES_UNIQUE).
A generator that writes connections as it goes violates both easily, and libCellML then refuses
the model -- which is what issue #343 reported on a 37-module cardiac electrophysiology model.

Three distinct ways it happened there, all covered below:

1. the same pair written from two module rows in *opposite* order -- (A, B) and (B, A) are the
   same connection to CellML, so an accumulator keyed on the ordered tuple misses it;
2. unit-converter connections, written through call sites that bypassed the per-row accumulator
   entirely (there are ~27 direct callers);
3. the same variable pair appended twice inside one connection.

These exercise the accumulator directly rather than through a full model: the reporter's model
needs 37 custom module types, and none of the shipped fixtures reproduce the bug.
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'src'))

from generators.CVSCellMLGenerator import CVS0DCellMLGenerator


class _Writer:
    """Minimal stand-in for the file handle the generator writes to."""

    def __init__(self):
        self.text = ""

    def writelines(self, lines):
        self.text += "".join(lines)


def _generator():
    """A generator with only the connection accumulator initialised.

    __init__ needs a parsed model, none of which is involved in connection bookkeeping.
    """
    gen = object.__new__(CVS0DCellMLGenerator)
    gen._reset_connections()
    return gen


def _flush(gen):
    wf = _Writer()
    getattr(gen, '_CVS0DCellMLGenerator__flush_connections')(wf)   # name-mangled private
    return wf.text


def _connections(text):
    out = []
    for block in re.findall(r'<connection>(.*?)</connection>', text, re.S):
        m = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block)
        pairs = re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block)
        out.append((m.groups(), pairs))
    return out


@pytest.mark.unit
def test_a_component_pair_written_in_both_orders_gives_one_connection():
    """(A, B) then (B, A) is one connection to CellML, not two.

    This is the reporter's 'Cad_module' <-> 'IcaT_module' error: each module row writes its own
    outgoing mappings, so a bidirectional pair got written once from each side.
    """
    gen = _generator()
    gen._add_connection('Cad_module', 'IcaT_module', [('Ca', 'Ca')])
    gen._add_connection('IcaT_module', 'Cad_module', [('i_CaT', 'i_CaT')])

    conns = _connections(_flush(gen))
    assert len(conns) == 1, conns
    comps, pairs = conns[0]
    assert set(comps) == {'Cad_module', 'IcaT_module'}
    assert len(pairs) == 2      # both mappings survive the merge


@pytest.mark.unit
def test_variables_from_a_reversed_call_are_swapped_to_match_the_orientation():
    """Merging a reversed call must swap its variables, or it connects the wrong ones.

    The merged block keeps the orientation of whichever call created it, so a later call arriving
    with the components the other way round describes its variables in the opposite order. Keeping
    them as given would map variable_1 of one component onto a variable living in the other.
    """
    gen = _generator()
    gen._add_connection('A', 'B', [('a_var', 'b_var')])
    gen._add_connection('B', 'A', [('b_other', 'a_other')])

    comps, pairs = _connections(_flush(gen))[0]
    assert comps == ('A', 'B')
    assert ('a_var', 'b_var') in pairs
    assert ('a_other', 'b_other') in pairs     # re-oriented: A's variable first


@pytest.mark.unit
def test_a_repeated_variable_pair_is_mapped_only_once():
    """MAP_VARIABLES_UNIQUE (2.16.3): the reporter's 'Nai' <-> 'Nai' error.

    Merging connections makes this visible even where the duplicate was previously harmless, so
    deduplicating the pairs is part of the same fix rather than a separate nicety.
    """
    gen = _generator()
    gen._add_connection('Naic2_module', 'IcaL_module', [('Nai', 'Nai')])
    gen._add_connection('Naic2_module', 'IcaL_module', [('Nai', 'Nai'), ('Ki', 'Ki')])

    _comps, pairs = _connections(_flush(gen))[0]
    assert pairs.count(('Nai', 'Nai')) == 1, pairs
    assert ('Ki', 'Ki') in pairs


@pytest.mark.unit
def test_repeated_unit_converter_connections_collapse_to_one():
    """The reporter's 'NTS_module' <-> 'unit_converter_nM_to_millimolar', which appeared 4x.

    Unit-converter connections are written from call sites that bypassed the old per-row
    accumulator, so every conversion emitted its own block.
    """
    gen = _generator()
    for var in ('ACh', 'NE', 'NPY'):
        gen._add_connection('NTS_module', 'unit_converter_nM_to_millimolar', [(var, var)])

    conns = _connections(_flush(gen))
    assert len(conns) == 1, conns
    assert len(conns[0][1]) == 3


@pytest.mark.unit
def test_blank_variable_names_are_skipped():
    """Callers pass ragged lists where a slot may be empty; an empty name is not a mapping."""
    gen = _generator()
    gen._add_connection('A', 'B', [('u', 'u'), ('', 'v'), ('w', None)])
    _comps, pairs = _connections(_flush(gen))[0]
    assert pairs == [('u', 'u')]


@pytest.mark.unit
def test_a_connection_with_no_usable_pairs_is_not_emitted():
    gen = _generator()
    gen._add_connection('A', 'B', [('', '')])
    assert _connections(_flush(gen)) == []


@pytest.mark.unit
def test_flushing_clears_the_accumulator_so_the_next_file_starts_empty():
    """The generator writes several CellML files per run; connections must not leak between them."""
    gen = _generator()
    gen._add_connection('A', 'B', [('u', 'u')])
    assert len(_connections(_flush(gen))) == 1
    assert _connections(_flush(gen)) == []


@pytest.mark.integration
def test_generated_model_has_no_duplicate_connections(generated_cellml_model_factory):
    """End-to-end guard: a generated model must satisfy both uniqueness rules.

    The unit tests pin the accumulator; this pins the property that actually matters, so a future
    writer added outside the accumulator is caught.
    """
    import collections

    cellml_path = generated_cellml_model_factory(
        "3compartment", "3compartment_parameters.csv", solver="CVODE_myokit")
    with open(cellml_path) as f:
        text = f.read()

    seen = collections.Counter()
    dup_pairs = []
    for block in re.findall(r'<connection>(.*?)</connection>', text, re.S):
        m = re.search(r'component_1="([^"]+)"\s+component_2="([^"]+)"', block)
        if not m:
            continue
        seen[frozenset(m.groups())] += 1
        mapped = re.findall(r'variable_1="([^"]+)"\s+variable_2="([^"]+)"', block)
        repeats = [k for k, v in collections.Counter(mapped).items() if v > 1]
        if repeats:
            dup_pairs.append((m.groups(), repeats))

    duplicated = {tuple(sorted(k)): v for k, v in seen.items() if v > 1}
    assert not duplicated, f"component pairs with more than one <connection>: {duplicated}"
    assert not dup_pairs, f"connections with a repeated <map_variables>: {dup_pairs}"
