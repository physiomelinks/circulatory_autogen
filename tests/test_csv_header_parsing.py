"""CSV inputs must be parsed by column header, not by column position (issue #159).

Position-based parsing fails silently and destructively: reorder two columns in a parameters CSV
and every value lands in the wrong field, producing a model that builds and runs and is simply
wrong. These tests pin the header-based behaviour by feeding the parsers column orders that
differ from the shipped files and requiring identical results.
"""
import pandas as pd
import pytest

from parsers.PrimitiveParsers import CSVFileParser


def _write(tmp_path, name, header, rows):
    path = tmp_path / name
    path.write_text('\n'.join([header] + rows) + '\n')
    return str(path)


@pytest.mark.unit
def test_parameters_csv_is_read_by_header_not_position(tmp_path):
    """Same rows, columns in a different order -> same parsed values."""
    parser = CSVFileParser()
    canonical = _write(
        tmp_path, 'canonical.csv',
        'variable_name,units,value,data_reference',
        ['R_pvn,Js_per_m6,1333000,Blanco_2013_Table_8',
         'q_C_init_pvn,m3,0.0001,user_defined'])
    shuffled = _write(
        tmp_path, 'shuffled.csv',
        'data_reference,value,variable_name,units',
        ['Blanco_2013_Table_8,1333000,R_pvn,Js_per_m6',
         'user_defined,0.0001,q_C_init_pvn,m3'])

    a = parser.get_data_as_dataframe(canonical).set_index('variable_name')
    b = parser.get_data_as_dataframe(shuffled).set_index('variable_name')

    for var in ('R_pvn', 'q_C_init_pvn'):
        for col in ('units', 'value', 'data_reference'):
            assert a.loc[var, col] == b.loc[var, col], (
                f"{var}.{col} differs between column orderings -- the parser is using position")


@pytest.mark.unit
def test_params_for_id_csv_is_read_by_header_not_position(tmp_path):
    """The params_for_id file is the other CSV whose column meaning matters (min vs max)."""
    parser = CSVFileParser()
    canonical = _write(
        tmp_path, 'pid_canonical.csv',
        'vessel_name,param_name,param_type,min,max,name_for_plotting',
        ['global,q_lv_init,const,200e-6,1500e-6,q_sbv',
         'aortic_root,C,const,1e-9,5e-8,C_ao'])
    shuffled = _write(
        tmp_path, 'pid_shuffled.csv',
        'max,name_for_plotting,vessel_name,min,param_type,param_name',
        ['1500e-6,q_sbv,global,200e-6,const,q_lv_init',
         '5e-8,C_ao,aortic_root,1e-9,const,C'])

    a = parser.get_data_as_dataframe_multistrings(canonical)
    b = parser.get_data_as_dataframe_multistrings(shuffled)

    for col in ('vessel_name', 'param_name', 'param_type', 'min', 'max', 'name_for_plotting'):
        assert list(a[col]) == list(b[col]), (
            f"column '{col}' differs between orderings -- min/max could be swapped silently")


@pytest.mark.unit
def test_a_missing_required_column_is_visible(tmp_path):
    """Dropping a column must not quietly shift the remaining ones into its place."""
    parser = CSVFileParser()
    path = _write(tmp_path, 'no_units.csv',
                  'variable_name,value,data_reference',
                  ['R_pvn,1333000,Blanco_2013_Table_8'])
    df = parser.get_data_as_dataframe(path)
    assert 'units' not in df.columns
    # the values that *are* present stayed with their own headers
    assert df.loc[0, 'value'] == '1333000'
    assert df.loc[0, 'data_reference'] == 'Blanco_2013_Table_8'


@pytest.mark.unit
def test_header_whitespace_is_stripped(tmp_path):
    """The shipped params_for_id files are column-aligned with padding spaces, so a header
    lookup only works because the parser strips them."""
    parser = CSVFileParser()
    path = _write(tmp_path, 'padded.csv',
                  'vessel_name,                  param_name,       param_type',
                  ['global,                        q_lv_init,        const'])
    df = parser.get_data_as_dataframe_multistrings(path)
    assert list(df.columns) == ['vessel_name', 'param_name', 'param_type']
