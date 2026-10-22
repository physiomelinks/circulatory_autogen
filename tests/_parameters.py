"""Parameters files in tests: nested JSON (``libcuflynx.utilities.parameters_file``), written from
rows or from the CSV text a test lays its rows out in."""

import csv
import io

from libcuflynx.utilities import parameters_file as pf

MODEL, MODULE = pf.MODEL, pf.MODULE


def rows_from_csv(text):
    """The rows of CSV text with a header (``variable_name,units,value,data_reference``...)."""
    reader = csv.DictReader(io.StringIO(text.strip() + '\n'))
    return [{k.strip(): (v or '').strip() for k, v in row.items() if k} for row in reader]


def write_parameters(path, rows, kind=MODEL):
    """Write ``rows`` (dicts, CSV text, or the text of a parameters file) as the parameters file
    ``path``; returns ``path``. In a parameterisation (``kind`` MODULE) a global is named
    ``global/<variable>``."""
    if isinstance(rows, str) and rows.lstrip().startswith('{'):
        with open(path, 'w') as f:
            f.write(rows)
        return path
    if isinstance(rows, str):
        rows = rows_from_csv(rows)
    pf.write_parameter_rows(str(path), list(rows), kind)
    return path


def read_parameters(path, kind=MODEL):
    """The rows of the parameters file ``path``, named as a user writes them."""
    return pf.read_parameter_rows(str(path), kind=kind)


def parameter_values(path, kind=MODEL):
    """``{name: value text}`` of the parameters file ``path``."""
    return {r['variable_name']: r['value'] for r in read_parameters(path, kind)}
