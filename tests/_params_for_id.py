"""params_for_id files in tests: JSON (``libcuflynx.utilities.params_for_id_file``), written from
entries or from the CSV text a test lays its rows out in."""

from libcuflynx.utilities import params_for_id_file as pfi


def write_params_for_id(path, rows):
    """Write ``rows`` (a list of entries, or CSV text ``vessel_name,param_name,min,max,...``) as
    the params_for_id file ``path``; returns ``path``."""
    doc = pfi.csv_to_document(rows) if isinstance(rows, str) else rows
    pfi.write_params_for_id(str(path), doc)
    return path


def params_for_id_entries(path):
    """The entries of the params_for_id file ``path``."""
    return list(pfi.read_params_for_id(str(path)).get('params') or [])
