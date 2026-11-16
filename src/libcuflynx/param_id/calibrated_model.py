"""The calibrated model: a model's CellML with a calibration's best fit written in.

A finished calibration writes ``<prefix>_calibrated.cellml`` at the top of the study's working
dir (its ``param_id_output_dir``), where the study manifest finds it as the study's ``model``
(``cuflynx_study._calibrated_model``). ``CVS0DParamID.write_calibrated_model`` writes it for a
calibration engine; :func:`write_calibrated_model` takes the parts, for a caller holding them
without one.
"""
import numpy as np

from libcuflynx.parsers.PrimitiveParsers import expand_modifier_param_vals
from libcuflynx.solver_wrappers.param_grouping import pair_names_with_values
from libcuflynx.utilities.utility_funcs import change_parameter_values_and_save

__all__ = ['CALIBRATED_MODEL_SUFFIX', 'calibrated_model_name', 'calibrated_parameter_values',
           'write_calibrated_model']

CALIBRATED_MODEL_SUFFIX = '_calibrated.cellml'


def calibrated_model_name(file_prefix):
    """The calibrated model's file name: ``<file_prefix>_calibrated.cellml``."""
    return f'{file_prefix}{CALIBRATED_MODEL_SUFFIX}'


def calibrated_parameter_values(param_id_info, best_param_vals):
    """``(names, values)``: each model parameter (``component/variable``) the best fit sets.

    A row naming several vessels gives each of them its one value, and a modifier row gives
    each target the value its modifier function makes of theta (``expand_modifier_param_vals``,
    which needs the row's baselines resolved, as a built calibration engine has them).
    """
    values = np.asarray(best_param_vals, dtype=float).ravel().tolist()
    expanded = expand_modifier_param_vals(param_id_info, values)
    names, out = [], []
    for row_names, value in zip(param_id_info['param_names'], expanded):
        for name, member_value in pair_names_with_values(row_names, value):
            names.append(str(name))
            out.append(float(member_value))
    return names, out


def write_calibrated_model(cellml_path, param_id_info, best_param_vals, output_path):
    """Write ``cellml_path`` with ``best_param_vals`` set to ``output_path``; returns it.

    The model is flattened (its imports resolved) and each parameter's initial value replaced
    (``utility_funcs.change_parameter_values_and_save``, which names a parameter it cannot
    find).
    """
    names, values = calibrated_parameter_values(param_id_info, best_param_vals)
    change_parameter_values_and_save(cellml_path, names, values, output_path)
    return output_path
