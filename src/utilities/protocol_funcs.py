"""
protocol_funcs.py — backward-compatible shim for protocol running utilities.

ProtocolRunner is now defined in src/protocol_runners/protocol_runner.py and
re-exported here so existing imports continue to work without change.

The module-level ``run_protocols`` function below is DEPRECATED.  Use the
ProtocolRunner class instead.
"""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), '..')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_ROOT_DIR = os.path.join(os.path.dirname(__file__), '../..')
_USER_INPUTS_DIR = os.path.join(_ROOT_DIR, 'user_run_files')

# Re-export ProtocolRunner from its canonical location.
from protocol_runners.protocol_runner import ProtocolRunner

__all__ = ['ProtocolRunner', 'run_protocols']


def _load_inp_data_dict(inp_data_dict=None):
    """Load user inputs YAML (with optional override), matching ProtocolRunner."""
    if inp_data_dict is not None:
        return inp_data_dict
    import yaml
    with open(os.path.join(_USER_INPUTS_DIR, 'user_inputs.yaml'), 'r') as fh:
        inp_data_dict = yaml.load(fh, Loader=yaml.FullLoader)
    override = inp_data_dict.get('user_inputs_path_override')
    if override:
        if os.path.exists(override):
            with open(override, 'r') as fh:
                inp_data_dict = yaml.load(fh, Loader=yaml.FullLoader)
        else:
            print(
                f"User inputs file not found at {override}\n"
                "Check user_inputs_path_override in user_inputs.yaml "
                "and set it to False to use the default location."
            )
            raise FileNotFoundError(override)
    return inp_data_dict


def _variable_index(var2idx, name):
    """Resolve a variable name to a result-list index (slash/dot tolerant)."""
    if name in var2idx:
        return var2idx[name]
    for alt in (name.replace('/', '.'), name.replace('.', '/')):
        if alt in var2idx:
            return var2idx[alt]
    raise KeyError(
        f"Variable {name!r} not found in model; "
        f"available names include: {list(var2idx.keys())[:8]}..."
    )


def run_protocols(model_path, variables_to_plot, protocol_info=None,
                  inp_data_dict=None, solver='CVODE_myokit'):
    """DEPRECATED: use ProtocolRunner instead.

    Run the protocols defined by *protocol_info* (or read from the obs-data JSON)
    and return time and result arrays for the requested variables.

    Delegates to :class:`ProtocolRunner` for simulation; only filters the
    full variable list down to *variables_to_plot* for backward compatibility.

    Parameters
    ----------
    model_path : str
    variables_to_plot : list[str]
    protocol_info : dict, optional
    inp_data_dict : dict, optional
        If None, loaded from user_run_files/user_inputs.yaml.
    solver : str, default 'CVODE_myokit'

    Returns
    -------
    t_list : list
        Time vector per experiment.
    res_list : list[list]
        Result arrays per experiment, one entry per name in *variables_to_plot*.
    """
    inp_data_dict = _load_inp_data_dict(inp_data_dict)

    runner = ProtocolRunner(
        model_path,
        inp_data_dict=inp_data_dict,
        solver=solver,
    )
    t_list, res_list_all, _ = runner.run_protocols(model_path, protocol_info)

    var2idx = runner.get_var2idx_dict()
    res_list = [
        [res_list_all[exp_idx][_variable_index(var2idx, name)]
         for name in variables_to_plot]
        for exp_idx in range(len(t_list))
    ]
    return t_list, res_list
