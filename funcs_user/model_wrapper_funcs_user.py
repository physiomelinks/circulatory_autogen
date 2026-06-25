"""
Template wrapper for ``model_type: python_user_defined``.

Use this when you already have your own Python ODE model and want to run it
through circulatory_autogen's calibration / sensitivity / identifiability
pipelines without expressing it as a CVS0D vessel-array CSV.

HOW TO USE
----------
1. Copy this file to ``funcs_user/{file_prefix}_wrapper.py`` where ``file_prefix``
   matches the one in your ``user_inputs.yaml`` (e.g. ``oscillator`` ->
   ``funcs_user/oscillator_wrapper.py``). Alternatively keep it anywhere and set
   ``model_wrapper_path: /abs/path/to/your_wrapper.py`` in ``user_inputs.yaml``.
2. In ``user_inputs.yaml`` set::

       model_type: python_user_defined
       solver: user_defined

3. Fill in the three required pieces below (``PARAMETERS``, ``STATES``,
   ``OUTPUT_NAMES``) and the ``rhs`` function. Optionally add ``compute_outputs``
   for observables that are algebraic functions of the states/parameters rather
   than states themselves.

NAMING
------
Parameter / state / output names use the canonical ``component/variable`` form.
They MUST match:
  * ``PARAMETERS`` keys  <->  ``{vessel_name}/{param_name}`` in
                              ``resources/{file_prefix}_params_for_id.csv``
                              (vessel_name -> component, param_name -> variable),
  * ``OUTPUT_NAMES``     <->  the ``operands`` entries in
                              ``resources/{file_prefix}_obs_data.json``.

WHAT THE FRAMEWORK DOES
-----------------------
The framework integrates ``rhs`` with scipy ``solve_ivp`` (the same machinery the
generated-python backend uses) and handles parameter sweeping, pre_time spin-up,
per-experiment resets and result extraction. You only describe the model.

Only ``PARAMETERS`` values are swept during calibration / SA (they are the
constants referenced in ``rhs``). Initial conditions in ``STATES`` are fixed.
"""

# 1. Parameter defaults. The optimiser overrides those listed in
#    {file_prefix}_params_for_id.csv; the rest stay at these defaults.
PARAMETERS = {
    "model/k": 1.0,
}

# 2. Initial state values (the ODE state vector, in this order).
STATES = {
    "model/x": 1.0,
}

# 3. Observable outputs referenced by the obs_data operands. These may be state
#    names (read directly) and/or algebraic names produced by compute_outputs().
OUTPUT_NAMES = ["model/x"]


def rhs(t, y, params):
    """Right-hand side of the ODE system: dy/dt = rhs(t, y, params).

    Args:
        t: current time (float).
        y: list of state values, in the order of STATES.
        params: dict of all parameter values (PARAMETERS overlaid with the
            current calibration/SA values), keyed by the PARAMETERS names.

    Returns:
        list of derivatives, one per state, in the order of STATES.
    """
    x = y[0]
    k = params["model/k"]
    dx_dt = -k * x
    return [dx_dt]


# OPTIONAL. Delete if every OUTPUT_NAMES entry is a state name.
def compute_outputs(t, y, params):
    """Compute algebraic outputs that are NOT themselves states.

    Args:
        t: current time (float).
        y: list of state values, in the order of STATES.
        params: dict of all parameter values.

    Returns:
        dict mapping output names (a subset of OUTPUT_NAMES that are not states)
        to their scalar value at time t.
    """
    return {}
