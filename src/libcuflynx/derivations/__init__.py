'''
Parameter derivations: how parameters are calculated from experimental data.

A derivations file, ``<name>_derivations.json`` beside the ``<name>_parameters.json`` it feeds
(schema ``schemas/derivations.schema.json``), lists derivations. Each reads inputs and sets
output parameters by one method::

    {"derivations": [
      {"name": "R_T0",
       "inputs": {"MAP": {"data_item": "MAP"}, "CVP": {"input_item": "CVP_Smith2001"},
                  "Q": {"parameter": "Systemic_T/Q_T0"}},
       "outputs": {"R_T0": "Systemic_T/R_T0"},
       "cellml": "derivations/systemic_resistance.cellml"},
      {"name": "alpha_eff",
       "inputs": {"FF0": {"input_item": "FF_Montanari2003"}},
       "outputs": {"alpha_eff": "renal/alpha_eff"},
       "python": "derivation_funcs.py:renal_alpha_eff"}]}

* An input is a model parameter (its path in the parameters file, possibly another derivation's
  output, so derivations chain), an obs_data ``input_item`` (a number never scored), or an
  obs_data ``data_item`` (a calibration target that is also an input). Its units are the
  source's.
* An output is a parameter of the parameters file. Each parameter is set by at most one
  derivation. Its leaf keeps ``value`` as the cached result and names the derivation:
  ``{"value": 2.0367e8, "units": "J_s_per_m6", "derivation": "R_T0"}``.
* ``cellml`` (:mod:`.cellml_method`): a CellML derivation model, inputs constants and outputs
  computed constants, units checked by libCellML and converted where they differ.
  ``python`` (:mod:`.python_method`): ``file.py:function``, called ``f(inputs, units)``.

Derivations are for data only: nothing here runs in a simulation. ``cuflynx-derive`` evaluates a
file, compares each output with its cached value (``CACHE_RTOL``) and with ``--write`` updates
them.
'''

from libcuflynx.derivations.cellml_method import (DerivationModelError, evaluate_cellml,
                                                  load_derivation_model)
from libcuflynx.derivations.core import (CACHE_RTOL, DERIVATIONS_SUFFIX, Derivation,
                                         DerivationError, DerivationResult, check_derivations,
                                         derivation_files, derivations_file_for,
                                         evaluate_derivations, load_derivations, load_obs_items,
                                         order_derivations, parameter_path,
                                         parameters_file_for, read_parameters, resolve_inputs,
                                         unresolved_inputs, write_derived_values)
from libcuflynx.derivations.python_method import (DerivationFunctionError,
                                                  build_derivation_funcs_dict, evaluate_python)

__all__ = [
    'CACHE_RTOL', 'DERIVATIONS_SUFFIX', 'Derivation', 'DerivationError', 'DerivationFunctionError',
    'DerivationModelError', 'DerivationResult', 'build_derivation_funcs_dict',
    'check_derivations', 'derivation_files', 'derivations_file_for', 'evaluate_cellml',
    'evaluate_derivations', 'evaluate_python', 'load_derivation_model', 'load_derivations',
    'load_obs_items',
    'order_derivations', 'parameter_path', 'parameters_file_for', 'read_parameters',
    'resolve_inputs', 'unresolved_inputs', 'write_derived_values',
]
