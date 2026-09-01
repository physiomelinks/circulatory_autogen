import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:  # optional dependency
    yaml = None
import json
import os, sys
import re
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# obs_data.json schema vocabularies
#
# Single source of truth for the value sets used by obs_data data_items, exposed
# via accessor functions so external tools (e.g. GUI editors) can populate their
# dropdowns by introspecting circulatory_autogen instead of hardcoding the lists
# — keeping them in sync as CA evolves. See get_valid_data_types() /
# get_valid_plot_types().
# ---------------------------------------------------------------------------

# Recognised data_item ``data_type`` values. ``timeseries`` is a deprecated
# alias for ``series`` and is intentionally not advertised here. Nor is
# ``prob_dist``, which was removed in issue #421: it described the shape of the
# ground truth rather than of the data, and an observable compared against a
# distribution is an ordinary ``constant`` whose cost_type takes
# ``prob_dist_params``.
VALID_DATA_TYPES = ("constant", "series", "frequency")

# Recognised ``plot_type`` values. ``None`` / ``""`` means "draw no marker".
VALID_PLOT_TYPES = (
    "horizontal",
    "vertical",
    "horizontal_from_min",
    "series",
    "frequency",
)


# The cost function a data_item gets when it does not name one. Single source of truth:
# PrimitiveParsers reads it rather than restating the literal, OMEXParsers.DEFAULT_COST_TYPE
# aliases it, and a front-end introspects it via get_default_cost_type() to label an empty
# cost-type picker honestly (CUFLynx #212) instead of hardcoding a fourth answer.
#
# gaussian_MLE, not MSE: CA used to give three different answers for the same question --
# MSE for an ordinary data_item, gaussian_MLE on OMEX import, and gaussian_MLE forced for
# Bayesian/MCMC (ensure_mle_cost_type_for_bayesian_inner). A default that changes depending
# on which door you came in is not a default. gaussian_MLE is the one that is already right
# for the probabilistic paths, and it uses the std a data_item is required to carry anyway.
DEFAULT_COST_TYPE = "gaussian_MLE"

# What the default used to be, so a run can tell the user what changed and how to pin it.
PREVIOUS_DEFAULT_COST_TYPE = "MSE"



#: obs_data entry keys superseded by the #466 vocabulary split, mapped to the key that now
#: carries the value. ``variable`` had two jobs -- it named the item *and* stood in as the operand
#: when ``operation`` was null -- so only its naming job moves here; the operand job moves to
#: ``operands``, which is why its advice names both.
LEGACY_OBS_ITEM_KEYS = {
    'variable': 'data_item_name',
    'name_for_plotting': 'trace_name_for_plotting',
}

LEGACY_OBS_KEY_ADVICE = {
    'variable': ("'variable' is deprecated: use 'data_item_name' for the item's identity -- it "
                 "must be unique, and it is what an operation_kwargs reference to another item "
                 "resolves against -- and 'operands' for the model variable the item reduces. "
                 "The old fallback, where a null 'operation' took its operand from 'variable', "
                 "has been removed."),
    'name_for_plotting': ("'name_for_plotting' is deprecated: it named two different things. Use "
                          "'trace_name_for_plotting' for the axis label of the trace, and "
                          "'item_name_for_plotting' for the item's own label (in sensitivity "
                          "tables and the like), which defaults to "
                          "'<trace_name_for_plotting> (<operation>)'."),
}


def obs_item_names(obs_info):
    """Each data_item's identity, i.e. what an operation_kwargs reference resolves against.

    One key, not a fallback chain. A dict assembled by hand elsewhere is brought into the
    current vocabulary by :func:`normalise_obs_info` at the boundary it arrives on, which is
    also where its author is told to move -- a silent fallback here told nobody anything, and
    kept the old spelling working indefinitely.

    Works on a ``prediction_info`` too, which is why that dict now spells its keys this way.
    """
    return _column(obs_info, "data_item_names")


def obs_item_labels(obs_info):
    """Each data_item's display label (the scalar feature). See ``obs_item_names``."""
    return _column(obs_info, "item_names_for_plotting")


def obs_trace_labels(obs_info):
    """Each data_item's trace label (the series it is drawn from). See ``obs_item_names``."""
    return _column(obs_info, "trace_names_for_plotting")


def _column(info, key):
    """The column at *key* as a list, or ``[]``.

    Tested against ``None`` rather than for truthiness. Several obs_info values are numpy
    arrays -- the ground-truth and std vectors, and the parameter bounds -- and
    ``array or []`` raises "the truth value of an array with more than one element is
    ambiguous". The keys these accessors read happen to be lists today, so the older
    spelling worked; it worked by luck, and a caller passing an array had no way to know.
    """
    value = (info or {}).get(key)
    return [] if value is None else list(value)


def obs_experiment_indices(info):
    """Which experiment each data_item belongs to, one per item.

    An accessor rather than a raw read for the same reason the label ones are: a downstream
    reader that goes through this is immune to the key being renamed, and CUFLynx reads this
    in two modules.
    """
    return _column(info, "experiment_idxs")


def obs_subexperiment_indices(info):
    """Which sub-experiment each data_item belongs to. See ``obs_experiment_indices``."""
    return _column(info, "subexperiment_idxs")


def obs_scalar_rows(info):
    """For each *constant* observable in order, the data_item row it came from.

    The bridge between two index spaces, and the one most easily misused: the cost vectors
    (``ground_truth_const``, ``std_const_vec``) are indexed by position in *this* list, while
    ``experiment_idxs`` and the labels are indexed by the values *in* it. Reading one with the
    other's counter is silently a different observable (#349).
    """
    return _column(info, "const_idx_to_obs_idx")


def obs_operand_lists(info):
    """Each item's operands -- the model variables it reduces -- as a list per item.

    Works on an ``obs_info`` and on a ``prediction_info`` alike, which is why
    ``prediction_info`` stores a list per item rather than the bare qname it used to.
    """
    return _column(info, "operands")


def migrate_legacy_obs_item_keys(items, where='data_items', variable_was_the_operand=False):
    """Rewrite an obs_data entry's superseded key names, warning once per key per file.

    Runs before schema validation, so everything downstream -- the series hydration helpers, the
    schema, ``process_obs_info`` -- sees only the current vocabulary. Returns a new list; the
    caller's dicts are not mutated, because an obs_data dict passed in by a user (or by
    ``ObsDataCreator``) is theirs, not ours.

    An entry that sets both a legacy key and its replacement is an error rather than a
    precedence rule: there is no reading of that which is not a mistake, and silently picking
    one would fit whichever the author did not mean.

    ``variable_was_the_operand`` is for ``prediction_items``, where the legacy ``variable`` held
    the model qname and there was no ``operands`` key at all -- so it seeds ``operands`` too.
    A data_item already states ``operands``, so there it only supplies the name.
    """
    if not isinstance(items, (list, tuple)):
        return items

    migrated = []
    seen_legacy = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            migrated.append(item)
            continue
        item = dict(item)
        for old, new in LEGACY_OBS_ITEM_KEYS.items():
            if old not in item:
                continue
            if new in item:
                raise ValueError(
                    f"{where}[{idx}] sets both '{old}' and its replacement '{new}'. "
                    f"Remove '{old}'. {LEGACY_OBS_KEY_ADVICE[old]}")
            item[new] = item.pop(old)
            seen_legacy.add(old)
        if (variable_was_the_operand and 'data_item_name' in item
                and not item.get('operands')):
            item['operands'] = [item['data_item_name']]
        migrated.append(item)

    for old in sorted(seen_legacy):
        warnings.warn(f"{where}: {LEGACY_OBS_KEY_ADVICE[old]}", DeprecationWarning, stacklevel=3)
    return migrated

#: Superseded keys of the parsed ``obs_info`` dict. This is a DIFFERENT layer from
#: :data:`LEGACY_OBS_ITEM_KEYS`, which renames keys of an obs_data *entry* -- the file a user
#: writes. These rename keys of the dict the parser hands the engine.
LEGACY_OBS_INFO_KEYS = {
    'obs_names': 'data_item_names',
    'names_for_plotting': 'item_names_for_plotting',
}

#: The same for ``prediction_info`` -- and the targets are deliberately NOT the same.
#:
#: ``names_for_plotting`` appears in both tables and means something different in each: the
#: *item* label in an obs_info, the *trace* label in a prediction_info. Merging the two tables
#: would silently move a trace label into the item-label slot, which is the kind of mistake that
#: shows up as a mislabelled axis months later rather than as a failure. ``names`` is worse: it
#: holds a model qname, so mapping it to ``data_item_names`` by analogy with obs_info's
#: ``obs_names`` would feed qnames to the uniqueness check while the solver was handed
#: identities. ``test_the_two_info_tables_disagree_on_purpose`` pins this apart.
LEGACY_PREDICTION_INFO_KEYS = {
    'names': 'operands',
    'names_for_plotting': 'trace_names_for_plotting',
}

LEGACY_INFO_KEY_ADVICE = {
    'obs_names': ("obs_info['obs_names'] is deprecated: use 'data_item_names', which is the "
                  "same list under the name the obs_data entry uses ('data_item_name')."),
    'names': ("prediction_info['names'] is deprecated: use 'operands', which holds the same "
              "model qnames as a list per item, matching obs_info['operands']."),
    'obs_info.names_for_plotting': (
        "obs_info['names_for_plotting'] is deprecated: it was the *item* label, so use "
        "'item_names_for_plotting'. The trace's axis label is 'trace_names_for_plotting'."),
    'prediction_info.names_for_plotting': (
        "prediction_info['names_for_plotting'] is deprecated: it was the *trace* label, so use "
        "'trace_names_for_plotting'. The item's own label is 'item_names_for_plotting'."),
}


def _advice(where, old):
    """The advice for *old*, disambiguated by dict when the key means two things."""
    return LEGACY_INFO_KEY_ADVICE.get(f'{where}.{old}') or LEGACY_INFO_KEY_ADVICE[old]


def _normalise_info(info, table, where):
    """Rewrite a parsed info dict's superseded keys, warning once per key.

    The obs_info twin of :func:`migrate_legacy_obs_item_keys`, and deliberately the same
    bargain: a copy rather than a mutation, because a dict handed in by a caller is theirs;
    an error rather than a precedence rule when both spellings are set, because no reading of
    that is not a mistake; and one warning per key, not per element.

    ``None`` passes through -- ``prediction_info`` is genuinely optional, and a caller that
    has none should not have to say so twice.
    """
    if info is None or not isinstance(info, dict):
        return info

    out = dict(info)
    seen_legacy = []
    for old, new in table.items():
        if old not in out:
            continue
        if new in out:
            raise ValueError(
                f"{where} sets both '{old}' and its replacement '{new}'. "
                f"Remove '{old}'. {_advice(where, old)}")
        out[new] = out.pop(old)
        seen_legacy.append(old)

    # `names` held a flat list of qnames; `operands` is a list *per item*, as it already is on
    # obs_info. Renaming the key without reshaping would leave one name covering two shapes,
    # which is the confusion this whole change exists to remove.
    if 'names' in table and 'names' in seen_legacy:
        out['operands'] = [list(n) if isinstance(n, (list, tuple)) else [n]
                           for n in out['operands']]

    for old in seen_legacy:
        warnings.warn(_advice(where, old), DeprecationWarning, stacklevel=3)
    return out


def normalise_obs_info(obs_info):
    """An ``obs_info`` in the current vocabulary. See :func:`_normalise_info`."""
    return _normalise_info(obs_info, LEGACY_OBS_INFO_KEYS, 'obs_info')


def normalise_prediction_info(prediction_info):
    """A ``prediction_info`` in the current vocabulary. See :func:`_normalise_info`."""
    return _normalise_info(prediction_info, LEGACY_PREDICTION_INFO_KEYS, 'prediction_info')


def get_default_cost_type():
    """The ``cost_type`` a data_item gets when it does not specify one."""
    return DEFAULT_COST_TYPE


def get_valid_data_types():
    """Return the recognised obs_data ``data_type`` values (excluding deprecated aliases)."""
    return list(VALID_DATA_TYPES)


def get_valid_plot_types():
    """Return the recognised obs_data ``plot_type`` values (excluding ``None``)."""
    return list(VALID_PLOT_TYPES)


class ObsDataCreator:
    """Builder for the observation-data structure used by calibration and SA.

    Produces the same structure as an ``obs_data.json`` file, in memory. Add the
    protocol info first, then one data item per observable, then retrieve the
    dict with [`get_obs_data_dict`][utilities.obs_data_helpers.ObsDataCreator.get_obs_data_dict]
    (or write it to disk with ``dump_to_path``)::

        obs = ObsDataCreator()
        obs.add_protocol_info(pre_times, sim_times, params_to_change)
        obs.add_data_item(entry)
        obs_data_dict = obs.get_obs_data_dict()

    The result is consumed by
    [`CVS0DParamID.set_ground_truth_data`][param_id.paramID.CVS0DParamID.set_ground_truth_data]
    and
    [`SensitivityAnalysis.set_ground_truth_data`][sensitivity_analysis.sensitivityAnalysis.SensitivityAnalysis.set_ground_truth_data].
    """

    def __init__(self):
        self.obs_data_dict = {}
        self.obs_data_dict['protocol_info'] = {}
        self.obs_data_dict['prediction_items'] = []
        self.obs_data_dict['data_items'] = []

    def add_protocol_info(self, pre_times, sim_times, params_to_change,
                          experiment_labels=None, offline_pre_time=None):
        """
        Add protocol information to the dictionary.
        pre_times: list of pre-simulation times for each experiment
        sim_times: 2D list of lists of simulation times for each experiment and subexperiment
        params_to_change: dictionary with parameter names as keys and list of lists of values Each parameter should have a value
        entry the same shape as sim_times.
        experiment_labels: list of labels for each experiment
        offline_pre_time: optional scalar; unlogged warmup before experiments (see parameter-identification docs)
        """
        # check pre_times is list and sim_times 2D list of lists
        if not isinstance(pre_times, list):
            raise ValueError("pre_times should be a list")
        if not isinstance(sim_times, list) or not all(isinstance(sublist, list) for sublist in sim_times):
            raise ValueError("sim_times should be a 2D list of lists")
        # check sizes of lists are correct
        num_exps = len(sim_times)
        if len(pre_times) != num_exps:
            raise ValueError("pre_times should have the same length as the number of experiments (number of rows of sim_times).")
        if not all(isinstance(entry, list) for entry in sim_times):
            raise ValueError("sim_times should be a 2D list with one row for each experiment and a column for each subexperiment.")
        if experiment_labels is not None:
            if len(experiment_labels) != num_exps:
                raise ValueError("experiment_labels should have the same length as the number of experiments (number of rows of sim_times).")
        else:
            # if experiment_labels is not provided, create default labels
            experiment_labels = [f'exp_{i}' for i in range(num_exps)]
        if type(params_to_change) is not dict:
            raise ValueError("params_to_change should be a dictionary with parameter names as keys and lists of values as values.")
        for param, values in params_to_change.items():
            if len(values) != num_exps:
                raise ValueError(f"Parameter {param} should have the same number of values as the number of experiments ({num_exps}).")
            if not all(isinstance(v, list) for v in values):
                raise ValueError(f"Parameter {param} should have a list of values for each subexperiment.")
            if not all(len(v) == len(sim_times[i]) for i, v in enumerate(values)):
                raise ValueError(f"Parameter {param} should have the same number of values as the number of subexperiments for each experiment.")
        
        # now add to dict
        self.obs_data_dict['protocol_info']['pre_times'] = pre_times
        self.obs_data_dict['protocol_info']['sim_times'] = sim_times
        self.obs_data_dict['protocol_info']['params_to_change'] = params_to_change
        self.obs_data_dict['protocol_info']['experiment_labels'] = experiment_labels
        if offline_pre_time is not None:
            self.obs_data_dict['protocol_info']['offline_pre_time'] = float(offline_pre_time)

    def add_prediction_item(self, variable, unit, experiment_idx):
        """
        Add a prediction item to the dictionary.
        variable: name of the variable to predict
        unit: unit of the variable
        experiment_idx: index of the experiment this prediction item belongs to
        """
        # check that experiment_idx is valid
        if experiment_idx < 0 or experiment_idx >= len(self.obs_data_dict['protocol_info']['sim_times']):
            raise ValueError(f"experiment_idx {experiment_idx} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")

        prediction_item = {
            'variable': variable,
            'unit': unit,
            'experiment_idx': experiment_idx
        }
        self.obs_data_dict['prediction_items'].append(prediction_item)

    #TODO Create functions for adding entries to the data_items
    def add_data_item(self, entry):
        """
        Add a data item to the dictionary.
        entry: dictionary containing the data item

        ``operation_kwargs`` (optional, default ``{}``) is a dict of keyword arguments passed to
        the ``operation`` func, on top of the ``operands`` it receives positionally, i.e.
        ``operation(*operands, **operation_kwargs)``. Keys must be keyword arguments of that func
        (an unknown key raises), and ``series_output`` is reserved for circulatory_autogen. A
        string value that matches the ``data_item_name`` of an earlier data item is replaced at
        run time by that observable's computed value. See issues #304 and #466 and the
        parameter-identification tutorial page.
        """
        required_keys = ['data_item_name', 'operands', 'unit', 'value', 'std']
        required_series_keys = ['obs_dt']
        optional_keys = ['trace_name_for_plotting', 'item_name_for_plotting', 'operation',
                         'operation_kwargs', 'cost_kwargs', 'weight', 'std',
                         'experiment_idx', 'subexperiment_idx']

        if 'operation_kwargs' in entry and not isinstance(entry['operation_kwargs'], dict):
            raise ValueError(
                f"'operation_kwargs' must be a dict of keyword arguments for the 'operation' "
                f"func, got {type(entry['operation_kwargs']).__name__}.")

        if 'cost_kwargs' in entry and not isinstance(entry['cost_kwargs'], dict):
            raise ValueError(
                f"'cost_kwargs' must be a dict of keyword arguments for the 'cost_type' "
                f"func, got {type(entry['cost_kwargs']).__name__}.")

        # `variable` did two jobs and is gone (#466); accept it for now so an existing script
        # keeps working, and say which key each of its jobs moved to.
        for legacy, current in (('variable', 'data_item_name'),
                                ('name_for_plotting', 'trace_name_for_plotting')):
            if legacy in entry:
                if current in entry:
                    raise ValueError(
                        f"data item sets both '{legacy}' and its replacement '{current}'. "
                        f"Remove '{legacy}'.")
                warnings.warn(LEGACY_OBS_KEY_ADVICE[legacy], DeprecationWarning, stacklevel=2)
                entry[current] = entry.pop(legacy)

        # Caught here rather than at parse time, which is where the uniqueness rule is
        # enforced: by then the offending call is long gone, and the message can only name the
        # collision, not the line that made it (#466).
        existing = {i.get('data_item_name') for i in self.obs_data_dict.get('data_items', [])}
        existing |= {i.get('data_item_name')
                     for i in self.obs_data_dict.get('prediction_items', [])}
        if entry.get('data_item_name') in existing:
            raise ValueError(
                f"data_item_name {entry['data_item_name']!r} is already used by another item. "
                f"Each item needs its own name; a shared spelling goes in "
                f"'trace_name_for_plotting' instead.")

        if 'trace_name_for_plotting' not in entry:
            operands = entry.get('operands') or []
            entry['trace_name_for_plotting'] = str(operands[0]) if operands \
                else entry['data_item_name']
        # check that trace_name_for_plotting only has one _ in it and remove if not
        if entry['trace_name_for_plotting'].count('_') > 1:
            print('Warning: trace_name_for_plotting contains multiple underscores, replacing with \_')
            entry['trace_name_for_plotting'] = re.sub('_', r'\_', entry['trace_name_for_plotting'])
        if 'operation' not in entry:
            entry['operation'] = None # default to None if not provided
        if 'weight' not in entry:
            entry['weight'] = 1.0 # default to 1.0 if not provided

        if 'subexperiment_idx' not in entry:
            entry['subexperiment_idx'] = 0  # default to 0 if not provided
        if 'experiment_idx' not in entry:
            entry['experiment_idx'] = 0  # default to 0 if not provided
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"Entry is missing required key: {key}")
        # check if value is a list or array and asign data_type accordingly
        if 'data_type' not in entry:
            if type(entry['value']) is list or type(entry['value']) is np.ndarray:
                entry['data_type'] = 'series'
                if 'obs_dt' in entry.keys():
                    pass
                elif 'dt' in entry.keys():
                    print("Warning: 'dt' for the time step of series data items is deprecated, ",
                        "please use 'obs_dt' instead. Setting 'obs_dt' to 'dt'.")
                    entry['obs_dt'] = entry['dt']
                    pass
                else:
                    raise ValueError(f"obs_dt is required for series entries")
            else:
                entry['data_type'] = 'constant'


        if self.obs_data_dict['protocol_info'] != {}:
            # check that experiment_idx and subexperiment_idx are valid if there is a protocol_info
            if entry['experiment_idx'] < 0 or entry['experiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times']):
                raise ValueError(f"experiment_idx {entry['experiment_idx']} is out of bounds for the number of experiments ({len(self.obs_data_dict['protocol_info']['sim_times'])}).")
            if entry['subexperiment_idx'] < 0 or entry['subexperiment_idx'] >= len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']]):
                raise ValueError(f"subexperiment_idx {entry['subexperiment_idx']} is out of bounds for the number of subexperiments in experiment {entry['experiment_idx']} ({len(self.obs_data_dict['protocol_info']['sim_times'][entry['experiment_idx']])}).")

        for key in entry.keys():
            if isinstance(entry[key], np.ndarray):
                entry[key] = entry[key].tolist()
            elif isinstance(entry[key], np.generic):
                entry[key] = entry[key].item()

        self.obs_data_dict['data_items'].append(entry)
    
    def get_obs_data_dict(self):
        """
        Returns the observation data dictionary.
        """
        return self.obs_data_dict

    def dump_to_path(self, output_path):
        """
        Dumps the observation data dictionary to a JSON file.
        """
        with open(output_path, 'w') as f:
            obs_data_dict = dict(self.obs_data_dict)
            if not obs_data_dict.get('prediction_items'):
                obs_data_dict.pop('prediction_items', None)
            json.dump(obs_data_dict, f, indent=2)
        print(f"Observation data dumped to {output_path}")
    
    def load_from_json_file(self, input_path):
        """
        Loads the observation data dictionary from a JSON file.
        input_path: path to the JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        self.obs_data_dict = data
        print(f"Observation data loaded from {input_path}")
        return data


def fill_protocol_info(
    obs_data: dict[str, Any] | list | None, protocol_info: dict[str, Any]
) -> dict[str, Any]:
    """Put ``protocol_info`` into an obs_data document, returning a new dict.

    An existing document's labels and colours are kept when they still fit the
    new schedule: those are the parts a user writes for themselves ("1 Hz
    pacing" reads better than "pacing, period 1000"), and re-deriving the timings
    is no reason to throw them away.

    A bare array of data_items -- the other accepted shape, and what the
    3compartment / heat_fenics studies ship -- becomes the object form carrying
    those same items, which is the only shape that can hold a protocol_info at
    all. ``dict(obs_data or {})`` used to raise on one, so a data-only file died
    rather than being updated.

    Here rather than beside the ``.mmt`` reader that first needed it: every key
    it writes -- ``protocol_info``, ``data_items``, ``experiment_labels``,
    ``experiment_colors`` -- is this module's vocabulary, and this is the module
    that migrates those names when they change (see
    :func:`migrate_legacy_obs_item_keys`). A copy of this living beside one
    *producer* of a protocol_info would be a second place to update and a
    second thing to forget; the EasyML reader produces one too, and so could
    anything else.
    """
    if isinstance(obs_data, list):
        obs_data = {"data_items": obs_data}
    out = dict(obs_data or {})
    existing = out.get("protocol_info") or {}
    merged = dict(protocol_info)
    n = len(protocol_info.get("sim_times", []))
    for key in ("experiment_labels", "experiment_colors"):
        kept = existing.get(key)
        if isinstance(kept, list) and len(kept) == n:
            merged[key] = kept
    out["protocol_info"] = merged
    return out
