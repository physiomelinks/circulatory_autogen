"""CasADi symbolic cost / gradient / observable evaluation for the param-id classes.

Every public function takes the param-id object as its first argument (``pid``), and reads
and writes state directly on it -- ``pid.cost_symb``, ``pid.cost_func`` and friends stay
instance attributes exactly as they were when this code lived in ``paramID.py``. Nothing
here imports ``param_id.paramID``: the dependency runs one way only, and the module is
importable on its own. This mirrors ``param_id.plot_outputs``, which is likewise called
with a param-id handle rather than importing the class.

``ParamID`` keeps thin delegating methods (``get_cost_ca``, ``get_jac_cost_ca``,
``get_obs_ca``, ``build_casadi_functions``) so external callers -- ``plot_outputs.py``, the
test-suite stubs that implement the same duck-typed contract -- are unaffected.

Generic methods this module calls back into on ``pid``:
  - ``get_cost_and_obs_from_params(param_vals, do_ad=True)``
  - ``get_obs_output_dict(obs_item, get_all_series, is_symbolic=True)``

CasADi is optional. The import is guarded here the same way ``param_id.math_backend`` and
``solver_wrappers`` guard theirs, so importing this module never fails; ``require_casadi()``
raises only once a CasADi path is actually taken.
"""
import hashlib

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None


#: Kept as a module constant so the solver factory and the run scripts can say the same thing
#: (#435). CasADi is an optional extra because it is 221 MB -- by far the largest thing in the
#: dependency set after autoemulate's torch -- and nothing but the casadi_python model type and
#: the symbolic-AD gradient path ever touches it.
CASADI_MISSING_MESSAGE = (
    "CasADi is required for symbolic or casadi_python workflows but is not installed. "
    'Install it with `pip install "libcuflynx[casadi]"` (it adds about 221 MB).')


def require_casadi():
    if ca is None:
        raise ImportError(CASADI_MISSING_MESSAGE)


def as_casadi_column(x):
    """Column vector, for either a numeric array or a casadi symbolic."""
    require_casadi()
    if isinstance(x, (np.ndarray, list, tuple)):
        return ca.DM(np.asarray(x, dtype=float).reshape(-1, 1))
    return ca.reshape(x, x.numel(), 1)


def content_hash(obj):
    """Stable content digest of nested lists/dicts/arrays, for cache keys.

    Not repr(): numpy truncates the repr of an array past ~1000 elements
    ('array([0, 1, 2, ..., 9997, 9998, 9999])'), so two different long observation series --
    exactly the thing this is used to distinguish -- can share a repr. Hash the raw bytes
    instead, tagging dtype and shape so arrays that differ only in those still separate.
    """
    digest = hashlib.blake2b(digest_size=16)

    def update(value):
        if isinstance(value, np.ndarray):
            digest.update(b'\x01' + str(value.dtype).encode() + str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (list, tuple)):
            digest.update(b'\x02')
            for item in value:
                update(item)
            digest.update(b'\x03')
        elif isinstance(value, dict):
            digest.update(b'\x04')
            for key in sorted(value, key=repr):
                update(key)
                update(value[key])
            digest.update(b'\x05')
        else:
            digest.update(b'\x06' + repr(value).encode())

    update(obj)
    return digest.hexdigest()


from libcuflynx.parsers.PrimitiveParsers import (expand_modifier_param_vals,
                                      modifier_weights_by_index, param_entry_labels)


def functions_cache_key(pid, param_names, get_all_series):
    """Signature of everything the CasADi symbolic graph is built from, so that the graph can
    be built once and re-evaluated at new parameter values.

    The free parameters enter the graph symbolically, so the key deliberately does NOT
    include their numeric values. Everything else does, because get_cost_and_obs_from_params
    bakes it into the SX graph as literal constants: the ground truth, the standard
    deviations, the per-experiment/sub weights, the cost type, and the params_to_change
    protocol values. Those are all reachable through public setters (set_ground_truth_data,
    set_obs_info, set_protocol_info) and by direct mutation of obs_info, so without them in
    the key a flow that re-points the data -- `set_ground_truth_data(A); run();
    set_ground_truth_data(B); run()`, staged sequential param-id, MCMC after calibration --
    would hit the cache and silently keep optimising against the *previous* data while
    reporting a plausible converged cost. Hashing them here rather than clearing the key in
    each setter also covers in-place mutation, which no setter can intercept.
    """
    pnames = tuple(n[0] if isinstance(n, (list, tuple)) else n for n in param_names)
    proto = pid.protocol_info or {}
    obs = pid.obs_info or {}
    data_sig = content_hash([
        obs.get('ground_truth_const'), obs.get('ground_truth_series'),
        obs.get('std_const_vec'), obs.get('std_series_vec'),
        obs.get('obs_dt'), obs.get('operands'), obs.get('operations'),
        obs.get('data_types'), obs.get('cost_type'), pid.cost_type,
        proto.get('scaled_weight_const_from_exp_sub'),
        proto.get('scaled_weight_series_from_exp_sub'),
        proto.get('params_to_change'),
    ])
    return (pnames, bool(get_all_series), float(pid.dt),
            repr(proto.get('sim_times')), repr(proto.get('pre_times')),
            repr(pid.solver_info.get('method') if pid.solver_info else None),
            data_sig)


def flatten_entries(param_id_info):
    """``(flat_member_names, per_entry_[(offset, weight), ...])`` for the params_for_id entries.

    One calibrated variable (theta) may govern several model constants: a grouped row shares
    one value across its members, a scale modifier applies ``theta * baseline_i``. The CasADi
    graph carries one SX symbol per *model constant*, so the jacobian is taken over every
    member and folded back per entry with ``d p_i/d theta`` -- the same affine weights the
    Myokit/FSA arm uses (``modifier_weights_by_index``: 1.0 for a shared-value group, the
    probed coefficient for a modifier). Both backends therefore answer the same question.
    """
    weights = modifier_weights_by_index(param_id_info)
    flat, entries = [], []
    for j, names in enumerate(param_id_info["param_names"]):
        members = list(names) if isinstance(names, (list, tuple)) else [names]
        w = weights.get(j)
        entry = []
        for i, member in enumerate(members):
            entry.append((len(flat), 1.0 if w is None else float(w[i])))
            flat.append(member)
        entries.append(entry)
    return flat, entries


def fold_entry_rows(matrix, entry_map, n_entries):
    """Fold a per-member jacobian (rows or columns indexed by member) into per-entry values.

    ``matrix`` is indexed ``[..., member]``; returns the same array with the last axis
    replaced by one column per calibrated variable, each the weighted sum over its members.
    """
    matrix = np.asarray(matrix, dtype=float)
    out = np.zeros(matrix.shape[:-1] + (n_entries,), dtype=float)
    for j, members in enumerate(entry_map):
        for offset, weight in members:
            out[..., j] += weight * matrix[..., offset]
    return out


def build_functions(pid, param_names, param_vals=None, get_all_series=False):
    require_casadi()
    # Flat member names + expanded values: a modifier's theta becomes fn(theta, baseline_i)
    # per target, exactly as the numeric protocol run does (expand_modifier_param_vals), so
    # the symbolic evaluation point and the simulated one are the same point.
    flat_names, entry_map = flatten_entries(pid.param_id_info)
    pid._casadi_entry_map = entry_map
    flat_vals = None
    if param_vals is not None:
        expanded = expand_modifier_param_vals(pid.param_id_info, param_vals)
        flat_vals = []
        for value in expanded:
            flat_vals.extend(value if isinstance(value, (list, tuple)) else [value])
    # Always refresh the numeric evaluation point (parameter values + init-seeded states).
    pid.sim_helper._create_param_subset(flat_names, flat_vals)

    # Build the symbolic graph once per structure. On a symbolic BDF / semi_implicit_euler
    # solve the graph is a mapaccum of thousands of rootfinder steps, and ca.gradient unrolls
    # sensitivities across all of them -- rebuilding it on every get_cost_ca / get_jac_cost_ca
    # call (twice per optimiser iteration) is what made long-run stiff multi-start impractical.
    cache_key = functions_cache_key(pid, param_names, get_all_series)
    if getattr(pid, '_casadi_functions_cache_key_val', None) == cache_key:
        return

    pid.cost_symb, pid.obs_dict_symb = pid.get_cost_and_obs_from_params(param_vals, do_ad=True)

    obs_outputs = []
    obs_meta = []

    for i, obs_item in enumerate(pid.obs_dict_symb):
        output_dict = pid.get_obs_output_dict(obs_item, get_all_series, is_symbolic=True)
        if get_all_series:
            obs_dict_item, obs_series_array_all = output_dict
            pid.obs_series_array_all_vec = ca.vertcat(*obs_series_array_all)
        else:
            obs_dict_item = output_dict

        for key in ['const', 'series', 'amp', 'phase']:
            val = obs_dict_item[key]

            if val is None:
                continue

            if isinstance(val, list):
                # series/amp/phase come back as a list with one entry per data item of
                # that type, each entry being a whole trace. Flatten each entry into the
                # obs vector but record the individual sizes, so get_obs can rebuild
                # the list of traces that the numpy path (and the plotting) expects.
                entry_sizes = []
                for entry in val:
                    entry_col = as_casadi_column(entry)
                    obs_outputs.append(entry_col)
                    entry_sizes.append(entry_col.size1())
                obs_meta.append((key, i, entry_sizes))
            else:
                obs_outputs.append(val)
                obs_meta.append((key, i, val.size1()))

    pid.obs_vec = ca.vertcat(*obs_outputs)
    pid.obs_meta = obs_meta

    # d(cost)/d(member) for every model constant; folded to d(cost)/d(theta) per calibrated
    # variable after evaluation (fold_entry_rows). The weights are constants, so folding the
    # numbers is identical to substituting theta symbolically and differentiating -- and it
    # keeps one code path for ungrouped rows, groups and modifiers.
    pid.jac_cost_symb = ca.gradient(pid.cost_symb, pid.sim_helper.variables_symb_subset)

    pid.cost_func = ca.Function('cost_func', [pid.sim_helper.states_symb, pid.sim_helper.variables_symb], [pid.cost_symb])

    if get_all_series:
        pid.obs_func = ca.Function('obs_func', [pid.sim_helper.states_symb, pid.sim_helper.variables_symb], [pid.obs_vec, pid.obs_series_array_all_vec])
    else:
        pid.obs_func = ca.Function('obs_func', [pid.sim_helper.states_symb, pid.sim_helper.variables_symb], [pid.obs_vec])

    pid.jac_cost_func = ca.Function('jac_cost_func', [pid.sim_helper.states_symb, pid.sim_helper.variables_symb], [pid.jac_cost_symb])

    pid._casadi_functions_cache_key_val = functions_cache_key(pid, param_names, get_all_series)


def get_jac_cost(pid, param_vals):
    param_names = pid.param_id_info["param_names"]
    build_functions(pid, param_names, param_vals)
    per_member = np.array(
        pid.jac_cost_func(pid.sim_helper.states, pid.sim_helper.variables)).flatten()
    # sum_i (dJ/dp_i) * dp_i/dtheta -- a no-op for ungrouped rows (one member, weight 1).
    return fold_entry_rows(per_member, pid._casadi_entry_map,
                           len(pid.param_id_info["param_names"]))


def get_cost(pid, param_vals):
    param_names = pid.param_id_info["param_names"]
    build_functions(pid, param_names, param_vals)
    cost = pid.cost_func(pid.sim_helper.states, pid.sim_helper.variables)
    return cost


def get_observable_sensitivities(pid, param_vals):
    """d(observable feature)/d(param) for the scalar (const) observables, via the exact CasADi
    jacobian of the symbolic observable vector w.r.t. the parameters.

    Returns ``{observable_label: {param_name: d(feature)/d(param)}}`` (see
    ``ParamID._observable_label``). This is the CasADi arm of
    ``ParamID.get_observable_sensitivities`` -- the backend-agnostic local-sensitivity
    accessor -- and mirrors how get_jac_cost differentiates the cost, but for each observable
    output instead of the aggregated cost.
    """
    import casadi as ca
    param_names = pid.param_id_info["param_names"]
    # One column per calibrated variable, keyed by entry label -- a grouped or modifier entry
    # reports d(feature)/d(theta) summed over its members, matching the Myokit FSA arm.
    flat_names = param_entry_labels(pid.param_id_info)
    build_functions(pid, param_names, param_vals)

    # jacobian of the flattened observable vector w.r.t. the differentiated parameter subset.
    jac_symb = ca.jacobian(pid.obs_vec, pid.sim_helper.variables_symb_subset)
    jac_func = ca.Function(
        'obs_jac_func',
        [pid.sim_helper.states_symb, pid.sim_helper.variables_symb],
        [jac_symb])
    jac = np.array(jac_func(pid.sim_helper.states, pid.sim_helper.variables))  # [rows x members]
    jac = fold_entry_rows(jac, pid._casadi_entry_map, len(flat_names))  # [rows x entries]

    # obs_meta slices obs_vec by (field, observable-item, size); the single 'const' entry holds
    # all scalar-feature rows in const-index order, so row idx+k is const observable k, whose
    # observable index is const_idx_to_obs_idx[k] (identical mapping to the Myokit arm). See
    # casadi_backend.get_obs, which slices obs_vec the same way.
    const_to_obs = pid.obs_info["const_idx_to_obs_idx"]
    out = {}
    idx = 0
    for key, _item, size in pid.obs_meta:
        n_rows = sum(size) if isinstance(size, list) else size
        if key == 'const':
            for k in range(n_rows):
                label = pid._observable_label(const_to_obs[k])
                out[label] = {flat_names[j]: float(jac[idx + k, j]) for j in range(len(flat_names))}
        idx += n_rows
    return out


def get_obs(pid, param_vals, get_all_series=False):
    param_names = pid.param_id_info["param_names"]
    build_functions(pid, param_names, param_vals, get_all_series)
    obs_val = pid.obs_func(pid.sim_helper.states, pid.sim_helper.variables)

    if get_all_series:
        obs_dict, obs_series_array_all = obs_val
        series_np = np.array(obs_series_array_all)

        obs_series_array_all_formatted = [
            [series_np[i, :] for i in range(series_np.shape[0])]
        ]
    else:
        obs_dict = obs_val
    obs_dict = np.array(obs_dict).flatten()

    obs = []

    num_items = len(pid.obs_dict_symb)
    for _ in range(num_items):
        obs.append({
            'const': None,
            'series': None,
            'amp': None,
            'phase': None,
        })

    idx = 0
    for key, i, size in pid.obs_meta:
        if isinstance(size, list):
            # a list of sizes means one trace per data item of this type
            traces = []
            for trace_size in size:
                traces.append(obs_dict[idx:idx+trace_size])
                idx += trace_size
            obs[i][key] = traces
            continue

        values = obs_dict[idx:idx+size]

        if size == 1:
            values = values[0]

        obs[i][key] = values

        idx += size

    if get_all_series:
        return obs, obs_series_array_all_formatted
    else:
        return obs
