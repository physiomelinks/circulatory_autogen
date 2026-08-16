"""Read what a param_id run wrote to disk (CUFLynx #210).

CA writes seven files across ``param_id/optimisers.py`` and ``param_id/paramID.py`` and, until
now, offered no reader. The only consumer hand-parsed them -- twice, in two files, with
different tolerance rules -- so the *format* was a contract that existed nowhere but in
comments in two repositories.

Everything a reader has to know, and therefore everything this module exists to stop anyone
else rediscovering:

* ``best_param_vals_history.csv`` holds **normalised** values (``param_norm_obj.normalise``),
  while the ``multi_start_*`` files hold **actual** ones (``_append_start_params``
  unnormalises before writing). Nothing in the format marks the asymmetry.
* ``best_cost_history.csv`` has **no header** and its row shape depends on the optimiser: the
  sorted top-10 for the genetic algorithm, a single scalar for ``sp_minimize``. Bayesian and
  CMA-ES write no cost history at all, so params-without-costs is normal, not corrupt.
* ``best_param_vals_history.csv`` *does* have a header (the parameter labels).
* gradients are ``dJ/dp`` in **real** space (``grad_norm / param_ranges``), and only the
  gradient-based optimisers write them.
* CA **appends and never truncates**, so a client must clear the files before a run --
  :func:`clear_run_history` is CA declaring which files are transient rather than the client
  hardcoding the list.
* the files may sit in a ``<case_type>_<prefix>`` subdirectory of the configured output dir.

Pure filesystem reads: no model, no solver, no simulation helper. Partial trailing rows are
skipped rather than raising, so it is safe to poll while a run is still writing.
"""
import csv
import glob
import json
import os

import numpy as np

# Written by the optimisers during a run. Transient: CA appends, so these must be removed
# before a new run or the new history is glued onto the old one.
HISTORY_FILES = (
    'best_cost_history.csv',
    'best_param_vals_history.csv',
    'best_gradient_history.csv',
    'multi_start_cost_history.csv',
    'multi_start_param_vals_history.csv',
    'multi_start_gradient_history.csv',
)

# Final results, written by _save_best_params. Not history, but part of what a reader wants.
RESULT_FILES = ('best_param_vals.npy', 'best_cost.npy')

# Where read_run_history gets the bounds it needs to denormalise best_param_vals_history.
BOUNDS_FILE = 'param_bounds.json'


def _rows(path, skip_header=False):
    """Numeric rows of a CSV, ignoring blanks and any trailing partial line.

    A row still being written is skipped rather than raising: this is polled mid-run.
    """
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, 'r') as handle:
        for idx, raw in enumerate(csv.reader(handle)):
            if skip_header and idx == 0:
                continue
            if not raw:
                continue
            try:
                out.append([float(cell) for cell in raw if str(cell).strip() != ''])
            except ValueError:
                # a header CA wrote, or a half-flushed final line
                continue
    return [row for row in out if row]


def _header(path):
    if not os.path.isfile(path):
        return []
    with open(path, 'r') as handle:
        for raw in csv.reader(handle):
            return [cell.strip() for cell in raw if str(cell).strip() != '']
    return []


def find_run_dir(output_dir):
    """The directory the run actually wrote to.

    CA may write into ``<output_dir>/<case_type>_<prefix>/`` rather than ``output_dir``
    itself, so accept either and let the caller pass whichever it configured.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None
    known = set(HISTORY_FILES) | set(RESULT_FILES)
    if any(os.path.isfile(os.path.join(output_dir, name)) for name in known):
        return output_dir
    candidates = [entry for entry in sorted(glob.glob(os.path.join(output_dir, '*')))
                  if os.path.isdir(entry)
                  and any(os.path.isfile(os.path.join(entry, name)) for name in known)]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # more than one case dir: newest by the results it wrote, so polling a re-run works
        return max(candidates, key=lambda d: max(
            (os.path.getmtime(os.path.join(d, name)) for name in known
             if os.path.isfile(os.path.join(d, name))), default=0.0))
    return output_dir


def save_param_bounds(param_id_info, output_dir):
    """Persist the parameter labels and bounds beside the run's outputs.

    Without these, ``best_param_vals_history.csv`` (which is normalised) cannot be turned
    back into parameter values from ``output_dir`` alone -- the reader would have to be handed
    a live ``param_id_info``, which is exactly the coupling #210 is removing. Rank-guarded by
    the caller.
    """
    if not param_id_info or not output_dir:
        return
    try:
        from libcuflynx.parsers.PrimitiveParsers import param_entry_labels
        labels = param_entry_labels(param_id_info)
    except Exception:
        labels = [str(n) for n in param_id_info.get('param_names', [])]
    payload = {
        'param_labels': list(labels),
        'param_mins': [float(v) for v in np.asarray(param_id_info['param_mins']).ravel()],
        'param_maxs': [float(v) for v in np.asarray(param_id_info['param_maxs']).ravel()],
    }
    with open(os.path.join(output_dir, BOUNDS_FILE), 'w') as handle:
        json.dump(payload, handle, indent=2)


def _bounds(run_dir, param_id_info):
    """(labels, mins, maxs) from the caller's param_id_info, else the persisted file."""
    if param_id_info:
        try:
            from libcuflynx.parsers.PrimitiveParsers import param_entry_labels
            labels = param_entry_labels(param_id_info)
        except Exception:
            labels = [str(n) for n in param_id_info.get('param_names', [])]
        return (list(labels),
                np.asarray(param_id_info['param_mins'], dtype=float).ravel(),
                np.asarray(param_id_info['param_maxs'], dtype=float).ravel())
    path = os.path.join(run_dir or '', BOUNDS_FILE)
    if os.path.isfile(path):
        with open(path, 'r') as handle:
            payload = json.load(handle)
        return (list(payload.get('param_labels') or []),
                np.asarray(payload.get('param_mins') or [], dtype=float),
                np.asarray(payload.get('param_maxs') or [], dtype=float))
    return [], None, None


def _group_starts(cost_rows, param_rows, grad_rows):
    """Regroup the ``start_idx, iteration, ...`` streams into one entry per start."""
    starts = {}

    def add(rows, key, scalar):
        for row in rows:
            if len(row) < 3:
                continue
            start_idx = int(row[0])
            entry = starts.setdefault(start_idx, {'cost': [], 'params': [], 'grad': []})
            entry[key].append(float(row[2]) if scalar else [float(v) for v in row[2:]])

    add(cost_rows, 'cost', True)
    add(param_rows, 'params', False)
    add(grad_rows, 'grad', False)
    return [starts[idx] for idx in sorted(starts)]


def read_run_history(output_dir, param_id_info=None):
    """Everything a GUI needs from a param_id run directory.

    Pure filesystem read, tolerant of partial rows (safe to poll mid-run), and it locates the
    ``<case_type>_<prefix>`` subdirectory itself.

    ``param_id_info`` is optional: bounds are read from ``param_bounds.json`` in the run dir
    when it is omitted (CA writes it via :func:`save_param_bounds`). Pass it to override, or
    for a run directory written before that file existed.

    Returns::

        {"param_labels": [str],
         "cost_history": [[float]],        # per generation, best first; [] if none written
         "param_history_norm": [[float]],  # as written -- normalised
         "param_history": [[float]],       # denormalised, or None if bounds unavailable
         "grad_history": [[float]],        # real-space dJ/dp; [] if none
         "starts": [{"cost": [float], "params": [[float]], "grad": [[float]]}],
         "best_param_vals": [float] | None,
         "best_cost": float | None,
         "run_dir": str | None}
    """
    run_dir = find_run_dir(output_dir)
    labels, mins, maxs = _bounds(run_dir, param_id_info)

    if run_dir is None:
        return {'param_labels': list(labels), 'cost_history': [], 'param_history_norm': [],
                'param_history': None, 'grad_history': [], 'starts': [],
                'best_param_vals': None, 'best_cost': None, 'run_dir': None}

    def path(name):
        return os.path.join(run_dir, name)

    # best_param_vals_history.csv carries the parameter labels; prefer them over the bounds
    # file's, since they describe the columns actually written.
    header = _header(path('best_param_vals_history.csv'))
    if header and not any(_is_number(cell) for cell in header):
        labels = header

    param_history_norm = _rows(path('best_param_vals_history.csv'), skip_header=bool(header))
    cost_history = _rows(path('best_cost_history.csv'))
    grad_history = _rows(path('best_gradient_history.csv'))

    param_history = None
    if param_history_norm and mins is not None and maxs is not None and mins.size:
        width = min(mins.size, maxs.size)
        param_history = [
            list(np.asarray(row[:width], dtype=float) * (maxs[:width] - mins[:width])
                 + mins[:width])
            for row in param_history_norm if len(row) >= width]

    starts = _group_starts(
        _rows(path('multi_start_cost_history.csv'), skip_header=True),
        _rows(path('multi_start_param_vals_history.csv'), skip_header=True),
        _rows(path('multi_start_gradient_history.csv'), skip_header=True))

    best_param_vals, best_cost = None, None
    if os.path.isfile(path('best_param_vals.npy')):
        best_param_vals = [float(v) for v in np.load(path('best_param_vals.npy')).ravel()]
    if os.path.isfile(path('best_cost.npy')):
        best_cost = float(np.load(path('best_cost.npy')).ravel()[0])

    return {'param_labels': list(labels), 'cost_history': cost_history,
            'param_history_norm': param_history_norm, 'param_history': param_history,
            'grad_history': grad_history, 'starts': starts,
            'best_param_vals': best_param_vals, 'best_cost': best_cost, 'run_dir': run_dir}


def _is_number(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def clear_run_history(output_dir):
    """Delete the transient history files, so a new run does not append onto an old one.

    CA declares which files are transient here, instead of every client hardcoding the list.
    The result files (``best_param_vals.npy`` / ``best_cost.npy``) are deliberately left
    alone: they are overwritten in place, and a cancelled run's best-so-far is worth keeping
    until a new one replaces it (issue #300).
    """
    run_dir = find_run_dir(output_dir)
    if run_dir is None:
        return
    for name in HISTORY_FILES:
        target = os.path.join(run_dir, name)
        if os.path.isfile(target):
            os.remove(target)
