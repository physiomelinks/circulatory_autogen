"""The study manifest: one file that says where every part of a study is, for CUFLynx.

A study is everything run from one user_inputs setup -- calibrations, UQ, sensitivity
analyses and emulators. Their outputs land in several directories (sensitivity writes
outside ``param_id_output_dir``; a shared emulator can sit anywhere) and the inputs they
were run from live elsewhere again (the model in ``generated_models/``, the user's funcs
wherever ``*_funcs_external_path`` points). CUFLynx can find the results of a run by
convention, but not the model, the funcs or the solver settings, and a study reopened
without them either fails to open or opens at the viewer's own solver defaults.

So every analysis records itself in ``<param_id_output_dir>/<study_name>_cuflynx_study.json``.
Dropping that file into CUFLynx (or pointing its outputs directory at the folder holding
it) reopens the whole study. The format is CUFLynx's ``study_manifest`` schema 1; the
keys are additive, so a reader that predates one ignores it.

    {"schema": 1, "study_name": ..., "file_prefix": ..., "study_dir": <abs>,
     "user_inputs": ..., "updated": <iso time>, "libcuflynx_version": ...,
     "model": ..., "base_model": ..., "obs_data": ..., "params_for_id": ...,
     "user_funcs": {"operation": ..., "cost": ..., "modifier": ...},
     "solver_info": {..., "solver": ..., "dt": ...}, "pre_time": ..., "sim_time": ...,
     "runs": [{"dir": ..., "method": ..., "analyses": [...], "status": ..., "obs_data": ...,
               "updated": ...}],
     "sensitivity": [{"dir": ..., "method": ..., "obs_data": ..., "updated": ...}],
     "emulator": ...,
     "workflow": {...}}

A calibration workflow's run (``libcuflynx.calibration_workflow``) writes one manifest for
the whole workflow, in its output directory, whose ``workflow`` key records the workflow as
run, its result, the target and each step (its model, obs_data and params_for_id, as the
workflow file names them, and its run directory and own manifest). Each step's manifest, a
single calibration CUFLynx can open on its own, has ``workflow``: ``{"manifest": ...,
"step": <id>}`` back to it.

Paths inside ``study_dir`` are stored relative to it, so the directory can be moved or
copied whole; paths outside it are absolute, so they keep resolving from wherever the
study ends up. ``study_dir`` itself is absolute so a copy of the manifest that is not
sitting in its study (a dropped file, or one embedded in an OMEX archive) can still be
resolved.

A calibration or MCMC run is entered when it starts, with ``status`` ``running``, so a run
that is cut short is still in the manifest with what it wrote each iteration; it becomes
``finished``, ``stopped`` (interrupted) or ``failed`` when it ends, and a killed process leaves
it ``running``.

The file is merged, not overwritten: each analysis upserts its own entry (keyed by
directory) and refreshes the study-level keys, so a sensitivity run does not erase the
calibration recorded before it, and two studies sharing one outputs directory keep
separate files. Writes are atomic because CUFLynx polls a study while it is running.

Writing is best effort and never fails a run. ``write_cuflynx_study: false`` in
user_inputs.yaml turns it off.

The JSON is written here rather than through CUFLynx's own module: the schema is the
contract between the two, and a study should be describable without the viewer installed.
"""

import json
import os
import tempfile
from datetime import datetime

__all__ = [
    'SCHEMA',
    'MANIFEST_SUFFIX',
    'manifest_path',
    'study_inputs',
    'update_study',
    'read_study',
]

#: The CUFLynx ``study_manifest`` schema this writes. Keys added since are additive.
SCHEMA = 1

MANIFEST_SUFFIX = '_cuflynx_study.json'

#: Single-path keys, stored relative inside the study and absolute outside it.
PATH_KEYS = ('model', 'base_model', 'obs_data', 'params_for_id', 'emulator', 'user_inputs')

#: Path keys inside the ``workflow`` block (and a step's link back to it).
WORKFLOW_PATH_KEYS = ('file', 'source', 'result', 'manifest', 'run', 'module_array',
                      'parameters', 'obs_data', 'params_for_id', 'generated_model',
                      'calibrated_parameters', 'sweep_dir', 'sweep_results')

#: Sections holding one entry per output directory.
ENTRY_SECTIONS = ('runs', 'sensitivity')

#: user_inputs key for each kind of user func file.
USER_FUNC_KEYS = {
    'operation': 'operation_funcs_external_path',
    'cost': 'cost_funcs_external_path',
    'modifier': 'modifier_funcs_external_path',
}


def _now():
    return datetime.now().isoformat(timespec='seconds')


def _version():
    try:
        from importlib.metadata import version
        return version('libcuflynx')
    except Exception:
        return None


def _rank():
    try:
        from libcuflynx.utilities.mpi_utils import get_MPI
        return get_MPI().COMM_WORLD.Get_rank()
    except Exception:
        return 0


def study_dir_for(inp):
    """The directory a study's manifest lives in: its ``param_id_output_dir``."""
    root = inp.get('param_id_output_dir')
    if not root:
        from libcuflynx.utilities.paths import default_param_id_output_dir
        root = default_param_id_output_dir()
    return os.path.abspath(root)


def manifest_path(inp):
    """``<param_id_output_dir>/<study_name>_cuflynx_study.json`` for this config."""
    name = inp.get('study_name') or inp.get('file_prefix') or inp.get('file_name_prefix') or 'study'
    return os.path.join(study_dir_for(inp), f'{name}{MANIFEST_SUFFIX}')


def _study_path(study_dir, value):
    """Relative when inside the study, absolute when anywhere else.

    A ``..`` path would be neither -- right only while the study stays exactly where it
    is -- so anything outside is recorded absolute.
    """
    full = os.path.abspath(str(value))
    try:
        relative = os.path.relpath(full, study_dir)
    except ValueError:
        # Windows refuses across drives; a different drive is as outside as it gets.
        return full
    return full if relative.split(os.sep)[0] == os.pardir else relative


def _solver_info(inp):
    """The solver settings a viewer needs to reproduce this study's traces.

    ``dt`` and ``solver`` are folded in beside the solver's own keys: that is the shape
    CUFLynx reads back (it pops ``dt`` out of ``solver_info``). ``dt`` is the *output*
    interval, and it matters on its own -- bounding CVODE with MaximumStep still leaves a
    coarse output grid that can sample straight over a fast transient.
    """
    info = dict(inp.get('solver_info') or {})
    if inp.get('solver') and 'solver' not in info:
        info['solver'] = inp['solver']
    if inp.get('dt') is not None:
        info['dt'] = inp['dt']
    return info or None


def _calibrated_model(study_dir, prefix, base_model):
    """The fitted model, if one has been written into the study, else None.

    Two places hold one: CUFLynx writes ``<prefix>_calibrated.cellml`` at the top of the
    outputs directory, and a generated pipeline bundle leaves CA's own fit-generation
    output at ``generated_models/<prefix>/<prefix>.cellml`` under it.
    """
    if not prefix:
        return None
    base = os.path.abspath(base_model) if base_model else None
    for candidate in (os.path.join(study_dir, f'{prefix}_calibrated.cellml'),
                      os.path.join(study_dir, 'generated_models', prefix, f'{prefix}.cellml')):
        if os.path.isfile(candidate) and os.path.abspath(candidate) != base:
            return candidate
    return None


def study_inputs(**values):
    """An ``inp``-shaped dict from keyword values, dropping the ones that are None.

    For callers that hold their settings as attributes rather than as the parsed
    user_inputs dict (a class constructed directly rather than through ``init_from_dict``).
    """
    return {key: value for key, value in values.items() if value is not None}


def _study_level(inp, study_dir):
    """The keys describing the study as a whole, from a parsed user_inputs dict."""
    prefix = inp.get('file_prefix') or inp.get('file_name_prefix')
    base_model = inp.get('model_path')
    calibrated = _calibrated_model(study_dir, prefix, base_model)
    funcs = {kind: inp[key] for kind, key in USER_FUNC_KEYS.items() if inp.get(key)}
    emulator = None
    if inp.get('use_emulator') and inp.get('emulator_dir'):
        emulator = inp['emulator_dir']
    return {
        'study_name': inp.get('study_name') or prefix,
        'file_prefix': prefix,
        'user_inputs': inp.get('user_inputs_path'),
        'model': calibrated or base_model,
        'base_model': base_model,
        'obs_data': inp.get('param_id_obs_path'),
        'params_for_id': inp.get('params_for_id_path'),
        'user_funcs': funcs or None,
        'solver_info': _solver_info(inp),
        'pre_time': inp.get('pre_time'),
        'sim_time': inp.get('sim_time'),
        'emulator': emulator,
    }


def _encode(study_dir, key, value):
    """A value as stored: paths made study-relative where that is honest."""
    if value is None:
        return None
    if key in PATH_KEYS:
        return _study_path(study_dir, value)
    if key == 'user_funcs':
        return {kind: _study_path(study_dir, path) for kind, path in value.items()}
    return value


def _encode_workflow(study_dir, value, key=None):
    '''A ``workflow`` block as stored: its paths (WORKFLOW_PATH_KEYS) made study-relative
    where that is honest, through nested dicts and lists.'''
    if isinstance(value, dict):
        return {k: _encode_workflow(study_dir, v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_workflow(study_dir, v, key) for v in value]
    if key in WORKFLOW_PATH_KEYS and isinstance(value, str) and value:
        return _study_path(study_dir, value)
    return value


def _read_raw(location):
    """The manifest as stored, or an empty dict when it is absent or unreadable."""
    if not os.path.isfile(location):
        return {}
    try:
        with open(location, encoding='utf-8') as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            return raw
        print(f'[cuflynx_study] WARNING: {location} is not a JSON object; rewriting it')
    except (OSError, ValueError) as error:
        print(f'[cuflynx_study] WARNING: could not read {location} ({error}); rewriting it')
    return {}


def _upsert(entries, entry):
    """Put ``entry`` first, merged into any existing entry for the same directory.

    Newest first, because CUFLynx opens a study on its first declared run. ``analyses``
    accumulates: one run directory holds both the calibration and the MCMC that followed
    it, since both use the same ``<method>_<prefix>_<obs>`` name.
    """
    kept, merged = [], dict(entry)
    for existing in entries or []:
        if isinstance(existing, dict) and existing.get('dir') == entry['dir']:
            analyses = list(existing.get('analyses') or [existing.get('method')])
            merged = {**existing, **entry}
            analyses += [a for a in entry.get('analyses') or [] if a not in analyses]
            merged['analyses'] = [a for a in analyses if a]
        elif isinstance(existing, dict):
            kept.append(existing)
    return [merged] + kept


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _write_atomic(location, record):
    directory = os.path.dirname(location)
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix='.cuflynx_study_', suffix='.json', dir=directory)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            json.dump(record, out, indent=2)
            out.write('\n')
        # mkstemp makes the file private (0600); the manifest is as readable as anything
        # else the run writes, so give it the mode a plain open() would have.
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, location)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def update_study(inp, section=None, entry=None, emulator=None, workflow=None):
    """Record this config's study, and optionally one analysis output, in its manifest.

    Args:
        inp: The parsed user_inputs dict, or an ``inp``-shaped dict from
            :func:`study_inputs`. Keys read: ``param_id_output_dir``, ``study_name``,
            ``file_prefix``, ``write_cuflynx_study``, ``user_inputs_path``, ``model_path``,
            ``param_id_obs_path``, ``params_for_id_path``, the ``*_funcs_external_path``
            keys, ``solver``, ``solver_info``, ``dt``, ``pre_time``, ``sim_time``,
            ``use_emulator`` and ``emulator_dir``.
        section: ``'runs'`` or ``'sensitivity'`` to record an output directory.
        entry: That output's ``{'dir': ..., 'method': ...}``; ``obs_data`` defaults to the
            config's.
        emulator: A trained emulator's directory, to declare as the study's emulator.
        workflow: The study's ``workflow`` block (a workflow's own manifest), or a step's
            ``{'manifest', 'step'}`` link to it; replaces the one stored.

    Returns:
        The manifest's path, or None when nothing was written (not rank 0, turned off,
        or the write failed -- which is reported, never raised).
    """
    if _rank() != 0 or inp.get('write_cuflynx_study') is False:
        return None
    try:
        location = manifest_path(inp)
        study_dir = os.path.dirname(location)
        record = _read_raw(location)
        record['schema'] = SCHEMA

        for key, value in _study_level(inp, study_dir).items():
            if value is not None:
                record[key] = _encode(study_dir, key, value)
        if emulator:
            record['emulator'] = _study_path(study_dir, emulator)
        if workflow is not None:
            record['workflow'] = _encode_workflow(study_dir, workflow)

        if section is not None:
            if section not in ENTRY_SECTIONS:
                raise ValueError(f'unknown study section {section!r}')
            item = dict(entry or {})
            if not item.get('dir'):
                raise ValueError(f'a {section} entry needs a dir')
            item['dir'] = _study_path(study_dir, item['dir'])
            obs = item.get('obs_data') or inp.get('param_id_obs_path')
            if obs:
                item['obs_data'] = _study_path(study_dir, obs)
            if item.get('method'):
                item.setdefault('analyses', [item['method']])
            item['updated'] = _now()
            record[section] = _upsert(record.get(section), item)

        record['study_dir'] = study_dir
        record['updated'] = _now()
        version = _version()
        if version:
            record['libcuflynx_version'] = version
        _write_atomic(location, record)
        return location
    except Exception as error:  # never fail an analysis over its description
        print(f'[cuflynx_study] WARNING: could not update the study manifest: {error}')
        return None


def read_study(path):
    """A manifest with its paths made absolute, from the file or the directory holding it.

    Relative paths resolve against the manifest's own directory, or against its recorded
    ``study_dir`` when that is where the files are (a copy that has been moved away from
    its study).
    """
    if os.path.isdir(path):
        found = sorted(name for name in os.listdir(path) if name.endswith(MANIFEST_SUFFIX))
        if len(found) != 1:
            raise FileNotFoundError(
                f'{path} holds {len(found)} *{MANIFEST_SUFFIX} files; name one')
        path = os.path.join(path, found[0])
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    here = os.path.dirname(os.path.abspath(path))
    base = raw.get('study_dir') or here
    if not os.path.isdir(base):
        base = here

    def resolve(value):
        return value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value))

    out = dict(raw)
    for key in PATH_KEYS:
        if out.get(key):
            out[key] = resolve(out[key])
    if out.get('user_funcs'):
        out['user_funcs'] = {kind: resolve(p) for kind, p in out['user_funcs'].items()}
    for section in ENTRY_SECTIONS:
        out[section] = [{**e, 'dir': resolve(e['dir']),
                         **({'obs_data': resolve(e['obs_data'])} if e.get('obs_data') else {})}
                        for e in out.get(section) or [] if isinstance(e, dict) and e.get('dir')]

    def resolve_workflow(value, key=None):
        if isinstance(value, dict):
            return {k: resolve_workflow(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_workflow(v, key) for v in value]
        if key in WORKFLOW_PATH_KEYS and isinstance(value, str) and value:
            return resolve(value)
        return value

    if isinstance(out.get('workflow'), dict):
        out['workflow'] = resolve_workflow(out['workflow'])
    return out
