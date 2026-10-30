"""The study manifest: one file that says where every part of a study is, for CUFLynx.

A study is everything run from one user_inputs setup -- calibrations, UQ, sensitivity
analyses and emulators. CUFLynx can find the results of a run by convention, but not the
model, the funcs or the solver settings, and a study reopened without them either fails to
open or opens at the viewer's own solver defaults.

So every analysis records itself in ``<param_id_output_dir>/<study_name>_cuflynx_study.json``.
That directory is the study's **working dir** (``<W>``): dropping the file into CUFLynx (or
pointing its working dir at the folder holding it) reopens the whole study. The format is
CUFLynx's ``study_manifest`` schema 1; the keys are additive, so a reader that predates one
ignores it.

    {"schema": 1, "study_name": ..., "file_prefix": ...,
     "user_inputs": ..., "updated": <iso time>, "libcuflynx_version": ...,
     "model": ..., "base_model": ..., "obs_data": ..., "params_for_id": ...,
     "user_funcs": {"operation": ..., "cost": ..., "modifier": ...},
     "system": {"module_array": ..., "parameters": ..., "module_library_dirs": [...],
                "external_modules_dir": ..., "use_builtin_modules": ...},
     "solver_info": {..., "solver": ..., "dt": ...}, "pre_time": ..., "sim_time": ...,
     "runs": [{"dir": ..., "method": ..., "analyses": [...], "status": ..., "obs_data": ...,
               "updated": ...}],
     "sensitivity": [{"dir": ..., "method": ..., "obs_data": ..., "updated": ...}],
     "emulator": ...,
     "workflow": {...}}

The working dir holds every input of the study as well as its outputs, so ``zip -r <W>`` is
the whole study and it opens wherever it is unzipped::

    <W>/
      <study>_cuflynx_study.json
      inputs/<kind>/            a copy of each input that was outside <W>: model/ (with the
                                CellML files it imports), obs_data/, params_for_id/,
                                user_inputs/ (the yaml as given, and the dated
                                user_inputs_<yymmdd>.yaml archives), funcs/, system/ (a
                                generated model's module array and parameters file, the
                                module versions it uses under modules/, and its
                                external_modules_dir under external_modules/),
                                emulator/, workflows/<wf>/ (a packed calibration workflow)
      <method>_<prefix>_<obs>/  calibration and UQ runs
      sensitivity/              Sobol and local sensitivity (the default sa_options output_dir)

An input is copied when the manifest is written (rank 0), once per content: a file already
in ``inputs/`` with the same name and content is reused, and a different file of the same
name goes into ``inputs/<kind>/<hash>/``. Every path in the manifest is relative to the
manifest's folder. A manifest nested inside a larger working dir (a calibration workflow's
step) is given that dir as ``cuflynx_working_dir``: its inputs are copied there, and its
paths may climb (``..``) as far as that dir and no further. An output directory outside the
working dir (an ``sa_options`` ``output_dir`` set to elsewhere) is recorded absolute.

A generated model's ``system`` holds what generating it again needs: the module array and
parameters file side by side (generation reads both from one ``resources_dir``), each module
library version the array uses (``calibration_workflow.pack.module_version_files``, as a packed
workflow holds them) and the ``external_modules_dir``. Built-in modules are not copied: they
come with libcuflynx, so ``use_builtin_modules`` and ``libcuflynx_version`` record them.
:func:`generation_inputs` gives the generation keys from a :func:`read_study` result.

``cuflynx_study_copy_inputs: false`` copies nothing: inputs outside the study are recorded by
absolute path, ``study_dir`` records where the study is, and the default sensitivity output
and the dated user_inputs archives go where they went before the working dir
(``sensitivity_outputs/<prefix>_SA_results`` and ``resources_dir``).

A calibration workflow's run (``libcuflynx.calibration_workflow``) writes one manifest for
the whole workflow, in its output directory, whose ``workflow`` key records the workflow as
run, its result, the target and each step (its model, obs_data and params_for_id, as the
packed copy in ``inputs/workflows/<wf>/`` names them, and its run directory and own
manifest). Each step's manifest, a single calibration CUFLynx can open on its own, has
``workflow``: ``{"manifest": ..., "step": <id>}`` back to it.

A calibration or MCMC run is entered when it starts, with ``status`` ``running``, so a run
that is cut short is still in the manifest with what it wrote each iteration; it becomes
``finished``, ``stopped`` (interrupted) or ``failed`` when it ends, and a killed process leaves
it ``running``.

The file is merged, not overwritten: each analysis upserts its own entry (keyed by
directory) and refreshes the study-level keys, so a sensitivity run does not erase the
calibration recorded before it, and two studies sharing one working dir keep separate
files. Writes are atomic because CUFLynx polls a study while it is running.

Writing is best effort and never fails a run. ``write_cuflynx_study: false`` in
user_inputs.yaml turns it off.

The JSON is written here rather than through CUFLynx's own module: the schema is the
contract between the two, and a study should be describable without the viewer installed.
"""

import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

__all__ = [
    'SCHEMA',
    'MANIFEST_SUFFIX',
    'INPUTS_DIR',
    'STUDY_INP_KEYS',
    'manifest_path',
    'copies_inputs',
    'working_dir',
    'sensitivity_dir',
    'user_inputs_archive_dir',
    'bring_in',
    'study_inputs',
    'update_study',
    'read_study',
    'generation_inputs',
]

#: The CUFLynx ``study_manifest`` schema this writes. Keys added since are additive.
SCHEMA = 1

MANIFEST_SUFFIX = '_cuflynx_study.json'

#: The working dir's folder of copied inputs.
INPUTS_DIR = 'inputs'

#: Single-path input keys, and the ``inputs/<kind>/`` folder a copy of each goes in.
PATH_KINDS = {'model': 'model', 'base_model': 'model', 'obs_data': 'obs_data',
              'params_for_id': 'params_for_id', 'emulator': 'emulator',
              'user_inputs': 'user_inputs'}
PATH_KEYS = tuple(PATH_KINDS)

#: The ``system`` block: the files a generated model was built from, by the inp key holding each.
SYSTEM_KEYS = {'module_array': 'vessels_csv_abs_path', 'parameters': 'parameters_csv_abs_path'}

#: The inp keys that say where a generated model's modules come from, recorded in ``system``.
MODULE_SOURCE_KEYS = ('module_library_dirs', 'external_modules_dir', 'use_builtin_modules')

#: The user_inputs keys about the manifest itself, beyond the ones ``update_study`` documents,
#: that a caller holding the parsed config passes on (``CVS0DParamID(study_settings=...)``).
STUDY_INP_KEYS = (('cuflynx_study_copy_inputs', 'cuflynx_working_dir')
                  + tuple(SYSTEM_KEYS.values()) + MODULE_SOURCE_KEYS)

#: Path keys inside the ``workflow`` block (and a step's link back to it), including the
#: ``outputs`` of each step.
WORKFLOW_PATH_KEYS = ('file', 'source', 'result', 'manifest', 'run', 'module_array',
                      'parameters', 'obs_data', 'params_for_id', 'generated_model',
                      'calibrated_parameters', 'param_id_output_dir', 'distribution_dir',
                      'sensitivity_dir', 'sobol_indices', 'emulator_dir')
#: The ones naming the workflow's inputs, copied in when they are outside the working dir.
WORKFLOW_INPUT_KEYS = ('file', 'source', 'module_array', 'parameters', 'obs_data',
                       'params_for_id')

#: Sections holding one entry per output directory.
ENTRY_SECTIONS = ('runs', 'sensitivity')

#: user_inputs key for each kind of user func file.
USER_FUNC_KEYS = {
    'operation': 'operation_funcs_external_path',
    'cost': 'cost_funcs_external_path',
    'modifier': 'modifier_funcs_external_path',
}

_XLINK_HREF = '{http://www.w3.org/1999/xlink}href'


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


def copies_inputs(inp):
    """Whether this config's study is a self-contained working dir: a manifest is written
    and ``cuflynx_study_copy_inputs`` is not false."""
    return (inp.get('write_cuflynx_study') is not False
            and inp.get('cuflynx_study_copy_inputs') is not False)


def working_dir(inp):
    """The working dir this config's inputs are copied into: ``cuflynx_working_dir`` when a
    caller nests the study in a larger one, else the study's own directory."""
    return os.path.abspath(inp.get('cuflynx_working_dir') or study_dir_for(inp))


def sensitivity_dir(inp):
    """``<param_id_output_dir>/sensitivity``: the default ``sa_options`` ``output_dir``."""
    return os.path.join(study_dir_for(inp), 'sensitivity')


def user_inputs_archive_dir(inp):
    """``<param_id_output_dir>/inputs/user_inputs``: where the dated user_inputs go."""
    return os.path.join(study_dir_for(inp), INPUTS_DIR, 'user_inputs')


def _inside(path, base):
    try:
        relative = os.path.relpath(path, base)
    except ValueError:
        # Windows refuses across drives; a different drive is as outside as it gets.
        return False
    return relative.split(os.sep)[0] != os.pardir


def _study_path(study_dir, value):
    """Relative when inside the study, absolute when anywhere else (no copying).

    A ``..`` path would be neither -- right only while the study stays exactly where it
    is -- so anything outside is recorded absolute.
    """
    full = os.path.abspath(str(value))
    return os.path.relpath(full, study_dir) if _inside(full, study_dir) else full


# -- copying inputs into the working dir ------------------------------------------------------

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cellml_imports(path, found):
    """Append to ``found`` every local file the CellML at ``path`` imports, recursively."""
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return
    for element in tree.iter():
        if not isinstance(element.tag, str) or element.tag.rsplit('}', 1)[-1] != 'import':
            continue
        href = element.get(_XLINK_HREF)
        if not href or '://' in href or os.path.isabs(href):
            continue
        target = os.path.normpath(os.path.join(os.path.dirname(path), href))
        if os.path.isfile(target) and target not in found:
            found.append(target)
            _cellml_imports(target, found)


def _bundle(path):
    """``(base, {relative path: source file}, main)``: the files ``path`` needs, relative to
    ``base``, and where ``path`` itself is among them. A file brings the CellML it imports;
    a directory brings its files (not hidden ones)."""
    if os.path.isdir(path):
        base = os.path.dirname(path)
        files = []
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            files += [os.path.join(root, n) for n in sorted(names) if not n.startswith('.')]
    else:
        files = [path]
        if path.endswith('.cellml'):
            _cellml_imports(path, files)
        base = os.path.commonpath([os.path.dirname(f) for f in files])
    return base, {os.path.relpath(f, base): f for f in files}, os.path.relpath(path, base)


def _same_file_in(directory, name, digest):
    """A file called ``name`` with content ``digest`` anywhere under ``directory``, or None."""
    for root, dirs, names in os.walk(directory):
        dirs.sort()
        if name in names and _sha256(os.path.join(root, name)) == digest:
            return os.path.join(root, name)
    return None


def _fits(target, files):
    """Whether ``files`` can go under ``target``: each is absent there or identical."""
    for relative, digest in files.items():
        existing = os.path.join(target, relative)
        if os.path.lexists(existing) and not (os.path.isfile(existing)
                                              and _sha256(existing) == digest):
            return False
    return True


#: (working dir, kind, source, mtime, size) -> its copy, so a run's repeated manifest writes
#: hash each input once.
_BROUGHT = {}


def bring_in(root, kind, path):
    """The copy of the input ``path`` under ``<root>/inputs/<kind>/``, made if need be.

    One copy per content: a lone file already anywhere under ``<root>/inputs/`` with the same
    name and content is reused; otherwise the file (with the CellML it imports, or a
    directory's files) goes into ``inputs/<kind>/``, or into ``inputs/<kind>/<hash>/`` when a
    different file of the same name is already there.
    """
    path = os.path.abspath(path)
    status = os.stat(path)
    key = (root, kind, path, status.st_mtime_ns, status.st_size)
    if key in _BROUGHT and os.path.exists(_BROUGHT[key]):
        return _BROUGHT[key]
    _, sources, main = _bundle(path)
    inputs = os.path.join(root, INPUTS_DIR)
    copy = None
    if len(sources) == 1 and not os.path.isdir(path) and os.path.isdir(inputs):
        copy = _same_file_in(inputs, os.path.basename(path), _sha256(path))
    if copy is None:
        copy = os.path.join(_place(root, kind, sources), main)
    _BROUGHT[key] = copy
    return copy


def _place(root, kind, sources):
    """Copy ``sources`` (``{relative path: source file}``) into ``<root>/inputs/<kind>/``, or
    into ``inputs/<kind>/<hash>/`` when a different file of one of those names is there; the
    directory they are in."""
    files = {relative: _sha256(source) for relative, source in sources.items()}
    target = os.path.join(root, INPUTS_DIR, kind)
    if not _fits(target, files):
        digest = hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()
        target = os.path.join(target, digest[:12])
    for relative, source in sources.items():
        destination = os.path.join(target, relative)
        if not os.path.exists(destination):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(source, destination)
    return target


def _module_files(system):
    """``(homes, files)`` of the module library versions ``system``'s module array uses
    (``pack.module_version_files``); none when the array cannot be read."""
    from libcuflynx.calibration_workflow.pack import module_version_files
    from libcuflynx.calibration_workflow.resolve import ResolvedArray
    from libcuflynx.utilities.config_schemas import read_module_array_records
    from libcuflynx.utilities.module_library import ModuleSources
    libraries = system.get('module_library_dirs') or []
    try:
        sources = ModuleSources(system)
        components = sources.component_registry()
        supermodules = sources.supermodule_registry()
        array = ResolvedArray(model=None, records=read_module_array_records(system['module_array']),
                              supermodules=supermodules)
        configs = array.config_paths(components, supermodules)
    except Exception as error:
        print(f'[cuflynx_study] WARNING: could not find the module versions of '
              f'{system["module_array"]} ({error}); copying every units file of its libraries '
              f'but none of their versions')
        configs = []
    homes, _, files = module_version_files(configs, libraries)
    return homes, files


def _external_home(externals):
    return ['external_modules'] if len(externals) == 1 else [
        os.path.join('external_modules', str(i)) for i in range(len(externals))]


def _system_sources(paths, system):
    """``system`` (the study-level block, absolute paths) with what is outside the working dir
    copied into ``inputs/system/`` as one directory, laid out as generation reads it: the
    module array and parameters file side by side, ``modules/`` (the library versions used,
    as ``module_version_files`` lays them out) and ``external_modules/``."""
    libraries = [os.path.abspath(d) for d in system.get('module_library_dirs') or []]
    external = system.get('external_modules_dir')
    externals = [os.path.abspath(d) for d in
                 ([] if not external else [external] if isinstance(external, str) else external)]
    files = (system['module_array'], system['parameters'])
    outside = [p for p in files + tuple(libraries + externals) if not _inside(p, paths.root)]
    if not outside:
        return system
    key = ('system', paths.root) + tuple(
        (p, os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in files) + (
        tuple(libraries), tuple(externals), system.get('use_builtin_modules'))
    if key in _BROUGHT and os.path.isdir(_BROUGHT[key][0]):
        target, sources = _BROUGHT[key]
    else:
        sources, copied = {}, dict(system)
        if any(not _inside(p, paths.root) for p in files):
            for name in SYSTEM_KEYS:
                sources[os.path.basename(system[name])] = system[name]
                copied[name] = os.path.basename(system[name])
        if any(not _inside(d, paths.root) for d in libraries):
            homes, module_files = _module_files(dict(system, module_library_dirs=libraries))
            sources.update(module_files)
            copied['module_library_dirs'] = homes
        if any(not _inside(d, paths.root) for d in externals):
            homes = _external_home(externals)
            for home, directory in zip(homes, externals):
                for name in os.listdir(directory) if os.path.isdir(directory) else []:
                    path = os.path.join(directory, name)
                    if os.path.isfile(path) and not name.startswith('.'):
                        sources[os.path.join(home, name)] = path
            copied['external_modules_dir'] = homes[0] if isinstance(external, str) else homes
        target = _place(paths.root, 'system', sources)
        sources = copied
        _BROUGHT[key] = (target, sources)
    out = dict(sources)
    for name in ('module_array', 'parameters', 'module_library_dirs', 'external_modules_dir'):
        value = out.get(name)
        if isinstance(value, list):
            out[name] = [v if os.path.isabs(v) else os.path.join(target, v) for v in value]
        elif value:
            out[name] = value if os.path.isabs(value) else os.path.join(target, value)
    return out


class _Paths:
    """How a manifest at ``study_dir`` stores paths, for one config."""

    def __init__(self, inp, study_dir):
        self.here = study_dir
        self.copy = copies_inputs(inp)
        self.root = working_dir(inp) if self.copy else study_dir

    def output(self, value):
        """An output: relative inside the working dir, absolute outside it (never copied)."""
        if not self.copy:
            return _study_path(self.here, value)
        full = os.path.abspath(str(value))
        return os.path.relpath(full, self.here) if _inside(full, self.root) else full

    def input(self, value, kind):
        """An input: copied into the working dir when it is outside, then relative."""
        if not self.copy:
            return _study_path(self.here, value)
        full = os.path.abspath(str(value))
        if not _inside(full, self.root):
            if not os.path.exists(full):
                print(f'[cuflynx_study] WARNING: {full} does not exist, so it is recorded '
                      f'by absolute path rather than copied into the working dir')
                return full
            full = bring_in(self.root, kind, full)
        return os.path.relpath(full, self.here)


# -- the manifest ---------------------------------------------------------------------------

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
    working dir, and a generated pipeline bundle leaves CA's own fit-generation output at
    ``generated_models/<prefix>/<prefix>.cellml`` under it.
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
    system = {name: inp[key] for name, key in SYSTEM_KEYS.items()
              if inp.get(key) and os.path.isfile(inp[key])}
    if system.get('module_array'):
        system.update({key: inp[key] for key in MODULE_SOURCE_KEYS
                       if inp.get(key) not in (None, [], '')})
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
        'system': system or None,
        'solver_info': _solver_info(inp),
        'pre_time': inp.get('pre_time'),
        'sim_time': inp.get('sim_time'),
        'emulator': emulator,
    }


def _encode(paths, key, value):
    """A study-level value as stored: its paths through ``paths``."""
    if value is None:
        return None
    if key in PATH_KINDS:
        return paths.input(value, PATH_KINDS[key])
    if key == 'user_funcs':
        return {kind: paths.input(path, 'funcs') for kind, path in value.items()}
    if key == 'system':
        return _encode_system(paths, value)
    return value


def _encode_system(paths, system):
    """The ``system`` block as stored: copied into ``inputs/system/`` (see
    :func:`_system_sources`) when it has a module array and parameters file, its paths
    relative."""
    if paths.copy and system.get('module_array') and system.get('parameters'):
        system = _system_sources(paths, system)
    out = {}
    for name, value in system.items():
        if name == 'use_builtin_modules':
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = [paths.input(v, 'system') for v in value]
        else:
            out[name] = paths.input(value, 'system')
    return out


def _encode_workflow(paths, value, key=None):
    '''A ``workflow`` block as stored: its paths (WORKFLOW_PATH_KEYS) through ``paths``,
    through nested dicts and lists.'''
    if isinstance(value, dict):
        return {k: _encode_workflow(paths, v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_workflow(paths, v, key) for v in value]
    if key in WORKFLOW_PATH_KEYS and isinstance(value, str) and value:
        if key in WORKFLOW_INPUT_KEYS:
            return paths.input(value, 'workflows')
        return paths.output(value)
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
            ``file_prefix``, ``write_cuflynx_study``, ``cuflynx_study_copy_inputs``,
            ``cuflynx_working_dir``, ``user_inputs_path``, ``model_path``,
            ``param_id_obs_path``, ``params_for_id_path``, the ``*_funcs_external_path``
            keys, ``vessels_csv_abs_path``, ``parameters_csv_abs_path``,
            ``module_library_dirs``, ``external_modules_dir``, ``use_builtin_modules``, ``solver``,
            ``solver_info``, ``dt``, ``pre_time``, ``sim_time``, ``use_emulator`` and
            ``emulator_dir``.
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
        paths = _Paths(inp, study_dir)
        record = _read_raw(location)
        record['schema'] = SCHEMA

        for key, value in _study_level(inp, study_dir).items():
            if value is not None:
                record[key] = _encode(paths, key, value)
        if emulator:
            record['emulator'] = paths.input(emulator, 'emulator')
        if workflow is not None:
            record['workflow'] = _encode_workflow(paths, workflow)

        if section is not None:
            if section not in ENTRY_SECTIONS:
                raise ValueError(f'unknown study section {section!r}')
            item = dict(entry or {})
            if not item.get('dir'):
                raise ValueError(f'a {section} entry needs a dir')
            item['dir'] = paths.output(item['dir'])
            obs = item.get('obs_data') or inp.get('param_id_obs_path')
            if obs:
                item['obs_data'] = paths.input(obs, 'obs_data')
            if item.get('method'):
                item.setdefault('analyses', [item['method']])
            item['updated'] = _now()
            record[section] = _upsert(record.get(section), item)

        if paths.copy:
            # every path is relative to the manifest's folder, wherever that ends up
            record.pop('study_dir', None)
        else:
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

    Relative paths resolve against the manifest's own directory, or, for a manifest that
    records ``study_dir`` (``cuflynx_study_copy_inputs: false``), against that when it
    exists (a copy that has been moved away from its study).
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
    if isinstance(out.get('user_funcs'), dict):
        out['user_funcs'] = {name: resolve(p) for name, p in out['user_funcs'].items() if p}
    if isinstance(out.get('system'), dict):
        out['system'] = {name: (p if isinstance(p, bool) else
                                [resolve(v) for v in p] if isinstance(p, list) else resolve(p))
                         for name, p in out['system'].items() if p is not None}
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


def generation_inputs(study):
    """The user_inputs keys that generate a study's model again from its ``system``: a
    :func:`read_study` result in, ``{'file_prefix', 'resources_dir', 'input_param_file'}``
    and the module source keys out, or None when the study records no generated model."""
    system = study.get('system') or {}
    if not system.get('module_array') or not system.get('parameters'):
        return None
    inp = {'file_prefix': study.get('file_prefix'),
           'resources_dir': os.path.dirname(system['module_array']),
           'input_param_file': os.path.relpath(system['parameters'],
                                               os.path.dirname(system['module_array']))}
    inp.update({key: system[key] for key in MODULE_SOURCE_KEYS if key in system})
    return inp
