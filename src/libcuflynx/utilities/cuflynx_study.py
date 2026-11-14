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
     "workflow": {...},
     "yaml_inputs": {<archived yaml>: {"base_model": ..., "obs_data": ..., ...}}}

The working dir holds every input of the study as well as its outputs, so a zip of it
(:func:`export_working_dir`) is the whole study and it opens wherever it is unzipped::

    <W>/
      <study>_cuflynx_study.json
      inputs/index.json         every stored input version: its kind, file, folder, content
                                hash, when it was first used and the runs that used it
      inputs/<kind>/<yyyy-mm-dd_HHMMSS>/
                                one version of an input that was outside <W>, by kind:
                                model (with the CellML files it imports), obs_data,
                                params_for_id, user_inputs (the yaml as given), funcs,
                                system (a generated model's module array and parameters
                                file, the module versions it uses under modules/, and its
                                external_modules_dir under external_modules/), emulator,
                                workflows/<wf> (a packed calibration workflow)
      inputs/user_inputs/user_inputs_<yymmdd>.yaml
                                the resolved config of each day's runs, dated by its name
      <method>_<prefix>_<obs>/  calibration and UQ runs
      sensitivity/              Sobol and local sensitivity (the default sa_options output_dir)

An input is copied when the manifest is written (rank 0), once per content, into a folder
named for the local date-time it was first used (``_2``, ``_3``... for a second content
first used in the same second). No copy sits outside a dated folder, so none looks like the
current version: the manifest names each run's own copies, and ``inputs/index.json``
(:func:`read_index`) lists every version. A content already stored (the same content hash)
is reused, so a rerun with unchanged inputs copies nothing. An app keeping the inputs it edits
(CUFLynx) stores them with :func:`store_input` and finds the one saved last with
:func:`latest_input`. Every path in the manifest is
relative to the manifest's folder. A manifest nested inside a larger working dir (a calibration workflow's
step) is given that dir as ``cuflynx_working_dir``: its inputs are copied there, and its
paths may climb (``..``) as far as that dir and no further.

An output directory outside the working dir (an ``sa_options`` ``output_dir`` set to elsewhere)
is allowed, but it does not travel with the working dir: a zip of ``<W>`` does not hold it. Its
``runs`` or ``sensitivity`` entry keeps the absolute path and has ``"outside": true``, the run
prints a warning naming it (rank 0, once per directory per process), and
:func:`outside_outputs` lists such entries, also for a zip made by hand, which holds them as they
are. Nothing re-runs a study from its manifest: :func:`read_study` resolves output directories
only to read their results, and :func:`generation_inputs` returns no output directory, so a study
opened elsewhere never writes into the recorded path.

:func:`export_working_dir` is how a working dir is shared. ``<W>`` itself is not changed; in the
zip, the archived user_inputs yamls name relative output directories (an outside one in the home
the working dir has for that output, with a warning) and name each input by its stored copy, the
version that yaml's run used (``yaml_inputs`` in the manifest), and the manifests and
``inputs/index.json`` hold no outside entry or absolute path. So a run of an unzipped yaml, through
``user_inputs_path_override``, reads and writes only inside the recipient's working dir.

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
import threading
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime

import yaml

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
    'store_input',
    'latest_input',
    'read_index',
    'study_inputs',
    'update_study',
    'read_study',
    'generation_inputs',
    'outside_outputs',
    'export_working_dir',
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
                      'sensitivity_dir', 'sobol_indices', 'emulator_dir', 'sweep_dir',
                      'sweep_results')
#: The ones naming the workflow's inputs, copied in when they are outside the working dir.
WORKFLOW_INPUT_KEYS = ('file', 'source', 'module_array', 'parameters', 'obs_data',
                       'params_for_id')

#: The study-level keys ``yaml_inputs`` records for each archived yaml: the inputs its run read.
YAML_INPUT_KEYS = ('base_model', 'obs_data', 'params_for_id', 'user_funcs', 'system', 'emulator')

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


def dated_user_inputs_name():
    """The name of today's archive of the resolved config: ``user_inputs_<yymmdd>.yaml``."""
    return f"user_inputs_{date.today().strftime('%y%m%d')}.yaml"


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


#: The working dir's list of the input versions it holds: ``inputs/index.json``.
INDEX_FILE = 'index.json'

#: A version's folder name: the local date-time it was first used, as the dated
#: ``user_inputs_<yymmdd>.yaml`` archives are dated in local time.
STAMP_FORMAT = '%Y-%m-%d_%H%M%S'


def _stamp():
    return datetime.now().replace(microsecond=0)


def _index_path(root):
    return os.path.join(root, INPUTS_DIR, INDEX_FILE)


def read_index(root):
    """The versions ``<root>/inputs/index.json`` lists, oldest first, or [] when there is none.

    Each is ``{"kind", "file", "folder", "hash", "first_used", "runs", "files"}``: ``folder``
    (relative to ``root``) holds one content of ``kind``, ``file`` is the input itself within
    it (None for a ``system`` bundle without one), ``hash`` is the content hash of every file in
    it (``files``: ``{relative path: sha256}``) and ``runs`` the output directories (relative to
    ``root`` when inside it) of the runs that used it. A version :func:`store_input` returned
    also has ``stored``, the local time it last did.
    """
    try:
        with open(_index_path(root), encoding='utf-8') as handle:
            versions = json.load(handle).get('versions')
    except (OSError, ValueError, AttributeError):
        return []
    return [v for v in versions if isinstance(v, dict)] if isinstance(versions, list) else []


def _write_index(root, versions):
    _write_atomic(_index_path(root), {'schema': 1, 'versions': versions})


def _holds(root, version, files):
    """Whether ``version``'s folder still holds ``files`` (``{relative path: sha256}``)."""
    folder = version.get('folder')
    if not folder:
        return False
    for relative, digest in files.items():
        path = os.path.join(root, folder, relative)
        if not os.path.isfile(path) or _sha256(path) != digest:
            return False
    return True


def _same_file_in(root, name, digest):
    """A stored file called ``name`` with content ``digest`` in any version the index lists,
    or None."""
    for version in read_index(root):
        for relative, stored in (version.get('files') or {}).items():
            if (os.path.basename(relative) == name and stored == digest
                    and _holds(root, version, {relative: digest})):
                return os.path.join(root, version['folder'], relative)
    return None


#: (working dir, kind, source, mtime, size) -> its copy, so a run's repeated manifest writes
#: hash each input once.
_BROUGHT = {}


def bring_in(root, kind, path):
    """The copy of the input ``path`` under ``<root>/inputs/<kind>/<first used>/``, made if
    need be.

    One copy per content: a lone file stored already (any kind) with the same name and
    content is reused, as is a stored version of ``kind`` with the same content; otherwise the
    file (with the CellML it imports, or a directory's files) goes into a new dated folder
    (:func:`_place`).
    """
    path = os.path.abspath(path)
    status = os.stat(path)
    key = (root, kind, path, status.st_mtime_ns, status.st_size)
    if key in _BROUGHT and os.path.exists(_BROUGHT[key]):
        return _BROUGHT[key]
    _, sources, main = _bundle(path)
    copy = None
    if len(sources) == 1 and not os.path.isdir(path):
        copy = _same_file_in(root, os.path.basename(path), _sha256(path))
    if copy is None:
        copy = os.path.join(_place(root, kind, sources, main), main)
    _BROUGHT[key] = copy
    return copy


def _content_hash(files):
    """A version's ``hash``, from its ``files`` (``{relative path: sha256}``)."""
    return hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()


def _place(root, kind, sources, main=None):
    """The folder holding ``sources`` (``{relative path: source file}``) in ``<root>/inputs/``:
    the version of ``kind`` the index lists with the same content, or else a new folder
    ``inputs/<kind>/<yyyy-mm-dd_HHMMSS>/`` (``_2``, ``_3``... when that second is taken) they
    are copied into and the index lists. ``main`` is the file the version is of."""
    files = {relative: _sha256(source) for relative, source in sources.items()}
    digest = _content_hash(files)
    versions = read_index(root)
    for version in versions:
        if (version.get('kind') == kind and version.get('hash') == digest
                and _holds(root, version, files)):
            return os.path.join(root, version['folder'])
    first_used = _stamp()
    parent = os.path.join(root, INPUTS_DIR, kind)
    name = first_used.strftime(STAMP_FORMAT)
    target, n = os.path.join(parent, name), 2
    while os.path.lexists(target):
        target, n = os.path.join(parent, f'{name}_{n}'), n + 1
    for relative, source in sources.items():
        destination = os.path.join(target, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)
    versions.append({'kind': kind, 'file': main, 'folder': os.path.relpath(target, root),
                     'hash': digest, 'first_used': first_used.isoformat(), 'runs': [],
                     'files': files})
    _write_index(root, versions)
    return target


#: Serialises :func:`store_input`'s read-modify-write of ``inputs/index.json`` within a process
#: (an app saving several inputs at once).
_INDEX_LOCK = threading.Lock()


def _version_holding(versions, root, path):
    """The version in ``versions`` whose folder holds ``path``, or None."""
    for version in versions:
        folder = version.get('folder')
        if folder and _inside(path, os.path.join(root, folder)):
            return version
    return None


def store_input(working_dir, kind, source, name=None):
    """Store ``source`` as an input of ``kind`` in ``working_dir``; its path relative to
    ``working_dir``.

    ``source`` is a file, a directory or bytes. A file is stored as ``name`` (default: its own
    name, with the CellML files it imports); a directory's files are stored as they are laid out
    in it, leaving out hidden files and folders (``.git``, ``.settings``...) as the working dir's
    zip does (:func:`export_working_dir`), ``name`` being the file among them the version is of
    (the folder is returned when there is none); bytes are stored
    as the file ``name``. Each content is stored once (:func:`_place`): the version of ``kind``
    holding the same files is reused, else a new folder ``inputs/<kind>/<yyyy-mm-dd_HHMMSS>/``
    holds it and ``inputs/index.json`` lists it. ``kind`` may have subfolders (``workflows/<wf>``).

    The version returned is marked ``stored`` with the time, so :func:`latest_input` gives the
    content saved last even when it is an earlier one saved again. A file or directory already
    inside ``working_dir`` is not copied, only marked when it is in a listed version.
    """
    root = os.path.abspath(working_dir)
    kind = str(kind).strip('/')
    if not kind or os.path.isabs(kind) or os.pardir in kind.split('/'):
        raise ValueError(f'not an input kind: {kind!r}')
    staging = None
    if isinstance(source, (bytes, bytearray)):
        if not name:
            raise ValueError('store_input needs a name for bytes')
        staging = tempfile.mkdtemp(prefix='cuflynx_store_')
        path = os.path.join(staging, os.path.basename(name))
        with open(path, 'wb') as handle:
            handle.write(source)
        sources, main = {os.path.basename(name): path}, os.path.basename(name)
    else:
        path = os.path.abspath(str(source))
        if os.path.isdir(path):
            sources = {}
            for d, dirs, names in os.walk(path):
                dirs[:] = [x for x in dirs if not x.startswith('.')]
                sources.update({os.path.relpath(os.path.join(d, n), path): os.path.join(d, n)
                                for n in names if not n.startswith('.')})
            main = os.path.normpath(name) if name else None
        elif name and os.path.basename(name) != os.path.basename(path):
            sources, main = {os.path.basename(name): path}, os.path.basename(name)
        else:
            _, sources, main = _bundle(path)
    try:
        with _INDEX_LOCK:
            if staging is None and _inside(path, root):
                stored = path
            else:
                folder = _place(root, kind, sources, main)
                stored = os.path.join(folder, main) if main else folder
            versions = read_index(root)
            version = _version_holding(versions, root, stored)
            if version is not None:
                version['stored'] = datetime.now().isoformat()
                _write_index(root, versions)
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)
    return os.path.relpath(stored, root)


def latest_input(working_dir, kind, name=None):
    """The absolute path of the stored input of ``kind`` (called ``name``, when given) that was
    stored last -- by :func:`store_input`, else by when it was first used -- or None.

    Only versions ``inputs/index.json`` lists are found, and only while their file exists.
    """
    root = os.path.abspath(working_dir)
    kind = str(kind).strip('/')
    best, best_key = None, None
    for position, version in enumerate(read_index(root)):
        if version.get('kind') != kind or not version.get('folder'):
            continue
        file = version.get('file')
        if name is not None and (not file or os.path.basename(file) != os.path.basename(name)):
            continue
        path = os.path.join(root, version['folder'], file) if file else \
            os.path.join(root, version['folder'])
        if not os.path.exists(path):
            continue
        key = (str(version.get('stored') or version.get('first_used') or ''), position)
        if best_key is None or key > best_key:
            best, best_key = path, key
    return best


def _record_run(root, used, run):
    """Add ``run`` (an output directory) to the ``runs`` of each version holding a path in
    ``used``."""
    versions = read_index(root)
    if not versions:
        return
    run = os.path.abspath(run)
    run = os.path.relpath(run, root) if _inside(run, root) else run
    changed = False
    for version in versions:
        folder = os.path.join(root, version.get('folder') or '')
        if version.get('folder') and any(_inside(p, folder) for p in used):
            runs = version.setdefault('runs', [])
            if run not in runs:
                runs.append(run)
                changed = True
    if changed:
        _write_index(root, versions)


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


def _derivation_sources(parameters):
    """``{path relative to the parameters file's directory: file}``: the
    ``<name>_derivations.json`` beside ``parameters`` and the files its derivations read (CellML
    with its imports, Python), as ``pack_workflow`` packs them; empty when there is none. A file
    outside that directory, or a derivations file that does not load, is warned about."""
    from libcuflynx.derivations import derivation_files, derivations_file_for
    source = derivations_file_for(parameters)
    if not os.path.isfile(source):
        return {}
    here, found = os.path.dirname(source), {os.path.basename(source): source}
    try:
        files = derivation_files(source)
    except (ValueError, OSError) as e:
        print(f'[cuflynx_study] WARNING: {source} does not load ({e}), so only it is copied, '
              f'not the files its derivations read')
        return found
    for path in files:
        relative = os.path.relpath(path, here)
        if relative.startswith(os.pardir):
            print(f'[cuflynx_study] WARNING: {source} reads {path}, outside its directory, so '
                  f'it is not copied into the working dir')
        elif os.path.isfile(path):
            found[relative] = path
    return found


def _system_sources(paths, system):
    """``system`` (the study-level block, absolute paths) with what is outside the working dir
    copied into ``inputs/system/`` as one directory, laid out as generation reads it: the
    module array and parameters file side by side (with the parameters file's derivations,
    :func:`_derivation_sources`), ``modules/`` (the library versions used, as
    ``module_version_files`` lays them out) and ``external_modules/``."""
    libraries = [os.path.abspath(d) for d in system.get('module_library_dirs') or []]
    external = system.get('external_modules_dir')
    externals = [os.path.abspath(d) for d in
                 ([] if not external else [external] if isinstance(external, str) else external)]
    files = (system['module_array'], system['parameters'])
    outside = [p for p in files + tuple(libraries + externals) if not _inside(p, paths.root)]
    if not outside:
        return system
    derivations = _derivation_sources(system['parameters'])
    key = ('system', paths.root) + tuple(
        (p, os.stat(p).st_mtime_ns, os.stat(p).st_size)
        for p in files + tuple(derivations.values())) + (
        tuple(libraries), tuple(externals), system.get('use_builtin_modules'))
    if key in _BROUGHT and os.path.isdir(_BROUGHT[key][0]):
        target, sources = _BROUGHT[key]
    else:
        sources, copied = {}, dict(system)
        if any(not _inside(p, paths.root) for p in files):
            for name in SYSTEM_KEYS:
                sources[os.path.basename(system[name])] = system[name]
                copied[name] = os.path.basename(system[name])
            sources.update(derivations)
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
        main = copied['module_array'] if copied['module_array'] in sources else None
        target = _place(paths.root, 'system', sources, main)
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
        #: Every input path recorded, as it is in the working dir: the versions a run used.
        self.used = []

    def output(self, value):
        """An output: relative inside the working dir, absolute outside it (never copied)."""
        if not self.copy:
            return _study_path(self.here, value)
        full = os.path.abspath(str(value))
        return os.path.relpath(full, self.here) if _inside(full, self.root) else full

    def outside(self, value):
        """Whether an output is outside the working dir, so a zip of it does not hold it."""
        return self.copy and not _inside(os.path.abspath(str(value)), self.root)

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
        self.used.append(full)
        return os.path.relpath(full, self.here)


#: The outside output directories warned about in this process, so each is warned about once
#: rather than at every manifest update of a running calibration.
_WARNED_OUTSIDE = set()


def _warn_outside(directory, root):
    if directory in _WARNED_OUTSIDE:
        return
    _WARNED_OUTSIDE.add(directory)
    print(f'[cuflynx_study] WARNING: the output directory {directory} is outside the working dir '
          f'{root}, so it will not travel with the working dir (a zip of {root} will not contain '
          f'it). The manifest records it by absolute path, flagged "outside": true.')


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


def _note_yaml_inputs(record, current, inp, paths):
    """Record in ``yaml_inputs`` the stored inputs this run read, for the yaml it was given and
    for today's archive of its resolved config, so an export can point each yaml at that run's
    versions rather than the latest. An archive names the last run of its day, as its content
    does. ``current`` holds the study-level keys of this run, as stored."""
    used = {key: current[key] for key in YAML_INPUT_KEYS if key in current}
    archive = os.path.join(user_inputs_archive_dir(inp), dated_user_inputs_name())
    names = [current.get('user_inputs')]
    if os.path.isfile(archive):
        names.append(os.path.relpath(archive, paths.here))
    noted = record.setdefault('yaml_inputs', {})
    for name in names:
        if name and not os.path.isabs(name):
            noted[name] = {**noted.get(name, {}), **used}


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

        current = {}
        for key, value in _study_level(inp, study_dir).items():
            if value is not None:
                current[key] = record[key] = _encode(paths, key, value)
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
            if paths.outside(entry['dir']):
                item['outside'] = True
                _warn_outside(item['dir'], paths.root)
            obs = item.get('obs_data') or inp.get('param_id_obs_path')
            if obs:
                item['obs_data'] = paths.input(obs, 'obs_data')
            if item.get('method'):
                item.setdefault('analyses', [item['method']])
            item['updated'] = _now()
            record[section] = _upsert(record.get(section), item)
            if paths.copy:
                _record_run(paths.root, paths.used, entry['dir'])

        if paths.copy:
            _note_yaml_inputs(record, current, inp, paths)
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


def outside_outputs(study):
    """The output directories a study records outside its working dir, which a zip of the
    working dir does not hold: ``[{'section', 'dir', 'method'}]``, newest first in each
    section, from a :func:`read_study` result (or a manifest's path, read with it)."""
    if isinstance(study, (str, os.PathLike)):
        study = read_study(study)
    return [{'section': section, 'dir': entry['dir'], 'method': entry.get('method')}
            for section in ENTRY_SECTIONS for entry in study.get(section) or []
            if isinstance(entry, dict) and entry.get('outside') and entry.get('dir')]


# -- sharing a working dir --------------------------------------------------------------------

#: Keys an archived user_inputs yaml loses in a zip: what the parser works out again, and what
#: only points back at this machine.
_EXPORT_DROPPED_KEYS = ('generated_models_subdir', 'param_id_output_dir_abs_path',
                        'user_inputs_path', 'user_inputs_path_override', 'cuflynx_working_dir',
                        'model_path', 'uncalibrated_model_path', 'params_for_id_path',
                        'vessels_csv_abs_path', 'parameters_csv_abs_path',
                        'vessels_0d_csv_abs_path', 'vessels_1d_csv_abs_path')

#: Keys with an absolute path that is not an input: caches and pipes in the system temp folder.
_EXPORT_MACHINE_KEYS = ('solver_info', 'coupler_pipe_dir')

#: Output keys a yaml cannot give relative to itself (the parser leaves them to the cwd), so a
#: zip's copy leaves them out, and the default in the working dir applies (``{}``: file_prefix).
_EXPORT_DEFAULTED_OUTPUTS = {
    'cpp_generated_models_dir': os.path.join('generated_models', '{}_cpp'),
    'cpp_output_dir': os.path.join('generated_models', '{}_cpp', 'simulation_outputs_cpp'),
}


def _is_absolute(value):
    return isinstance(value, str) and os.path.isabs(value)


def _absolute_values(value, found):
    """Add to ``found`` every absolute path in ``value``, a parsed yaml, at any depth."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in _EXPORT_MACHINE_KEYS:
                _absolute_values(item, found)
    elif isinstance(value, list):
        for item in value:
            _absolute_values(item, found)
    elif _is_absolute(value):
        found.add((value, None))


def _portable_inputs(out, data, used, folder, inputs):
    """Point ``out``, a yaml in ``folder`` of the working dir, at the stored inputs of its run.

    ``used`` holds them relative to the working dir, None for one not in it. An input with no
    copy keeps the path it had, so a run elsewhere stops at it rather than falling back on a
    default. ``inputs`` gets ``(path as the yaml had it, its copy or None)``. The parser takes
    ``resources_dir`` and ``param_id_obs_path`` relative to the yaml, as it does the funcs, the
    model, the module dirs and ``emulator_dir``, and ``input_param_file`` and
    ``params_for_id_file`` relative to ``resources_dir``.
    """
    def point(key, copy, holder=out, original=None):
        given = holder.get(key)
        names = [given] if isinstance(given, str) else list(given or [])
        names = names or ([original] if original else [])
        copies = copy if isinstance(copy, list) else [copy]
        if None in copies:
            inputs.update((name, None) for name in names)
            if original and not given:
                holder[key] = original
            return
        relative = [os.path.relpath(path, folder) for path in copies]
        holder[key] = relative if isinstance(copy, list) else relative[0]
        if names != relative:
            names = names if len(names) == len(copies) else [key] * len(copies)
            inputs.update(zip(names, copies))

    system = used.get('system') or {}
    params_for_id = used.get('params_for_id')
    resources = next((os.path.dirname(path) for path in (
        system.get('module_array'), system.get('parameters'), params_for_id) if path), None)
    point('resources_dir', resources)
    if resources and system.get('parameters'):
        out['input_param_file'] = os.path.relpath(system['parameters'], resources)
    if resources and params_for_id:
        out['params_for_id_file'] = os.path.relpath(params_for_id, resources)
    for key in ('module_library_dirs', 'external_modules_dir'):
        if key in system or out.get(key):
            point(key, system.get(key))
    point('param_id_obs_path', used.get('obs_data'))
    funcs = used.get('user_funcs') or {}
    for kind, key in USER_FUNC_KEYS.items():
        if kind in funcs or out.get(key):
            point(key, funcs.get(kind))
    model_key = ('external_model_path' if out.get('model_type') == 'external_python'
                 else 'generated_model_path')
    if 'base_model' in used or data.get('model_path'):
        point(model_key, used.get('base_model'), original=data.get('model_path'))
    if out.get('use_emulator'):
        settings = dict(out.get('emulator_settings') or {})
        if 'emulator' in used or settings.get('emulator_dir'):
            point('emulator_dir', used.get('emulator'), holder=settings)
        out['emulator_settings'] = settings

    left = set()
    _absolute_values({key: value for key, value in out.items() if out.get('couple_to_1d')
                      or key != 'cpp_1d_model_config_path'}, left)
    inputs.update(left)


def _portable_user_inputs(data, folder, root, moved, used, inputs):
    """A parsed archived yaml, held in ``folder`` of the working dir ``root``, as a zip holds it.

    The parser takes ``param_id_output_dir`` and ``generated_models_dir`` relative to the yaml
    and ``sa_options`` ``output_dir`` and ``param_reduction_options`` ``sensitivity_dir`` relative to
    ``<W>/sensitivity`` (its ``output_dir`` relative to its ``sensitivity_dir``). ``moved`` gets
    ``{outside directory: its path relative to the working dir}``; ``used`` and ``inputs`` are
    those of :func:`_portable_inputs`.
    """
    def place(original, default):
        original = os.path.normpath(original)
        if _inside(original, root):
            return os.path.relpath(original, root)
        return moved.setdefault(original, default)

    up = os.path.relpath(root, os.path.join(root, folder))
    out = {key: value for key, value in data.items() if key not in _EXPORT_DROPPED_KEYS}
    out['param_id_output_dir'] = up

    generated = out.get('generated_models_dir')
    if not generated or _is_absolute(generated):
        relative = place(generated, 'generated_models') if generated else 'generated_models'
        out['generated_models_dir'] = os.path.join(up, relative)

    sa_options = out.get('sa_options')
    if isinstance(sa_options, dict) and _is_absolute(sa_options.get('output_dir')):
        original = os.path.normpath(sa_options['output_dir'])
        relative = place(original, os.path.join('sensitivity', os.path.basename(original)))
        out['sa_options'] = dict(sa_options, output_dir=os.path.relpath(relative, 'sensitivity'))

    reduction = out.get('param_reduction_options')
    if isinstance(reduction, dict) and any(
            _is_absolute(reduction.get(key)) for key in ('sensitivity_dir', 'output_dir')):
        # the parser reads sensitivity_dir relative to <W>/sensitivity, output_dir to sensitivity_dir
        reduction = dict(reduction)
        sens = reduction.get('sensitivity_dir')
        if _is_absolute(sens):
            sens = os.path.normpath(sens)
            sens_rel = place(sens, os.path.join('sensitivity', os.path.basename(sens)))
            reduction['sensitivity_dir'] = os.path.relpath(sens_rel, 'sensitivity')
        else:
            # unset, the parser takes the Sobol run's output_dir (already rewritten above)
            sens = sens or (out.get('sa_options') or {}).get('output_dir') or '.'
            sens_rel = os.path.join('sensitivity', sens)
        if _is_absolute(reduction.get('output_dir')):
            target = os.path.normpath(reduction['output_dir'])
            target_rel = place(target, os.path.join(sens_rel, os.path.basename(target)))
            reduction['output_dir'] = os.path.relpath(target_rel, sens_rel)
        out['param_reduction_options'] = reduction

    settings = out.get('emulator_settings')
    if (isinstance(settings, dict) and _is_absolute(settings.get('emulator_dir'))
            and not out.get('use_emulator')):
        from libcuflynx.emulators.emulator_trainer import resolve_emulator_dir
        kept = {key: value for key, value in settings.items() if key != 'emulator_dir'}
        default = resolve_emulator_dir(dict(out, emulator_settings=kept, param_id_output_dir=root))
        place(settings['emulator_dir'], os.path.relpath(default, root))
        out['emulator_settings'] = kept

    for key, default in _EXPORT_DEFAULTED_OUTPUTS.items():
        if _is_absolute(out.get(key)):
            place(out.pop(key), default.format(out.get('file_prefix')))
    _portable_inputs(out, data, used, folder, inputs)
    return out


def _portable_yaml(path, relative, root, moved, used, inputs):
    """The text of the archived yaml at ``path`` as a zip holds it, or None to copy it as it is."""
    try:
        with open(path, encoding='utf-8') as handle:
            data = yaml.load(handle, Loader=yaml.FullLoader)
        if not isinstance(data, dict):
            return None
        out = _portable_user_inputs(data, os.path.dirname(relative), root, moved, used, inputs)
        return yaml.safe_dump(out, default_flow_style=False, sort_keys=False)
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        print(f'[cuflynx_study] WARNING: could not read {path} ({error}), so it goes in the zip '
              f'as it is and may name folders outside the working dir')
        return None


def _drop_absolute(value, found):
    """``value`` without the absolute paths in it, at any depth, which are added to ``found``."""
    def absolute(item):
        if _is_absolute(item):
            found.add(item)
            return True
        return False

    if isinstance(value, dict):
        return {k: _drop_absolute(v, found) for k, v in value.items() if not absolute(v)}
    if isinstance(value, list):
        return [_drop_absolute(v, found) for v in value if not absolute(v)]
    return value


def _portable_manifest(raw, location, moved, found):
    """A manifest as a zip holds it: no outside entry and no absolute path."""
    if raw.get('study_dir'):
        raise ValueError(f'{location} records its study_dir, so its inputs are not in the working '
                         f'dir (cuflynx_study_copy_inputs: false) and a zip would not hold them')
    out = dict(raw)
    for section in ENTRY_SECTIONS:
        if section not in raw:
            continue
        out[section] = []
        for entry in raw[section]:
            directory = entry.get('dir') if isinstance(entry, dict) else None
            if not (isinstance(entry, dict) and (entry.get('outside') or _is_absolute(directory))):
                out[section].append(entry)
            elif directory:
                directory = os.path.normpath(directory)
                moved.setdefault(directory, os.path.join(
                    'sensitivity' if section == 'sensitivity' else '',
                    os.path.basename(directory)))
    return _drop_absolute(out, found)


def _held_inputs(raw, location, root):
    """A manifest's ``yaml_inputs``, its yamls and inputs relative to the working dir ``root``,
    an input that is not there being None."""
    here = os.path.dirname(location)

    def held(value):
        if isinstance(value, dict):
            return {key: held(item) for key, item in value.items()}
        if isinstance(value, list):
            return [held(item) for item in value]
        if not isinstance(value, str):
            return value
        full = os.path.normpath(os.path.join(here, value))
        return os.path.relpath(full, root) if _inside(full, root) and os.path.exists(full) else None

    return {os.path.relpath(os.path.normpath(os.path.join(here, name)), root): held(used)
            for name, used in (raw.get('yaml_inputs') or {}).items()}


def _portable_index(raw, changed):
    """``inputs/index.json`` as a zip holds it: no absolute run, and the hash of each file
    ``changed`` (``{path relative to the working dir: bytes}``) as the zip holds it."""
    for version in raw.get('versions') or []:
        if not isinstance(version, dict):
            continue
        version['runs'] = [run for run in version.get('runs') or [] if not _is_absolute(run)]
        files = version.get('files') or {}
        stored = {relative: os.path.join(version.get('folder') or '', relative)
                  for relative in files}
        rewritten = {r: p for r, p in stored.items() if p in changed}
        for relative, path in rewritten.items():
            files[relative] = hashlib.sha256(changed[path]).hexdigest()
        if rewritten:
            version['hash'] = _content_hash(files)
    return raw


def _write_zip(zip_path, root, files, changed):
    """Write ``files`` (relative to ``root``) to ``zip_path`` atomically, in their order, with
    the content in ``changed`` in place of the file's own."""
    directory = os.path.dirname(zip_path)
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix='.cuflynx_study_', suffix='.zip', dir=directory)
    os.close(handle)
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
            for relative in files:
                path, name = os.path.join(root, relative), relative.replace(os.sep, '/')
                if relative in changed:
                    info = zipfile.ZipInfo.from_file(path, name, strict_timestamps=False)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, changed[relative])
                else:
                    archive.write(path, name)
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, zip_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def export_working_dir(working_dir, zip_path):
    """Write a zip of ``working_dir`` to ``zip_path`` to share it; ``working_dir`` is not changed.

    The zip's paths are relative to the working dir, in sorted order, without hidden files; an
    empty folder (a run's output directory that nothing has been written to) stays. Only
    in the zip, so that a run from it reads and writes only in the recipient's working dir:

    - every archived user_inputs yaml (``inputs/user_inputs/``) names its output directories
      relative to where it is unzipped. ``param_id_output_dir`` is the unzipped working dir. An
      outside ``sa_options`` ``output_dir`` becomes ``sensitivity/<its name>``, as does an
      outside ``param_reduction_options`` ``sensitivity_dir``; an outside ``output_dir`` becomes
      ``<sensitivity_dir>/<its name>`` and one inside the working dir stays where it is,
      ``generated_models_dir`` ``generated_models`` and ``emulator_dir`` the default under
      ``emulators/``, and a yaml the parser cannot read is kept as it is, with a warning;
    - every such yaml names its inputs by their stored copies, the versions its own run used
      (``yaml_inputs`` in the manifest). The model goes in ``generated_model_path``, or
      ``external_model_path`` for ``external_python``. An input with no copy keeps its absolute
      path, with a warning, so a run elsewhere stops at it;
    - every manifest drops its ``"outside"`` entries and any absolute path;
    - ``inputs/index.json`` drops absolute ``runs`` and has the hashes of the rewritten files.

    Each outside directory gets one warning (rank 0) naming it and its relative path, and saying
    the results there are not in the zip, and each rewritten input one naming its copy.

    Returns:
        ``[{'original', 'relative'}]``, one per path left behind, ``relative`` (to the working
        dir) being None for one with no output home, such as a ``--write-calibrated`` file, then
        one per input rewritten or left absolute, ``{'original', 'relative', 'input': True}``,
        ``relative`` being its copy or None; [] on ranks other than 0, which write nothing.

    Raises:
        FileNotFoundError: ``working_dir`` holds no manifest.
        ValueError: a manifest records ``study_dir`` (``cuflynx_study_copy_inputs: false``).
    """
    if _rank() != 0:
        return []
    root = os.path.abspath(working_dir)
    zip_path = os.path.abspath(zip_path)
    files = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        held = [os.path.join(folder, name) for name in names
                if not name.startswith('.') and os.path.join(folder, name) != zip_path]
        files += [os.path.relpath(path, root) for path in held or ([folder] * (not dirs))]
    files.sort()
    manifests = [f for f in files if f.endswith(MANIFEST_SUFFIX)]
    if not manifests:
        raise FileNotFoundError(f'{root} holds no *{MANIFEST_SUFFIX}')

    records = {}
    for relative in manifests:
        with open(os.path.join(root, relative), encoding='utf-8') as handle:
            records[relative] = json.load(handle)
    used = {}
    for relative in sorted(manifests, key=lambda name: str(records[name].get('updated'))):
        used.update(_held_inputs(records[relative], os.path.join(root, relative), root))

    changed, moved, found, inputs = {}, {}, set(), set()
    archive = os.path.join(INPUTS_DIR, 'user_inputs') + os.sep
    for relative in files:
        if relative.startswith(archive) and relative.endswith(('.yaml', '.yml')):
            text = _portable_yaml(os.path.join(root, relative), relative, root, moved,
                                  used.get(relative, {}), inputs)
            if text is not None:
                changed[relative] = text.encode('utf-8')
    for relative in manifests:
        record = _portable_manifest(records[relative], os.path.join(root, relative), moved, found)
        changed[relative] = (json.dumps(record, indent=2) + '\n').encode('utf-8')
    index = os.path.relpath(_index_path(root), root)
    if index in files:
        with open(_index_path(root), encoding='utf-8') as handle:
            record = _portable_index(json.load(handle), changed)
        changed[index] = (json.dumps(record, indent=2) + '\n').encode('utf-8')
    _write_zip(zip_path, root, files, changed)

    left_behind = [{'original': original, 'relative': relative}
                   for original, relative in sorted(moved.items())]
    for item in left_behind:
        print(f'[cuflynx_study] WARNING: the output directory {item["original"]} is outside the '
              f'working dir {root}, so {zip_path} names it by the relative path '
              f'{item["relative"]} (relative to the working dir) instead, and the results there '
              f'are not in the zip. Your own files are unchanged.')
    for path in sorted(found - set(moved)):
        print(f'[cuflynx_study] WARNING: {path} is outside the working dir {root}, so the '
              f'manifests in {zip_path} leave it out and it is not in the zip. Your own files '
              f'are unchanged.')
        left_behind.append({'original': path, 'relative': None})
    for path, copy in sorted(inputs, key=lambda item: (item[0], item[1] or '')):
        if copy is None:
            print(f'[cuflynx_study] WARNING: the input {path} has no copy in the working dir '
                  f'{root}, so the yamls in {zip_path} that name it keep that path, and a run of '
                  f'them on another machine stops there. Your own files are unchanged.')
        else:
            print(f'[cuflynx_study] WARNING: the yamls in {zip_path} name the input {path} by its '
                  f'copy {copy} (relative to the working dir) instead. Your own files are '
                  f'unchanged.')
        left_behind.append({'original': path, 'relative': copy, 'input': True})
    return left_behind
