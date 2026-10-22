'''
Packing a calibration workflow: every file it needs to run, and none of its results, in one
directory, so the workflow can be run again anywhere (CUFLynx zips it into an OMEX archive).

Layout of the packed directory::

    calibration_workflow.json     the workflow, its paths rewritten to the files below
    inputs/<step_id>/             each step's obs_data, params_for_id, and module array with
                                  its parameters file (a file several steps share is packed
                                  once, under the first)
    inputs/_target/               the target's module array and parameters file, with its
                                  variable_mapping.csv if it has one
    modules/                      each module version the workflow uses, laid out as in its
                                  library, and every units file of that library
                                  (modules/<n>/ for the n-th of several libraries)
    settings/                     the files and directories its settings name
                                  (external_modules_dir, *_funcs_external_path)

Built-in modules are not packed: they come with libcuflynx.
'''

import copy
import json
import os
import shutil

from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.spec import (PATH_SETTINGS, WORKFLOW_FILE_NAME,
                                                  load_workflow)

INPUTS_DIR = 'inputs'
MODULES_DIR = 'modules'
SETTINGS_DIR = 'settings'
TARGET_OWNER = '_target'


def _ignore_hidden(directory, names):
    return [n for n in names if n.startswith('.')]


class _Packer:
    def __init__(self, dest_dir):
        self.dest_dir = os.path.abspath(dest_dir)
        self.packed = {}           # realpath of a source -> its path in the pack
        self.files = set()         # paths written, relative to dest_dir

    def file(self, source, owner):
        '''Copy ``source`` into ``inputs/<owner>/`` (once); its path in the pack.'''
        real = os.path.realpath(source)
        if real in self.packed:
            return self.packed[real]
        stem, ext = os.path.splitext(os.path.basename(source))
        relative, n = os.path.join(INPUTS_DIR, owner, stem + ext), 1
        while relative in self.files:
            n += 1
            relative = os.path.join(INPUTS_DIR, owner, f'{stem}_{n}{ext}')
        self._copy(source, relative)
        self.packed[real] = relative
        return relative

    def setting(self, source, key):
        '''Copy a settings file or directory into ``settings/<key>/``; its path in the pack.'''
        real = os.path.realpath(source)
        if real in self.packed:
            return self.packed[real]
        relative = os.path.join(SETTINGS_DIR, key, os.path.basename(os.path.normpath(source)))
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(self.dest_dir, relative),
                            ignore=_ignore_hidden, dirs_exist_ok=True)
            self.files |= {os.path.relpath(os.path.join(root, f), self.dest_dir)
                           for root, _, names in os.walk(os.path.join(self.dest_dir, relative))
                           for f in names}
        else:
            self._copy(source, relative)
        self.packed[real] = relative
        return relative

    def _copy(self, source, relative):
        destination = os.path.join(self.dest_dir, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)
        self.files.add(relative)

    def tree(self, source, relative):
        destination = os.path.join(self.dest_dir, relative)
        shutil.copytree(source, destination, ignore=_ignore_hidden, dirs_exist_ok=True)
        for root, _, names in os.walk(destination):
            self.files |= {os.path.relpath(os.path.join(root, f), self.dest_dir) for f in names}


def _library_of(path, library_dirs):
    '''The index of the library directory ``path`` is in, or None.'''
    real = os.path.realpath(path)
    for i, library in enumerate(library_dirs):
        root = os.path.realpath(library)
        if os.path.commonpath([real, root]) == root:
            return i
    return None


def pack_workflow(source, dest_dir, module_library_dirs=None):
    '''Copy what the calibration workflow ``source`` (a calibration_workflow.json path) needs to
    run into ``dest_dir`` (see the module docstring), checking it first as a run would.

    Returns ``{'workflow': the packed workflow file, 'files': [every file written, relative to
    dest_dir], 'modules': [the module version directories packed, relative to dest_dir]}``.
    Running the packed workflow gives what running ``source`` does.
    '''
    workflow = load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    if workflow.raw is None or workflow.path is None:
        raise ValueError('pack_workflow packs a workflow file; save the workflow first.')
    os.makedirs(dest_dir, exist_ok=True)
    if os.listdir(dest_dir):
        raise ValueError(f'{dest_dir} is not empty; pack into a new directory.')
    packer = _Packer(dest_dir)
    raw = copy.deepcopy(workflow.raw)

    def model(spec_model, owner):
        if spec_model.kind == 'library':
            return spec_model.as_dict()
        array = packer.file(spec_model.module_array, owner)
        packed = {'module_array': array,
                  'parameters': packer.file(spec_model.parameters, owner)}
        return packed

    def settings(raw_settings, spec_settings):
        for key in PATH_SETTINGS:
            if spec_settings.get(key):
                raw_settings[key] = packer.setting(spec_settings[key], key)

    raw['target'] = model(workflow.target, TARGET_OWNER)
    if workflow.target.kind == 'array':
        mapping = resolved.target.file('variable_mapping.csv')
        if os.path.isfile(mapping):
            packer.file(mapping, TARGET_OWNER)
    settings(raw.setdefault('settings', {}), workflow.settings)
    if not raw['settings']:
        del raw['settings']
    for raw_step, step in zip(raw['steps'], workflow.steps):
        if step.model is None:
            continue                   # a reduction names no files
        raw_step['model'] = model(step.model, step.id)
        raw_step['obs_data'] = packer.file(step.obs_data, step.id)
        if step.params_for_id:
            raw_step['params_for_id'] = packer.file(step.params_for_id, step.id)
        if raw_step.get('settings'):
            settings(raw_step['settings'], step.settings)

    libraries = list(workflow.module_library_dirs)
    if libraries:
        homes = ([MODULES_DIR] if len(libraries) == 1 else
                 [os.path.join(MODULES_DIR, str(i)) for i in range(len(libraries))])
        configs = set(resolved.module_configs())
        for step in workflow.steps:
            if step.model is not None:
                configs |= set(resolved.steps[step.id].model.config_paths(
                    resolved.components, resolved.supermodules))
        versions = set()
        for config in configs:
            i = _library_of(config, libraries)
            if i is not None:          # not a built-in or external_modules_dir module
                versions.add((i, os.path.dirname(os.path.realpath(config))))
        packed_versions = []
        for i, version_dir in sorted(versions):
            relative = os.path.join(homes[i], os.path.relpath(
                version_dir, os.path.realpath(libraries[i])))
            packer.tree(version_dir, relative)
            packed_versions.append(relative)
        # units are looked up across the whole library, so every units file goes
        for i, library in enumerate(libraries):
            for root, dirs, names in os.walk(library):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in names:
                    if name.endswith('units.cellml') and not name.startswith('.'):
                        path = os.path.join(root, name)
                        packer._copy(path, os.path.join(homes[i],
                                                        os.path.relpath(path, library)))
        raw['module_library_dirs'] = homes
    else:
        packed_versions = []

    workflow_path = os.path.join(packer.dest_dir, WORKFLOW_FILE_NAME)
    with open(workflow_path, 'w') as f:
        json.dump(raw, f, indent=1)
        f.write('\n')
    packer.files.add(WORKFLOW_FILE_NAME)
    load_workflow(workflow_path)       # the packed file still parses
    return {'workflow': workflow_path, 'files': sorted(packer.files),
            'modules': packed_versions}
