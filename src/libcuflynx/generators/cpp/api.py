'''
The ``api`` block of a module config entry: how a generated model talks to another model.

An entry with ``"module_format": "external_api"`` (or any entry that carries an ``api`` dict)
describes, in data, the calls that couple the generated 0D model to something outside it:

* ``role: consumer``  -- the generated code calls out to the other model (e.g. the FV 1D solver
  over named pipes). ``calls`` lists each exchange and when it happens.
* ``role: provider``  -- the generated code *is* called by the other model (e.g. a 3D heart
  driving it through a lifex ``Circulation``-style class). ``functions`` lists the methods the
  generated class exposes and which variable each one sets or gets. A provider is a row of the
  module array, connected to CellML modules through its ports; its functions name its own port
  variables (or ``component/variable``).
  ``transport: python``: the other model is a Python class (e.g. a FEniCS model). The generated
  C++ is also built as a shared library, and ``libcuflynx.coupling`` steps it together with the
  class. The class's inputs and outputs are the module's own port variables: each one connected
  to a CellML variable the 0D model computes is an input to the class, each one connected to a
  boundary condition of the 0D model is an output of the class (see externals.infer_directions).
  A port connected to several modules exchanges one value per module.
* ``role: process``   -- another program run alongside the generated model (e.g. the FV 1D solver):
  how it is launched (``program``, ``coordinator``) and the pipes it talks over (``channels``,
  ``message_length``). A consumer names it with ``"process": "<vessel_type>"`` and then lists only
  its ``calls``; the channels come from the process entry, so they are written down once. The C++
  generator writes the coordinator's run configuration (``coupler_config.json``) from it.

Anything received from the other model becomes a libCellML external variable.
'''

import copy
import os

# Version of the C interface (templates/model0d_capi.cpp.j2) that libcuflynx.coupling loads.
CAPI_VERSION = 1

API_ROLES = ('consumer', 'provider', 'process')
PROCESS_COORDINATORS = ('coupler',)
API_TRANSPORTS = ('named_pipe', 'cpp_class', 'python')
# keys a python-transport api block may carry (anything else is most likely a typo)
PYTHON_API_KEYS = {'name', 'description', 'role', 'transport', 'exchange', 'python', 'coupling_dt',
                   'subiterations', 'tol', 'relaxation', 'variables', '_config_dir'}
VARIABLE_DIRECTIONS = ('to_external', 'from_external')
API_EXCHANGES = ('per_rhs', 'per_time_step')
CALL_WHEN = ('init', 'step_start', 'rhs_start', 'rhs', 'step_end')
CALL_KINDS = ('send', 'recv', 'send_recv')
FUNCTION_KINDS = ('set', 'get', 'set_state', 'set_indexed', 'get_indexed', 'time',
                  'time_discretization', 'step', 'noop', 'extrapolate')

# Factor that converts an API value into the model's SI value: model = api * factor.
API_UNIT_FACTORS = {
    'mmHg': 133.322387415,
    'Pa': 1.0,
    'J_per_m3': 1.0,
    'dyne_per_cm2': 0.1,
    'ml': 1e-6,
    'm3': 1.0,
    'ml_per_s': 1e-6,
    'm3_per_s': 1.0,
    'second': 1.0,
    's': 1.0,
    'dimensionless': 1.0,
    'mmHg_per_ml': 133.322387415 / 1e-6,
    'J_per_m6': 1.0,
    'mM': 1.0,
    'millimolar': 1.0,
    'mol_per_m3': 1.0,
    'uM': 1e-3,
    'mol_per_s': 1.0,
    'nmol_per_s': 1e-9,
    'm2_per_s': 1.0,
    'per_s': 1.0,
}


class APIConfigError(ValueError):
    pass


def is_api(value):
    """vessels_df stores a missing module-config field as the *string* "None", never NaN."""
    return isinstance(value, dict)


def unit_factor(entry):
    """model_value = api_value * factor, for a function/call entry of an api block."""
    if 'factor' in entry:
        return float(entry['factor'])
    units = entry.get('api_units')
    if units is None:
        return 1.0
    if units not in API_UNIT_FACTORS:
        raise APIConfigError(f"Unknown api_units '{units}'. Known units: {sorted(API_UNIT_FACTORS)}. "
                             f"Give an explicit 'factor' instead.")
    return API_UNIT_FACTORS[units]


def validate_api_block(api, where=''):
    """Check the structure of an api block. Raises APIConfigError with the offending path."""
    ctx = f' in {where}' if where else ''
    if not isinstance(api, dict):
        raise APIConfigError(f'api block{ctx} must be a dict, got {type(api).__name__}')
    for key in ('name', 'role', 'transport'):
        if key not in api:
            raise APIConfigError(f"api block{ctx} is missing '{key}'")
    if api['role'] not in API_ROLES:
        raise APIConfigError(f"api role{ctx} must be one of {API_ROLES}, got {api['role']!r}")
    if api['transport'] not in API_TRANSPORTS:
        raise APIConfigError(f"api transport{ctx} must be one of {API_TRANSPORTS}, got {api['transport']!r}")
    exchange = api.get('exchange', 'per_time_step')
    if exchange not in API_EXCHANGES:
        raise APIConfigError(f"api exchange{ctx} must be one of {API_EXCHANGES}, got {exchange!r}")

    if api['role'] == 'process':
        _validate_process(api, ctx)
        return True

    if api['transport'] == 'named_pipe':
        if 'channels' not in api and 'process' in api:
            # the channels come from the process entry: the calls are checked against them once
            # resolve_process_apis has run, and only their own structure here
            channels = None
        else:
            _validate_channels(api, ctx)
            channels = api['channels']
        calls = api.get('calls')
        if not isinstance(calls, list) or not calls:
            raise APIConfigError(f"named_pipe api{ctx} needs a non-empty 'calls' list")
        for call in calls:
            _validate_call(call, channels, ctx)
    elif api['transport'] == 'python':
        _validate_python(api, ctx)
    elif api['transport'] == 'cpp_class':
        functions = api.get('functions')
        if not isinstance(functions, list) or not functions:
            raise APIConfigError(f"cpp_class api{ctx} needs a non-empty 'functions' list")
        for fn in functions:
            if 'name' not in fn or 'kind' not in fn:
                raise APIConfigError(f"api function{ctx} needs 'name' and 'kind': {fn}")
            if fn['kind'] not in FUNCTION_KINDS:
                raise APIConfigError(f"api function '{fn['name']}'{ctx} has unknown kind {fn['kind']!r}; "
                                     f"expected one of {FUNCTION_KINDS}")
            if fn['kind'] in ('set', 'get', 'set_state') and 'variable' not in fn:
                raise APIConfigError(f"api function '{fn['name']}'{ctx} of kind {fn['kind']} needs 'variable'")
            if fn['kind'] in ('set_indexed', 'get_indexed', 'extrapolate'):
                if 'chamber_enum' not in api:
                    raise APIConfigError(f"api function '{fn['name']}'{ctx} is indexed but the api has no 'chamber_enum'")
            unit_factor(fn)
    return True


def python_model_path(api):
    '''Absolute path of a python-transport api's model file (relative to its module config).'''
    path = os.path.expanduser(str(api['python']['file']))
    if not os.path.isabs(path):
        path = os.path.join(api.get('_config_dir', os.getcwd()), path)
    return os.path.abspath(path)


def _validate_python(api, ctx):
    if api['role'] != 'provider':
        raise APIConfigError(f"python api{ctx} must have role 'provider' (the generated model is a library the "
                             f"Python model is stepped with), got {api['role']!r}")
    unknown = set(api) - PYTHON_API_KEYS
    if unknown:
        raise APIConfigError(f"python api{ctx} has unknown key(s) {sorted(unknown)}; allowed: "
                             f"{sorted(k for k in PYTHON_API_KEYS if not k.startswith('_'))}")
    py = api.get('python')
    if not isinstance(py, dict) or not isinstance(py.get('file'), str) or not isinstance(py.get('class'), str):
        raise APIConfigError(f"python api{ctx} needs \"python\": {{\"file\": \"<model>.py\", \"class\": "
                             f"\"<ClassName>\"}} (the file relative to the module config)")
    if '_config_dir' in api and not os.path.isfile(python_model_path(api)):
        raise APIConfigError(f"python api{ctx}: model file {python_model_path(api)} not found "
                             f"(api.python.file is relative to the module config's folder)")
    for key in ('coupling_dt', 'tol'):
        if key in api and not (isinstance(api[key], (int, float)) and api[key] > 0):
            raise APIConfigError(f"python api{ctx}: '{key}' must be a positive number, got {api[key]!r}")
    if 'subiterations' in api and not (isinstance(api['subiterations'], int) and api['subiterations'] >= 0):
        raise APIConfigError(f"python api{ctx}: 'subiterations' must be a non-negative integer")
    if 'relaxation' in api and not (isinstance(api['relaxation'], (int, float)) and 0 < api['relaxation'] <= 1):
        raise APIConfigError(f"python api{ctx}: 'relaxation' must be in (0, 1]")
    for var, spec in (api.get('variables') or {}).items():
        if not isinstance(spec, dict):
            raise APIConfigError(f"python api{ctx}: variables['{var}'] must be a dict")
        if 'direction' in spec and spec['direction'] not in VARIABLE_DIRECTIONS:
            raise APIConfigError(f"python api{ctx}: variables['{var}'].direction must be one of "
                                 f"{VARIABLE_DIRECTIONS}")
        unit_factor(spec)


def _validate_channels(api, ctx):
    channels = api.get('channels')
    if not isinstance(channels, dict) or not channels:
        raise APIConfigError(f"named_pipe api{ctx} needs a 'channels' dict")
    for cname, ch in channels.items():
        if not isinstance(ch, dict) or not ({'send', 'recv'} & set(ch)):
            raise APIConfigError(f"channel '{cname}'{ctx} needs a 'send' and/or 'recv' pipe name")


def _validate_process(api, ctx):
    if api['transport'] != 'named_pipe':
        raise APIConfigError(f"process api{ctx} must use transport 'named_pipe', got {api['transport']!r}")
    _validate_channels(api, ctx)
    if api.get('coordinator') not in PROCESS_COORDINATORS:
        raise APIConfigError(f"process api{ctx} needs a 'coordinator', one of {PROCESS_COORDINATORS}")
    program = api.get('program')
    if not isinstance(program, dict) or 'script' not in program or \
            not ({'package', 'path'} & set(program)):
        raise APIConfigError(f"process api{ctx} needs a 'program' with a 'script' and a 'package' or 'path'")
    if 'calls' in api:
        raise APIConfigError(f"process api{ctx} lists no calls: the consumers that use it do")


def _validate_call(call, channels, ctx):
    for key in ('name', 'kind', 'when', 'channel'):
        if key not in call:
            raise APIConfigError(f"api call{ctx} is missing '{key}': {call}")
    if call['kind'] not in CALL_KINDS:
        raise APIConfigError(f"api call '{call['name']}'{ctx} has unknown kind {call['kind']!r}")
    whens = call['when'] if isinstance(call['when'], list) else [call['when']]
    for w in whens:
        if w not in CALL_WHEN:
            raise APIConfigError(f"api call '{call['name']}'{ctx} has unknown 'when' {w!r}; expected {CALL_WHEN}")
    if call['kind'] in ('send', 'send_recv') and not ('send' in call or 'send_by_input' in call):
        raise APIConfigError(f"api call '{call['name']}'{ctx} needs 'send' or 'send_by_input'")
    if call['kind'] in ('recv', 'send_recv') and 'recv' not in call:
        raise APIConfigError(f"api call '{call['name']}'{ctx} needs 'recv'")
    if channels is None:
        return
    if call['channel'] not in channels:
        raise APIConfigError(f"api call '{call['name']}'{ctx} uses undeclared channel {call['channel']!r}")
    ch = channels[call['channel']]
    if call['kind'] in ('send', 'send_recv') and 'send' not in ch:
        raise APIConfigError(f"api call '{call['name']}'{ctx} sends on channel '{call['channel']}' which has no send pipe")
    if call['kind'] in ('recv', 'send_recv') and 'recv' not in ch:
        raise APIConfigError(f"api call '{call['name']}'{ctx} receives on channel '{call['channel']}' which has no recv pipe")


def call_whens(call):
    return call['when'] if isinstance(call['when'], list) else [call['when']]


def resolve_process_apis(apis):
    """Give every consumer that names a ``process`` that process's channels and message length.

    ``apis`` is a list of (where, api dict) pairs, edited in place. A consumer's own channels, if
    any, are added to (and override) the process's. Each consumer also gets ``process_api``: the
    process entry's api block, which the C++ generator uses for the coordinator's configuration.
    """
    processes = {}
    for where, api in apis:
        if api.get('role') == 'process':
            processes[where[0]] = api
    for where, api in apis:
        name = api.get('process')
        if name is None or api.get('role') == 'process':
            continue
        if name not in processes:
            raise APIConfigError(f"api in module config {where} names process {name!r}, but no module config "
                                 f"entry with vessel_type {name!r} has an api with role 'process'.")
        proc = processes[name]
        if proc['transport'] != api['transport']:
            raise APIConfigError(f"api in module config {where} uses transport {api['transport']!r} but its "
                                 f"process {name!r} uses {proc['transport']!r}.")
        own = api.get('_own_channels', api.get('channels', {}))
        api['_own_channels'] = own
        api['channels'] = {**copy.deepcopy(proc['channels']), **own}
        api.setdefault('message_length', proc.get('message_length', 2))
        api['process_api'] = proc
        validate_api_block(api, where=f'module config {where}')


def validate_module_config_apis(module_df):
    """Validate every api block in a loaded module-config DataFrame (called when configs load),
    and resolve the consumers that name a process (see resolve_process_apis)."""
    if 'api' not in module_df.columns:
        return
    apis = []
    for row in module_df.itertuples():
        api = getattr(row, 'api')
        if is_api(api):
            where = (row.vessel_type, row.BC_type)
            validate_api_block(api, where=f"module config {where}")
            apis.append((where, api))
    resolve_process_apis(apis)

