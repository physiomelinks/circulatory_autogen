'''
The ``api`` block of a module config entry: how a generated model talks to another model.

An entry with ``"module_format": "external_api"`` (or any entry that carries an ``api`` dict)
describes, in data, the calls that couple the generated 0D model to something outside it:

* ``role: consumer``  -- the generated code calls out to the other model (e.g. the FV 1D solver
  over named pipes). ``calls`` lists each exchange and when it happens.
* ``role: provider``  -- the generated code *is* called by the other model (e.g. a 3D heart
  driving it through a lifex ``Circulation``-style class). ``functions`` lists the methods the
  generated class exposes and which variable each one sets or gets. A provider is a row of the
  vessel array, connected to CellML modules through its ports; its functions name its own port
  variables (or ``component/variable``).

Anything received from the other model becomes a libCellML external variable.
'''

API_ROLES = ('consumer', 'provider')
API_TRANSPORTS = ('named_pipe', 'cpp_class')
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

    if api['transport'] == 'named_pipe':
        channels = api.get('channels')
        if not isinstance(channels, dict) or not channels:
            raise APIConfigError(f"named_pipe api{ctx} needs a 'channels' dict")
        for cname, ch in channels.items():
            if not isinstance(ch, dict) or not ({'send', 'recv'} & set(ch)):
                raise APIConfigError(f"channel '{cname}'{ctx} needs a 'send' and/or 'recv' pipe name")
        calls = api.get('calls')
        if not isinstance(calls, list) or not calls:
            raise APIConfigError(f"named_pipe api{ctx} needs a non-empty 'calls' list")
        for call in calls:
            _validate_call(call, channels, ctx)
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
    if call['channel'] not in channels:
        raise APIConfigError(f"api call '{call['name']}'{ctx} uses undeclared channel {call['channel']!r}")
    ch = channels[call['channel']]
    if call['kind'] in ('send', 'send_recv') and 'send' not in ch:
        raise APIConfigError(f"api call '{call['name']}'{ctx} sends on channel '{call['channel']}' which has no send pipe")
    if call['kind'] in ('recv', 'send_recv') and 'recv' not in ch:
        raise APIConfigError(f"api call '{call['name']}'{ctx} receives on channel '{call['channel']}' which has no recv pipe")
    if call['kind'] in ('send', 'send_recv') and not ('send' in call or 'send_by_input' in call):
        raise APIConfigError(f"api call '{call['name']}'{ctx} needs 'send' or 'send_by_input'")
    if call['kind'] in ('recv', 'send_recv') and 'recv' not in call:
        raise APIConfigError(f"api call '{call['name']}'{ctx} needs 'recv'")


def call_whens(call):
    return call['when'] if isinstance(call['when'], list) else [call['when']]


def validate_module_config_apis(module_df):
    """Validate every api block in a loaded module-config DataFrame (called when configs load)."""
    if 'api' not in module_df.columns:
        return
    for row in module_df.itertuples():
        api = getattr(row, 'api')
        if is_api(api):
            validate_api_block(api, where=f"module config ({row.vessel_type}, {row.BC_type})")

