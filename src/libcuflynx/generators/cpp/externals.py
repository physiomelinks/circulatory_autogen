'''
Collect the external variables of a generated model and the code that fills them.

Every value that comes from outside the CellML model -- a boundary condition received from the
FV 1D solver, the 1D blood volume, a delayed variable, a value set by a 3D model through an API --
is a libCellML external variable. The generated C code asks for it through its
``externalVariable(voi, states, rates, variables, index)`` callback, and the C++ wrapper answers
from one cache (``ext_cache``) that the code built here keeps up to date.

Nothing in this module is specific to one coupled model: named-pipe exchanges are generated from
the ``calls`` of an ``api`` block (see ``api.py`` and the FV1D entries in
``resources/coupling_modules_config.json``).
'''

import re
from dataclasses import dataclass, field

from libcuflynx.utilities.config_schemas import is_heart_vessel_type
from libcuflynx.generators.cpp.api import is_api, call_whens, unit_factor, APIConfigError

HOOKS = ('init', 'step_start', 'rhs_start', 'rhs', 'step_end')


class ExternalsError(RuntimeError):
    pass


@dataclass
class ModelRef:
    '''A variable of the flat model, resolved after analysis to a state or variables index.'''
    variable: object
    label: str
    kind: str = None  # 'state' | 'variable'
    index: int = None
    symbol: str = None  # named index in the generated C, e.g. S_heart_module_q_lv / V_heart_u_lv_ext

    @property
    def name(self):
        '''The index as written in generated code: its named constant, or the number.'''
        return self.symbol if self.symbol else str(self.index)

    def cpp(self, states='s', variables='v'):
        if self.kind == 'state':
            return f'{states}[{self.name}]'
        return f'{variables}[{self.name}]'


@dataclass
class ExternalSpec:
    '''A variable made external in the analyser, plus whatever the handler needs to fill it.'''
    ref: ModelRef
    source: str               # 'pipe' | 'delay' | 'api'
    deps: list = field(default_factory=list)   # ModelRefs the analyser must compute first
    initial: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class PipeConnection:
    key: str                  # key in conn_1d_0d_info ("1", "2", ...)
    vessel: str
    side: str                 # 'inlet' | 'outlet' of the 0D module
    input_quantity: str       # 'flow' | 'pressure' received from the other model
    input_ref: ModelRef
    output_ref: ModelRef
    control_ref: ModelRef = None
    api: dict = None


def flat_variable(flat_model, component_name, variable_name):
    comp = flat_model.component(component_name, True)
    if comp is None:
        raise ExternalsError(f"Component '{component_name}' not found in the flattened model.")
    var = comp.variable(variable_name)
    if var is None:
        raise ExternalsError(f"Variable '{variable_name}' not found in component '{component_name}' "
                             f"of the flattened model.")
    return var


# ---------------------------------------------------------------------------------------------
# Named-pipe APIs (FV 1D coupling)
# ---------------------------------------------------------------------------------------------

# The heart has several vessel ports on each side; which one a 1D vessel connects to is decided
# by name, as the CellML generator does (CVSCellMLGenerator vessel-port matching).
HEART_PORT_BY_NEIGHBOUR = {
    'outlet': [(('aorta', 'aortic_root'), 1, 'u_root'), (('par',), 1, 'u_par')],
    'inlet': [(('ivc',), 0, 'v_ivc'), (('svc',), 0, 'v_svc'), (('pvn',), 0, 'v_pvn')],
}


def _fv1d_neighbours(vessels_df, row, side):
    names = row.out_vessels if side == 'outlet' else row.inp_vessels
    out = []
    for name in names:
        match = vessels_df.loc[vessels_df['name'] == name]
        if len(match) == 1 and is_api(match.iloc[0].get('api', 'None')) and \
                match.iloc[0]['api'].get('transport') == 'named_pipe':
            out.append(match.iloc[0])
    return out


def _select_vessel_port(row, side, neighbour_name):
    ports = row.exit_ports if side == 'outlet' else row.entrance_ports
    vessel_ports = [p for p in ports if p['port_type'] == 'vessel_port']
    if len(vessel_ports) == 1:
        return vessel_ports[0]
    if row.name == 'heart' or str(row.module_type).startswith('heart') or is_heart_vessel_type(row.vessel_type):
        for keys, var_pos, var_name in HEART_PORT_BY_NEIGHBOUR[side]:
            if any(k in neighbour_name for k in keys):
                for p in vessel_ports:
                    if p['variables'][var_pos] == var_name:
                        return p
    raise ExternalsError(
        f"Module '{row.name}' has {len(vessel_ports)} vessel ports on its {side} side, and the one "
        f"connected to 1D vessel '{neighbour_name}' can't be identified.")


def _control_port_variable(row, side, port_type):
    ports = row.exit_ports if side == 'outlet' else row.entrance_ports
    for p in ports:
        if p['port_type'] == port_type:
            return p['variables'][0]
    return None


def collect_named_pipe_connections(vessels_df, conn_1d_0d_info, flat_model):
    '''Build the pipe connections (and their external variables) from conn_1d_0d_info.

    conn_1d_0d_info is written by CSV0DModelParser.split_0d_1d_module_array; this fills in its
    cellml_idx / port_idx / port_state0_or_var1 / R_T_variable_idx entries once indices exist.
    '''
    connections, volume_specs, externals = [], [], []
    if not conn_1d_0d_info:
        return connections, volume_specs, externals

    for key, info in conn_1d_0d_info.items():
        idx0 = int(info['vess0d_idx'])
        if idx0 < 0 or idx0 >= len(vessels_df):
            raise ExternalsError(f'conn_1d_0d_info[{key}] refers to 0D vessel index {idx0}, '
                                 f'which is outside the 0D module array.')
        row = vessels_df.iloc[idx0]

        if info.get('port_volume_sum') == 1:
            api = row.get('api', 'None')
            if not is_api(api):
                raise ExternalsError(f"Volume-sum module '{row['name']}' has no api block in its module config.")
            recv_calls = [c for c in api['calls'] if c['kind'] in ('recv', 'send_recv')]
            var_name = recv_calls[0]['recv'][0]
            ref = ModelRef(flat_variable(flat_model, row['name'], var_name), f"{row['name']}/{var_name}")
            spec = ExternalSpec(ref, 'pipe', meta={'conn_key': key, 'role': 'volume_sum'})
            externals.append(spec)
            volume_specs.append({'key': key, 'spec': spec, 'api': api})
            continue

        side = 'inlet' if info['cellml_bc_in0_or_out1'] == 0 else 'outlet'
        tup = next(vessels_df.iloc[[idx0]].itertuples())
        neighbours = _fv1d_neighbours(vessels_df, row, side)
        if not neighbours:
            raise ExternalsError(f"conn_1d_0d_info[{key}]: 0D module '{row['name']}' has no 1D (named-pipe api) "
                                 f"neighbour on its {side} side. Check the module array.")
        if len(neighbours) > 1:
            by_index = [n for n in neighbours if re.fullmatch(rf"FV1D_0*{info['vess1d_idx']}", n['name'])]
            neighbours = by_index if len(by_index) == 1 else neighbours[:1]
        neighbour = neighbours[0]
        api = neighbour['api']

        port = _select_vessel_port(tup, side, neighbour['name'])
        flow_var, pressure_var = port['variables'][0], port['variables'][1]
        bc_letter = tup.BC_type[0] if side == 'inlet' else tup.BC_type[1]
        if bc_letter not in ('v', 'p'):
            raise ExternalsError(f"Module '{row['name']}' has BC type {tup.BC_type}; its {side} must be a "
                                 f"flow (v) or pressure (p) boundary condition to couple to a 1D vessel.")
        input_quantity = 'flow' if bc_letter == 'v' else 'pressure'
        expected = 0 if input_quantity == 'flow' else 1
        if info['cellml_bc_flow0_or_press1'] != expected:
            raise ExternalsError(f"conn_1d_0d_info[{key}] says the 0D side receives "
                                 f"{'flow' if info['cellml_bc_flow0_or_press1'] == 0 else 'pressure'}, but module "
                                 f"'{row['name']}' BC type {tup.BC_type} receives {input_quantity} on its {side}.")
        in_name = flow_var if input_quantity == 'flow' else pressure_var
        out_name = pressure_var if input_quantity == 'flow' else flow_var

        comp = row['name']
        input_ref = ModelRef(flat_variable(flat_model, comp, in_name), f'{comp}/{in_name}')
        output_ref = ModelRef(flat_variable(flat_model, comp, out_name), f'{comp}/{out_name}')
        control_ref = None
        control_name = _control_port_variable(tup, side, 'FV_resistance_port')
        if control_name is not None:
            control_ref = ModelRef(flat_variable(flat_model, comp, control_name), f'{comp}/{control_name}')

        conn = PipeConnection(key, comp, side, input_quantity, input_ref, output_ref, control_ref, api)
        connections.append(conn)
        externals.append(ExternalSpec(input_ref, 'pipe', meta={'conn_key': key, 'role': 'bc'}))

    return connections, volume_specs, externals


def fill_conn_info(conn_1d_0d_info, connections, volume_specs):
    for conn in connections:
        info = conn_1d_0d_info[conn.key]
        info['cellml_idx'] = conn.input_ref.index
        info['port_idx'] = conn.output_ref.index
        info['port_state0_or_var1'] = 0 if conn.output_ref.kind == 'state' else 1
        info['R_T_variable_idx'] = conn.control_ref.index if conn.control_ref is not None else -1
    for vol in volume_specs:
        conn_1d_0d_info[vol['key']]['cellml_idx'] = vol['spec'].ref.index


def _value_expr(token, conn=None):
    '''C++ expression for one element of a call's "send" list.'''
    if isinstance(token, (int, float)):
        return repr(float(token))
    if token in ('$voi',):
        return 'voiLoc'
    if token in ('$dt', '$dt_stage'):
        return 'dtLoc'
    if token.startswith('port.'):
        which = token.split('.', 1)[1]
        if conn is None:
            raise APIConfigError(f"'{token}' can only be used in a per-connection call")
        if which == 'output':
            return conn.output_ref.cpp()
        if which == 'input':
            return conn.input_ref.cpp()
        if which in ('flow', 'pressure'):
            ref = conn.input_ref if which == conn.input_quantity else conn.output_ref
            return ref.cpp()
        raise APIConfigError(f'Unknown port value {token!r}')
    if token.startswith('control.'):
        port_type, _, default = token.split('.', 1)[1].partition('|')
        if conn is not None and conn.control_ref is not None:
            return conn.control_ref.cpp()
        return repr(float(default or 0.0))
    raise APIConfigError(f'Unknown value {token!r} in an api call')


def _recv_targets(tokens, buffer, conn=None, var_refs=None):
    lines = []
    for pos, token in enumerate(tokens):
        if token == '$ignore':
            continue
        if token == '$dt':
            lines.append(f'dt = {buffer}[{pos}];')
        elif token.startswith('port.'):
            which = token.split('.', 1)[1]
            if which not in ('input', conn.input_quantity):
                raise APIConfigError(f"A connection can only receive its input ({conn.input_quantity}), not {token!r}")
            lines.append(f'setExternal({conn.input_ref.name}, {buffer}[{pos}]);')
        else:
            ref = (var_refs or {}).get(token)
            if ref is None:
                raise APIConfigError(f'Unknown receive target {token!r}')
            lines.append(f'setExternal({ref.name}, {buffer}[{pos}]);')
    return lines


def build_named_pipe_code(connections, volume_specs):
    '''C++ for the named-pipe transport: pipe members, open/close, and code for each hook.'''
    apis = []
    for conn in connections:
        if conn.api not in apis:
            apis.append(conn.api)
    if len(apis) > 1:
        names = {a['name'] for a in apis}
        if len(names) > 1:
            raise ExternalsError(f'More than one named-pipe api in one model is not supported yet: {names}')
    if not apis:
        return None
    api = apis[0]
    msg_len = int(api.get('message_length', 2))
    channels = api['channels']

    calls = list(api['calls'])
    for vol in volume_specs:
        calls += [dict(c, _volume=vol) for c in vol['api']['calls']]
    # Only the channels some call uses are opened: the coupler creates e.g. the volume FIFO only
    # when the model has a volume sum, and opening a missing FIFO fails.
    used_channels = {c['channel'] for c in calls}

    # Pipes, in the order they must be opened (all writers, then all readers, in channel order;
    # a FIFO open blocks until the other end opens, so the order must match the coupler's).
    conn_keys = [c.key for c in connections]
    send_pipes, recv_pipes = [], []
    for cname, ch in channels.items():
        if cname not in used_channels:
            continue
        indexed = ch.get('indexed_by') == 'connection'
        for direction, bucket in (('send', send_pipes), ('recv', recv_pipes)):
            if direction not in ch:
                continue
            if indexed:
                for k in conn_keys:
                    bucket.append({'member': f'pipe_{direction}_{cname}_{k}', 'name': ch[direction].replace('{i}', k),
                                   'channel': cname, 'conn': k})
            else:
                bucket.append({'member': f'pipe_{direction}_{cname}', 'name': ch[direction],
                               'channel': cname, 'conn': None})

    def pipe(direction, cname, key=None):
        suffix = f'_{key}' if (key is not None and channels[cname].get('indexed_by') == 'connection') else ''
        return f'pipe_{direction}_{cname}{suffix}'

    hooks = {h: [] for h in HOOKS}
    for call in calls:
        for when in call_whens(call):
            code = hooks[when]
            per_conn = call.get('per') == 'connection'
            if per_conn:
                # all sends, then all receives -- the coupler relays every 0D message before any reply
                if call['kind'] in ('send', 'send_recv'):
                    for conn in connections:
                        tokens = call['send_by_input'][conn.input_quantity] if 'send_by_input' in call else call['send']
                        vals = [_value_expr(t, conn) for t in tokens] + ['0.0'] * (msg_len - len(tokens))
                        code.append(f'{{ double msg[{msg_len}] = {{{", ".join(vals)}}}; '
                                    f'pipeWrite({pipe("send", call["channel"], conn.key)}, msg); }}')
                if call['kind'] in ('recv', 'send_recv'):
                    for conn in connections:
                        code.append(f'{{ double msg[{msg_len}]; pipeRead({pipe("recv", call["channel"], conn.key)}, msg);')
                        code += ['  ' + l for l in _recv_targets(call['recv'], 'msg', conn)]
                        code.append('}')
            else:
                if call['kind'] in ('send', 'send_recv'):
                    tokens = call['send']
                    vals = [_value_expr(t) for t in tokens] + ['0.0'] * (msg_len - len(tokens))
                    code.append(f'{{ double msg[{msg_len}] = {{{", ".join(vals)}}}; '
                                f'pipeWrite({pipe("send", call["channel"])}, msg); }}')
                if call['kind'] in ('recv', 'send_recv'):
                    var_refs = {}
                    if '_volume' in call:
                        spec = call['_volume']['spec']
                        var_refs = {t: spec.ref for t in call['recv'] if not t.startswith('$')}
                    code.append(f'{{ double msg[{msg_len}]; pipeRead({pipe("recv", call["channel"])}, msg);')
                    code += ['  ' + l for l in _recv_targets(call['recv'], 'msg', None, var_refs)]
                    code.append('}')

    return {
        'api_name': api['name'],
        'message_length': msg_len,
        'send_pipes': send_pipes,
        'recv_pipes': recv_pipes,
        'hooks': hooks,
    }


# ---------------------------------------------------------------------------------------------
# Delays
# ---------------------------------------------------------------------------------------------

def collect_delays(vessels_df, flat_model):
    delays = []
    if 'delay_info' not in vessels_df.columns:
        return delays
    for row in vessels_df.itertuples():
        info = row.delay_info
        if not isinstance(info, dict):
            continue
        for vtd, dv, amount in zip(info['variables_to_delay'], info['delayed_variables'], info['delay_amounts']):
            comp = row.name
            delayed = ModelRef(flat_variable(flat_model, comp, dv), f'{comp}/{dv}')
            source = ModelRef(flat_variable(flat_model, comp, vtd), f'{comp}/{vtd}')
            amount_ref = ModelRef(flat_variable(flat_model, comp, amount), f'{comp}/{amount}')
            delays.append(ExternalSpec(delayed, 'delay', deps=[source, amount_ref],
                                       meta={'source': source, 'amount': amount_ref}))
    return delays


# ---------------------------------------------------------------------------------------------
# Provider APIs (a class the other model calls, e.g. lifex Circulation): an external module in
# the module array, coupled to CellML modules through its ports
# ---------------------------------------------------------------------------------------------

def provider_port_refs(vessels_df, prow, flat_model):
    """Map each variable on the ports of an external (api) module to the CellML variable it is
    connected to.

    Ports connect by port_type exactly as between CellML modules: an entrance port of the api
    module takes the same-typed exit (or general) port of a module in its inp_vessels, an exit port
    feeds the same-typed entrance (or general) port of a module in its out_vessels, and variables
    pair up by position. Returns {api variable name: ModelRef}.
    """
    refs = {}

    def partner_ports(names, kinds):
        for name in names:
            match = vessels_df.loc[vessels_df['name'] == name]
            if len(match) != 1 or match.iloc[0]['module_format'] != 'cellml':
                continue
            partner = match.iloc[0]
            for kind in kinds:
                for port in partner[kind] if isinstance(partner[kind], list) else []:
                    yield partner['name'], port

    sides = [('entrance_ports', prow['inp_vessels'], ('exit_ports', 'general_ports')),
             ('exit_ports', prow['out_vessels'], ('entrance_ports', 'general_ports')),
             ('general_ports', list(prow['inp_vessels']) + list(prow['out_vessels']),
              ('general_ports', 'entrance_ports', 'exit_ports'))]
    for own_kind, neighbours, partner_kinds in sides:
        for port in prow[own_kind] if isinstance(prow[own_kind], list) else []:
            found = [(comp, pp) for comp, pp in partner_ports(neighbours, partner_kinds)
                     if pp['port_type'] == port['port_type']]
            if not found:
                raise ExternalsError(f"Port '{port['port_type']}' of external module '{prow['name']}' is not "
                                     f"connected to a CellML module with a matching port; check the module array.")
            comp, pp = found[0]
            if len(pp['variables']) != len(port['variables']):
                raise ExternalsError(f"Port '{port['port_type']}' has {len(port['variables'])} variables on "
                                     f"'{prow['name']}' but {len(pp['variables'])} on '{comp}'.")
            for own_var, partner_var in zip(port['variables'], pp['variables']):
                refs[own_var] = ModelRef(flat_variable(flat_model, comp, partner_var), f'{comp}/{partner_var}')
    return refs


def collect_provider_apis(vessels_df, flat_model):
    """External modules (module-array rows) whose api block has role: provider.

    Their api functions name the module's own port variables (resolved through the ports to the
    connected CellML modules), or an absolute "component/variable".
    Returns a list of dicts: {api, vessel, set_specs, get_refs, state_refs}.
    """
    providers = []
    if 'api' not in vessels_df.columns:
        return providers
    for _, prow in vessels_df.iterrows():
        api = prow['api']
        if not is_api(api) or api.get('role') != 'provider' or api.get('transport') != 'cpp_class':
            continue
        port_refs = provider_port_refs(vessels_df, prow, flat_model)

        def ref_for(name, _refs=port_refs, _row=prow['name']):
            if '/' in name:
                comp, _, var = name.rpartition('/')
                return ModelRef(flat_variable(flat_model, comp, var), f'{comp}/{var}')
            if name not in _refs:
                raise ExternalsError(f"api variable '{name}' of '{_row}' is not on one of its ports "
                                     f"(or give it as component/variable).")
            return _refs[name]

        set_specs, get_refs, state_refs = {}, {}, {}
        for fn in api['functions']:
            kind = fn['kind']
            if kind == 'set':
                if fn['variable'] not in set_specs:
                    init = fn.get('initial')
                    set_specs[fn['variable']] = ExternalSpec(ref_for(fn['variable']), 'api',
                                                             initial=float(init) if init is not None else None,
                                                             meta={'function': fn['name']})
            elif kind == 'get':
                get_refs.setdefault(fn['variable'], ref_for(fn['variable']))
            elif kind == 'set_state':
                state_refs.setdefault(fn['variable'], ref_for(fn['variable']))
        for chamber, fields in api.get('chamber_enum', {}).get('values', {}).items():
            for field_name, var in fields.items():
                if field_name in ('pressure_set', 'set'):
                    if var not in set_specs:
                        set_specs[var] = ExternalSpec(ref_for(var), 'api', initial=None, meta={'chamber': chamber})
                else:
                    get_refs.setdefault(var, ref_for(var))
        providers.append({'api': api, 'vessel': prow['name'], 'set_specs': set_specs,
                          'get_refs': get_refs, 'state_refs': state_refs})
    return providers


# ---------------------------------------------------------------------------------------------
# Python external models (api transport "python"): rows of the module array whose api block
# names a Python class. Their port variables are exchanged with the connected CellML modules
# through the C interface (templates/model0d_capi.cpp.j2) that libcuflynx.coupling drives.
# ---------------------------------------------------------------------------------------------

@dataclass
class ExchangeVariable:
    '''One port variable of a Python external model, and the 0D variable(s) it is connected to.'''
    name: str                 # "<row>/<port variable>"
    row: str
    variable: str
    refs: list                # one ModelRef per connected 0D module, in neighbour order
    neighbours: list          # the connected 0D modules' names, in the same order
    direction: str = None     # 'to_external' | 'from_external'
    units: str = ''
    factor: float = 1.0
    specs: list = field(default_factory=list)  # ExternalSpecs (from_external only)


def python_rows(vessels_df):
    '''Module-array rows whose api block has transport "python".'''
    if 'api' not in vessels_df.columns:
        return []
    return [row for _, row in vessels_df.iterrows()
            if is_api(row['api']) and row['api'].get('transport') == 'python']


def _all_port_refs(vessels_df, prow, flat_model):
    '''{own port variable: [(neighbour name, ModelRef), ...]} over every CellML module connected to
    each port of the external row (entrance ports: inp_vessels; exit ports: out_vessels; general
    ports: both), in the order the module array lists them. Variables pair up by position.'''
    refs = {}
    sides = [('entrance_ports', list(prow['inp_vessels']), ('exit_ports', 'general_ports')),
             ('exit_ports', list(prow['out_vessels']), ('entrance_ports', 'general_ports')),
             ('general_ports', list(prow['inp_vessels']) + list(prow['out_vessels']),
              ('general_ports', 'entrance_ports', 'exit_ports'))]
    for own_kind, neighbours, partner_kinds in sides:
        for port in prow[own_kind] if isinstance(prow[own_kind], list) else []:
            found = []
            for name in neighbours:
                match = vessels_df.loc[vessels_df['name'] == name]
                if len(match) != 1 or match.iloc[0]['module_format'] != 'cellml':
                    continue
                partner = match.iloc[0]
                for kind in partner_kinds:
                    for pp in partner[kind] if isinstance(partner[kind], list) else []:
                        if pp['port_type'] == port['port_type']:
                            found.append((name, pp))
                            break
                    else:
                        continue
                    break
            if not found:
                raise ExternalsError(
                    f"Port '{port['port_type']}' of external module '{prow['name']}' is not connected to any "
                    f"CellML module with a matching port. Name the module(s) in its "
                    f"{'inp_vessels' if own_kind == 'entrance_ports' else 'out_vessels' if own_kind == 'exit_ports' else 'inp/out_vessels'} "
                    f"and check they have a '{port['port_type']}' port.")
            for name, pp in found:
                if len(pp['variables']) != len(port['variables']):
                    raise ExternalsError(f"Port '{port['port_type']}' has {len(port['variables'])} variables on "
                                         f"'{prow['name']}' but {len(pp['variables'])} on '{name}'.")
                for own_var, partner_var in zip(port['variables'], pp['variables']):
                    if own_var in refs and any(n == name for n, _ in refs[own_var]):
                        raise ExternalsError(f"'{prow['name']}/{own_var}' is on more than one port connected "
                                             f"to '{name}'.")
                    refs.setdefault(own_var, []).append(
                        (name, ModelRef(flat_variable(flat_model, name, partner_var), f'{name}/{partner_var}')))
    return refs


def variable_kinds(flat_model):
    '''Analyse the flat model with no external variables and return a function giving, for a
    variable of the flat model, 'constant' (a parameter or an unconnected boundary condition),
    'state' or 'computed'.'''
    import libcellml
    import libcuflynx.utilities.libcellml_helper_funcs as cellml
    analyser = libcellml.Analyser()
    analyser.analyseModel(flat_model)
    am = cellml.get_analysed_model(analyser)
    constant_type = libcellml.AnalyserVariable.Type.CONSTANT

    def kind(variable):
        for astate in am.states():
            if am.areEquivalentVariables(astate.variable(), variable):
                return 'state'
        for av in am.variables():
            if am.areEquivalentVariables(av.variable(), variable):
                return 'constant' if av.type() == constant_type else 'computed'
        return None
    return kind


def collect_python_exchange(vessels_df, flat_model):
    '''The exchange table of every python-transport row, with each variable's direction:

    * connected to a CellML variable the 0D model computes (a state or an equation's result):
      ``to_external``, the Python model receives it;
    * connected to a constant (a boundary condition left open for the external model, whose
      parameter value is the starting value): ``from_external``, the Python model sets it, and it
      becomes a libCellML external variable of the generated C.

    ``api.variables[<var>].direction`` overrides the inference. Returns (rows, exchange) where
    rows is [{row, api, variables: [ExchangeVariable]}].
    '''
    rows, exchange = [], []
    py_rows = python_rows(vessels_df)
    if not py_rows:
        return rows, exchange
    kind = variable_kinds(flat_model)
    for prow in py_rows:
        api = prow['api']
        overrides = api.get('variables') or {}
        units_of = {v[0]: v[1] for v in prow['variables_and_units']} \
            if isinstance(prow['variables_and_units'], list) else {}
        port_refs = _all_port_refs(vessels_df, prow, flat_model)
        unknown = set(overrides) - set(port_refs)
        if unknown:
            raise ExternalsError(f"api.variables of '{prow['name']}' names {sorted(unknown)}, which are not "
                                 f"variables on its ports ({sorted(port_refs)}).")
        row_vars = []
        for var, pairs in port_refs.items():
            refs = [r for _, r in pairs]
            spec = overrides.get(var, {})
            direction = spec.get('direction')
            if direction is None:
                kinds = {kind(r.variable) for r in refs}
                if None in kinds:
                    raise ExternalsError(f"'{prow['name']}/{var}': a connected variable is not in the analysed model.")
                directions = {'from_external' if k == 'constant' else 'to_external' for k in kinds}
                if len(directions) > 1:
                    raise ExternalsError(
                        f"'{prow['name']}/{var}' connects to {[r.label for r in refs]}, some computed by the 0D "
                        f"model and some not; set api.variables['{var}'].direction.")
                direction = directions.pop()
            x = ExchangeVariable(name=f"{prow['name']}/{var}", row=prow['name'], variable=var, refs=refs,
                                 neighbours=[n for n, _ in pairs], direction=direction,
                                 units=refs[0].variable.units().name() if refs[0].variable.units() else
                                 units_of.get(var, ''),
                                 factor=unit_factor(spec) if spec else 1.0)
            if direction == 'from_external':
                x.specs = [ExternalSpec(r, 'api', initial=None, meta={'exchange': x.name}) for r in refs]
            row_vars.append(x)
            exchange.append(x)
        rows.append({'row': prow['name'], 'api': api, 'variables': row_vars})
    return rows, exchange
