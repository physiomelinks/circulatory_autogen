import ast
import pathlib
import re
import warnings

import pytest

from libcuflynx.parsers.PrimitiveParsers import (
    YamlFileParser,
    migrate_legacy_solver_info_keys,
    validate_solver_info,
    warn_if_casadi_nonzero_pre_time,
    PARAM_ID_METHODS,
    valid_param_id_methods,
    param_id_method_options,
    get_solver_info_default,
    SOLVER_SCHEMA,
    SOLVER_INFO_FIELDS,
    solver_info_fields,
    gradient_sources,
    ANALYSIS_OPTIONS,
    analysis_options,
    uq_options,
    _SOLVER_INTEGRATOR_KEYS,
    _CASADI_ADJOINT_METHODS,
    MODEL_TYPE_ALIASES,
    normalise_model_type,
)

# The descriptor shape shared by optimiser_options, solver_info fields, and analysis options.
_DESCRIPTOR_TYPES = {'int', 'float', 'bool', 'str', 'dict', 'enum'}


def _assert_descriptors_well_formed(context, options, valid_types=_DESCRIPTOR_TYPES):
    """Every option/field descriptor must carry the fields a settings UI relies on."""
    assert isinstance(options, list) and options, f'{context}: must be a non-empty list'
    seen = set()
    for opt in options:
        key = opt.get('name')
        assert key and key not in seen, f'{context}: missing/duplicate name {key!r}'
        seen.add(key)
        assert opt.get('type') in valid_types, f'{context}.{key}: bad type {opt.get("type")!r}'
        assert isinstance(opt.get('required'), bool), f'{context}.{key}: required must be bool'
        assert 'default' in opt, f'{context}.{key}: needs a default (None if none)'
        assert opt.get('description'), f'{context}.{key}: needs a description'
        if opt['type'] == 'enum':
            assert opt.get('default') in opt.get('choices', []), \
                f'{context}.{key}: enum default not in choices'
    return seen


def test_param_id_methods_schema_matches_dispatch():
    """PARAM_ID_METHODS is the discoverable list of calibration methods surfaced to downstream
    tools (e.g. the CUFLynx settings UI), so it must stay in sync with the param_id_method
    dispatch in OpencorParamID.run(). If a method is added/removed there, update this set."""
    assert set(PARAM_ID_METHODS.keys()) == {
        'genetic_algorithm', 'CMA-ES', 'bayesian', 'sp_minimize', 'multi_start_sp_minimize'
    }
    for name, meta in PARAM_ID_METHODS.items():
        assert meta.get('label') and meta.get('description')
        assert isinstance(meta.get('gradient_based'), bool)
    # aliases are surfaced by valid_param_id_methods (the dispatch accepts CMAES / cmaes for CMA-ES)
    assert set(valid_param_id_methods()) >= set(PARAM_ID_METHODS.keys()) | {'CMAES', 'cmaes'}


def test_param_id_method_options_are_well_formed():
    """Every method exposes its optimiser_options settings so a tool can auto-populate a settings
    form. Each option descriptor must carry the fields the UI relies on, with consistent types."""
    for name, meta in PARAM_ID_METHODS.items():
        _assert_descriptors_well_formed(name, meta.get('options'))
    # aliases resolve to the same options as their canonical method
    assert param_id_method_options('CMAES') == param_id_method_options('CMA-ES')
    assert param_id_method_options('not_a_method') == []


def test_param_id_method_options_match_optimiser_reads():
    """The advertised options must be the ones the optimiser classes actually read from
    optimiser_options -- otherwise a tool would offer settings that do nothing (or omit real
    ones). Guards against PARAM_ID_METHODS drifting from optimisers.py."""
    def names(method):
        return {opt['name'] for opt in param_id_method_options(method)}

    # Keys each optimiser reads from optimiser_options (see param_id/optimisers.py).
    assert names('genetic_algorithm') == {'num_calls_to_function', 'cost_convergence',
                                          'max_patience', 'num_elite', 'num_survivors',
                                          'num_mutations_per_survivor', 'num_cross_breed',
                                          'objective_function', 'use_relative_cost_tolerance',
                                          'relative_cost_tolerance'}
    assert names('CMA-ES') == {'num_calls_to_function', 'sigma0', 'cost_convergence',
                               'max_patience'}
    assert names('bayesian') == {'num_calls_to_function'}
    assert names('sp_minimize') == {'cost_convergence'}
    assert names('multi_start_sp_minimize') == {
        'num_starts', 'start_sampling', 'include_init_point', 'seed', 'fd_step',
        'no_new_starts_on_convergence', 'convergence_cluster_tol_frac', 'cost_convergence'}
    # multi-start is a superset of sp_minimize's gradient-descent settings
    assert names('sp_minimize') <= names('multi_start_sp_minimize')


def test_the_ga_stopping_rules_are_listed_together():
    """Option order is form order in a front-end that renders this schema, so it is part of the
    contract rather than incidental. The relative-tolerance pair is the same decision as
    cost_convergence and max_patience -- when to stop -- and belongs with them, not parked past
    the population sizes where a reader takes it for something unrelated."""
    order = [opt['name'] for opt in param_id_method_options('genetic_algorithm')]
    stopping = ['cost_convergence', 'max_patience',
                'use_relative_cost_tolerance', 'relative_cost_tolerance']
    first = order.index(stopping[0])
    assert order[first:first + len(stopping)] == stopping
    # and the switch comes before the value it gates, which reads backwards otherwise
    assert order.index('use_relative_cost_tolerance') < order.index('relative_cost_tolerance')


def test_solver_info_fields_schema_well_formed():
    """SOLVER_INFO_FIELDS lets a tool auto-populate the solver settings form; every solver that
    can be selected must have a well-formed field list, and it must be exposed on SOLVER_SCHEMA."""
    for solver, fields in SOLVER_INFO_FIELDS.items():
        _assert_descriptors_well_formed(solver, fields)
    # every solver offered by SOLVER_SCHEMA has a solver_info field list
    all_solvers = {s for solvers in SOLVER_SCHEMA['solvers_by_model_type'].values() for s in solvers}
    assert all_solvers <= set(SOLVER_INFO_FIELDS), \
        f'solvers without solver_info fields: {all_solvers - set(SOLVER_INFO_FIELDS)}'
    assert SOLVER_SCHEMA['solver_info_fields_by_solver'] is SOLVER_INFO_FIELDS
    assert solver_info_fields('CVODE_myokit') and solver_info_fields('not_a_solver') == []


@pytest.mark.unit
def test_the_schema_offers_current_model_type_names_only():
    """A renamed model_type is accepted but never *advertised*.

    SOLVER_SCHEMA is what a downstream tool builds its menu from (CUFLynx reads it
    to populate the "Generated model format" dropdown), so a name on its way out
    must not appear there -- otherwise the rename never finishes, because the GUI
    keeps writing the old spelling into new configs.
    """
    for old, current in MODEL_TYPE_ALIASES.items():
        assert old not in SOLVER_SCHEMA['model_types'], old
        assert old not in SOLVER_SCHEMA['solvers_by_model_type'], old
        assert old not in SOLVER_SCHEMA['default_solver_by_model_type'], old
        # The name it maps to has to be a real one, or the alias sends configs nowhere.
        assert current in SOLVER_SCHEMA['model_types'], current

    # 'cellml' specifically: it is the default, so it must be selectable.
    assert 'cellml' in SOLVER_SCHEMA['model_types']


@pytest.mark.unit
def test_the_old_cellml_only_spelling_still_resolves():
    """`cellml_only` was the default model_type, so it is written in every
    user_inputs.yaml that predates the rename -- including the dated copies CA
    archives beside every run. It has to keep working."""
    assert normalise_model_type('cellml_only') == 'cellml'
    # Everything else passes through untouched: this translates, it does not
    # validate. An invalid name must reach the caller's own check unchanged, so
    # the error names what the user actually wrote.
    assert normalise_model_type('cellml') == 'cellml'
    assert normalise_model_type('casadi_python') == 'casadi_python'
    assert normalise_model_type('not_a_model_type') == 'not_a_model_type'
    assert normalise_model_type(None) is None


@pytest.mark.unit
def test_a_config_written_before_the_rename_still_parses(tmp_path, base_user_inputs):
    """The end-to-end version of the above: the whole point of the alias is that a
    yaml saying `cellml_only` still resolves a model path rather than exiting."""
    inp = dict(base_user_inputs)
    inp['model_type'] = 'cellml_only'
    inp['generated_models_dir'] = str(tmp_path / 'generated')
    parsed = YamlFileParser().parse_user_inputs_file(
        inp, obs_path_needed=False, do_generation_with_fit_parameters=False)
    assert parsed['model_type'] == 'cellml'
    assert parsed['model_path'].endswith('.cellml')


def test_solver_integrator_keys_derived_from_schema():
    """_SOLVER_INTEGRATOR_KEYS (used by validate_solver_info) is derived from SOLVER_INFO_FIELDS,
    so the accepted keys and the advertised settings cannot drift. Locks the exact key sets that
    validation enforces today -- if a field is added to the schema, update these sets too."""
    for solver, fields in SOLVER_INFO_FIELDS.items():
        assert _SOLVER_INTEGRATOR_KEYS[solver] == {f['name'] for f in fields}
    cvode = {'MaximumStep', 'MaximumNumberOfSteps', 'rtol', 'atol'}
    assert _SOLVER_INTEGRATOR_KEYS['CVODE_opencor'] == cvode
    # CVODE_myokit is deliberately not in that family: myokit.Simulation exposes only
    # set_max_step_size / set_min_step_size / set_tolerance, so myokit_helper never reads
    # MaximumNumberOfSteps and advertising it would render a dead control downstream.
    assert _SOLVER_INTEGRATOR_KEYS['CVODE_myokit'] == {'MaximumStep', 'rtol', 'atol'}
    assert _SOLVER_INTEGRATOR_KEYS['solve_ivp'] == {
        'rtol', 'atol', 'max_step', 'vectorized', 'dense_output', 'jac'}
    assert _SOLVER_INTEGRATOR_KEYS['casadi_integrator'] == {
        'reltol', 'abstol', 'rtol', 'atol', 'max_num_steps', 'max_step_size', 'max_step',
        'options'}
    # 'max_step' comes with the stiff BDF methods (bdf_newton / bdf_tape / bdf_kernel).
    # 'gradient_strategy' selects tape vs kernel for semi_implicit_signed (issue #346).
    assert _SOLVER_INTEGRATOR_KEYS['aadc_semi_implicit'] == {
        'tol', 'threads', 'max_step', 'gradient_strategy', 'jac_lag'}


def test_schema_settings_are_actually_read_by_the_code():
    """Every setting the schemas advertise must be read somewhere in src/.

    The schemas are CUFLynx's contract -- it builds its settings forms by reading them -- so a
    setting no code consumes becomes a control the user can change with no effect and no way to
    tell. The sibling tests above cannot catch that: _SOLVER_INTEGRATOR_KEYS is *derived from*
    SOLVER_INFO_FIELDS, so they compare the schema against a copy of itself. A phantom
    'gradient_method' on aadc_semi_implicit passed them for precisely that reason (AD vs FD is
    chosen by the do_ad flag, and the AD backend follows from model_type -- nothing ever read
    it). Check the schema against the source instead.

    The search is deliberately repo-wide rather than per-solver-file, because a setting is not
    always consumed by its own helper. PrimitiveParsers.py is excluded because that is where the
    schema declares the names in the first place.

    This is the weak, repo-wide half of the check; the per-solver half is
    test_each_solver_setting_is_read_by_that_solvers_consumer below (issue #330).
    """
    src_dir = pathlib.Path(__file__).resolve().parent.parent / 'src'
    corpus = '\n'.join(
        path.read_text(errors='ignore')
        for path in src_dir.rglob('*.py')
        if path.name != 'PrimitiveParsers.py' and 'obsolete' not in path.parts
    )

    def never_read(names):
        return [n for n in names if f'"{n}"' not in corpus and f"'{n}'" not in corpus]

    unread_solver = {
        solver: never_read([f['name'] for f in fields])
        for solver, fields in SOLVER_INFO_FIELDS.items()
    }
    unread_solver = {k: v for k, v in unread_solver.items() if v}
    assert not unread_solver, (
        f'solver_info settings advertised to CUFLynx but read nowhere in src/: {unread_solver}. '
        'Either wire the setting up or remove it from SOLVER_INFO_FIELDS.')

    unread_method = {
        method: never_read([o['name'] for o in spec.get('options', [])])
        for method, spec in PARAM_ID_METHODS.items()
    }
    unread_method = {k: v for k, v in unread_method.items() if v}
    assert not unread_method, (
        f'optimiser options advertised to CUFLynx but read nowhere in src/: {unread_method}. '
        'Either wire the option up or remove it from PARAM_ID_METHODS.')


# ---------------------------------------------------------------------------
# Per-solver, consumption-aware schema check (issue #330)
# ---------------------------------------------------------------------------
# A setting must be read by *its own* solver's backend, not merely appear somewhere in src/:
# protocol_runner.py relays every key into get_simulation_helper's solver_info whether or not the
# backend does anything with it, so a repo-wide substring search counts a pass-through as a read.
#
# The solver -> backend-module mapping below is DERIVED from get_simulation_helper's own dispatch
# rather than hand-written. A hand-maintained map is the very failure mode this check exists to
# prevent: it drifts silently, and a solver whose entry went stale gets checked against the wrong
# file -- or, if its entry is simply missing, is not checked at all.

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / 'src'
_RELAY_FILES = ('libcuflynx/protocol_runners/protocol_runner.py',
                'libcuflynx/parsers/PrimitiveParsers.py')

# Genuine cross-module consumers, listed explicitly because they are not reachable from the
# factory's dispatch. Keep this list short and justified -- every entry is a hole in the
# derivation.
_EXTRA_CONSUMERS = {
    # AADC's settings are split: the integrator knobs are read by the helper the factory returns,
    # but the gradient knobs (jac_lag, gradient_strategy) are read by the tape/kernel gradient
    # paths, which the forward-solve helper never touches.
    'aadc_semi_implicit': ['libcuflynx/param_id/aadc_backend.py'],
}

# Solvers with no Python backend at all, so the factory has nothing to dispatch to. cpp models are
# never run through a SimulationHelper: solver_info is baked into the emitted C++ by the cpp branch
# of script_generate_with_new_architecture, which hands the values to CVSCppGenerator. Those two
# files are therefore the real consumer, and the fields are checked against them exactly as a
# helper module would be -- this is a different consumer, not an exemption.
_GENERATED_CODE_CONSUMERS = {
    solver: ['libcuflynx/scripts/script_generate_with_new_architecture.py',
             'libcuflynx/generators/CVSCppGenerator.py']
    for solver in ('CVODE', 'RK4', 'PETSC')
}


def _import_aliases(tree):
    """{local alias: src-relative module path} for `from x.y import Z as Alias` imports."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                if name.asname:
                    aliases[name.asname] = node.module.replace('.', '/') + '.py'
    return aliases


def _local_string_lists(func):
    """{var: [strings]} for `name = ['a', 'b']` assignments inside `func`.

    The dispatch tests membership against these (`solver in casadi_solvers`), so resolving them
    is what lets the derivation see solvers that are not compared to a literal.
    """
    out = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
            values = [e.value for e in node.value.elts
                      if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if values and len(values) == len(node.value.elts):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out[target.id] = values
    return out


def _derive_solver_consumers():
    """Read solver -> backend module out of `get_simulation_helper`'s if/elif dispatch."""
    factory_path = _SRC_DIR / 'libcuflynx' / 'solver_wrappers' / '__init__.py'
    tree = ast.parse(factory_path.read_text())
    aliases = _import_aliases(tree)
    func = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == 'get_simulation_helper')
    string_lists = _local_string_lists(func)

    def solvers_matched_by(test):
        """The solver names a branch condition selects (`== 'x'` or `in some_list`)."""
        names = []
        for node in ast.walk(test):
            if not isinstance(node, ast.Compare):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id == 'solver'):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                    names.append(comparator.value)
                elif isinstance(op, ast.In):
                    if isinstance(comparator, ast.Name):
                        names.extend(string_lists.get(comparator.id, []))
                    elif isinstance(comparator, (ast.List, ast.Tuple)):
                        names.extend(e.value for e in comparator.elts
                                     if isinstance(e, ast.Constant))
        return names

    def modules_returned_by(body):
        """The helper modules a branch constructs and returns."""
        modules = []
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call) \
                        and isinstance(node.value.func, ast.Name) \
                        and node.value.func.id in aliases:
                    modules.append(aliases[node.value.func.id])
        return modules

    consumers = {}

    def walk_chain(statements):
        for stmt in statements:
            if isinstance(stmt, ast.If):
                modules = modules_returned_by(stmt.body)
                for solver in solvers_matched_by(stmt.test):
                    for module in modules:
                        consumers.setdefault(solver, [])
                        if module not in consumers[solver]:
                            consumers[solver].append(module)
                walk_chain(stmt.orelse)  # the elif chain

    walk_chain(func.body)
    return consumers


def _setting_names_read(source):
    """String constants used as a *setting name* in `source`, via the AST rather than a substring
    search.

    A substring search over the file text counts a name that only appears in a docstring, a
    comment or an error message -- myokit_helper's own docstring lists 'Key supported solver_info
    keys', which would vouch for a key nothing reads. Only these positions count:

      solver_info['x'] / opts['x'] = ...     subscript
      solver_info.get('x') / .pop('x')       lookup
      'x' in solver_info                     membership
      key == 'x'                             the translating forwarders (rtol -> RelativeTolerance)
      [... 'x' ...] / {'x': default}         key allow-lists and default blocks

    Comments and docstrings are not any of these, so they stop counting as evidence.
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            names.add(node.slice.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ('get', 'pop') and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
        elif isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(node.left, ast.Constant) \
                        and isinstance(node.left.value, str):
                    names.add(node.left.value)
                elif isinstance(op, ast.Eq):
                    for side in (node.left, comparator):
                        if isinstance(side, ast.Constant) and isinstance(side.value, str):
                            names.add(side.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.add(element.value)
    return names


def _consumers_by_solver():
    """The full solver -> consumer-module map: derived, plus the two documented additions."""
    consumers = {solver: list(modules)
                 for solver, modules in _derive_solver_consumers().items()}
    for solver, extra in _EXTRA_CONSUMERS.items():
        consumers.setdefault(solver, []).extend(extra)
    for solver, modules in _GENERATED_CODE_CONSUMERS.items():
        consumers.setdefault(solver, []).extend(modules)
    return consumers


@pytest.mark.unit
def test_solver_consumer_map_is_derived_from_the_factory_dispatch():
    """Guard on the derivation itself.

    If get_simulation_helper is restructured so the AST walk stops recognising its dispatch, the
    map silently empties and every per-solver check below passes vacuously. Pin that the
    derivation still finds each solver the factory really dispatches, and that it resolves to the
    backend that solver actually uses.
    """
    derived = _derive_solver_consumers()
    assert derived, ('the dispatch in get_simulation_helper could not be read; '
                     'test_each_solver_setting_is_read_by_that_solvers_consumer would pass '
                     'vacuously')
    # Spot-check the two shapes the dispatch uses: `solver == 'x'` and `solver in some_list`.
    assert derived['CVODE_myokit'] == ['libcuflynx/solver_wrappers/myokit_helper.py']
    assert derived['casadi_integrator'] == ['libcuflynx/solver_wrappers/casadi_python_solver_helper.py']
    # solve_ivp is dispatched via a list rather than an equality.
    assert derived['solve_ivp'] == ['libcuflynx/solver_wrappers/python_solver_helper.py']
    for solver, modules in derived.items():
        assert solver in SOLVER_INFO_FIELDS, (
            'get_simulation_helper dispatches solver ' + solver + ' but SOLVER_INFO_FIELDS does '
            'not declare its settings, so a front-end cannot build its form')
        for module in modules:
            assert (_SRC_DIR / module).exists(), module


@pytest.mark.unit
def test_every_advertised_solver_has_a_consumer_to_check_against():
    """No solver may sit outside the check.

    Before #330 the cpp solvers were exempt, which is how they came to advertise MaximumStep,
    rtol and atol that the generated C++ never reads. An unmapped solver is not a small gap: it
    is a solver whose entire settings form is unverified.
    """
    unmapped = sorted(set(SOLVER_INFO_FIELDS) - set(_consumers_by_solver()))
    assert not unmapped, (
        'solvers advertising settings with no consumer to check them against: ' + str(unmapped)
        + '. Add the backend to get_simulation_helper (preferred, since the map is derived from '
        'it), or record the consumer in _EXTRA_CONSUMERS / _GENERATED_CODE_CONSUMERS with a '
        'comment saying why the derivation cannot see it.')


@pytest.mark.unit
def test_each_solver_setting_is_read_by_that_solvers_consumer():
    """Per-solver, consumption-aware version of the repo-wide check above (issue #330).

    A solver_info setting is CUFLynx's contract: it renders a control for it. If the backend that
    owns the solver never reads the key, the user gets a control that silently does nothing --
    and the repo-wide check cannot see that, because it only asks whether the *name* appears
    anywhere in src/, and every key appears in the relay in protocol_runner.py. That is how
    CVODE_myokit came to advertise MaximumNumberOfSteps (myokit.Simulation exposes only
    set_max_step_size / set_min_step_size / set_tolerance) and how the cpp solvers came to
    advertise MaximumStep/rtol/atol (the emitted C++ hardcodes its tolerances and steps at
    dt_solver).

    CVODE_opencor forwards solver_info wholesale into OpenCOR's odeSolverProperties() instead of
    reading each key by name, so its evidence is the alias branches (rtol -> RelativeTolerance)
    and its default block rather than four separate lookups. That is still a read of the name in
    code, which is what _setting_names_read requires.
    """
    phantom = {}
    for solver, consumers in _consumers_by_solver().items():
        read = set()
        for rel in consumers:
            path = _SRC_DIR / rel
            assert path.exists(), 'consumer file missing for ' + solver + ': ' + rel
            read |= _setting_names_read(path.read_text(errors='ignore'))
        missing = [f['name'] for f in SOLVER_INFO_FIELDS[solver] if f['name'] not in read]
        if missing:
            phantom[solver] = missing

    assert not phantom, (
        'solver_info settings advertised to CUFLynx that their own backend never reads: '
        + str(phantom) + '. CUFLynx renders a control for each, so an unread one is a knob the '
        'user can turn with no effect and no way to tell. Either wire it up in the consumer, or '
        'drop it from SOLVER_INFO_FIELDS.')


@pytest.mark.unit
def test_the_relay_does_not_count_as_reading_a_setting():
    """Guard on the guard: if a relay ever became a mapped consumer, the per-solver check above
    would silently degrade back into the repo-wide one it replaced.

    protocol_runner.py setdefault()s every key into the solver_info it forwards, so it mentions
    all of them while consuming none.
    """
    for relay in _RELAY_FILES:
        for solver, consumers in _consumers_by_solver().items():
            assert relay not in consumers, (
                relay + ' relays solver_info wholesale and must never be treated as a consumer '
                '(mapped for ' + solver + ')')


@pytest.mark.unit
def test_a_setting_named_only_in_prose_does_not_count_as_read():
    """The second gap #330 records: the old check was a substring search over the file text, so a
    key mentioned in a docstring, a comment or an error message vouched for itself."""
    prose_only = '''
"""MaximumNumberOfSteps is supported by this backend."""
# MaximumStep is also mentioned here
def run(solver_info):
    raise ValueError("set rtol to fix this")
'''
    assert _setting_names_read(prose_only) == set()

    genuinely_read = '''
def run(solver_info):
    a = solver_info["MaximumStep"]
    b = solver_info.get("rtol")
    if "atol" in solver_info:
        pass
    return a, b
'''
    assert {'MaximumStep', 'rtol', 'atol'} <= _setting_names_read(genuinely_read)


def test_analysis_options_schema_well_formed():
    """The non-calibration analysis modes (sensitivity, UQ, identifiability) expose their option
    blocks the same way, so a tool can auto-populate their settings forms too."""
    assert set(ANALYSIS_OPTIONS) == {'sensitivity_analysis', 'uq', 'identifiability_analysis',
                                     'emulation'}
    for mode, meta in ANALYSIS_OPTIONS.items():
        assert meta.get('label') and meta.get('enable_flag') and meta.get('options_key')
        _assert_descriptors_well_formed(mode, meta.get('options'))
    # option names the analysis code actually reads (sensitivityAnalysis.py / paramID.py / IA)
    def names(mode):
        return {o['name'] for o in analysis_options(mode)}
    # gradient_method and fd_rel_step are read by run_local_sensitivity for method
    # 'local' (#338): which arm differentiates, and the finite-difference step.
    assert names('sensitivity_analysis') == {
        'method', 'sample_type', 'num_samples', 'gradient_method', 'fd_rel_step'}
    # 'uq', not 'mcmc': MCMC is one UQ method, and 'method' is the seam the others are added at.
    assert names('uq') == {'method', 'library', 'num_steps', 'num_walkers', 'burn_in',
                           'num_tune', 'pymc_method', 'chain_save_every'}
    assert ANALYSIS_OPTIONS['uq']['options_key'] == 'UQ_options'
    assert names('identifiability_analysis') == {'method', 'gradient_source', 'sub_method'}
    assert names('emulation') == {
        'emulator_dir', 'models', 'num_train_samples', 'reuse_samples', 'sample_type',
        'log_scale_params', 'random_seed', 'test_fraction', 'n_splits', 'n_iter', 'min_r2',
        'out_of_bounds', 'fd_rel_step'}
    assert analysis_options('not_a_mode') == []
    # the enabling flags match the documented user_inputs feature flags
    assert {m['enable_flag'] for m in ANALYSIS_OPTIONS.values()} == {
        'do_sensitivity', 'do_uq', 'do_ia', 'do_emulation'}
    # Emulation alone carries a second flag, because it has a train step and a use step:
    # do_emulation fits the surrogate, use_emulator makes the analyses evaluate it (#333).
    assert ANALYSIS_OPTIONS['emulation']['use_flag'] == 'use_emulator'
    assert {m.get('use_flag') for m in ANALYSIS_OPTIONS.values()} == {None, 'use_emulator'}


def _option(mode, name):
    return next(o for o in analysis_options(mode) if o['name'] == name)


def test_reuse_samples_is_a_tickbox_that_says_what_it_does_not_do():
    """The setting that refits saved samples instead of re-running the simulations.

    A bool descriptor is what makes CUFLynx render a tickbox, so the type is load-bearing
    rather than cosmetic. The description has to carry the other half of the story too: it
    reuses a *previous* design, so num_train_samples/sample_type/log_scale_params stop
    applying, and there has to be a previous training run to reuse. A user who reads only the
    label would otherwise expect a fresh design of num_train_samples points.
    """
    opt = _option('emulation', 'reuse_samples')
    assert opt['type'] == 'bool', 'a bool is what a settings form renders as a tickbox'
    assert opt['default'] is False, 'reuse must be opted into; a run defaults to simulating'
    assert opt['required'] is False
    description = opt['description']
    assert 'num_train_samples' in description and 'sample_type' in description, \
        'the description must say which design settings stop applying'
    assert 'reuse_samples false' in description or 'reuse_samples: false' in description, \
        'the description must name the way out: a first run without it'

    # ... and the trainer actually reads it, so the tickbox is not decoration.
    trainer_src = (pathlib.Path(__file__).resolve().parents[1] / 'src' / 'libcuflynx' / 'emulators'
                   / 'emulator_trainer.py').read_text(encoding='utf-8')
    assert "'reuse_samples'" in trainer_src


def test_every_uq_option_the_code_reads_is_advertised():
    """A UQ setting CA reads but does not declare cannot be reached from a front-end at all.

    CUFLynx builds its UQ form from ANALYSIS_OPTIONS['uq'] and hardcodes nothing, so an option
    missing here is an option nobody can set: it stays at whatever default the ``.get()`` call
    carries, silently. That is how the pyMC backend shipped selectable but unconfigurable --
    ``library: pymc`` was advertised while ``num_tune`` and ``pymc_method``, the only two
    settings that backend adds, were not.

    Scanning the source rather than restating a list is the point. The name set above is
    hand-maintained, so it agrees with whatever it was last edited to say; this reads what the
    code actually asks ``UQ_options`` for, and so fails when a read is added without a
    descriptor, which is the direction the mistake goes in.
    """
    src_dir = pathlib.Path(__file__).resolve().parent.parent / 'src'
    # UQ_options['x'], UQ_options.get('x'), self.UQ_options.get("x", default)
    pattern = re.compile(r"""UQ_options(?:\.get\(|\[)\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]""")
    read = set()
    for path in src_dir.rglob('*.py'):
        if 'obsolete' in path.parts:
            continue
        read.update(pattern.findall(path.read_text(encoding='utf-8')))

    advertised = {o['name'] for o in analysis_options('uq')}
    assert read, 'found no UQ_options reads at all -- the pattern has stopped matching'
    assert read <= advertised, (
        f'UQ_options read by the code but not declared in ANALYSIS_OPTIONS["uq"]: '
        f'{sorted(read - advertised)}')


def test_uq_options_are_filtered_by_sampler_library():
    """A settings form should offer what the chosen sampler actually reads, and nothing else.

    ``num_tune`` and ``pymc_method`` are pyMC's alone. Offering them under emcee is not a
    cosmetic wart: the user sets a tuning count that nothing reads and an algorithm that will
    not run, and has no way to tell those from the settings that do apply.
    """
    everything = {o['name'] for o in analysis_options('uq')}
    pymc_only = {'num_tune', 'pymc_method'}

    assert {o['name'] for o in uq_options('pymc')} == everything
    assert {o['name'] for o in uq_options('emcee')} == everything - pymc_only
    # No library named at all is "show me the schema", not "show me nothing".
    assert {o['name'] for o in uq_options()} == everything


def test_an_unknown_sampler_library_still_gets_the_shared_options():
    """A front-end newer than the CA it is pointed at (or older) must degrade to the settings
    that are certainly right, rather than to an empty form with no way to configure a run."""
    assert {o['name'] for o in uq_options('zeus')} == \
        {o['name'] for o in analysis_options('uq')} - {'num_tune', 'pymc_method'}


def test_library_specific_uq_options_are_read_only_in_that_librarys_arm():
    """The annotation has to match the dispatch, or the form hides a setting that matters.

    Restating "num_tune and pymc_method are pyMC's" in the test would only agree with itself.
    This reads ``_build_sampler``, which is where the choice is actually made, and requires that
    an option marked for one library is read inside that library's branch and not before it --
    so moving a read out of the pyMC arm without dropping the annotation fails here.
    """
    import inspect

    from libcuflynx.param_id.paramID import OpencorMCMC

    source = inspect.getsource(OpencorMCMC._build_sampler)
    before_pymc, marker, pymc_arm = source.partition("if library == 'pymc'")
    assert marker, "_build_sampler no longer dispatches on library == 'pymc'"

    annotated = [o for o in analysis_options('uq') if o.get('libraries')]
    assert annotated, 'no UQ option is annotated with the libraries that read it'
    for opt in annotated:
        assert opt['libraries'] == ['pymc'], (
            f"{opt['name']}: this test only knows how to check the pyMC arm; extend it "
            'alongside a new library-specific option')
        assert opt['name'] in pymc_arm, (
            f"{opt['name']} is advertised as pyMC-only but _build_sampler's pyMC arm never "
            'reads it')
        assert opt['name'] not in before_pymc, (
            f"{opt['name']} is advertised as pyMC-only but is read before the pyMC arm, so "
            'another backend reads it too and the form must offer it there')


def test_closed_set_analysis_options_are_enums_with_choices():
    """Every option whose consumer dispatches on a fixed set of values must be declared
    'enum' with those values in 'choices' -- not a free 'str'.

    The schema is what front-ends build their settings forms from, so a free string
    becomes a text box for what is really a menu: the user can type something that
    only fails once the run is under way, and a GUI cannot offer a dropdown without
    hardcoding the list (which then drifts from CA).

    Choices are pinned to the dispatch sites, so adding a branch there without
    updating the schema fails here:
      * sample_type -> sobolSA._generate_samples (raises ValueError otherwise)
      * sub_method  -> utility_funcs.calculate_hessian
      * method      -> sensitivityAnalysis / identifiabilityAnalysis
    """
    expected = {
        ('sensitivity_analysis', 'method'): ['sobol', 'local'],
        ('sensitivity_analysis', 'sample_type'): ['saltelli', 'sobol'],
        ('identifiability_analysis', 'method'): ['Laplace', 'profile_likelihood'],
        ('identifiability_analysis', 'gradient_source'): ['FD', 'AD', 'FSA'],
        # sub_method's 'AD' branch in calculate_hessian raises NotImplementedError, so it is
        # deliberately absent from sub_method's choices -- AD is now reached via gradient_source
        # instead (the Fisher-information path), not the calculate_hessian sub_method.
        ('identifiability_analysis', 'sub_method'): ['parabola_fit', 'numdifftools_finite_diff'],
        # Only what is implemented may be offered: a menu entry a front-end can select but CA
        # cannot run is the same defect as a setting nothing reads. Extend these as the SMC /
        # surrogate methods and the pyMC backend land.
        ('uq', 'method'): ['mcmc'],
        # 'zeus' is still accepted by _build_sampler for backwards compatibility but is
        # deliberately not advertised: it is not a CA dependency and there is no extra that
        # installs it, so offering it in a menu would be a control that fails for most users.
        ('uq', 'library'): ['emcee', 'pymc'],
        # -> PyMCSampler.__init__, which raises on anything else.
        ('uq', 'pymc_method'): ['mcmc', 'smc'],
        # -> EmulatorTrainer.design (raises ValueError otherwise)
        ('emulation', 'sample_type'): ['sobol', 'latin_hypercube', 'random'],
        # -> EmulatorBundle.check_bounds. 'error' is the default deliberately: outside its
        # training box an emulator extrapolates with no error estimate at all.
        ('emulation', 'out_of_bounds'): ['error', 'warn', 'clip'],
    }
    for (mode, name), choices in expected.items():
        opt = _option(mode, name)
        assert opt['type'] == 'enum', f'{mode}.{name} should be enum, got {opt["type"]!r}'
        assert opt['choices'] == choices, f'{mode}.{name} choices drifted from the dispatch'
        assert opt['default'] in opt['choices'], f'{mode}.{name} default not selectable'


def test_sample_type_choices_match_sobolsa_dispatch():
    """Guards the pairing directly: each declared sample_type must be a branch in
    sobol_SA.generate_samples, and an unknown one must still raise.

    Reads the source rather than importing sobolSA, which pulls in SALib/mpi4py and
    would make a pure schema test depend on the analysis stack being installed.
    """
    import re
    from pathlib import Path

    src_file = (Path(__file__).resolve().parents[1] / 'src' / 'libcuflynx'
                / 'sensitivity_analysis' / 'sobolSA.py')
    src = src_file.read_text()
    body = src[src.index('def generate_samples'):]
    body = body[:body.index('\n    def ', 1)]  # just this method

    for choice in _option('sensitivity_analysis', 'sample_type')['choices']:
        assert re.search(rf'sample_type"?\'?\]?\s*==\s*[\'"]{choice}[\'"]', body), \
            f'sample_type {choice!r} is offered but generate_samples does not dispatch on it'
    assert 'raise ValueError' in body, 'generate_samples should still reject an unknown sample_type'


def test_cost_func_metadata_discovers_builtins():
    """The obs-data editor discovers valid cost_type values + flags at runtime (costs are a
    user-extensible registry, not a static schema)."""
    from libcuflynx.funcs.cost_funcs_user import cost_func_metadata
    meta = cost_func_metadata()
    # built-in costs are all present
    assert {'gaussian_MLE', 'MSE', 'AE', 'multimodal_gaussian', 'additive', 'norm_additive'} \
        <= set(meta)
    for name, flags in meta.items():
        assert set(flags) == {'is_MLE', 'is_combiner', 'differentiable'}
        assert all(isinstance(v, bool) for v in flags.values())
    assert meta['gaussian_MLE']['is_MLE'] and meta['gaussian_MLE']['differentiable']
    assert meta['additive']['is_combiner']
    assert not meta['MSE']['is_MLE']


def test_cost_registry_excludes_organisational_accessors():
    """The organisational helpers in cost_funcs_user (register/build/get accessors and the
    cost_func_metadata accessor) must not be registered as selectable cost functions -- otherwise
    a bogus 'cost_func_metadata' cost shows up and even self-references in its own output (#259)."""
    from libcuflynx.funcs.cost_funcs_user import get_cost_funcs_dict_for_mode, cost_func_metadata
    costs = get_cost_funcs_dict_for_mode("numpy")
    for accessor in ('cost_func_metadata', 'get_cost_funcs_dict_for_mode',
                     'build_cost_funcs_dict', 'register_cost_funcs'):
        assert accessor not in costs, f'{accessor} should not be a registered cost function'
    # the real costs are still there
    assert {'gaussian_MLE', 'MSE', 'AE'} <= set(costs)
    # and the metadata accessor no longer lists itself
    assert 'cost_func_metadata' not in cost_func_metadata()


def test_statically_defaulted_options_advertise_their_default():
    """If the code substitutes a fixed value when an option is absent, that value belongs in the
    schema's `default` -- a `None` default renders as a blank required field in front-ends even
    though CA supplies the value at run time (#277). Pins the options with a known static default;
    genuinely-required options (GA's num_calls_to_function, which raises if absent) stay None."""
    def opt(options, name):
        return next(o for o in options if o['name'] == name)

    num_samples = opt(analysis_options('sensitivity_analysis'), 'num_samples')
    assert num_samples['default'] == 32 and num_samples['required'] is False

    # bayesian falls back to a 10000-call budget; CMA-ES already advertised the same default.
    for method in ('bayesian', 'CMA-ES'):
        ncalls = opt(param_id_method_options(method), 'num_calls_to_function')
        assert ncalls['default'] == 10000 and ncalls['required'] is False

    # the genetic algorithm has no fallback (it raises if the key is missing), so None/required
    # is the correct, honest descriptor -- do not give it a phantom default.
    ga = opt(param_id_method_options('genetic_algorithm'), 'num_calls_to_function')
    assert ga['default'] is None and ga['required'] is True

    # The GA population settings DO have a fallback the code substitutes, so the schema must
    # advertise it (a front-end pre-fills these). GeneticAlgorithmOptimiser._population_sizes
    # reads these very values, so schema and code cannot drift.
    ga_opts = param_id_method_options('genetic_algorithm')
    for name, expected in (('num_elite', 12), ('num_survivors', 48),
                           ('num_mutations_per_survivor', 12), ('num_cross_breed', 120)):
        descriptor = opt(ga_opts, name)
        assert descriptor['default'] == expected and descriptor['required'] is False, name


def test_ga_population_debug_defaults_advertised_and_match_the_optimiser():
    """DEBUG substitutes a quick-run population, so each GA population option advertises it as
    `debug_default` (#313) -- otherwise a tool can't show/pass the DEBUG values without hardcoding
    them. The advertised values must equal what GeneticAlgorithmOptimiser derives (and runs) under
    DEBUG, so schema and code cannot drift."""
    from libcuflynx.param_id.optimisers import GeneticAlgorithmOptimiser

    def opt(options, name):
        return next(o for o in options if o['name'] == name)

    ga_opts = param_id_method_options('genetic_algorithm')
    for name, expected in (('num_elite', 4), ('num_survivors', 6),
                           ('num_mutations_per_survivor', 2), ('num_cross_breed', 10)):
        assert opt(ga_opts, name)['debug_default'] == expected, name

    # the optimiser derives its DEBUG population from these very descriptors
    derived = GeneticAlgorithmOptimiser._debug_population()
    for name in GeneticAlgorithmOptimiser._POPULATION_KEYS:
        assert derived[name] == opt(ga_opts, name)['debug_default'], name

    # debug_default is confined to the GA population knobs; no other advertised option carries one
    for method, meta in PARAM_ID_METHODS.items():
        for o in meta['options']:
            if 'debug_default' in o:
                assert method == 'genetic_algorithm' and o['name'] in \
                    GeneticAlgorithmOptimiser._POPULATION_KEYS, (method, o['name'])


def test_casadi_integrator_rejects_maximum_step_keys():
    with pytest.raises(ValueError, match="MaximumStep"):
        validate_solver_info('casadi_integrator', {
            'solver': 'casadi_integrator',
            'method': 'cvodes',
            'MaximumStep': 0.001,
        })

    with pytest.raises(ValueError, match="MaximumNumberOfSteps"):
        validate_solver_info('casadi_integrator', {
            'solver': 'casadi_integrator',
            'method': 'cvodes',
            'MaximumNumberOfSteps': 5000,
        })


def test_casadi_integrator_accepts_cvodes_options():
    validate_solver_info('casadi_integrator', {
        'solver': 'casadi_integrator',
        'method': 'cvodes',
        'max_step_size': 0.0001,
        'max_num_steps': 50000,
        'reltol': 1e-8,
        'abstol': 1e-10,
    })


def test_casadi_integrator_accepts_bdf_max_step():
    """The symbolic bdf method reads solver_info['max_step'] (internal sub-step cap), distinct
    from max_step_size. It must validate -- previously it was rejected as an unsupported key."""
    validate_solver_info('casadi_integrator', {
        'solver': 'casadi_integrator',
        'method': 'bdf',
        'max_step': 0.0005,
        'max_step_size': 0.001,
    })
    assert any(f['name'] == 'max_step' for f in solver_info_fields('casadi_integrator'))


def test_cellml_solver_accepts_maximum_step_keys():
    validate_solver_info('CVODE_myokit', {
        'solver': 'CVODE_myokit',
        'method': 'CVODE',
        'MaximumStep': 0.001,
    })
    validate_solver_info('CVODE_opencor', {
        'solver': 'CVODE_opencor',
        'method': 'CVODE',
        'MaximumStep': 0.001,
        'MaximumNumberOfSteps': 5000,
    })


def test_myokit_rejects_maximum_number_of_steps_after_migration():
    """It has no such knob, so a config that still carries it is migrated (with a
    warning) rather than validated -- reaching validation means it was set anew."""
    with pytest.raises(ValueError, match='MaximumNumberOfSteps'):
        validate_solver_info('CVODE_myokit', {
            'solver': 'CVODE_myokit',
            'MaximumNumberOfSteps': 5000,
        })


def test_cpp_rk4_accepts_maximum_number_of_steps():
    """MaximumNumberOfSteps is the one integrator key the cpp path really forwards: the generate
    script reads it and CVSCppGenerator emits it as CVODE's mxsteps."""
    validate_solver_info('RK4', {
        'solver': 'RK4',
        'method': 'RK4',
        'dt_solver': 1e-4,
        'MaximumNumberOfSteps': 5000,
    })


@pytest.mark.parametrize('solver', ['CVODE', 'RK4', 'PETSC'])
def test_cpp_solvers_reject_the_setting_the_generated_code_never_reads(solver):
    """The generated C++ integrates at the fixed dt_solver, so there is no maximum-step-size knob
    for MaximumStep to control. Advertising it made CUFLynx draw a dead control (issue #330)."""
    with pytest.raises(ValueError, match='MaximumStep'):
        validate_solver_info(solver, {'solver': solver, 'MaximumStep': 0.001})


@pytest.mark.parametrize('solver', ['CVODE', 'RK4', 'PETSC'])
def test_cpp_solvers_accept_tolerances_now_that_the_generator_emits_them(solver):
    """rtol/atol were removed in #330 because the emitted C++ hardcoded its tolerances; #398
    wired them through, so they are a real setting again."""
    validate_solver_info(solver, {
        'solver': solver,
        'dt_solver': 1e-4,
        'MaximumNumberOfSteps': 5000,
        'rtol': 1e-8,
        'atol': 1e-10,
    })


@pytest.mark.parametrize('solver', ['CVODE', 'RK4', 'PETSC'])
def test_cpp_maximum_step_is_migrated_with_a_warning_not_rejected(solver, capsys):
    """A config written for a CVODE backend must keep running -- it just has to say which of its
    settings stopped applying, rather than failing validation on the way in."""
    migrated = migrate_legacy_solver_info_keys(solver, {
        'solver': solver,
        'dt_solver': 1e-4,
        'MaximumStep': 0.001,
        'MaximumNumberOfSteps': 5000,
        'rtol': 1e-8,
        'atol': 1e-10,
    })
    # MaximumStep goes; the tolerances stay, because the generator emits them (#398).
    assert set(migrated) == {'solver', 'dt_solver', 'MaximumNumberOfSteps', 'rtol', 'atol'}
    validate_solver_info(solver, migrated)  # the migrated config is accepted

    warned = capsys.readouterr().out
    assert 'MaximumStep' in warned, 'MaximumStep was dropped silently'
    assert 'dt_solver' in warned, 'the MaximumStep warning should name the setting to use instead'
    for kept in ('rtol', 'atol'):
        assert kept not in warned, kept + ' is wired up and must not warn'


@pytest.mark.parametrize('solver', ['CVODE', 'RK4', 'PETSC'])
def test_cpp_maximum_step_alone_becomes_the_solver_step(solver, capsys):
    """MaximumStep on its own is the only step the user gave, so it has to survive.

    It used to be dropped here, and the parser's own MaximumStep -> dt_solver fallback runs
    *after* this -- so nothing was left for it to find and generation died with
    ``KeyError: 'dt_solver'`` on every cpp config that had not already been rewritten to the new
    key, including the cpp autogeneration test.
    """
    migrated = migrate_legacy_solver_info_keys(solver, {
        'solver': solver,
        'MaximumStep': 0.001,
    })
    assert migrated == {'solver': solver, 'dt_solver': 0.001}
    validate_solver_info(solver, migrated)

    warned = capsys.readouterr().out
    assert 'MaximumStep' in warned and 'dt_solver' in warned, (
        'a renamed setting must name what it became'
    )


def test_a_cpp_config_with_only_maximum_step_still_generates():
    """The end the bug was actually felt at: the value has to reach the generator.

    ``script_generate_with_new_architecture`` reads ``solver_info['dt_solver']`` directly, so a
    config that names the step the old way must arrive with that key present rather than raising
    KeyError part-way through generation.
    """
    parsed = YamlFileParser().parse_user_inputs_file(
        {
            'file_prefix': '3compartment',
            'input_param_file': '3compartment_parameters.csv',
            'model_type': 'cpp',
            'solver': 'RK4',
            'solver_info': {'MaximumStep': 0.001},
        },
        obs_path_needed=False,
        do_generation_with_fit_parameters=False,
    )
    assert parsed['solver_info']['dt_solver'] == 0.001


def test_solve_ivp_rejects_maximum_step_keys():
    with pytest.raises(ValueError, match="MaximumStep"):
        validate_solver_info('solve_ivp', {
            'solver': 'solve_ivp',
            'method': 'BDF',
            'MaximumStep': 0.001,
        })


def test_migrate_legacy_solver_info_keys_for_solve_ivp():
    migrated = migrate_legacy_solver_info_keys('solve_ivp', {
        'MaximumStep': 0.0001,
        'MaximumNumberOfSteps': 5000,
        'method': 'BDF',
    })
    assert migrated == {'method': 'BDF', 'max_step': 0.0001}
    validate_solver_info('solve_ivp', {'solver': 'solve_ivp', **migrated})


def test_migrate_legacy_solver_info_keys_for_casadi_integrator():
    migrated = migrate_legacy_solver_info_keys('casadi_integrator', {
        'MaximumStep': 0.0001,
        'MaximumNumberOfSteps': 5000,
        'method': 'cvodes',
    })
    assert migrated == {
        'method': 'cvodes',
        'max_step_size': 0.0001,
        'max_num_steps': 5000,
    }
    validate_solver_info('casadi_integrator', {'solver': 'casadi_integrator', **migrated})


def test_migrate_drops_maximum_number_of_steps_for_myokit(capsys):
    """myokit has no max-step-count knob, so the key is dropped rather than renamed --
    but never in silence: a setting that stops taking effect must say so."""
    migrated = migrate_legacy_solver_info_keys('CVODE_myokit', {
        'MaximumStep': 0.0001,
        'MaximumNumberOfSteps': 5000,
        'rtol': 1e-8,
    })
    assert migrated == {'MaximumStep': 0.0001, 'rtol': 1e-8}
    validate_solver_info('CVODE_myokit', {'solver': 'CVODE_myokit', **migrated})

    warning = capsys.readouterr().out
    assert 'WARNING' in warning
    assert 'MaximumNumberOfSteps' in warning
    assert 'CVODE_myokit' in warning
    # Names what to use instead, since there is no direct replacement.
    assert 'MaximumStep' in warning
    assert 'rtol' in warning


def test_migrate_is_silent_when_there_is_nothing_to_migrate(capsys):
    """A config already using the right keys must not be nagged."""
    migrate_legacy_solver_info_keys('CVODE_myokit', {'MaximumStep': 0.0001, 'atol': 1e-8})
    assert capsys.readouterr().out == ''


def test_migrate_names_the_replacement_key_when_renaming(capsys):
    migrate_legacy_solver_info_keys('solve_ivp', {'MaximumStep': 0.0001})
    out = capsys.readouterr().out
    assert 'MaximumStep' in out and 'max_step' in out

    migrate_legacy_solver_info_keys('casadi_integrator', {'MaximumNumberOfSteps': 500})
    out = capsys.readouterr().out
    assert 'MaximumNumberOfSteps' in out and 'max_num_steps' in out


def test_setting_both_names_for_one_setting_is_an_error():
    """Not a warning: preferring either value would hide the other from a user who
    believes it is in effect, and nothing can tell which one they meant."""
    with pytest.raises(ValueError) as exc:
        migrate_legacy_solver_info_keys('casadi_integrator', {
            'MaximumNumberOfSteps': 500,
            'max_num_steps': 999,
        })
    msg = str(exc.value)
    # Both names, both values, and which one to delete.
    assert 'MaximumNumberOfSteps' in msg and 'max_num_steps' in msg
    assert '500' in msg and '999' in msg
    assert 'Remove' in msg

    with pytest.raises(ValueError, match='MaximumStep'):
        migrate_legacy_solver_info_keys('solve_ivp', {
            'MaximumStep': 0.001,
            'max_step': 0.002,
        })


def test_duplicate_names_error_even_when_the_values_agree():
    """The config still says one thing twice; leaving it means the next edit to
    either name silently does nothing."""
    with pytest.raises(ValueError):
        migrate_legacy_solver_info_keys('casadi_integrator', {
            'MaximumNumberOfSteps': 500,
            'max_num_steps': 500,
        })


def test_dt_solver_alongside_the_new_key_is_not_a_duplicate(capsys):
    """dt_solver is a framework key used only as a *fallback* source, not another
    spelling of max_step, so pairing the two is legitimate and silent."""
    migrated = migrate_legacy_solver_info_keys('solve_ivp', {
        'dt_solver': 0.01,
        'max_step': 0.002,
    })
    assert migrated == {'dt_solver': 0.01, 'max_step': 0.002}
    assert capsys.readouterr().out == ''


def test_myokit_default_solver_info_needs_no_migration():
    """CA's own cellml default must validate for myokit, not just opencor --
    otherwise the default config would be rejected by its own validator."""
    from libcuflynx.parsers.PrimitiveParsers import _solver_info_default_for

    defaults = _solver_info_default_for('cellml', 'CVODE_myokit')
    assert 'MaximumNumberOfSteps' not in defaults
    assert defaults['MaximumStep'] == 0.001
    validate_solver_info('CVODE_myokit', defaults)

    # opencor keeps it: it is a real setting there.
    opencor = _solver_info_default_for('cellml', 'CVODE_opencor')
    assert opencor['MaximumNumberOfSteps'] == 5000
    validate_solver_info('CVODE_opencor', opencor)


def test_parse_user_inputs_migrates_legacy_keys_for_python_model():
    parsed = YamlFileParser().parse_user_inputs_file({
        'file_prefix': '3compartment',
        'input_param_file': '3compartment_parameters.csv',
        'model_type': 'python',
        'solver': 'solve_ivp',
        'solver_info': {
            'MaximumStep': 0.0001,
            'MaximumNumberOfSteps': 5000,
            'method': 'BDF',
        },
        'dt': 0.01,
        'pre_time': 0.0,
        'sim_time': 1.0,
    }, obs_path_needed=False)
    assert parsed['solver_info']['max_step'] == 0.0001
    assert 'MaximumStep' not in parsed['solver_info']
    assert 'MaximumNumberOfSteps' not in parsed['solver_info']


def test_parse_user_inputs_warns_for_casadi_nonzero_pre_time():
    with pytest.warns(UserWarning, match='does not support nonzero pre_time'):
        YamlFileParser().parse_user_inputs_file({
            'file_prefix': '3compartment_nonstiff',
            'input_param_file': '3compartment_nonstiff_parameters.csv',
            'model_type': 'casadi_python',
            'solver': 'casadi_integrator',
            'solver_info': {'method': 'cvodes', 'max_num_steps': 50000},
            'dt': 0.01,
            'pre_time': 0.5,
            'sim_time': 0.3,
        }, obs_path_needed=False)


def test_warn_if_casadi_nonzero_pre_time_ignores_other_model_types():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warn_if_casadi_nonzero_pre_time('python', pre_time=0.5)
    assert len(caught) == 0


def test_sa_method_choices_each_have_a_dispatch_handler():
    """Every sensitivity_analysis 'method' choice in the schema must have a matching
    run_<method>_sensitivity handler on SensitivityAnalysis, and the run dispatcher must derive
    its valid set from the schema (not a hardcoded list). This locks the schema <-> dispatch
    correspondence the run message now relies on, so adding a method to the schema without a
    handler (or vice versa) fails here."""
    from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis, sa_method_choices

    choices = sa_method_choices()
    assert choices, "no sensitivity_analysis method choices found in the schema"
    # sa_method_choices reads straight from the schema accessor.
    assert choices == _option('sensitivity_analysis', 'method')['choices']
    for method in choices:
        assert hasattr(SensitivityAnalysis, f'run_{method}_sensitivity'), (
            f"schema advertises sa method {method!r} but SensitivityAnalysis has no "
            f"run_{method}_sensitivity handler")


def test_gradient_sources_well_formed_and_match_get_gradient_dispatch():
    """gradient_sources(model_type, solver) must advertise exactly the sources CA can actually
    produce, keyed to the top-level do_ad flag, so a front-end can build a gradient menu without
    hand-mirroring CA's rules. Pinned to OpencorParamID.get_gradient's dispatch: the AD-capable
    model types are AD_GRADIENT_MODEL_TYPES, and cellml+CVODE_myokit gets FSA.
    """
    from libcuflynx.param_id.optimisers import AD_GRADIENT_MODEL_TYPES

    _REQUIRED_KEYS = {'value', 'label', 'do_ad', 'requires_all_differentiable', 'description'}

    def _check_shape(srcs):
        assert srcs, 'at least finite differences must always be offered'
        for s in srcs:
            assert set(s) == _REQUIRED_KEYS, s
            assert s['value'] in {'FD', 'AD', 'FSA'}
            assert isinstance(s['do_ad'], bool)
            assert isinstance(s['requires_all_differentiable'], bool)
        # Finite differences is always present, and is the only do_ad=False source.
        fd = [s for s in srcs if s['value'] == 'FD']
        assert len(fd) == 1 and fd[0]['do_ad'] is False
        assert [s for s in srcs if not s['do_ad']] == fd
        # Only CasADi AD needs all-differentiable.
        for s in srcs:
            assert s['requires_all_differentiable'] == (
                s['value'] == 'AD' and 'CasADi' in s['label'])

    # Every AD-capable model type offers an AD source with do_ad=True (matching get_gradient).
    for mt in AD_GRADIENT_MODEL_TYPES:
        srcs = gradient_sources(mt)
        _check_shape(srcs)
        ad = [s for s in srcs if s['value'] == 'AD']
        assert len(ad) == 1 and ad[0]['do_ad'] is True, mt

    # cellml gets the Myokit CVODES FSA source only with the Myokit solver.
    myokit = gradient_sources('cellml', 'CVODE_myokit')
    _check_shape(myokit)
    fsa = [s for s in myokit if s['value'] == 'FSA']
    assert len(fsa) == 1 and fsa[0]['do_ad'] is True

    # No analytic source for cellml under a non-FSA solver, or for non-AD model types.
    for mt, sv in [('cellml', 'CVODE_opencor'), ('python', 'solve_ivp'), ('cpp', 'RK4')]:
        srcs = gradient_sources(mt, sv)
        _check_shape(srcs)
        assert [s['value'] for s in srcs] == ['FD'], (mt, sv)


def test_per_integrator_ad_fsa_and_default_method_schema():
    """ad_suitable_methods / fsa_suitable_methods / default_method_by_solver let a front-end gate
    its Gradient menu on the selected integrator (issue #298). Lock their shape and, crucially,
    that ad_suitable_methods is derived from _CASADI_ADJOINT_METHODS (so the flag and the
    adjoint-warning cannot drift) and that the referenced methods actually exist in the schema."""
    ad = SOLVER_SCHEMA['ad_suitable_methods']
    fsa = SOLVER_SCHEMA['fsa_suitable_methods']
    default_method = SOLVER_SCHEMA['default_method_by_solver']

    casadi_methods = SOLVER_SCHEMA['methods_by_solver']['casadi_integrator']
    # AD-suitable = every casadi_integrator method that is NOT an adjoint-sensitivity method.
    assert ad['casadi_integrator'] == [m for m in casadi_methods
                                       if m not in _CASADI_ADJOINT_METHODS]
    assert ad['casadi_integrator'] == ['collocation', 'rk', 'semi_implicit_euler', 'bdf']
    # the adjoint methods are exactly the ones excluded
    assert set(casadi_methods) - set(ad['casadi_integrator']) == set(_CASADI_ADJOINT_METHODS)

    # fsa_suitable methods exist in the schema for their solver.
    for solver, methods in fsa.items():
        assert solver in SOLVER_SCHEMA['methods_by_solver'], solver
        assert set(methods) <= set(SOLVER_SCHEMA['methods_by_solver'][solver]), (solver, methods)
    assert fsa['CVODE_myokit'] == ['CVODE']

    # a default method must be one of that solver's methods, and AD-suitable where AD applies.
    for solver, m in default_method.items():
        assert m in SOLVER_SCHEMA['methods_by_solver'][solver], (solver, m)
    assert default_method['casadi_integrator'] == 'bdf'
    assert default_method['casadi_integrator'] in ad['casadi_integrator']


def test_gradient_sources_gates_analytic_source_on_integrator_method():
    """gradient_sources(..., method=...) drops the analytic source when the selected integrator
    cannot produce it -- CasADi AD for the adjoint methods (cvodes/idas), FSA for a non-CVODE
    method -- and keeps it otherwise. This is the per-integrator gate downstream tools apply."""
    # CasADi AD: offered for the symbolic/AD-suitable methods, gone for the adjoint ones.
    for m in SOLVER_SCHEMA['ad_suitable_methods']['casadi_integrator']:
        assert 'AD' in [s['value'] for s in gradient_sources('casadi_python', method=m)], m
    for m in _CASADI_ADJOINT_METHODS:
        assert [s['value'] for s in gradient_sources('casadi_python', method=m)] == ['FD'], m
    # method=None (or an unknown method) leaves AD offered.
    assert 'AD' in [s['value'] for s in gradient_sources('casadi_python')]
    assert 'AD' in [s['value'] for s in gradient_sources('casadi_python', method='not_a_method')]

    # Myokit FSA: offered for the CVODE method (the only, and FSA-suitable, method).
    assert 'FSA' in [s['value'] for s in
                     gradient_sources('cellml', 'CVODE_myokit', method='CVODE')]
    assert 'FSA' in [s['value'] for s in gradient_sources('cellml', 'CVODE_myokit')]

    # AADC AD has no per-integrator gate (its tape method is independent of this menu).
    assert 'AD' in [s['value'] for s in gradient_sources('aadc_python', method='semi_implicit')]


@pytest.mark.unit
def test_aadc_ad_suitable_methods_match_what_the_tape_enforces():
    """The advertised AD-suitable AADC methods must be exactly those aadc_backend accepts.

    Before issue #336 the schema had no aadc_semi_implicit entry at all, so a tool reading it
    could not tell which methods the tape can record -- and since methods_by_solver lists
    'adaptive_rk45' first, a front-end defaulting to "first offered" picked the one method AD can
    never use, failing only once a calibration had started.
    """
    from libcuflynx.parsers.PrimitiveParsers import AADC_TAPE_CONSISTENT_METHODS

    all_methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']
    advertised = SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']

    # Derived from AADC_AD_METHODS -- the tape-replayable methods PLUS the stiff BDF methods,
    # which reach a gradient by their own dispatch rather than the standard tape. Asserting
    # against AADC_TAPE_CONSISTENT_METHODS alone would exclude the BDF methods, which is exactly
    # the gap this replaced.
    from libcuflynx.parsers.PrimitiveParsers import AADC_AD_METHODS
    assert advertised == [m for m in all_methods if m in AADC_AD_METHODS]
    assert set(AADC_TAPE_CONSISTENT_METHODS) <= set(advertised)
    assert set(advertised) <= set(all_methods), (advertised, all_methods)
    # the adaptive integrator is the one that must NOT be advertised
    assert 'adaptive_rk45' in all_methods and 'adaptive_rk45' not in advertised

    # and the runtime check enforces precisely this set
    from libcuflynx.param_id.aadc_backend import TAPE_CONSISTENT_METHODS
    assert tuple(TAPE_CONSISTENT_METHODS) == tuple(AADC_TAPE_CONSISTENT_METHODS)


@pytest.mark.unit
def test_aadc_default_method_is_ad_suitable():
    """A default the AD path cannot use is a poor default for a tape-gradient backend."""
    default = SOLVER_SCHEMA['default_method_by_solver']['aadc_semi_implicit']
    assert default in SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit'], default
    assert default in SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit'], default


@pytest.mark.unit
def test_gradient_sources_gates_aadc_ad_on_a_tape_consistent_method():
    """AD must not be offered for an AADC method the tape cannot record."""
    from libcuflynx.parsers.PrimitiveParsers import gradient_sources

    def values(method):
        return [s['value'] for s in gradient_sources('aadc_python', 'aadc_semi_implicit',
                                                     method=method)]

    # the adaptive integrator: FD only, no AD
    assert values('adaptive_rk45') == ['FD']
    # every tape-consistent method still offers AD
    for m in SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']:
        assert 'AD' in values(m), m
    # unspecified or unknown method leaves AD offered, matching the casadi branch
    assert 'AD' in values(None)
    assert 'AD' in values('some_unknown_method')


@pytest.mark.unit
def test_aadc_bdf_methods_are_advertised_as_ad_suitable():
    """The stiff BDF methods are AD-capable and must be advertised as such.

    They do not go through the standard replay tape: aadc_backend.cost_and_grad dispatches each
    to its own gradient implementation *before* the AADC_TAPE_CONSISTENT_METHODS check, so they
    are AD-capable without being members of that tuple. Deriving ad_suitable_methods from the
    tuple alone left a tool refusing AD for exactly the stiff-model methods, which is the
    combination the BDF work exists to enable.
    """
    from libcuflynx.parsers.PrimitiveParsers import (
        AADC_AD_METHODS, AADC_BDF_AD_METHODS, AADC_TAPE_CONSISTENT_METHODS)
    import inspect
    from libcuflynx.param_id import aadc_backend

    advertised = SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']
    all_methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']

    # both routes to a gradient are advertised, and nothing else is invented
    assert AADC_AD_METHODS == AADC_TAPE_CONSISTENT_METHODS + AADC_BDF_AD_METHODS
    assert advertised == [m for m in all_methods if m in AADC_AD_METHODS]
    for m in AADC_BDF_AD_METHODS:
        assert m in advertised, f"{m} has a gradient implementation but is not advertised"
    # the adaptive integrator still must not be offered
    assert 'adaptive_rk45' not in advertised

    # each advertised BDF method really is dispatched to its own gradient path: the signed
    # scheme through the strategy resolver, anything else by name in cost_and_grad itself
    src = inspect.getsource(aadc_backend.cost_and_grad)
    for m in AADC_BDF_AD_METHODS:
        routed = aadc_backend.resolve_gradient_strategy(m, {}) is not None
        named = m.upper() + "_METHOD" in src or repr(m) in src
        assert routed or named, (
            f"{m} is advertised as AD-suitable but cost_and_grad does not dispatch it")


@pytest.mark.unit
def test_forward_methods_are_exactly_what_the_aadc_dispatch_can_integrate():
    """methods_by_solver is a superset: not every method can run a plain forward solve.

    A GUI building a "run a simulation" menu from methods_by_solver can offer a method that
    cannot solve -- issue #346, where picking one broke every interactive simulation. This key
    lists only what run() dispatches, and the loop below checks that claim against the source.
    forward_methods_by_solver is the list to build that menu from.
    """
    import inspect
    from libcuflynx.parsers.PrimitiveParsers import AADC_FORWARD_METHODS
    from libcuflynx.solver_wrappers import aadc_python_solver_helper as helper

    advertised = SOLVER_SCHEMA['forward_methods_by_solver']['aadc_semi_implicit']
    assert advertised == list(AADC_FORWARD_METHODS)
    assert set(advertised) <= set(SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit'])
    # semi_implicit_signed was the one method with a gradient but no forward branch, so a
    # calibration using it could not simulate its own best fit. It has one now.
    assert 'semi_implicit_signed' in advertised

    # every advertised forward method really has a branch in run()
    run_src = inspect.getsource(helper.SimulationHelper.run)
    for method in AADC_FORWARD_METHODS:
        assert repr(method) in run_src or f"'{method}'" in run_src, (
            f"{method} is advertised as forward-capable but run() does not dispatch it")


@pytest.mark.unit
def test_the_unknown_method_error_lists_what_the_dispatch_accepts():
    """The hand-written list had gone stale, omitting implicit_newton and bdf_newton.

    That cost real debugging time downstream: the stale text matched an older checkout exactly,
    so a genuine finding looked like a local environment problem (issue #346).
    """
    import inspect
    from libcuflynx.solver_wrappers import aadc_python_solver_helper as helper

    src = inspect.getsource(helper.SimulationHelper.run)
    assert "AADC_FORWARD_METHODS" in src, "error text should be derived, not hand-written"


@pytest.mark.unit
def test_semi_implicit_signed_replaces_the_two_bdf_gradient_names():
    """bdf_tape and bdf_kernel were one integrator with two execution strategies.

    Both step x += dt*f/(1 - dt*diag J); they differ only in where the loop runs (an AADC tape,
    or a C++ kernel replay that falls back to the tape). Advertising them as separate integrators
    made a GUI offer two 'methods' that no forward solve accepts.
    """
    from libcuflynx.parsers.PrimitiveParsers import (
        AADC_BDF_AD_METHODS, AADC_LEGACY_METHOD_ALIASES)

    methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']
    assert 'semi_implicit_signed' in methods
    for gone in ('bdf_tape', 'bdf_kernel'):
        assert gone not in methods, f"{gone} is a gradient strategy, not a method"
        assert gone in AADC_LEGACY_METHOD_ALIASES, "old configs must keep working"
    # still AD-capable, by its own gradient path
    assert 'semi_implicit_signed' in AADC_BDF_AD_METHODS
    assert 'semi_implicit_signed' in SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']

    strategy = [f for f in SOLVER_INFO_FIELDS['aadc_semi_implicit']
                if f['name'] == 'gradient_strategy']
    assert strategy and strategy[0]['choices'] == ['tape', 'kernel']


@pytest.mark.unit
def test_legacy_bdf_names_resolve_through_the_dispatch_not_just_the_alias_table():
    """"Old configs keep working" has to be checked against the code that runs them.

    Asserting only that 'bdf_tape' appears in AADC_LEGACY_METHOD_ALIASES tests a declaration:
    the table could stay correct while the dispatch that consumes it was rewritten to ignore a
    name, and the guarantee would break with every test still green. So drive the resolver.
    """
    from libcuflynx.param_id.aadc_backend import resolve_gradient_strategy

    assert resolve_gradient_strategy('bdf_tape', {}) == ('semi_implicit_signed', 'tape')
    assert resolve_gradient_strategy('bdf_kernel', {}) == ('semi_implicit_signed', 'kernel')

    # the canonical name honours the setting, and defaults to the tape
    assert resolve_gradient_strategy('semi_implicit_signed', {}) == ('semi_implicit_signed', 'tape')
    assert resolve_gradient_strategy(
        'semi_implicit_signed', {'gradient_strategy': 'kernel'}) == ('semi_implicit_signed', 'kernel')

    # anything else is not this scheme, and must fall through to the other gradient paths
    for other in ('bdf_newton', 'rk4', 'semi_implicit', 'adaptive_rk45'):
        assert resolve_gradient_strategy(other, {}) is None

    with pytest.raises(ValueError, match='gradient_strategy'):
        resolve_gradient_strategy('semi_implicit_signed', {'gradient_strategy': 'nonsense'})


@pytest.mark.unit
def test_a_legacy_name_warns_rather_than_silently_dropping_a_conflicting_strategy():
    """'bdf_kernel' fixes the strategy, so gradient_strategy='tape' alongside it cannot be
    honoured. The name wins (it is the more specific statement of intent), but silently
    discarding a setting the user wrote is the failure mode this whole change is fixing."""
    from libcuflynx.param_id.aadc_backend import resolve_gradient_strategy

    with pytest.warns(UserWarning, match='ignored'):
        resolved = resolve_gradient_strategy('bdf_kernel', {'gradient_strategy': 'tape'})
    assert resolved == ('semi_implicit_signed', 'kernel')

    # no warning when they agree
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolve_gradient_strategy(
            'bdf_tape', {'gradient_strategy': 'tape'}) == ('semi_implicit_signed', 'tape')


@pytest.mark.unit
def test_the_aadc_default_method_can_actually_integrate():
    """A default has to produce a number before it is AD-friendly.

    'rk4' was chosen in #336 for tape-consistency without checking it could integrate: on
    3compartment it raises OverflowError at dt 1e-3, 1e-4 and 1e-5, while implicit_newton lands
    within 2% of CVODE_myokit (issue #346).
    """
    from libcuflynx.parsers.PrimitiveParsers import AADC_FORWARD_METHODS

    default = SOLVER_SCHEMA['default_method_by_solver']['aadc_semi_implicit']
    assert default in AADC_FORWARD_METHODS, "the default must be forward-solvable"
    assert default in SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']
    assert default != 'rk4', "rk4 cannot integrate a stiff model; see issue #346"


@pytest.mark.unit
def test_stiff_suitable_methods_are_real_methods_and_exclude_the_measured_failures():
    """Which integrators can be trusted on a stiff model.

    Measured on 3compartment against CVODE_myokit (issue #346): rk4 overflows at three step
    sizes, adaptive_rk45 does not return, implicit_euler_ift completes but is 84% low, while
    semi_implicit (+6.7%) and implicit_newton (-1.9%) are usable.

    implicit_euler_ift is the entry worth defending: it is excluded *despite* completing,
    because returning a smooth trace that is wrong by a factor of six is worse than raising --
    and it remains in ad_suitable_methods, so a gradient calibration would use it silently.
    """
    stiff = SOLVER_SCHEMA['stiff_suitable_methods']

    # every entry must be a method that solver actually offers
    for solver, methods in stiff.items():
        known = SOLVER_SCHEMA['methods_by_solver'].get(solver)
        assert known is not None, f"{solver} is not in methods_by_solver"
        for m in methods:
            assert m in known, f"{solver}: {m!r} is not one of its methods"

    # and every solver must have an entry, so a consumer can tell "assessed, nothing qualifies"
    # (an empty list) from "not in the table at all" -- a missing key would otherwise read as
    # "no stiff-safe method" and silently hide the solver from a stiff-model menu
    assert set(stiff) == set(SOLVER_SCHEMA['methods_by_solver']), (
        "every solver needs a stiff_suitable_methods entry, empty if nothing qualifies")

    aadc = stiff['aadc_semi_implicit']
    assert aadc == ['semi_implicit', 'semi_implicit_signed', 'implicit_newton']
    for excluded in ('rk4', 'adaptive_rk45', 'implicit_euler_ift'):
        assert excluded not in aadc, f"{excluded} is not usable on a stiff model; see issue #346"

    # explicit solve_ivp methods must not be advertised as stiff-capable
    for excluded in ('RK45', 'RK23', 'DOP853', 'forward_euler'):
        assert excluded not in stiff['solve_ivp']
    # nor CasADi's explicit rk
    assert 'rk' not in stiff['casadi_integrator']


@pytest.mark.unit
def test_every_default_method_is_stiff_suitable_where_the_solver_has_a_stiff_set():
    """A default lands on whatever a user gets without choosing, and the models this framework
    generates are stiff -- so the default must be one of the trustworthy ones."""
    stiff = SOLVER_SCHEMA['stiff_suitable_methods']
    for solver, default in SOLVER_SCHEMA['default_method_by_solver'].items():
        if solver in stiff:
            assert default in stiff[solver], (
                f"default for {solver} is {default!r}, which is not stiff-suitable")


# --------------------------------------------------------- Myokit CVODES tolerance mapping
#
# Myokit's signature is set_tolerance(abs_tol, rel_tol) -- absolute FIRST, the reverse of the
# rtol-then-atol order CA's schema lists. The helper passed them positionally in schema order,
# so each reached the other argument. A symmetric pair cannot catch that, so every test here
# uses asymmetric values and asserts which value reached which keyword.


class _RecordingSimulation:
    """Stands in for myokit.Simulation; records what reached set_tolerance."""

    def __init__(self):
        self.tolerance_calls = []

    def set_tolerance(self, abs_tol=1e-6, rel_tol=1e-4):
        self.tolerance_calls.append({'abs_tol': abs_tol, 'rel_tol': rel_tol})


def _apply(solver_info, fsa_enabled=False):
    from libcuflynx.solver_wrappers.myokit_helper import apply_cvodes_tolerances
    sim = _RecordingSimulation()
    effective = apply_cvodes_tolerances(sim, solver_info, fsa_enabled)
    return sim.tolerance_calls, effective


@pytest.mark.unit
def test_asymmetric_tolerances_reach_the_right_myokit_arguments():
    """rtol must arrive as rel_tol and atol as abs_tol. Swapped, this call would apply
    abs 1e-4 / rel 1e-6 -- measured bit-identical to Myokit's own defaults on 3compartment,
    which is how the swap stayed invisible."""
    calls, effective = _apply({'rtol': 1e-4, 'atol': 1e-6})
    assert calls == [{'abs_tol': 1e-6, 'rel_tol': 1e-4}]
    assert effective == (1e-6, 1e-4)


@pytest.mark.unit
def test_no_tolerances_means_cas_own_defaults_are_applied():
    """With neither set (and no FSA), CA's declared defaults are applied -- abs stays at the
    1e-8 floor previous users ran at (so existing models do not start failing), rel relaxes
    to 1e-6. Declared and effective are the same thing, whichever front door the run came
    through."""
    calls, effective = _apply({})
    assert calls == [{'abs_tol': 1e-8, 'rel_tol': 1e-6}]
    assert effective == (1e-8, 1e-6)


@pytest.mark.unit
def test_fsa_tightens_to_1e8_when_the_user_set_none():
    """CVODES sensitivities are only as accurate as the state solve; the default tolerance's
    noise floor swamps small sensitivities."""
    calls, effective = _apply({}, fsa_enabled=True)
    assert calls == [{'abs_tol': 1e-8, 'rel_tol': 1e-8}]
    assert effective == (1e-8, 1e-8)


@pytest.mark.unit
def test_explicit_tolerances_win_over_the_fsa_tightening():
    calls, _ = _apply({'rtol': 1e-5, 'atol': 1e-7}, fsa_enabled=True)
    assert calls == [{'abs_tol': 1e-7, 'rel_tol': 1e-5}]


@pytest.mark.unit
def test_a_partial_setting_fills_the_partner_with_cas_default():
    """set_tolerance takes both values, so setting only one needs a partner: CA's declared
    default for the other, the same value the schema advertises."""
    assert _apply({'rtol': 1e-5})[0] == [{'abs_tol': 1e-8, 'rel_tol': 1e-5}]
    assert _apply({'atol': 1e-9})[0] == [{'abs_tol': 1e-9, 'rel_tol': 1e-6}]


@pytest.mark.unit
def test_a_failed_solve_names_the_stability_knobs_and_their_values():
    """A failed solve must tell the user which numbers to turn: the effective MaximumStep,
    atol and rtol, and that decreasing them may help stability."""
    from libcuflynx.solver_wrappers.myokit_helper import stability_hint
    hint = stability_hint({'MaximumStep': 0.001}, (1e-8, 1e-6))
    assert 'MaximumStep is 0.001' in hint
    assert 'atol is 1e-08' in hint
    assert 'rtol is 1e-06' in hint
    assert 'decreasing these' in hint and 'stability' in hint


@pytest.mark.unit
def test_the_stability_hint_reports_an_unset_maximum_step_honestly():
    """No MaximumStep in solver_info means the integrator step is unbounded -- say so, rather
    than inventing a number the user never set."""
    from libcuflynx.solver_wrappers.myokit_helper import stability_hint
    hint = stability_hint({}, (1e-8, 1e-6))
    assert 'MaximumStep is unset (unbounded)' in hint


@pytest.mark.unit
def test_myokit_set_tolerance_still_takes_abs_and_rel_keywords():
    """The keyword call is the whole fix; a Myokit rename must break loudly here, not revert
    the helper to positional guessing."""
    myokit = pytest.importorskip('myokit')
    import inspect
    params = inspect.signature(myokit.Simulation.set_tolerance).parameters
    assert 'abs_tol' in params and 'rel_tol' in params


@pytest.mark.unit
def test_cvode_myokit_schema_declares_cas_defaults():
    """Front-ends seed interactive solves from these declared defaults (they deliberately do
    not restate them), so the declaration is the one place the default is decided -- and it
    must equal what the helper applies when the user sets neither, or declared and effective
    drift apart. abs keeps the 1e-8 floor previous users ran at; rel relaxes to 1e-6."""
    from libcuflynx.solver_wrappers.myokit_helper import CA_DEFAULT_ABS_TOL, CA_DEFAULT_REL_TOL
    fields = {f['name']: f for f in solver_info_fields('CVODE_myokit')}
    assert fields['rtol']['default'] == CA_DEFAULT_REL_TOL == 1e-6
    assert fields['atol']['default'] == CA_DEFAULT_ABS_TOL == 1e-8


# ---------------------------------------------------------------------------
# Legacy mcmc_options / do_mcmc spelling (renamed to UQ_options / do_uq)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_legacy_mcmc_option_names_are_migrated_with_a_warning(capsys):
    """MCMC became one method of UQ, so its settings moved to UQ_options/do_uq. A config written
    for the old names must keep running and be told, once, what to rename -- the same
    migrate-with-a-warning treatment solver_info keys get."""
    from libcuflynx.parsers.PrimitiveParsers import _normalise_uq_option_names

    inp = {
        'do_mcmc': True,
        'mcmc_options': {'num_steps': 11},
        'debug_mcmc_options': {'num_steps': 3},
    }
    _normalise_uq_option_names(inp)

    assert inp == {
        'do_uq': True,
        'UQ_options': {'num_steps': 11},
        'debug_UQ_options': {'num_steps': 3},
    }, 'the values must survive the rename, and the old keys must not linger'

    warned = capsys.readouterr().out
    for old_key, new_key in [('mcmc_options', 'UQ_options'),
                             ('debug_mcmc_options', 'debug_UQ_options'),
                             ('do_mcmc', 'do_uq')]:
        assert old_key in warned and new_key in warned, old_key + ' was renamed silently'


@pytest.mark.unit
def test_a_config_using_only_the_new_uq_names_is_silent():
    """The migration must not nag a config that is already correct."""
    from libcuflynx.parsers.PrimitiveParsers import _normalise_uq_option_names

    inp = {'do_uq': False, 'UQ_options': {'method': 'mcmc'}}
    before = dict(inp)
    _normalise_uq_option_names(inp)
    assert inp == before


@pytest.mark.unit
def test_setting_both_uq_spellings_is_refused():
    """Not a stale key to migrate but a contradiction: the two values can disagree, and
    silently preferring either would hide one from a user who believes it is in effect."""
    from libcuflynx.parsers.PrimitiveParsers import _normalise_uq_option_names

    with pytest.raises(ValueError, match='mcmc_options'):
        _normalise_uq_option_names({'mcmc_options': {'num_steps': 1},
                                    'UQ_options': {'num_steps': 2}})


@pytest.mark.unit
def test_deprecated_mcmc_options_kwarg_still_reaches_UQ_options(capsys):
    """The public CVS0DParamID(mcmc_options=...) kwarg keeps working, and passing both spellings
    is refused for the same reason as above."""
    from libcuflynx.param_id.paramID import _resolve_UQ_options

    assert _resolve_UQ_options(None, {'num_steps': 5}) == {'num_steps': 5}
    assert 'deprecated' in capsys.readouterr().out

    assert _resolve_UQ_options({'num_steps': 9}, None) == {'num_steps': 9}
    assert capsys.readouterr().out == '', 'the new spelling must be silent'

    with pytest.raises(ValueError, match='not both'):
        _resolve_UQ_options({'num_steps': 9}, {'num_steps': 5})


@pytest.mark.unit
def test_a_config_that_names_no_solver_gets_one_a_pip_install_can_provide():
    """`cellml` must default to CVODE_myokit, not CVODE_opencor.

    OpenCOR's Python module is not on PyPI, so `CVODE_opencor` is the one backend a
    `pip install libcuflynx` cannot supply. Defaulting to it meant a user who wrote a
    minimal config and left `solver` out was routed straight at the unavailable one --
    invisible in a checkout, because the shipped `user_inputs.yaml` sets it explicitly,
    and invisible in CI for the same reason.

    Pinned because nothing else pins it. The change that fixed this shipped no test, and
    upstream then renamed the key this value hangs off (`cellml_only` -> `cellml`) in a
    separate branch. Merging the two would have restored the old default silently, with
    every suite still green -- three independent reads (the schema, the solver_info
    default, and the implicit fallback in `parse_user_inputs_file`) all agreeing on the
    wrong answer.
    """
    assert SOLVER_SCHEMA['default_solver_by_model_type']['cellml'] == 'CVODE_myokit'
    assert get_solver_info_default('cellml')['solver'] == 'CVODE_myokit'
    # Still *offered*, just not the default: it works fine inside OpenCOR, and removing it
    # would make `solver: CVODE_opencor` fail validation instead of reaching the message
    # that names CVODE_myokit as the replacement.
    assert 'CVODE_opencor' in SOLVER_SCHEMA['solvers_by_model_type']['cellml']


@pytest.mark.unit
def test_the_defaulted_solver_comes_from_the_schema_for_every_model_type():
    """One source for "which solver if you name none", not a ladder beside the schema.

    SOLVER_SCHEMA is what CUFLynx and every other consumer reads, so a second copy in
    parse_user_inputs_file is a second answer waiting to disagree -- and it had. The
    ladder defaulted `casadi_python` to 'cvodes', which is a *method* of the
    casadi_integrator solver rather than a solver; `get_simulation_helper` accepts only
    'casadi_integrator' for that model type. Nothing caught it because a defaulted solver
    skips the validation the explicit path goes through.
    """
    defaults = SOLVER_SCHEMA['default_solver_by_model_type']

    # Every model type has one, and it is a solver that model type actually offers.
    for model_type, solvers in SOLVER_SCHEMA['solvers_by_model_type'].items():
        assert model_type in defaults, f'{model_type} has no default solver'
        assert defaults[model_type] in solvers, (
            f'{model_type} defaults to {defaults[model_type]!r}, which is not one of its '
            f'solvers {solvers} -- a method name here is the mistake to look for')


@pytest.mark.unit
def test_the_legacy_cvode_alias_leaves_cpp_alone():
    """'CVODE' is a legacy spelling for CellML *and* a current solver for cpp.

    Before the split into CVODE_opencor / CVODE_myokit, 'CVODE' meant the CellML backend.
    The compatibility rewrite applied to every model type, so a cpp study that named its
    own valid solver had it silently replaced with a CellML one.
    """
    assert 'CVODE' in SOLVER_SCHEMA['solvers_by_model_type']['cpp']
    assert 'CVODE' not in SOLVER_SCHEMA['solvers_by_model_type']['cellml']
