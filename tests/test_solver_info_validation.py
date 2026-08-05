import pathlib
import warnings

import pytest

from parsers.PrimitiveParsers import (
    YamlFileParser,
    migrate_legacy_solver_info_keys,
    validate_solver_info,
    warn_if_casadi_nonzero_pre_time,
    PARAM_ID_METHODS,
    valid_param_id_methods,
    param_id_method_options,
    SOLVER_SCHEMA,
    SOLVER_INFO_FIELDS,
    solver_info_fields,
    gradient_sources,
    ANALYSIS_OPTIONS,
    analysis_options,
    _SOLVER_INTEGRATOR_KEYS,
    _CASADI_ADJOINT_METHODS,
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
    builtin_methods = {name for name in PARAM_ID_METHODS if not name.startswith('calisim_')}
    assert builtin_methods == {
        'genetic_algorithm', 'CMA-ES', 'bayesian', 'sp_minimize', 'multi_start_sp_minimize'
    }
    # The calisim_* methods are generated (one per calisim optimisation engine/method pair, see
    # param_id/calisim_methods.py) rather than listed here: which ones exist depends on the
    # installed calisim/openturns, so pin the shape, not the full set. They all dispatch through
    # the same is_calisim_method() branch of OpencorParamID.run().
    from param_id.calisim_methods import is_calisim_method
    calisim_methods = {name for name in PARAM_ID_METHODS if name.startswith('calisim_')}
    assert 'calisim_optuna_tpes' in calisim_methods
    assert all(is_calisim_method(name) for name in calisim_methods)
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
                                          'num_mutations_per_survivor', 'num_cross_breed'}
    assert names('CMA-ES') == {'num_calls_to_function', 'sigma0', 'cost_convergence',
                               'max_patience'}
    assert names('bayesian') == {'num_calls_to_function'}
    assert names('sp_minimize') == {'cost_convergence'}
    assert names('multi_start_sp_minimize') == {
        'num_starts', 'start_sampling', 'include_init_point', 'seed', 'fd_step',
        'no_new_starts_on_convergence', 'convergence_cluster_tol_frac', 'cost_convergence'}
    # multi-start is a superset of sp_minimize's gradient-descent settings
    assert names('sp_minimize') <= names('multi_start_sp_minimize')
    # the calisim backends (param_id/calisim_wrapper.py) all read the same block, plus
    # acquisition_func on the surrogate-based engines
    calisim_common = {'num_calls_to_function', 'cost_convergence', 'n_init', 'random_seed',
                      'n_jobs', 'method_kwargs'}
    assert names('calisim_optuna_tpes') == calisim_common
    assert names('calisim_emukit') == calisim_common | {'acquisition_func'}


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
        'tol', 'threads', 'max_step', 'gradient_strategy'}


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

    KNOWN LIMITATION -- this check is per *setting*, not per *solver*, and a name that merely
    passes through counts as "read". Both together let CVODE_myokit advertise
    MaximumNumberOfSteps for a long time: the name appears in protocol_runner.py, but only to be
    relayed into get_simulation_helper's solver_info, and myokit_helper drops it on the floor
    (myokit.Simulation exposes only set_max_step_size / set_min_step_size / set_tolerance). This
    docstring previously cited that relay as proof the setting was read, which is exactly the
    reasoning to distrust. Tightening this to per-solver, consumption-aware checking is the real
    fix; until then, a name appearing in the corpus is necessary but NOT sufficient.
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


def test_analysis_options_schema_well_formed():
    """The non-calibration analysis modes (sensitivity, MCMC, identifiability) expose their option
    blocks the same way, so a tool can auto-populate their settings forms too."""
    assert set(ANALYSIS_OPTIONS) == {'sensitivity_analysis', 'mcmc', 'identifiability_analysis'}
    for mode, meta in ANALYSIS_OPTIONS.items():
        assert meta.get('label') and meta.get('enable_flag') and meta.get('options_key')
        _assert_descriptors_well_formed(mode, meta.get('options'))
    # option names the analysis code actually reads (sensitivityAnalysis.py / paramID.py / IA)
    def names(mode):
        return {o['name'] for o in analysis_options(mode)}
    assert names('sensitivity_analysis') == {'method', 'sample_type', 'num_samples'}
    assert names('mcmc') == {'num_steps', 'num_walkers'}
    assert names('identifiability_analysis') == {'method', 'gradient_source', 'sub_method'}
    assert analysis_options('not_a_mode') == []
    # the enabling flags match the documented user_inputs feature flags
    assert {m['enable_flag'] for m in ANALYSIS_OPTIONS.values()} == {
        'do_sensitivity', 'do_mcmc', 'do_ia'}


def _option(mode, name):
    return next(o for o in analysis_options(mode) if o['name'] == name)


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

    src_file = Path(__file__).resolve().parents[1] / 'src' / 'sensitivity_analysis' / 'sobolSA.py'
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
    from funcs_user.cost_funcs_user import cost_func_metadata
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
    from funcs_user.cost_funcs_user import get_cost_funcs_dict_for_mode, cost_func_metadata
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
    from param_id.optimisers import GeneticAlgorithmOptimiser

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
    validate_solver_info('RK4', {
        'solver': 'RK4',
        'method': 'RK4',
        'MaximumStep': 0.001,
        'MaximumNumberOfSteps': 5000,
    })


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
    """CA's own cellml_only default must validate for myokit, not just opencor --
    otherwise the default config would be rejected by its own validator."""
    from parsers.PrimitiveParsers import _solver_info_default_for

    defaults = _solver_info_default_for('cellml_only', 'CVODE_myokit')
    assert 'MaximumNumberOfSteps' not in defaults
    assert defaults['MaximumStep'] == 0.001
    validate_solver_info('CVODE_myokit', defaults)

    # opencor keeps it: it is a real setting there.
    opencor = _solver_info_default_for('cellml_only', 'CVODE_opencor')
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
    from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis, sa_method_choices

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
    model types are AD_GRADIENT_MODEL_TYPES, and cellml_only+CVODE_myokit gets FSA.
    """
    from param_id.optimisers import AD_GRADIENT_MODEL_TYPES

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

    # cellml_only gets the Myokit CVODES FSA source only with the Myokit solver.
    myokit = gradient_sources('cellml_only', 'CVODE_myokit')
    _check_shape(myokit)
    fsa = [s for s in myokit if s['value'] == 'FSA']
    assert len(fsa) == 1 and fsa[0]['do_ad'] is True

    # No analytic source for cellml_only under a non-FSA solver, or for non-AD model types.
    for mt, sv in [('cellml_only', 'CVODE_opencor'), ('python', 'solve_ivp'), ('cpp', 'RK4')]:
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
                     gradient_sources('cellml_only', 'CVODE_myokit', method='CVODE')]
    assert 'FSA' in [s['value'] for s in gradient_sources('cellml_only', 'CVODE_myokit')]

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
    from parsers.PrimitiveParsers import AADC_TAPE_CONSISTENT_METHODS

    all_methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']
    advertised = SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']

    # Derived from AADC_AD_METHODS -- the tape-replayable methods PLUS the stiff BDF methods,
    # which reach a gradient by their own dispatch rather than the standard tape. Asserting
    # against AADC_TAPE_CONSISTENT_METHODS alone would exclude the BDF methods, which is exactly
    # the gap this replaced.
    from parsers.PrimitiveParsers import AADC_AD_METHODS
    assert advertised == [m for m in all_methods if m in AADC_AD_METHODS]
    assert set(AADC_TAPE_CONSISTENT_METHODS) <= set(advertised)
    assert set(advertised) <= set(all_methods), (advertised, all_methods)
    # the adaptive integrator is the one that must NOT be advertised
    assert 'adaptive_rk45' in all_methods and 'adaptive_rk45' not in advertised

    # and the runtime check enforces precisely this set
    from param_id.aadc_backend import TAPE_CONSISTENT_METHODS
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
    from parsers.PrimitiveParsers import gradient_sources

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
    from parsers.PrimitiveParsers import (
        AADC_AD_METHODS, AADC_BDF_AD_METHODS, AADC_TAPE_CONSISTENT_METHODS)
    import inspect
    from param_id import aadc_backend

    advertised = SOLVER_SCHEMA['ad_suitable_methods']['aadc_semi_implicit']
    all_methods = SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit']

    # both routes to a gradient are advertised, and nothing else is invented
    assert AADC_AD_METHODS == AADC_TAPE_CONSISTENT_METHODS + AADC_BDF_AD_METHODS
    assert advertised == [m for m in all_methods if m in AADC_AD_METHODS]
    for m in AADC_BDF_AD_METHODS:
        assert m in advertised, f"{m} has a gradient implementation but is not advertised"
    # the adaptive integrator still must not be offered
    assert 'adaptive_rk45' not in advertised

    # each advertised BDF method really is dispatched to its own gradient path
    src = inspect.getsource(aadc_backend.cost_and_grad)
    for m in AADC_BDF_AD_METHODS:
        assert m.upper() + "_METHOD" in src or repr(m) in src, (
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
    from parsers.PrimitiveParsers import AADC_FORWARD_METHODS
    from solver_wrappers import aadc_python_solver_helper as helper

    advertised = SOLVER_SCHEMA['forward_methods_by_solver']['aadc_semi_implicit']
    assert advertised == list(AADC_FORWARD_METHODS)
    assert 'semi_implicit_signed' not in advertised
    assert set(advertised) <= set(SOLVER_SCHEMA['methods_by_solver']['aadc_semi_implicit'])

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
    from solver_wrappers import aadc_python_solver_helper as helper

    src = inspect.getsource(helper.SimulationHelper.run)
    assert "AADC_FORWARD_METHODS" in src, "error text should be derived, not hand-written"


@pytest.mark.unit
def test_semi_implicit_signed_replaces_the_two_bdf_gradient_names():
    """bdf_tape and bdf_kernel were one integrator with two execution strategies.

    Both step x += dt*f/(1 - dt*diag J); they differ only in where the loop runs (an AADC tape,
    or a C++ kernel replay that falls back to the tape). Advertising them as separate integrators
    made a GUI offer two 'methods' that no forward solve accepts.
    """
    from parsers.PrimitiveParsers import (
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
def test_the_aadc_default_method_can_actually_integrate():
    """A default has to produce a number before it is AD-friendly.

    'rk4' was chosen in #336 for tape-consistency without checking it could integrate: on
    3compartment it raises OverflowError at dt 1e-3, 1e-4 and 1e-5, while implicit_newton lands
    within 2% of CVODE_myokit (issue #346).
    """
    from parsers.PrimitiveParsers import AADC_FORWARD_METHODS

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

    aadc = stiff['aadc_semi_implicit']
    assert aadc == ['semi_implicit', 'implicit_newton']
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
