"""Write a self-contained, runnable pipeline bundle from a ``user_inputs.yaml``.

A configuration is not a run. ``user_inputs.yaml`` says what to do, but reproducing
it later -- on another machine, in a paper's supplementary material, six months on --
also needs the model, the observations, the parameter bounds, and something to drive
them in the right order. This writes all of that into one folder:

    user_inputs_<yymmdd>.yaml    the configuration, with every path made relative
    run_pipeline.py              the driver; each stage gated by a ``do_*`` flag
    plot_outputs.py              the figures, including the posterior ones
    plot_utilities.py            the machinery those figures are drawn with
    generated_models/<prefix>/   the CellML model
    resources/                   obs_data.json, params_for_id.csv, any user funcs

``run_pipeline.py`` is a *static* script: everything specific to a study lives in
the yaml beside it, so the script stays readable and there is no generated code to
audit. It finds its own configuration (the newest ``user_inputs_*.yaml`` next to
it) and resolves circulatory_autogen from ``--ca-src``, ``$CIRCULATORY_AUTOGEN_SRC``
or an installed ``libcuflynx``, in that order.

Stages, each gated by its flag in the yaml::

    do_simulation   run the model and dump its outputs
    do_emulation    train a surrogate of the observable features
    do_sensitivity  Sobol indices
    do_calibration  parameter identification
    do_uq           MCMC        (or do_ia for Laplace identifiability)

Usage::

    cuflynx-generate-pipeline --user-inputs my_run.yaml --output-dir ./bundle
    cd bundle && python run_pipeline.py

This began in CUFLynx, which exports the same bundle from the GUI's settings. The
scripts belong with the engine that has to keep working with them, so they live
here; CUFLynx keeps the part that is genuinely its own -- turning GUI settings into
a user_inputs dict -- and calls this to write the bundle.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import date

import yaml

from libcuflynx.scripts import _cli

PIPELINE_SCRIPT = '''#!/usr/bin/env python3
"""Reproducible circulatory_autogen pipeline.

This follows the circulatory_autogen "generation and calibration" tutorial:
build ONE config dict (``inp_data_dict``) from the sibling user_inputs_*.yaml,
then drive each stage with the class ``init_from_dict(...)`` constructors. Each
stage runs only if its ``do_*`` flag is set in the yaml — flip them there.

This folder is self-contained:
    user_inputs_<date>.yaml      the run configuration (edit the do_* flags here)
    generated_models/<prefix>/   the CellML model
    resources/                   obs_data.json + params_for_id.csv
    output/                      results are written here

Usage:
    python run_pipeline.py
    # needs the engine: `pip install libcuflynx` (any environment with it will do)

    python run_pipeline.py --ca-src /path/to/circulatory_autogen/src
    # or set CIRCULATORY_AUTOGEN_SRC -- a checkout wins over an installed package,
    # so this is how you run the bundle against your own circulatory_autogen
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    matches = sorted(glob.glob(os.path.join(HERE, "user_inputs_*.yaml")))
    if not matches:
        sys.exit("No user_inputs_*.yaml found next to this script.")
    with open(matches[-1]) as fh:
        return yaml.safe_load(fh)


def resolve_ca_src():
    """Where circulatory_autogen is coming from, or None for an installed package.

    Three arrangements, in the order the app itself uses (see ca_imports):

      1. ``--ca-src`` or ``CIRCULATORY_AUTOGEN_SRC`` naming a real directory wins.
         A checkout the user deliberately pointed at is always preferred, so someone
         developing against their own circulatory_autogen keeps doing so.
      2. Otherwise, ``libcuflynx`` installed as an ordinary package is enough.
      3. Only if neither is there does this give up, and then it names both ways out
         rather than only the checkout.

    Arrangement 2 used to be missing entirely, so a user who ran
    ``pip install libcuflynx``, configured no CA directory in the GUI and exported a
    pipeline got a bundle that could not run: it exited 1 asking for a checkout they
    had no reason to have. ``ca_imports`` learned to resolve an installed package when
    the distribution went to PyPI; this script carries a deliberate duplicate of that
    rule (see the ca_imports docstring) and the duplicate was not updated with it.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ca-src", default=None)
    args, _ = ap.parse_known_args()

    explicit = args.ca_src
    configured = explicit or os.environ.get("CIRCULATORY_AUTOGEN_SRC")

    if configured:
        if os.path.isdir(configured):
            return configured
        if explicit:
            # Named on the command line: a wrong path is a mistake to report, not
            # something to paper over by quietly running a different engine.
            sys.exit("--ca-src %r is not a directory." % (configured,))
        # A bundle often outlives the machine it was made on, so a stale
        # CIRCULATORY_AUTOGEN_SRC in someone's profile is ordinary rather than an
        # error. Say so, and carry on if there is an installed package to carry on
        # with.
        sys.stderr.write(
            "warning: CIRCULATORY_AUTOGEN_SRC=%r is not a directory; looking for an "
            "installed libcuflynx instead.\\n" % (configured,)
        )

    if installed_libcuflynx_available():
        return None

    sys.exit(
        "circulatory_autogen was not found. Either install the engine:\\n"
        "    pip install libcuflynx\\n"
        "or point this bundle at a checkout of it:\\n"
        "    python run_pipeline.py --ca-src /path/to/circulatory_autogen/src\\n"
        "    # or set CIRCULATORY_AUTOGEN_SRC in the environment"
    )


def installed_libcuflynx_available():
    """Whether ``libcuflynx`` can be imported without help from ``--ca-src``.

    ``find_spec`` rather than an import: this only has to answer whether the package
    is present, and importing it costs seconds the failure path should not pay. It
    raises rather than returning None when a parent is missing or is not a package,
    so catching that is part of the answer.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(CA_NAMESPACE) is not None
    except (ImportError, AttributeError, ValueError):
        return False


CA_NAMESPACE = "libcuflynx"

# CA's own top-level packages: anything else keeps its bare name (an
# ``operation_funcs`` must not become ``libcuflynx.operation_funcs``).
CA_PACKAGES = frozenset({
    "checks", "coupler", "emulators", "generators", "identifiabilty_analysis",
    "models", "param_id", "parsers", "protocol_runners", "scripts",
    "sensitivity_analysis", "solver1d", "solver_wrappers", "utilities",
})

# Modules whose namespaced spelling is not the flat one with the prefix glued on
# (CA #433 moved the funcs into the package; operation_funcs was only ever
# reachable by bare name off a directory).
RELOCATED_MODULES = {
    "cost_funcs_user": CA_NAMESPACE + ".funcs.cost_funcs_user",
    "operation_funcs_user": CA_NAMESPACE + ".funcs.operation_funcs_user",
    "modifier_funcs_user": CA_NAMESPACE + ".funcs.modifier_funcs_user",
    "operation_funcs": CA_NAMESPACE + ".param_id.operation_funcs",
}


def ca_candidates(module):
    """Both spellings of CA module ``module``, most-preferred first."""
    if module in RELOCATED_MODULES:
        return [RELOCATED_MODULES[module], module]
    if module.split(".", 1)[0] not in CA_PACKAGES:
        return [module]
    return [CA_NAMESPACE + "." + module, module]


def ca_import(module):
    """Import a circulatory_autogen module, in either of its two layouts.

    CA moved every module under a ``libcuflynx.`` namespace (CA #437) and older
    checkouts are flat, so an export has to run against both: try the namespaced
    spelling first, then the flat one. (Mirrors ``apps/api/ca_imports.py`` in
    CUFLynx, which is where this rule is documented; the tables above are pinned
    against it by ``tests/test_ca_import_parity.py``.)
    """
    import importlib

    errors = []
    for cand in ca_candidates(module):
        try:
            return importlib.import_module(cand)
        except ModuleNotFoundError as exc:
            # Only try the other spelling when *this* one is absent. A module
            # that is present but missing a dependency of its own must say so.
            if not (exc.name and (cand == exc.name or cand.startswith(exc.name + "."))):
                raise
            errors.append("%r (%s)" % (cand, exc))
    # ImportError, not sys.exit: callers below probe for optional CA features
    # inside ``except Exception`` and a SystemExit would sail straight past them.
    raise ImportError(
        "circulatory_autogen module %r could not be imported (tried %s). "
        "Point --ca-src at the 'src' folder of a circulatory_autogen clone."
        % (module, " and ".join(errors))
    )


def ca_from(module, *names):
    """``from <module> import <names>``; one name returns it, several a tuple.

    A name the module does not have raises **ImportError**, not AttributeError.
    The probes below are the "older CA -> fall back" idiom and they catch
    ImportError; a bare getattr let an AttributeError sail straight past them, so
    a feature the connected CA simply predates crashed the exported run instead
    of degrading it.
    """
    mod = ca_import(module)
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise ImportError(
            "circulatory_autogen's %r has no %s -- this circulatory_autogen "
            "predates it." % (module, ", ".join(missing))
        )
    values = tuple(getattr(mod, n) for n in names)
    return values[0] if len(names) == 1 else values


def _uq_block(cfg):
    """The UQ option block from the yaml, under either spelling.

    CA renamed do_mcmc/mcmc_options to do_uq/UQ_options once MCMC became one method of
    uncertainty quantification rather than the whole of it. An export made before that
    rename still runs: both spellings are read here, and only the new one is written.
    """
    return dict(cfg.get("UQ_options") or cfg.get("mcmc_options") or {})


def _do_uq(cfg):
    return bool(cfg.get("do_uq", cfg.get("do_mcmc", False)))


def _uq_options_key():
    """Whether this CA's init_from_dict reads UQ_options or the older mcmc_options.

    Called rather than cached at import: circulatory_autogen is only importable once
    --ca-src has been resolved onto sys.path.
    """
    try:
        import inspect

        CVS0DParamID = ca_from("param_id.paramID", "CVS0DParamID")

        if "UQ_options" in inspect.signature(CVS0DParamID.__init__).parameters:
            return "UQ_options"
    except Exception:
        pass
    return "mcmc_options"


def obs_data_items(doc):
    """The data_items of an obs_data document, whichever shape it is in.

    circulatory_autogen accepts two: an object with a ``data_items`` key, and a
    bare array *of* data_items (the data-only form the 3compartment and
    heat_fenics studies ship). Assuming the object form here crashed the MLE
    rewrite below with "'list' object has no attribute 'get'".

    Restated in this generated script rather than imported: the export has to run
    on a machine that has circulatory_autogen and nothing of CUFLynx. It mirrors
    ``obs_data.data_items_of`` in the app -- keep the two in step.
    """
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        items = doc.get("data_items") or []
        return items if isinstance(items, list) else []
    return []


def obs_protocol_info(doc):
    """The protocol_info of an obs_data document, or {} (the data-only form has
    none -- it is run with manual time). Mirrors ``obs_data.protocol_info_of``."""
    if isinstance(doc, dict) and isinstance(doc.get("protocol_info"), dict):
        return doc["protocol_info"]
    return {}


def build_inp_data_dict(cfg, output_dir):
    """Turn the exported yaml into a circulatory_autogen ``inp_data_dict`` with
    every path resolved to an absolute location inside this export folder. This is
    the dict the ``init_from_dict`` constructors consume (see the CA tutorial)."""
    resources = os.path.join(HERE, cfg.get("resources_dir", "resources"))
    generated_models_dir = os.path.join(HERE, "generated_models")
    solver_info = dict(cfg.get("solver_info", {}))
    solver_info.setdefault("solver", cfg.get("solver"))

    inp = {
        "file_prefix": cfg["file_prefix"],
        "input_param_file": cfg.get("input_param_file", cfg["file_prefix"] + "_parameters.csv"),
        "model_type": cfg.get("model_type", "cellml"),
        # The CellML lives at generated_models/<prefix>/<prefix>.cellml — the layout
        # circulatory_autogen resolves model_path to, so every stage agrees.
        "model_path": os.path.join(generated_models_dir, cfg["file_prefix"], cfg["model_file"]),
        "generated_models_dir": generated_models_dir,
        "resources_dir": resources,
        "param_id_output_dir": output_dir,
        "solver_info": solver_info,
        "dt": float(cfg.get("dt", 0.01)),
        "sim_time": float(cfg.get("sim_time", 2.0)),
        "pre_time": float(cfg.get("pre_time", 0.0)),
        "param_id_method": cfg.get("param_id_method", "genetic_algorithm"),
        "do_ad": bool(cfg.get("do_ad", False)),
        "optimiser_options": dict(cfg.get("optimiser_options", {})),
        _uq_options_key(): dict(_uq_block(cfg)),
        "sa_options": {**cfg.get("sa_options", {}), "output_dir": output_dir},
        "DEBUG": False,
    }
    if cfg.get("param_id_obs_path"):
        inp["param_id_obs_path"] = os.path.join(HERE, cfg["param_id_obs_path"])
        # Run the simulation over the same protocol window as calibration/SA and the
        # live app: when obs_data carries a protocol_info, its pre/sim times take
        # precedence over the yaml. The SA/calibration init_from_dict constructors
        # already do this internally; get_simulation_helper_from_inp_data_dict reads
        # only inp["pre_time"]/["sim_time"], so without this the simulation would run
        # an unwarmed, wrong-length window and its outputs wouldn't match the obs_data.
        try:
            proto = obs_protocol_info(json.loads(open(inp["param_id_obs_path"]).read()))
            pre = (proto.get("pre_times") or [None])[0]
            sim = (proto.get("sim_times") or [[None]])[0][0]
            if pre is not None:
                inp["pre_time"] = float(pre)
            if sim is not None:
                inp["sim_time"] = float(sim)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass
    if cfg.get("params_for_id_file"):
        inp["params_for_id_path"] = os.path.join(resources, cfg["params_for_id_file"])
    # The emulator keys, which nothing copied before there was an emulator stage.
    # emulator_dir is resolved against this folder when it is relative, so a bundle
    # carrying a trained emulator stays self-contained wherever it is unpacked.
    for key in ("do_emulation", "use_emulator", "emulator_settings"):
        if key in cfg:
            inp[key] = cfg[key]
    settings = inp.get("emulator_settings")
    if isinstance(settings, dict) and settings.get("emulator_dir"):
        settings = dict(settings)
        if not os.path.isabs(settings["emulator_dir"]):
            settings["emulator_dir"] = os.path.join(HERE, settings["emulator_dir"])
        inp["emulator_settings"] = settings
    # User-authored operation / cost / modifier funcs travel with the export, so
    # point CA at the copies in this folder rather than wherever they lived on
    # the machine that produced it (CA #303, #383). Without these the run dies on
    # the first data_item or params_for_id entry naming a func the user wrote.
    # Matched by suffix rather than by a fixed list, so a kind CUFLynx grows
    # needs no edit to this generated script. external_model_path rides along:
    # an external_python model is user-authored Python that travels with the
    # study for exactly the same reason, and CA reads it from its own key.
    for key in [
        k for k in cfg
        if k.endswith("_funcs_external_path") or k == "external_model_path"
    ]:
        if cfg.get(key):
            inp[key] = os.path.join(HERE, cfg[key])
    return inp


def mle_obs_data(obs_path, out_dir, cost_type="gaussian_MLE"):
    """MCMC / Laplace need ln L = -cost, so write a copy of the obs_data with every
    data_item's cost_type set to an MLE cost (mirrors uq_runner._mle_obs_path)."""
    obs = json.loads(open(obs_path).read())
    for item in obs_data_items(obs):
        if isinstance(item, dict):
            item["cost_type"] = cost_type
    out = os.path.join(out_dir, "uq_obs_data.json")
    open(out, "w").write(json.dumps(obs))
    return out


def flat_param_names(param_id):
    return [g[0] if isinstance(g, (list, tuple)) else g for g in param_id.get_param_names()]


def as_floats(series):
    """A flat list of floats from one variable's results, whatever holds them.

    get_all_results_dict() hands back the backend's own container: CasADi
    returns DM matrices, which are deliberately not iterable, while the others
    return arrays. get_results(..., flatten=True) used to do this flattening,
    so it has to happen here now that the outputs come from one call.
    """
    if hasattr(series, "full"):  # CasADi DM
        series = series.full().ravel()
    elif hasattr(series, "ravel"):  # numpy
        series = series.ravel()
    return [float(v) for v in series]


def save_all_outputs(out_dir, sim_helper, exp_idx=0):
    """Save a run's traces the way circulatory_autogen saves them.

    An ``all_outputs_exp_<i>.npz`` of ``{variable: series}`` -- the same file and
    the same shape CA writes for a calibrated best fit, so the plotting script
    has one reader for both and no CUFLynx-only JSON exists anywhere in the
    bundle (CUFLynx #210). ``get_all_results_dict`` excludes time, which is the
    one separate ask; it is stored under a "time"-suffixed key, which is how the
    abscissa is identified in CA's own files.
    """
    data = {name: np.asarray(as_floats(series))
            for name, series in sim_helper.get_all_results_dict().items()}
    data["time"] = np.asarray(as_floats(sim_helper.get_results(["time"], flatten=True)[0]))
    np.savez(os.path.join(out_dir, "all_outputs_exp_%d.npz" % exp_idx), **data)


def save_uq_samples(out_dir, flat, qnames):
    """Persist the posterior samples plus their labels.

    The samples *are* the result, so they are what is stored -- numeric, in CA's
    own .npy idiom. The histograms are derived by the plotting script, so the
    bundle carries no summary format of CUFLynx's own devising.
    """
    np.save(os.path.join(out_dir, "uq_posterior_samples.npy"), np.asarray(flat, dtype=float))
    with open(os.path.join(out_dir, "uq_param_names.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([[name] for name in qnames])


def main():
    # None means "an installed libcuflynx answers already" -- leave sys.path alone
    # rather than prepending a bogus entry.
    ca_src = resolve_ca_src()
    if ca_src:
        sys.path.insert(0, ca_src)
    cfg = load_config()

    output_dir = os.path.join(HERE, "output")
    os.makedirs(output_dir, exist_ok=True)
    inp = build_inp_data_dict(cfg, output_dir)

    # python / casadi_python backends run a generated .py model: build it from the
    # bundled CellML, alongside where circulatory_autogen expects the model.
    if inp["model_type"] in ("python", "casadi_python"):
        PythonGenerator = ca_from("generators.PythonGenerator", "PythonGenerator")

        cellml_path = os.path.join(HERE, "generated_models", cfg["file_prefix"], cfg["model_file"])
        inp["model_path"] = PythonGenerator(
            cellml_path,
            output_dir=os.path.dirname(cellml_path),
            module_name=cfg["file_prefix"],
            casadi_compat=(inp["model_type"] == "casadi_python"),
        ).generate()

    # ---- 1) Simulation -----------------------------------------------------
    if cfg.get("do_simulation"):
        print("=== simulation ===", flush=True)
        get_simulation_helper_from_inp_data_dict = ca_from(
            "solver_wrappers", "get_simulation_helper_from_inp_data_dict")

        sim_helper = get_simulation_helper_from_inp_data_dict(inp)
        sim_helper.run()
        save_all_outputs(output_dir, sim_helper)
        # Released before the next stage builds its own: a helper holds a
        # compiled model, and every stage below constructs one of its own.
        if hasattr(sim_helper, "close_simulation"):
            sim_helper.close_simulation()

    # ---- 2) Emulator training ---------------------------------------------
    # Before the analyses, because `use_emulator` makes them evaluate the trained
    # emulator instead of the solver. Training itself always runs the real solver.
    if cfg.get("do_emulation"):
        print("=== emulator training ===", flush=True)
        EmulatorTrainer = ca_from("emulators.emulator_trainer", "EmulatorTrainer")

        trainer = EmulatorTrainer.init_from_dict(inp)
        bundle = trainer.train()

        # train() returns the bundle on rank 0 and None everywhere else -- only
        # rank 0 fits. So only rank 0 can report on it: checking the return on
        # every rank would have each non-root rank decide nothing was trained.
        # A failure to fit raises rather than returning None, so there is nothing
        # to test for here beyond that.
        if getattr(trainer, "rank", 0) == 0 and bundle is not None:
            rows = sorted(bundle.error_stats(),
                          key=lambda r: (r.get("r2") is None, r.get("r2", 0.0)))
            print("  worst held-out R2 per feature:", flush=True)
            for row in rows[:10]:
                print("    %-44s %s" % (str(row.get("label"))[:44], row.get("r2")),
                      flush=True)
            print("  min_r2 refuses an emulator below it at use time; raise "
                  "emulator_settings.num_train_samples if these are weak.",
                  flush=True)

    # ---- 3) Sensitivity analysis ------------------------------------------
    if cfg.get("do_sensitivity"):
        print("=== sensitivity analysis ===", flush=True)
        SensitivityAnalysis = ca_from(
            "sensitivity_analysis.sensitivityAnalysis", "SensitivityAnalysis")

        sa_agent = SensitivityAnalysis.init_from_dict(inp)
        sa_agent.run_sensitivity_analysis(inp["sa_options"])
        # circulatory_autogen leaves its indices as CSVs and figures; the
        # plotting script wants them the way every other stage reports, so this
        # stage writes its own summary too. Without it the exported
        # plot_analysis() heatmap had nothing to read and never drew.
        # Nothing written here: circulatory_autogen already wrote
        # all_outputs_n<N>_Sobol_indices.csv into this directory, and the
        # plotting script reads that (CUFLynx #210). This stage used to write a
        # summary of its own beside CA's, which is a second format to keep in
        # step for no gain.

    # ---- 4) Calibration ----------------------------------------------------
    best_param_vals = None  # reused by UQ below when available
    calibrated = None  # the engine itself, likewise
    if cfg.get("do_calibration"):
        print("=== calibration ===", flush=True)
        CVS0DParamID = ca_from("param_id.paramID", "CVS0DParamID")

        param_id = CVS0DParamID.init_from_dict(inp)
        param_id.run()
        param_id.plot_outputs()
        best_param_vals = param_id.get_best_param_vals()
        # Kept for UQ below: it wants exactly this engine, and building a
        # second one compiles the model again.
        calibrated = param_id

    # ---- 5) Uncertainty quantification ------------------------------------
    if _do_uq(cfg) or cfg.get("do_ia"):
        method = "mcmc" if _do_uq(cfg) else "laplace"
        print(f"=== uncertainty quantification ({method}) ===", flush=True)
        paramID_module = ca_import("param_id.paramID")
        CVS0DParamID, ensure_mle_cost_type_for_bayesian_inner = ca_from(
            "param_id.paramID", "CVS0DParamID", "ensure_mle_cost_type_for_bayesian_inner")

        # MCMC / Laplace need ln L = -cost, so use an MLE obs copy + MLE cost_type.
        uq_key = _uq_options_key()
        cost_type = inp[uq_key].get("cost_type", "gaussian_MLE")
        uq_inp = dict(inp)
        uq_inp["param_id_obs_path"] = mle_obs_data(inp["param_id_obs_path"], output_dir, cost_type)
        uq_inp["optimiser_options"] = {**inp["optimiser_options"], "cost_type": cost_type}
        uq_inp[uq_key] = {**inp[uq_key], "cost_type": cost_type}

        # UQ needs a best fit: reuse the calibration above, else run one now.
        if best_param_vals is None:
            print("  running a calibration first to get the best fit for UQ", flush=True)
            calib = CVS0DParamID.init_from_dict(uq_inp)
            calib.run()
            best_param_vals = calib.get_best_param_vals()
            calibrated = calib  # so Laplace below reuses it rather than rebuilding
        best_param_vals = np.asarray(best_param_vals, dtype=float)

        if method == "mcmc":
            # Reuse the calibration engine when there is one: run_UQ promotes it
            # in place, so UQ samples with the model already compiled. Older CA
            # only offers run_mcmc() on an object built with mcmc_instead=True,
            # which forces a second engine and a second compile.
            if calibrated is not None and hasattr(calibrated, "run_UQ"):
                mcmc = calibrated
            else:
                mcmc = CVS0DParamID.init_from_dict({**uq_inp, "mcmc_instead": True})
            mcmc.set_best_param_vals(best_param_vals)
            ensure_mle_cost_type_for_bayesian_inner(paramID_module.mcmc_object, uq_inp)
            if hasattr(mcmc, "run_UQ"):
                mcmc.run_UQ(uq_inp.get(uq_key))
            else:
                mcmc.run_mcmc()
            if getattr(mcmc, "rank", 0) == 0:
                save_uq_samples(output_dir, mcmc.get_mcmc_samples()[0], flat_param_names(mcmc))
        else:
            IdentifiabilityAnalysis = ca_from(
                "identifiabilty_analysis.identifiabilityAnalysis", "IdentifiabilityAnalysis")

            # Reuse the calibration's engine when there is one: Laplace only
            # needs a built param_id, and constructing a second CVS0DParamID
            # compiles the model a second time for no gain.
            cvs = calibrated or CVS0DParamID.init_from_dict(uq_inp)
            ia = IdentifiabilityAnalysis.init_from_dict(uq_inp, cvs.param_id)
            ia.set_best_param_vals(best_param_vals)
            ensure_mle_cost_type_for_bayesian_inner(cvs.param_id, uq_inp)
            ia.run({"method": "Laplace"})
            if getattr(ia, "rank", 0) == 0:
                # CA renamed `mean_Lapalace` -> `mean_Laplace`; prefer the corrected
                # name, fall back to the old spelling for older CA versions.
                laplace_mean = getattr(ia, "mean_Laplace", None)
                if laplace_mean is None:
                    laplace_mean = ia.mean_Lapalace
                samples = np.random.multivariate_normal(
                    laplace_mean, ia.covariance_matrix_Laplace, size=100000
                )
                save_uq_samples(output_dir, samples, flat_param_names(cvs))

    print(f"Done. Outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
'''

PLOT_UTILITIES_SCRIPT = '''#!/usr/bin/env python3
"""Finding and loading a CUFLynx run's data. Machinery for plot_outputs.py.

Nothing here decides how anything *looks*. It locates the run directory, reads
the files a run leaves behind, and lays panels out on a grid; every colour,
label, axis and limit lives in plot_outputs.py, which is the file to edit.

Kept separate so that editing a plot never means reading past code that has
nothing to do with plots. You should not need to open this file.
"""

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Filled in by load_plotting_libs(), so importing this module stays cheap and a
# missing matplotlib is reported by plot_outputs rather than at import time.
plt = None
np = None

PLOTS_DIRNAME = "pyscript_plots"


def _default_out():
    """Where to look for run data, when nobody says.

    Two layouts reach this script. run_pipeline.py writes into `output/` beside
    it, so that is preferred. But CUFLynx also drops these scripts straight into
    the outputs directory the user chose, where the run data is in
    circulatory_autogen's own `<method>_<model>_<hash>_obs_data/` folders and
    there is no `output/` at all.
    """
    from_env = os.environ.get("CUFLYNX_OUTPUT_DIR")
    if from_env:
        return from_env
    beside = os.path.join(HERE, "output")
    return beside if os.path.isdir(beside) else HERE


OUT = _default_out()
PLOTS = os.path.join(OUT, PLOTS_DIRNAME)


def set_output_dir(path):
    """Point everything at another run directory."""
    global OUT, PLOTS
    OUT = os.path.abspath(path)
    PLOTS = os.path.join(OUT, PLOTS_DIRNAME)
    return OUT


def output_dir_from_argv(argv):
    """The directory named on the command line, or None."""
    args = [a for a in argv if not a.startswith("-")]
    flagged = [a.split("=", 1)[1] for a in argv if a.startswith("--output-dir=")]
    if "--output-dir" in argv:
        idx = argv.index("--output-dir")
        flagged += argv[idx + 1 : idx + 2]
    chosen = (flagged or args or [None])[0]
    return chosen


def load_plotting_libs():
    """Import matplotlib and numpy, or explain what to install."""
    global plt, np
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError as exc:
        raise SystemExit(
            "This script needs matplotlib and numpy, and could not import them "
            f"({exc}). Install them into the Python you are running this with: "
            "python -m pip install matplotlib numpy"
        )
    plt, np = _plt, _np
    return plt, np


# ---------------------------------------------------------------------------
# Finding things
# ---------------------------------------------------------------------------
def find(name):
    """The first file called `name` anywhere under the run directory."""
    matches = glob.glob(os.path.join(OUT, "**", name), recursive=True)
    return matches[0] if matches else None


def resolve_name(names, wanted):
    """Find `wanted` among a run's variable names, whichever way it is spelled.

    Three differences, all cosmetic and all fatal if ignored: obs_data writes
    ``aortic_root/v`` with a slash, the saved npz writes a dot, and
    circulatory_autogen's flat CellML calls the component ``aortic_root_module``.
    """
    if wanted in names:
        return wanted
    text = str(wanted)
    for sep in ("/", "."):
        if sep not in text:
            continue
        comp, var = text.split(sep, 1)
        bare = comp[:-7] if comp.endswith("_module") else comp
        for candidate_comp in (comp, bare, bare + "_module"):
            for out_sep in (".", "/"):
                candidate = f"{candidate_comp}{out_sep}{var}"
                if candidate in names:
                    return candidate
        if var in names:
            return var
    return None


def pick(series, name):
    """The array for `name`, or None if this run did not record it."""
    key = resolve_name(series, name)
    return series[key] if key else None


def is_time(operand):
    """Whether an operand names the time axis rather than a fitted series."""
    tail = str(operand).replace("/", ".").split(".")[-1].strip().lower()
    return tail in ("time", "t")


def tex(label, operation=""):
    """A panel or bar label, with the operation kept out of maths mode's way."""
    if not operation:
        return f"${label}$"
    return f"${label}$ ({operation.replace('_', ' ')})"


def plain(label):
    """A label safe to hand matplotlib as-is.

    Names like ``V_mid_m_Kv4_2`` are not maths. Wrapped in ``$`` by ``tex`` above,
    mathtext reads the second underscore as a second subscript and raises
    "Double subscript", which loses the whole figure over one axis label.
    Outside maths mode matplotlib renders the string literally, so the fix is to
    not wrap it -- only a stray ``$`` would reopen maths mode.
    """
    return str(label).replace("$", "")


# ---------------------------------------------------------------------------
# Reading what a run leaves behind
# ---------------------------------------------------------------------------
def cost_history():
    """Rows of the cost history, newest column first, or []."""
    path = find("best_cost_history.csv")
    if not path:
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            values = []
            for cell in row:
                try:
                    values.append(float(cell))
                except ValueError:
                    values = []
                    break
            if values:
                rows.append(values)
    return rows


def param_bounds():
    """``{label: (min, max)}`` from the study's own params_for_id, or {}.

    The bounds come from the file the *user* wrote, in ``resources/`` -- the same
    numbers they typed in the editor -- rather than from a copy of them written
    beside the outputs. Labels are matched loosely because the history header
    writes a member qname with "/" replaced by a space.
    """
    bounds = {}
    for name in ("params_for_id.json", "params_for_id.csv"):
        path = os.path.join(HERE, "resources", name)
        if not os.path.isfile(path):
            continue
        try:
            if name.endswith(".json"):
                with open(path, encoding="utf-8-sig") as fh:
                    entries = json.load(fh).get("params") or []
                for entry in entries:
                    targets = entry.get("targets") or entry.get("modifies") or []
                    labels = [entry.get("name")] + list(targets)
                    for label in labels:
                        if label and entry.get("min") is not None:
                            bounds[str(label)] = (float(entry["min"]), float(entry["max"]))
            else:
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    for row in csv.DictReader(fh):
                        label = "%s/%s" % (row["vessel_name"].strip(),
                                           row["param_name"].strip())
                        bounds[label] = (float(row["min"]), float(row["max"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        break
    # The history header writes "vessel param", not "vessel/param".
    for label, value in list(bounds.items()):
        bounds.setdefault(label.replace("/", " "), value)
    return bounds


def param_history():
    """``(generations, [(name, values), ...])`` of the fitted parameters.

    circulatory_autogen writes this file **normalised** to each parameter's
    [min, max] -- 0 is its lower bound, 1 its upper -- while the multi-start
    history files hold actual values. Nothing in the file says so, so the values
    are converted back here using the bounds from the study's own params_for_id
    (see :func:`param_bounds`); a parameter whose bounds cannot be found is left
    normalised rather than silently mis-scaled, and :func:`param_history_units`
    says which happened.
    """
    path = find("best_param_vals_history.csv")
    if not path:
        return [], [], False
    rows = []
    header = []
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.reader(fh)):
            try:
                rows.append([float(c) for c in row])
            except ValueError:
                if i == 0:
                    header = row
    if not rows:
        return [], [], False
    columns = list(zip(*rows))
    names = header if len(header) == len(columns) else [f"p{i}" for i in range(len(columns))]
    bounds = param_bounds()
    series = []
    for name, column in zip(names, columns):
        low_high = bounds.get(name.strip())
        if low_high is None:
            series.append((name, column, False))
            continue
        low, high = low_high
        series.append((name, tuple(low + v * (high - low) for v in column), True))
    return list(range(len(rows))), [(n, c) for n, c, _ in series], \
        all(scaled for _, _, scaled in series) and bool(series)


def _matrix_csv(path):
    """``(row_labels, col_labels, {row: {col: float|None}})`` from a labelled CSV."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        return [], [], {}
    header = [c.strip() for c in rows[0]][1:]
    labels, table = [], {}
    for row in rows[1:]:
        label = row[0].strip()
        labels.append(label)
        values = {}
        for col, cell in zip(header, row[1:]):
            try:
                values[col] = float((cell or "").strip())
            except ValueError:
                # circulatory_autogen marks a failed evaluation NaN deliberately,
                # to keep it distinct from a real zero.
                values[col] = None
        table[label] = values
    return labels, header, table


def sensitivity_indices():
    """Sensitivity indices from circulatory_autogen's own CSVs, or None.

    Sobol first (``all_outputs_n<N>_Sobol_indices.csv``: a Parameter column, then
    ``S1_<output>`` / ``ST_<output>`` per output), then the local-sensitivity
    matrix. Read from CA's files rather than from a summary the pipeline wrote
    beside them -- there is no CUFLynx-authored results format any more, so a run
    produced by CA's own scripts plots exactly like one produced here (#210).
    """
    matches = [m for m in sorted(glob.glob(os.path.join(OUT, "**", "*Sobol_indices.csv"),
                                           recursive=True))
               if "2nd_order" not in os.path.basename(m)]
    if matches:
        params, columns, table = _matrix_csv(matches[0])
        indices, output_names = {}, []
        for column in columns:
            kind, _, out_name = column.partition("_")
            if kind not in ("S1", "ST") or not out_name:
                continue
            if out_name not in output_names:
                output_names.append(out_name)
            indices.setdefault(kind, {}).setdefault(out_name, {})
            for param in params:
                indices[kind][out_name][param] = table[param].get(column)
        if indices:
            return {"indices": indices, "output_names": output_names,
                    "param_names": params}

    path = find("local_sensitivity_relative.csv")
    if not path:
        return None
    output_names, param_names, table = _matrix_csv(path)
    return {"indices": {"local": table}, "output_names": output_names,
            "param_names": param_names}


def uq_posteriors(bins=40):
    """Per-parameter posterior summary + histogram, or None.

    Binned here from the samples the run persisted, so the bundle stores the
    result (samples, in CA's .npy idiom) rather than a summary of it.
    """
    path = find("uq_posterior_samples.npy")
    names_path = find("uq_param_names.csv")
    if not path or not names_path:
        return None
    flat = np.load(path, allow_pickle=False)
    with open(names_path, newline="", encoding="utf-8") as fh:
        qnames = [row[0].strip() for row in csv.reader(fh) if row]
    out = []
    for i, qname in enumerate(qnames):
        if i >= flat.shape[1]:
            break
        col = np.asarray(flat[:, i], dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        counts, edges = np.histogram(col, bins=bins)
        out.append({
            "qname": qname,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "bins": [float(x) for x in edges],
            "counts": [int(x) for x in counts],
        })
    return out or None


def obs_data_items(doc):
    """The data_items of an obs_data document, whichever shape it is in.

    circulatory_autogen accepts two: an object with a ``data_items`` key, and a
    bare array *of* data_items (the data-only form the 3compartment and
    heat_fenics studies ship). Only the object form used to be recognised here,
    so a data-only study exported plots with no observed targets at all.

    Restated in this generated script rather than imported: it has to run on a
    machine that has none of CUFLynx. Mirrors ``obs_data.data_items_of`` in the
    app -- keep the two in step.
    """
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        items = doc.get("data_items") or []
        return items if isinstance(items, list) else []
    return []


def latest_obs_data():
    """The obs_data.json belonging to this run, or None.

    A run directory keeps a dated copy per attempt; the newest is the one the
    saved vectors came from.
    """
    matches = sorted(
        glob.glob(os.path.join(OUT, "**", "*obs_data*.json"), recursive=True),
        key=os.path.getmtime,
    )
    for path in reversed(matches):
        try:
            with open(path, encoding="utf-8-sig") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if obs_data_items(doc):
            return doc
    return None


def observed(doc=None):
    """One entry per fitted data_item: variable, label, operation, value, series.

    ``series`` is the operands with any time operand removed, so two data_items
    that reduce to the same trace are two targets on one curve rather than two
    curves.
    """
    doc = doc if doc is not None else latest_obs_data()
    out = []
    for item in obs_data_items(doc):
        operands = list(item.get("operands") or [])
        series = tuple(o for o in operands if not is_time(o))
        variable = series[0] if series else (operands[0] if operands else None)
        out.append(
            {
                "variable": variable,
                "series": series or (variable,),
                "label": (item.get("trace_name_for_plotting")
                          or item.get("name_for_plotting")
                          or item.get("data_item_name")
                          or item.get("variable") or variable),
                "operation": item.get("operation") or "",
                "value": item.get("value"),
                "experiment": int(item.get("experiment_idx", 0) or 0),
            }
        )
    return out


def all_outputs_files():
    """The ``all_outputs_*exp_<i>.npz`` traces present, best fit preferred.

    circulatory_autogen writes ``all_outputs_with_best_param_vals_exp_<i>.npz``
    after a calibration; a simulation-only run writes ``all_outputs_exp_<i>.npz``
    in the same shape. The best fit wins when both exist -- it is the run worth
    looking at, and plotting both would draw the same variables twice.
    """
    files = sorted(
        glob.glob(os.path.join(OUT, "**", "all_outputs_*exp_*.npz"), recursive=True)
    )
    # CA also writes a `_plot.npz` variant; prefer the full one when both exist.
    files = [f for f in files if not f.endswith("_plot.npz")] or files
    best = [f for f in files if "with_best_param_vals" in os.path.basename(f)]
    return best or files


def best_fit_runs():
    """``[(experiment, time, {variable: series}), ...]`` from the saved npz files."""
    files = all_outputs_files()
    runs = []
    for path in files:
        stem = os.path.basename(path)
        exp = "".join(c for c in stem.split("exp_")[-1] if c.isdigit()) or "0"
        data = np.load(path, allow_pickle=True)
        names = list(data.keys())
        time_key = next((n for n in names if n.endswith("time")), None)
        time = data[time_key] if time_key else np.arange(len(data[names[0]]))
        runs.append((exp, time, {n: data[n] for n in names if n != time_key}))
    return runs


def error_vector(name):
    """A saved error vector as a flat array, or None."""
    path = find(name)
    if not path:
        return None
    values = np.asarray(np.load(path, allow_pickle=True), dtype=float).ravel()
    return values if len(values) else None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def save(fig, filename, dpi=150):
    """Write a figure into the plots directory and close it.

    ``bbox_inches="tight"`` because a long observable name, a rotated tick label
    or a legend placed below the axes is otherwise cropped at the figure edge --
    silently, since the file is still written.
    """
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def grid(n, cols=3, fig_w=5.0, fig_h=3.4):
    """A figure with `n` axes laid out in a grid; unused axes are hidden."""
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w * cols, fig_h * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, list(axes.flat[:n])


def paginate(items, per_page):
    """`items` split into pages, so a 456-variable model is readable."""
    return [items[i : i + per_page] for i in range(0, len(items), per_page)]


# What this script can draw, and therefore what it looks for before claiming
# there is nothing to do.
#: Everything here is a file circulatory_autogen writes, or (for the posterior
#: samples) a numeric result in its .npy idiom. There is deliberately no
#: CUFLynx-authored results format left to look for (#210).
INPUTS = (
    "best_cost_history.csv",
    "best_param_vals_history.csv",
    "local_sensitivity_relative.csv",
    "uq_posterior_samples.npy",
    "percent_error_vec.npy",
    "std_error_vec.npy",
)


def nothing_to_plot():
    """The inputs, if none of them are anywhere under OUT; else an empty list."""
    if any(find(name) for name in INPUTS):
        return []
    if glob.glob(os.path.join(OUT, "**", "*Sobol_indices.csv"), recursive=True):
        return []
    if all_outputs_files():
        return []
    return list(INPUTS)


def run_sections(sections):
    """Draw each section, reporting failures without losing the others.

    A malformed results.json should not cost you the simulation plots that
    rendered perfectly well.
    """
    failures = []
    for section in sections:
        try:
            section()
        except Exception as exc:  # noqa: BLE001 - report and carry on
            failures.append(f"{getattr(section, '__name__', section)}: {exc}")
    return failures


# ---------------------------------------------------------------------------
# Posterior predictive (libcuflynx.param_id.posterior_predictive)
# ---------------------------------------------------------------------------
def mcmc_chain():
    """The raw chain, ``(steps, walkers, params)``, or None."""
    import numpy as np

    path = os.path.join(OUT, "mcmc_chain.npy")
    if not os.path.isfile(path):
        return None
    chain = np.load(path, allow_pickle=True)
    return chain if chain.ndim == 3 else None


def posterior_samples(burn_in=0.5):
    """The chain flattened after burn-in, ``(draws, params)``, or None.

    The walkers start scattered over the prior box, so the early steps say more
    about where they were initialised than about the posterior.
    """
    chain = mcmc_chain()
    if chain is None:
        return None
    start = int(chain.shape[0] * burn_in) if burn_in < 1 else int(burn_in)
    start = min(max(start, 0), max(chain.shape[0] - 1, 0))
    return chain[start:].reshape(-1, chain.shape[2])


def posterior_predictive():
    """What ``posterior_predictive.save()`` wrote, or None if the check was not run."""
    import numpy as np

    path = os.path.join(OUT, "posterior_predictive.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def posterior_predictive_coverage():
    """The coverage summary written beside the samples, or None."""
    path = os.path.join(OUT, "posterior_predictive_coverage.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def parameter_names():
    """Names for the calibrated parameters, in chain column order."""
    path = os.path.join(OUT, "param_names.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as handle:
            rows = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
        return rows or None
    except OSError:
        return None
'''

PLOTTING_SCRIPT = '''#!/usr/bin/env python3
"""Plots from a CUFLynx run — yours to edit.

    python plot_outputs.py                       # find the run data automatically
    python plot_outputs.py --output-dir <dir>    # a specific run directory
    CUFLYNX_OUTPUT_DIR=<dir> python plot_outputs.py

Every figure is drawn by a function in this file, and each one is ordinary
matplotlib. Finding and loading the data is `plot_utilities.py`, which you
should not need to open.

WHAT TO EDIT
    STYLE                one place for colours, sizes and dpi
    panel_*              one function per fitted observable, named after it
    PANELS               which of those panels appear, and in what order
    plot_*               one function per figure -- best fit, progress,
                         error bars, analysis, simulation traces
    FIGURES              which figures get drawn at all

To change one plot, edit its function. To drop it, remove it from FIGURES.
To add one, write a function and add it.
"""

import os
import sys

import plot_utilities as util

# ---------------------------------------------------------------------------
# STYLE — shared by every figure below
# ---------------------------------------------------------------------------
STYLE = {
    "palette": ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000", "#a142f4", "#e84a5f"],
    "target_colour": "#333333",
    # Dash patterns for observed-value lines, so several on one axes stay
    # tellable apart in grey scale as well as in colour.
    "target_dashes": [(4, 2), (1, 1.6), (6, 2, 1, 2), (3, 1, 1, 1), (8, 3)],
    "panel_cols": 3,
    "panel_size": (5.0, 3.4),   # inches, per panel
    "dpi": 150,
    # The pipeline logs every model variable, so one figure of 456 panels is
    # unusable. Traces are paginated at this many per page.
    "panels_per_page": 12,
}

PALETTE = STYLE["palette"]
TARGET_COLOUR = STYLE["target_colour"]
TARGET_DASHES = STYLE["target_dashes"]

# Bound to matplotlib/numpy in main(), so this file reads like a normal script.
plt = None
np = None


def colour(i):
    return PALETTE[i % len(PALETTE)]


def pick(series, name):
    """The array recorded for `name`, or None. Spelling differences handled."""
    return util.pick(series, name)


# <<PANELS>>


# ---------------------------------------------------------------------------
# BEST FIT — the calibrated traces, with what they were fitted to
# ---------------------------------------------------------------------------
def plot_best_fit():
    """One figure per experiment, one panel per entry in PANELS."""
    for exp, t, series in util.best_fit_runs():
        panels = PANELS or _discovered_panels(exp, series)
        if not panels:
            _plot_all_traces(t, series, f"best_fit_exp{exp}")
            continue
        fig, axes = util.grid(
            len(panels), STYLE["panel_cols"], *STYLE["panel_size"]
        )
        for ax, panel in zip(axes, panels):
            panel(ax, t, series)
        fig.tight_layout()
        util.save(fig, f"best_fit_exp{exp}.png", STYLE["dpi"])


# ---------------------------------------------------------------------------
# ERROR BARS — how far each observable ended up from its target
# ---------------------------------------------------------------------------
def plot_error_bars():
    """Sorted and signed: "which is worst, and in which direction"."""
    labels = [util.tex(o["label"], o["operation"]) for o in util.observed()]
    for name, title, unit, filename in (
        ("percent_error_vec.npy", "Best fit: error per observable", "error (%)",
         "calibration_percent_error.png"),
        ("std_error_vec.npy", "Best fit: error in standard deviations", "error (std)",
         "calibration_std_error.png"),
    ):
        values = util.error_vector(name)
        if values is None:
            continue
        names = labels if len(labels) == len(values) else [str(i) for i in range(len(values))]
        order = np.argsort(values)
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(values) + 3), 4))
        ax.barh(
            range(len(values)),
            values[order],
            color=["#c0504d" if v < 0 else PALETTE[0] for v in values[order]],
        )
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([names[i] for i in order], fontsize=9)
        ax.axvline(0, color="#333", lw=1)
        ax.set_xlabel(unit)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        util.save(fig, filename, STYLE["dpi"])


# ---------------------------------------------------------------------------
# PROGRESS — how the calibration got there
# ---------------------------------------------------------------------------
def plot_progress():
    costs = util.cost_history()
    if costs:
        best = [row[0] for row in costs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(best)), best, color=PALETTE[0], lw=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("generation")
        ax.set_ylabel("best cost")
        ax.set_title("Cost")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        util.save(fig, "progress_cost.png", STYLE["dpi"])

    generations, params, physical = util.param_history()
    if params:
        fig, axes = util.grid(len(params), STYLE["panel_cols"], 4.5, 3.0)
        for i, (ax, (name, values)) in enumerate(zip(axes, params)):
            ax.plot(generations, values, color=colour(i), lw=1.4)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("generation")
            # circulatory_autogen writes this history normalised to [min, max].
            # It is converted back using the bounds in the study's own
            # params_for_id; when those cannot be found the values stay
            # normalised, and the axis label says so rather than implying a
            # physical value.
            if physical:
                ax.set_ylabel("value", fontsize=7)
            else:
                ax.set_ylabel("normalised value", fontsize=7)
                ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        util.save(fig, "progress_params.png", STYLE["dpi"])


# ---------------------------------------------------------------------------
# ANALYSIS — sensitivity indices and UQ posteriors
# ---------------------------------------------------------------------------
def plot_analysis():
    """Sensitivity heatmap and UQ posteriors.

    Both come from circulatory_autogen's own files -- its Sobol / local
    sensitivity CSVs, and the posterior samples the run persisted -- so a run
    directory produced by CA's own scripts plots exactly like one produced by
    this pipeline (CUFLynx #210).
    """
    res = util.sensitivity_indices() or {}

    # Sensitivity: indices are {kind: {output: {param: value}}}.
    indices = res.get("indices")
    if indices:
        kind = "local" if "local" in indices else ("ST" if "ST" in indices else next(iter(indices)))
        by_out = indices[kind]
        outs = res.get("output_names") or list(by_out.keys())
        params = res.get("param_names") or sorted({p for o in by_out.values() for p in o})
        mat = np.array(
            [[by_out.get(o, {}).get(p, np.nan) for o in outs] for p in params], dtype=float
        )
        # A local index is signed -- which way a parameter pushes an output is
        # half the answer -- so it gets a diverging map centred on zero.
        signed = kind == "local"
        vmax = np.nanmax(np.abs(mat)) or 1.0
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * len(outs), 1 + 0.4 * len(params)))
        im = ax.imshow(
            mat, aspect="auto", cmap="coolwarm" if signed else "viridis",
            vmin=-vmax if signed else 0, vmax=vmax,
        )
        ax.set_xticks(range(len(outs)))
        ax.set_xticklabels(outs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=6)
        ax.set_title(f"Sensitivity ({kind})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        util.save(fig, "analysis_sensitivity.png", STYLE["dpi"])

    # UQ posteriors, binned from the samples the run persisted.
    uq_params = util.uq_posteriors()
    if uq_params:
        n = len(uq_params)
        fig, axes = plt.subplots(n, 1, figsize=(5, 2 * n), squeeze=False)
        for i, param in enumerate(uq_params):
            ax = axes[i][0]
            edges = np.array(param["bins"])
            counts = np.array(param["counts"])
            centres = (
                0.5 * (edges[:-1] + edges[1:])
                if len(edges) == len(counts) + 1
                else np.arange(len(counts))
            )
            width = (centres[1] - centres[0]) if len(centres) > 1 else 1
            ax.bar(centres, counts, width=width, color=PALETTE[0], alpha=0.6)
            ax.axvline(param["mean"], color=PALETTE[5])
            ax.set_title(param.get("qname", f"param {i}"), fontsize=7)
        fig.tight_layout()
        util.save(fig, "analysis_uq.png", STYLE["dpi"])


def _plot_all_traces(t, outputs, stem):
    """A panel per variable, paginated.

    A panel each, not one shared axes: model variables span wildly different
    scales -- pressures ~1e4, flows ~1e-4, valve states 0/1 -- so on a common
    linear axis all but the largest collapse onto zero.
    """
    names = list(outputs)
    if not names:
        return
    pages = util.paginate(names, STYLE["panels_per_page"])
    for page_no, page in enumerate(pages, start=1):
        fig, axes = util.grid(len(page), STYLE["panel_cols"], 4.5, 2.6)
        for i, (ax, name) in enumerate(zip(axes, page)):
            values = outputs[name]
            ax.plot(t[: len(values)], values[: len(t)], color=colour(i), lw=1.1)
            ax.set_title(name, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        page_suffix = f"_p{page_no}" if len(pages) > 1 else ""
        util.save(fig, f"{stem}{page_suffix}.png", STYLE["dpi"])


def _discovered_panels(exp, series):
    """Panels built at run time, when this script was written without obs_data.

    Grouped by series, so a trace fitted on its mean and its max is one panel
    with two targets rather than two panels of the same curve.
    """
    wanted = [o for o in util.observed() if o["experiment"] == int(exp)] or util.observed()
    order, by_series = [], {}
    for item in wanted:
        key = util.resolve_name(series, item["variable"])
        if key is None:
            continue
        if key not in by_series:
            by_series[key] = {"label": item["label"], "targets": []}
            order.append(key)
        by_series[key]["targets"].append(item)

    def make(key, i):
        def panel(ax, t, series_):
            group = by_series[key]
            values = series_[key]
            ax.plot(t[: len(values)], values[: len(t)], color=colour(i), lw=1.4, label="best fit")
            for j, target in enumerate(group["targets"]):
                value = target["value"]
                if isinstance(value, (int, float)):
                    ax.axhline(
                        value, color=TARGET_COLOUR, lw=1.1,
                        dashes=TARGET_DASHES[j % len(TARGET_DASHES)],
                        label=f"{target['operation'] or 'observed'} = {value:.4g}",
                    )
            ax.set_title(util.tex(group["label"]), fontsize=10)
            ax.set_xlabel("time")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="best")

        return panel

    return [make(key, i) for i, key in enumerate(order)]


# ---------------------------------------------------------------------------
def plot_corner():
    """Pairwise posterior. Where a fit is unidentifiable, this is where it shows.

    A ridge running diagonally across a pair means the two parameters trade off
    against each other and only their combination is determined -- a marginal
    histogram of either one hides that completely.
    """
    samples = util.posterior_samples()
    if samples is None or samples.shape[0] < 2:
        return

    names = util.parameter_names() or [
        "p%d" % i for i in range(samples.shape[1])]
    labels = [util.plain(n.split("/")[-1]) for n in names[: samples.shape[1]]]

    try:
        import corner
    except ImportError:
        # corner is a CA dependency, but the bundle may be run somewhere leaner.
        print("  corner is not installed; drawing marginals instead")
        n = samples.shape[1]
        fig, axes = util.grid(n, cols=4, fig_w=3.0, fig_h=2.2)
        for i, ax in enumerate(axes[:n]):
            ax.hist(samples[:, i], bins=40, color="#2a78d6")
            ax.set_title(labels[i], fontsize=8)
        util.save(fig, "posterior_marginals.png")
        return

    fig = corner.corner(
        samples, labels=labels, show_titles=True,
        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 8},
        quantiles=[0.05, 0.5, 0.95], color="#2a78d6",
    )
    util.save(fig, "posterior_corner.png", dpi=140)


# ---------------------------------------------------------------------------
def plot_posterior_predictive():
    """Model predictions from posterior draws against what was measured.

    One row per observable: the bar is the model's central interval over the
    draws, the marker is the measurement and its error bar is +/- one std. An
    observable whose bar misses its marker is one the fit does not reproduce,
    whatever the cost said.
    """
    data = util.posterior_predictive()
    if data is None:
        return

    preds = data["predictions"]
    truth = data["ground_truth"]
    std = data["std"]
    labels = [str(x) for x in data["labels"]]

    usable = ~np.all(np.isnan(preds), axis=0)
    if not usable.any():
        return
    idx = np.where(usable)[0]

    # Everything is plotted in units of the measurement's own std, so 84
    # observables on different scales can share one axis and "inside the error
    # bar" means the same distance everywhere.
    scale = np.where(np.abs(std[idx]) > 0, np.abs(std[idx]), 1.0)
    lo = (np.nanpercentile(preds[:, idx], 2.5, axis=0) - truth[idx]) / scale
    mid = (np.nanmedian(preds[:, idx], axis=0) - truth[idx]) / scale
    hi = (np.nanpercentile(preds[:, idx], 97.5, axis=0) - truth[idx]) / scale

    positions = list(range(len(idx)))
    for page, rows in enumerate(util.paginate(positions, 28)):
        rows = np.asarray(rows)
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.28 * len(rows) + 1.4)))
        y = np.arange(len(rows))[::-1]

        ax.axvspan(-1, 1, color="#e8e7e0", zorder=0, label="measured +/- 1 std")
        ax.axvline(0, color="#78776d", linewidth=1, zorder=1)
        ax.hlines(y, lo[rows], hi[rows], color="#2a78d6", linewidth=3,
                  alpha=0.55, zorder=2, label="model 95% interval")
        ax.plot(mid[rows], y, "o", color="#2a78d6", markersize=5, zorder=3,
                label="model median")

        ax.set_yticks(y)
        ax.set_yticklabels([util.plain(labels[idx[i]]) for i in rows],
                           fontsize=7)
        ax.set_xlabel("(model - measured) / measurement std")
        ax.set_title("Posterior predictive against the data", loc="left")
        ax.legend(fontsize=7, loc="lower right")
        util.save(fig, "posterior_predictive_%d.png" % page)


# ---------------------------------------------------------------------------
def plot_coverage():
    """Coverage against its nominal level.

    An 80% interval that contains 80% of the observations is calibrated. Far
    below and the posterior is too narrow or biased; far above and it is too wide
    to be saying much.
    """
    summary = util.posterior_predictive_coverage()
    if not summary:
        return
    levels = (summary.get("coverage") or {}).get("levels") or {}
    if not levels:
        return

    ordered = sorted(levels.items(), key=lambda kv: float(kv[0]))
    nominal = [float(k) for k, _ in ordered]
    predictive = [v["predictive_coverage"] for _, v in ordered]
    data_interval = [v["data_interval_coverage"] for _, v in ordered]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(ordered))
    bars = [
        ax.bar(x - 0.2, predictive, 0.38, color="#2a78d6",
               label="data inside model interval"),
        ax.bar(x + 0.2, data_interval, 0.38, color="#eb6834",
               label="model median inside data interval"),
    ]
    # Labelled directly: a coverage of 0 is a bar of no height, and an unlabelled
    # empty slot reads as "not measured" rather than "none of them".
    for group in bars:
        for rect in group:
            ax.annotate("%.0f%%" % (100 * rect.get_height()),
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color="#26251f")
    for i, level in enumerate(nominal):
        ax.hlines(level, i - 0.45, i + 0.45, color="#26251f", linewidth=2,
                  linestyle="--", zorder=3,
                  label="nominal" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(["%.0f%%" % (100 * level) for level in nominal])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of observables")
    ax.set_xlabel("credible level")
    n_obs = (summary.get("coverage") or {}).get("num_observables")
    used_emulator = summary.get("used_emulator")
    title = "Coverage over %s observables" % n_obs
    if used_emulator:
        title += "  (emulator, not solver)"
    ax.set_title(title, loc="left")
    # Below the axes: inside, it sat on top of the nominal lines it is explaining.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False)
    util.save(fig, "posterior_coverage.png")


# ---------------------------------------------------------------------------
# The figures, in order. Comment one out to stop drawing it.
# ---------------------------------------------------------------------------
# plot_best_fit draws the traces for every run, calibrated or not: a
# simulation-only run leaves the same all_outputs npz a calibration does, so
# there is no separate "simulation outputs" figure to draw (CUFLynx #210).
FIGURES = [
    plot_best_fit,
    plot_progress,
    plot_error_bars,
    plot_analysis,
    # These three draw nothing unless the run wrote a chain / ran the posterior
    # predictive check, so they are safe to leave on for a calibration-only run.
    plot_corner,
    plot_posterior_predictive,
    plot_coverage,
]


def main():
    global plt, np

    chosen = util.output_dir_from_argv(sys.argv[1:])
    if chosen:
        util.set_output_dir(chosen)

    if not os.path.isdir(util.OUT):
        raise SystemExit(f"No such directory: {util.OUT}")
    missing = util.nothing_to_plot()
    if missing:
        raise SystemExit(
            f"Nothing to plot in {util.OUT} — found none of {', '.join(missing)}. "
            f"Run run_pipeline.py first, or point this at a run directory: "
            f"python plot_outputs.py --output-dir <dir>"
        )

    plt, np = util.load_plotting_libs()
    os.makedirs(util.PLOTS, exist_ok=True)

    failures = util.run_sections(FIGURES)
    print(f"Plots written to {util.PLOTS}")
    for failure in failures:
        print(f"WARNING: {failure}")


if __name__ == "__main__":
    main()
'''

def _data_items_of(document):
    """The ``data_items`` of an obs_data document, in either accepted shape.

    A document may be the object form (``{"data_items": [...]}``) or a bare
    array of items -- the shipped 3compartment and heat_fenics obs_data are bare
    arrays. Tolerant on purpose: this is only used to name the generated panels,
    so anything else reads as "no items" rather than raising. The parser is what
    refuses a malformed document, with a message naming the problem.
    """
    if isinstance(document, dict):
        items = document.get('data_items')
        return items if isinstance(items, list) else []
    if isinstance(document, list):
        return document
    return []


def _identifier(text):
    """A readable Python identifier from a label like ``v_{AR}``."""
    cleaned = []
    for ch in str(text):
        cleaned.append(ch if (ch.isalnum() or ch == "_") else "_")
    name = "".join(cleaned).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    if not name or name[0].isdigit():
        name = f"panel_{name}" if name else "panel"
    return name


def _panel_functions(obs_data):
    """Generate one named panel function per fitted series.

    The alternative -- a loop over whatever obs_data happens to be next to the
    data -- produces a script that works and cannot be edited: to change one
    panel you have to understand the loop that draws all of them. Here each
    panel is a few lines of ordinary matplotlib with the variable names already
    written in, so changing one is changing one.

    Accepts either obs_data shape (object or bare array of data_items): the app
    hands over the object form, but a caller reading a study's file straight off
    disk has whichever the user wrote.
    """
    items = _data_items_of(obs_data)
    if not items:
        return (
            "# No obs_data was available when this script was written, so there are\n"
            "# no generated panels here. plot_best_fit() falls back to discovering\n"
            "# them from the obs_data.json in the run directory. Re-export with a\n"
            "# model loaded to get one named function per observable instead.\n"
            "PANELS = []"
        )

    # Group exactly as the drawing code does: one panel per series, with a time
    # operand ignored, so several operations on one trace share an axes.
    groups: list[dict] = []
    index: dict[tuple, dict] = {}
    for item in items:
        operands = list(item.get("operands") or [])
        series = tuple(
            o for o in operands
            if str(o).replace("/", ".").split(".")[-1].strip().lower() not in ("time", "t")
        )
        variable = series[0] if series else (operands[0] if operands else None)
        if not variable:
            continue
        key = series or (variable,)
        group = index.get(key)
        if group is None:
            group = {
                "variable": variable,
                "label": (item.get("trace_name_for_plotting")
                          or item.get("name_for_plotting")
                          or item.get("data_item_name")
                          or variable),
                "described": (item.get("data_item_name")
                              or item.get("variable") or ""),
                "targets": [],
            }
            index[key] = group
            groups.append(group)
        group["targets"].append(item)

    used: set[str] = set()
    blocks: list[str] = []
    names: list[str] = []
    for panel_idx, group in enumerate(groups):
        name = f"panel_{_identifier(group['label'])}"
        suffix = 2
        while name in used:
            name = f"panel_{_identifier(group['label'])}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)

        described = group["described"]
        title = f"${group['label']}$"
        lines = [
            f"def {name}(ax, t, series):",
            f'    """{described or group["label"]} — from {group["variable"]}."""',
            f'    y = pick(series, {group["variable"]!r})',
            "    if y is None:",
            f'        ax.set_title({title!r} + " (not recorded)")',
            "        return",
            f"    ax.plot(t[: len(y)], y[: len(t)], color=PALETTE[{panel_idx % 6}], "
            f'lw=1.4, label="best fit")',
        ]
        for i, target in enumerate(group["targets"]):
            value = target.get("value")
            operation = (target.get("operation") or "observed").replace("_", " ")
            if isinstance(value, (int, float)):
                lines.append(
                    f"    ax.axhline({value!r}, color=TARGET_COLOUR, lw=1.1, "
                    f"dashes=TARGET_DASHES[{i % 5}], "
                    f'label="{operation} = {value:.4g}")'
                )
        lines += [
            f"    ax.set_title({title!r}, fontsize=10)",
            '    ax.set_xlabel("time")',
            "    ax.grid(alpha=0.25)",
            '    ax.legend(fontsize=7, loc="best")',
        ]
        blocks.append("\n".join(lines))

    listing = "\n".join(f"    {n}," for n in names)
    blocks.append(
        "# The figure, in order. Comment a line out to drop that panel.\n"
        f"PANELS = [\n{listing}\n]"
    )
    return "\n\n\n".join(blocks)


def render_pipeline_script() -> str:
    """The standalone pipeline driver (reads the sibling dated yaml)."""
    return PIPELINE_SCRIPT


PLOT_UTILITIES_NAME = "plot_utilities.py"


PLOT_UTILITIES_NAME = 'plot_utilities.py'
PLOTTING_SCRIPT_NAME = 'plot_outputs.py'


def render_plot_utilities():
    """The machinery half: finding the run, reading its files, laying out axes.

    Split from the script the user edits so that changing a plot never means
    reading past code that has nothing to do with plots.
    """
    return PLOT_UTILITIES_SCRIPT


def render_plotting_script(obs_data=None):
    """The half the user edits, and the one they run.

    With an ``obs_data`` document the best-fit panels are written out as named
    functions with the variables filled in, so the script is something to edit
    rather than something to read around. Without one it still works, finding
    the panels in the run directory at draw time.
    """
    return PLOTTING_SCRIPT.replace("# <<PANELS>>", _panel_functions(obs_data))


PIPELINE_SCRIPT_NAME = 'run_pipeline.py'

#: Where a bundle lands when --output-dir is not given.
DEFAULT_OUTPUT_DIR = 'pipeline_bundle'

#: Keys naming a user-authored Python file that has to travel with the bundle: an
#: obs_data item names its operation by name, so a study using a func the user wrote
#: is not reproducible unless the func ships too (CA #303, #383).
_EXTERNAL_FILE_SUFFIX = '_funcs_external_path'
_EXTERNAL_FILE_KEYS = ('external_model_path',)

#: Copied into the bundle rather than described by an absolute path.
_RESOURCE_KEYS = ('param_id_obs_path', 'params_for_id_path')

#: Absolute locations that describe the machine the config came from, not the run.
#: run_pipeline.py rebuilds each of them relative to wherever the bundle is opened.
_MACHINE_PATH_KEYS = (
    'generated_models_dir', 'param_id_output_dir', 'external_modules_dir',
    'generated_models_subdir', 'param_id_output_dir_abs_path',
    'uncalibrated_model_path', 'vessels_csv_abs_path', 'parameters_csv_abs_path',
    'cpp_generated_models_dir', 'cpp_1d_model_config_path',
)


class PipelineBundleError(ValueError):
    """The configuration does not describe a bundle that could be written."""


def dated_suffix(today=None):
    return (today or date.today()).strftime('%y%m%d')


def render_pipeline_script():
    """The standalone pipeline driver. Static -- the study lives in the yaml."""
    return PIPELINE_SCRIPT


def _model_path(inp_data_dict):
    """The CellML to ship, from ``model_path`` or the conventional layout."""
    model_path = inp_data_dict.get('model_path')
    if model_path:
        return model_path
    prefix = inp_data_dict.get('file_prefix')
    generated = inp_data_dict.get('generated_models_dir')
    if not (prefix and generated):
        raise PipelineBundleError(
            'no model_path, and file_prefix/generated_models_dir are not both set, '
            'so the model to bundle cannot be located')
    return os.path.join(generated, prefix, prefix + '.cellml')


def _copy_into(source, directory, what):
    if not os.path.isfile(source):
        raise PipelineBundleError('%s not found: %s' % (what, source))
    os.makedirs(directory, exist_ok=True)
    name = os.path.basename(source)
    shutil.copyfile(source, os.path.join(directory, name))
    return name


def write_pipeline_bundle(inp_data_dict, output_dir, today=None):
    """Write the bundle described by ``inp_data_dict`` into ``output_dir``.

    Returns the paths written, relative to ``output_dir``.

    The configuration is rewritten as it is copied: every input becomes a path
    relative to the bundle, and the absolute directories that describe the machine
    it came from are dropped, because run_pipeline.py rebuilds them from its own
    location. That is what makes the folder movable.
    """
    if not isinstance(inp_data_dict, dict):
        raise PipelineBundleError('user inputs did not parse to a mapping of settings')
    if not inp_data_dict.get('file_prefix'):
        raise PipelineBundleError('user inputs has no file_prefix')

    prefix = inp_data_dict['file_prefix']
    output_dir = os.path.abspath(output_dir)
    resources = os.path.join(output_dir, 'resources')
    model_dir = os.path.join(output_dir, 'generated_models', prefix)
    os.makedirs(resources, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    cfg = dict(inp_data_dict)
    written = []

    cfg['model_file'] = _copy_into(_model_path(inp_data_dict), model_dir, 'model')
    written.append('generated_models/%s/%s' % (prefix, cfg['model_file']))
    cfg.pop('model_path', None)

    if inp_data_dict.get('param_id_obs_path'):
        name = _copy_into(inp_data_dict['param_id_obs_path'], resources, 'obs_data')
        cfg['param_id_obs_path'] = 'resources/' + name
        written.append('resources/' + name)

    if inp_data_dict.get('params_for_id_path'):
        name = _copy_into(
            inp_data_dict['params_for_id_path'], resources, 'params_for_id')
        # The generated script names this one by basename, not by path: it always
        # resolves inside resources_dir.
        cfg['params_for_id_file'] = name
        cfg.pop('params_for_id_path', None)
        written.append('resources/' + name)

    for key, value in list(inp_data_dict.items()):
        if not value or not isinstance(value, str):
            continue
        if key.endswith(_EXTERNAL_FILE_SUFFIX) or key in _EXTERNAL_FILE_KEYS:
            name = _copy_into(value, resources, key)
            cfg[key] = 'resources/' + name
            written.append('resources/' + name)

    for key in _MACHINE_PATH_KEYS:
        cfg.pop(key, None)
    cfg['resources_dir'] = 'resources'

    # A trained emulator is an input like any other when the bundle is meant to
    # reproduce a run that used one; an absolute dir would not survive the move.
    settings = cfg.get('emulator_settings')
    if isinstance(settings, dict) and settings.get('emulator_dir'):
        emulator_dir = settings['emulator_dir']
        if not os.path.isdir(emulator_dir) and cfg.get('use_emulator'):
            # The one way this bundle can look complete and not be: use_emulator
            # says the analyses evaluate an emulator, and the only pointer to it
            # is a path on the machine this was written on.
            print(
                'WARNING: use_emulator is set but no emulator was found at %s, so '
                'none is bundled and the yaml still names an absolute path. Train '
                'it and write the bundle again, or the folder will not run '
                'elsewhere.' % emulator_dir, file=sys.stderr)
        if os.path.isdir(emulator_dir):
            target = os.path.join(output_dir, 'emulator')
            if os.path.abspath(emulator_dir) != target:
                shutil.copytree(emulator_dir, target, dirs_exist_ok=True)
            settings = dict(settings)
            settings['emulator_dir'] = 'emulator'
            cfg['emulator_settings'] = settings
            written.append('emulator/')

    yaml_name = 'user_inputs_%s.yaml' % dated_suffix(today)
    with open(os.path.join(output_dir, yaml_name), 'w') as file:
        yaml.safe_dump(cfg, file, default_flow_style=False, sort_keys=False)
    written.insert(0, yaml_name)

    script_path = os.path.join(output_dir, PIPELINE_SCRIPT_NAME)
    with open(script_path, 'w') as file:
        file.write(render_pipeline_script())
    os.chmod(script_path, 0o755)
    written.insert(1, PIPELINE_SCRIPT_NAME)

    # The plots travel with the run for the same reason the model does: a folder
    # that reproduces a study but cannot draw it is half a bundle. The panels
    # read the run directory, so a calibration-only run simply draws fewer of
    # them.
    obs_document = None
    obs_source = inp_data_dict.get('param_id_obs_path')
    if obs_source and os.path.isfile(obs_source):
        try:
            with open(obs_source, encoding='utf-8-sig') as file:
                obs_document = json.load(file)
        except (OSError, ValueError):
            # Only used to name the best-fit panels; the script finds them at
            # draw time otherwise.
            obs_document = None

    for name, contents in (
            (PLOT_UTILITIES_NAME, render_plot_utilities()),
            (PLOTTING_SCRIPT_NAME, render_plotting_script(obs_document))):
        path = os.path.join(output_dir, name)
        with open(path, 'w') as file:
            file.write(contents)
        os.chmod(path, 0o755)
        written.insert(2, name)

    return written


def generate_pipeline_script(inp_data_dict=None, output_dir=None):
    """Stage entry point: write a bundle for the configured run."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser

    if inp_data_dict is None:
        yaml_parser = YamlFileParser()
        inp_data_dict = yaml_parser.parse_user_inputs_file(
            None, obs_path_needed=False, do_generation_with_fit_parameters=False)
    if not output_dir:
        raise PipelineBundleError('no output directory given')

    written = write_pipeline_bundle(inp_data_dict, output_dir)
    print('Wrote a runnable pipeline bundle to %s' % os.path.abspath(output_dir))
    for name in written:
        print('  %s' % name)
    print('\nRun it with:\n  cd %s && python %s'
          % (output_dir, PIPELINE_SCRIPT_NAME))
    return written


def main(argv=None):
    """Entry point for the ``cuflynx-generate-pipeline`` command."""
    parser = _cli.build_parser(
        'Write a self-contained folder that reproduces this configuration: the model, '
        'the observations, the parameter bounds, the configuration itself, and a '
        'run_pipeline.py that drives every stage its do_* flags enable.')
    # Optional, with a named default: a required option would be rejected by argparse
    # before --user-inputs is even looked at, so `--user-inputs <missing>` would report
    # the wrong problem.
    parser.add_argument(
        '--output-dir', dest='output_dir', metavar='DIR',
        default=DEFAULT_OUTPUT_DIR,
        help='directory to write the bundle into, created if absent '
             '(default: %(default)s, under the current directory)')
    args = parser.parse_args(argv)
    inp_data_dict = _cli.load_user_inputs(args)

    try:
        generate_pipeline_script(inp_data_dict, args.output_dir)
    except PipelineBundleError as exc:
        print('error: %s' % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
