"""Write a self-contained, runnable pipeline bundle from a ``user_inputs.yaml``.

A configuration is not a run. ``user_inputs.yaml`` says what to do, but reproducing
it later -- on another machine, in a paper's supplementary material, six months on --
also needs the model, the observations, the parameter bounds, and something to drive
them in the right order. This writes all of that into one folder:

    user_inputs_<yymmdd>.yaml    the configuration, with every path made relative
    run_pipeline.py              the driver; each stage gated by a ``do_*`` flag
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
