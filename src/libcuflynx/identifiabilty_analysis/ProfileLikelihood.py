"""Profile likelihood for parameter identifiability.

For each calibrated entry in ``param_id_info``, the entry is held at a sweep of values around its
best fit and the other entries are re-calibrated at each value with the configured
``param_id_method``. The best cost at each held value is the profile.

The cost relates to the log-likelihood the same way as everywhere else in the package
(``ParamID.get_lnlikelihood_from_params``): for ``@is_MLE`` costs,
``-ln L = cost * _lnlikelihood_denorm_factor``. The 95% profile-likelihood interval is where
``-ln L`` stays within ``NLL_THRESHOLD_95`` of its minimum, i.e. where the cost stays within
``NLL_THRESHOLD_95 / _lnlikelihood_denorm_factor`` of it.

So that the interval always has this meaning, the profile (the best fit it is measured from and
every re-fit) runs on ``@is_MLE`` costs, as Laplace does: an observable whose cost has an MLE
form (``MLE_FORMS``) is switched to it by ``ensure_mle_cost_type_for_bayesian_inner``, with a
warning naming the observable, and the calibration's own costs are put back afterwards. An
observable whose cost has no MLE form is an error.
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from libcuflynx.param_id.paramID import ensure_mle_cost_type_for_bayesian_inner
from libcuflynx.parsers.PrimitiveParsers import analysis_options
from libcuflynx.utilities.mpi_utils import get_MPI as _get_MPI
from libcuflynx.utilities.utility_funcs import Normalise_class

MPI = _get_MPI()

#: Rise in -ln L that bounds the 95% profile-likelihood interval for one parameter (Wilks):
#: 2 * (-ln L) rises by the chi-square(1 dof) 0.95 quantile, 3.841, so -ln L rises by 1.921.
NLL_THRESHOLD_95 = 0.5 * float(chi2.ppf(0.95, df=1))

#: The MLE form of each cost that is not ``@is_MLE`` itself: the ``@is_MLE`` cost with the same
#: minimiser (MSE is 2 * gaussian_MLE), which ``ensure_mle_cost_type_for_bayesian_inner`` puts in
#: its place. Any other cost without ``@is_MLE`` (such as AE) has no likelihood to profile.
MLE_FORMS = {"MSE": "gaussian_MLE"}


def profile_likelihood_options(ia_options):
    """``num_points``, ``range_factor`` and ``optimiser_options`` from ``ia_options``, checked.

    Defaults come from ``ANALYSIS_OPTIONS['identifiability_analysis']``.
    """
    defaults = {o["name"]: o["default"] for o in analysis_options("identifiability_analysis")}
    num_points = ia_options.get("num_points", defaults["num_points"])
    range_factor = ia_options.get("range_factor", defaults["range_factor"])
    optimiser_options = ia_options.get("optimiser_options", defaults["optimiser_options"])
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, (int, np.integer))
        or num_points < 2
    ):
        raise ValueError(f"ia_options num_points must be an integer >= 2, got {num_points!r}.")
    if (
        isinstance(range_factor, bool)
        or not isinstance(range_factor, (int, float))
        or not 0 < range_factor <= 1
    ):
        raise ValueError(
            f"ia_options range_factor must be a number in (0, 1], got {range_factor!r}."
        )
    if optimiser_options is not None and not isinstance(optimiser_options, dict):
        raise ValueError(
            f"ia_options optimiser_options must be a mapping, got {optimiser_options!r}."
        )
    return {
        "num_points": int(num_points),
        "range_factor": float(range_factor),
        "optimiser_options": optimiser_options or {},
    }


def param_label(param_name):
    """Display label for an entry's names: one qname, or a group's qnames joined by '+'."""
    if isinstance(param_name, (list, tuple, np.ndarray)):
        return "+".join(str(name) for name in param_name)
    return str(param_name)


def _file_stem(param_idx, param_name):
    return f"param_{param_idx}_" + param_label(param_name).replace("/", "_").replace("+", "__")


class _HeldEntry:
    """The param_id's cost with entry ``idx`` held at ``value``, over the other entries.

    This is the ``cost_obj`` a profile point's optimiser calibrates (see
    ``ParamID.make_optimiser``). Each call puts ``value`` back into the vector at ``idx`` and
    hands the full vector to the param_id, so the param_id and its backends always see the
    parameter set they were built for, and a grouped or modifier entry is held as a whole by
    ``expand_modifier_param_vals``. ``param_init`` is the warm start.
    """

    def __init__(self, param_id, idx, value, param_init):
        self.param_id = param_id
        self.idx = int(idx)
        self.value = float(value)
        self.param_init = np.asarray(param_init, dtype=float)
        self.best_param_vals = None

    def full(self, param_vals):
        return np.insert(np.asarray(param_vals, dtype=float), self.idx, self.value)

    def _reduced(self, gradient):
        return np.delete(np.asarray(gradient, dtype=float).ravel(), self.idx)

    def get_cost_from_params(self, param_vals):
        return self.param_id.get_cost_from_params(self.full(param_vals))

    def get_cost(self, param_vals):
        return self.param_id.get_cost(self.full(param_vals))

    def get_lnlikelihood_lnprior_from_params(self, param_vals):
        return self.param_id.get_lnlikelihood_lnprior_from_params(self.full(param_vals))

    def get_gradient(self, param_vals):
        return self._reduced(self.param_id.get_gradient(self.full(param_vals)))

    def get_cost_and_gradient(self, param_vals):
        cost, gradient = self.param_id.get_cost_and_gradient(self.full(param_vals))
        return cost, self._reduced(gradient)

    def set_best_param_vals(self, param_vals):
        self.best_param_vals = np.asarray(param_vals, dtype=float)


class ProfileLikelihood:
    """Profile likelihood analysis around a best fit.

    For each calibrated entry:

    1. hold the entry at ``num_points`` values within ``range_factor`` of its ``[min, max]``
       range either side of the best fit;
    2. re-calibrate the other entries at each held value, warm-started from the optimum at the
       neighbouring held value (the sweep runs outwards from the best fit);
    3. record the cost at each re-calibrated optimum.

    A grouped entry is held as a whole (every target takes the value) and a modifier entry is
    held at theta, applied through its modifier function, as in calibration.

    ``optimiser_options`` are merged over the inner param_id's own for the re-calibrations, so a
    cheaper budget (``num_calls_to_function``) can be set for the profile points.
    """

    def __init__(
        self,
        param_id,
        param_id_info,
        output_dir,
        num_points=50,
        range_factor=0.2,
        optimiser_options=None,
    ):
        self.param_id = param_id
        self.param_id_info = param_id_info
        self.output_dir = output_dir
        self.num_points = num_points
        self.range_factor = range_factor
        self.optimiser_options = optimiser_options or {}

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        self.profile_results = {}
        self.best_param_vals = None
        self.best_cost = None

    def set_best_param_vals(self, best_param_vals):
        """Set the best-fit parameter values the profiles are taken around."""
        self.best_param_vals = np.array(best_param_vals, dtype=float)

    # ---- likelihood scaling -------------------------------------------------------------------

    def _nll_per_cost(self):
        """Factor turning the cost into -ln L (see ``ParamID.get_lnlikelihood_from_params``)."""
        return float(getattr(self.param_id, "_lnlikelihood_denorm_factor", 1.0))

    def _use_mle_costs(self):
        """Put every observable on an ``@is_MLE`` cost; return the settings to restore.

        Raises ``ValueError`` naming each observable whose cost has no MLE form. Returns
        ``None`` when the param_id has no observables, else ``(obs_info cost_type, cost_type)``
        as they were.
        """
        pid = self.param_id
        obs_info = getattr(pid, "obs_info", None)
        if obs_info is None:
            return None
        costs = getattr(pid, "cost_funcs_dict", None) or {}
        original = list(obs_info.get("cost_type") or [])
        names = obs_info.get("data_item_names") or [f"observable {i}" for i in range(len(original))]
        no_form = [
            f"'{names[i]}' (cost '{ct}')"
            for i, ct in enumerate(original)
            if not getattr(costs.get(ct), "is_MLE", False) and ct not in MLE_FORMS
        ]
        if no_form:
            forms = ", ".join(f"'{ct}' -> '{mle}'" for ct, mle in MLE_FORMS.items())
            raise ValueError(
                "Profile likelihood needs a likelihood cost for every observable, and these "
                f"costs have no MLE form: {', '.join(no_form)}. Use an @is_MLE cost (such as "
                f"gaussian_MLE) or a cost with an MLE form ({forms})."
            )

        saved = (obs_info.get("cost_type"), getattr(pid, "cost_type", None))
        ensure_mle_cost_type_for_bayesian_inner(pid, {})
        if self.rank == 0:
            for name, old, new in zip(names, original, obs_info["cost_type"]):
                if old != new:
                    warnings.warn(
                        f"profile likelihood: converting cost '{old}' to its MLE form "
                        f"'{new}' for {name}",
                        UserWarning,
                        stacklevel=3,
                    )
        return saved

    def confidence_interval_95(self, param_values, costs, best_val=None, best_cost=None):
        """The 95% profile-likelihood interval, in cost units.

        Returns ``{'lower', 'upper', 'cost_threshold'}``: the held values where the profile
        crosses ``min cost + NLL_THRESHOLD_95 / _lnlikelihood_denorm_factor``, linearly
        interpolated. A bound the sweep does not reach is ``None`` (the interval is open on that
        side within the sweep). The minimum includes the best fit itself when given.
        """
        param_values = np.asarray(param_values, dtype=float)
        costs = np.asarray(costs, dtype=float)
        finite = np.isfinite(costs)
        if not np.any(finite):
            return {"lower": None, "upper": None, "cost_threshold": None}
        min_cost = float(np.min(costs[finite]))
        if best_cost is not None and np.isfinite(best_cost):
            min_cost = min(min_cost, float(best_cost))
        threshold = min_cost + NLL_THRESHOLD_95 / self._nll_per_cost()

        centre = int(np.argmin(np.where(finite, costs, np.inf)))
        if best_val is not None:
            centre = int(np.argmin(np.abs(param_values - best_val)))

        def crossing(indices):
            for a, b in zip(indices[:-1], indices[1:]):
                ya, yb = costs[a], costs[b]
                if ya <= threshold < yb or (np.isfinite(ya) and not np.isfinite(yb)):
                    if not np.isfinite(yb):
                        return float(param_values[a])
                    xa, xb = param_values[a], param_values[b]
                    return float(xa + (threshold - ya) * (xb - xa) / (yb - ya))
            return None

        upper = crossing(list(range(centre, len(costs))))
        lower = crossing(list(range(centre, -1, -1)))
        return {"lower": lower, "upper": upper, "cost_threshold": threshold}

    # ---- one profile point --------------------------------------------------------------------

    def _optimize_with_fixed_parameter(self, fixed_param_idx, fixed_value, param_init=None):
        """Re-calibrate the other entries with entry ``fixed_param_idx`` held at ``fixed_value``.

        Collective: every rank runs the optimiser.

        Args:
            fixed_param_idx: Index of the held entry.
            fixed_value: Value to hold it at (theta for a modifier entry).
            param_init: Starting point for the other entries; defaults to their best fit.

        Returns:
            ``(cost, optimum)``: the cost at the re-calibrated optimum, and that optimum (the
            other entries' values, in order).
        """
        pid = self.param_id
        if param_init is None:
            param_init = np.delete(self.best_param_vals, fixed_param_idx)
        held = _HeldEntry(pid, fixed_param_idx, fixed_value, param_init)
        if pid.num_params == 1:
            # Nothing else to re-calibrate: the profile is the cost itself.
            return float(held.get_cost(np.empty(0))), np.empty(0)

        info = pid.param_id_info
        keep = [i for i in range(pid.num_params) if i != fixed_param_idx]
        mins = np.asarray(info["param_mins"], dtype=float)[keep]
        maxs = np.asarray(info["param_maxs"], dtype=float)[keep]
        reduced_info = {
            "param_names": [info["param_names"][i] for i in keep],
            "param_mins": mins,
            "param_maxs": maxs,
        }
        # output_dir None: a profile point's optimiser writes no files. The profile's own JSON
        # and plots are the output.
        optimiser = pid.make_optimiser(
            held,
            reduced_info,
            Normalise_class(mins, maxs),
            None,
            optimiser_options={**(pid.optimiser_options or {}), **self.optimiser_options},
        )
        optimiser.run()
        optimum = np.array(optimiser.best_param_vals, dtype=float)
        return float(held.get_cost(optimum)), optimum

    def _profile_single_parameter(self, param_idx):
        """Sweep entry ``param_idx`` outwards from its best fit, warm-starting each point."""
        best_val = float(self.best_param_vals[param_idx])
        param_min = self.param_id.param_id_info["param_mins"][param_idx]
        param_max = self.param_id.param_id_info["param_maxs"][param_idx]

        range_width = (param_max - param_min) * self.range_factor
        sweep_min = max(param_min, best_val - range_width)
        sweep_max = min(param_max, best_val + range_width)

        param_values = np.linspace(sweep_min, sweep_max, self.num_points)
        costs = np.full(self.num_points, np.inf)
        optima = [None] * self.num_points

        start = int(np.argmin(np.abs(param_values - best_val)))
        best_others = np.delete(self.best_param_vals, param_idx)
        for order in (range(start, self.num_points), range(start - 1, -1, -1)):
            # Up from the best fit, then down from the point nearest it.
            x0 = best_others if order.start == start or optima[start] is None else optima[start]
            for i in order:
                if self.rank == 0:
                    print(f"  Evaluating point {i + 1}/{self.num_points}: {param_values[i]:.6g}")
                costs[i], optima[i] = self._optimize_with_fixed_parameter(
                    param_idx, param_values[i], param_init=x0
                )
                if np.isfinite(costs[i]) and np.all(np.isfinite(optima[i])):
                    x0 = optima[i]

        param_name = self.param_id.param_id_info["param_names"][param_idx]
        ci_95 = self.confidence_interval_95(param_values, costs, best_val, self.best_cost)
        return {
            "param_values": param_values,
            "costs": costs,
            "param_name": param_name,
            "best_val": best_val,
            "best_cost": self.best_cost,
            "ci_95": ci_95,
        }

    # ---- output -------------------------------------------------------------------------------

    def _save_results(self):
        """Write one JSON per entry and ``summary.json`` under ``profile_likelihood/``."""
        results_dir = os.path.join(self.output_dir, "profile_likelihood")
        os.makedirs(results_dir, exist_ok=True)

        for param_idx, profile in self.profile_results.items():
            filepath = os.path.join(
                results_dir, _file_stem(param_idx, profile["param_name"]) + ".json"
            )
            best_cost = profile.get("best_cost")
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "param_values": np.asarray(profile["param_values"]).tolist(),
                        "costs": np.asarray(profile["costs"]).tolist(),
                        "param_name": profile["param_name"],
                        "best_val": float(profile["best_val"]),
                        "best_cost": None if best_cost is None else float(best_cost),
                        "nll_per_cost": self._nll_per_cost(),
                        "ci_95": profile.get("ci_95"),
                    },
                    f,
                    indent=2,
                )

        summary = {
            "num_params": len(self.profile_results),
            "num_points": self.num_points,
            "range_factor": self.range_factor,
            "nll_threshold_95": NLL_THRESHOLD_95,
            "parameters": [profile["param_name"] for profile in self.profile_results.values()],
        }
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _y_limits(costs):
        """Finite y-limits with 5% padding, or None when no cost is finite."""
        finite = np.asarray(costs, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        ymin, ymax = float(finite.min()), float(finite.max())
        pad = (ymax - ymin) * 0.05 or max(abs(ymax) * 0.05, 1e-12)
        return ymin - pad, ymax + pad

    def _draw_profile(self, ax, profile):
        label = param_label(profile["param_name"])
        ax.plot(profile["param_values"], profile["costs"], "b-o", linewidth=2, markersize=3)
        ax.axvline(profile["best_val"], color="r", linestyle="--", alpha=0.7, label="Best fit")
        ci_95 = profile.get("ci_95")
        if ci_95 and ci_95.get("cost_threshold") is not None:
            ax.axhline(ci_95["cost_threshold"], color="g", linestyle=":", label="95% threshold")
            values = profile["param_values"]
            lower = ci_95["lower"] if ci_95["lower"] is not None else values[0]
            upper = ci_95["upper"] if ci_95["upper"] is not None else values[-1]
            ax.axvspan(lower, upper, alpha=0.2, color="green", label="95% CI")
        ax.set_xlabel(label)
        ax.set_ylabel("Cost")
        ax.set_title(f"Profile: {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_profiles(self):
        """Plot every profile on one page, and each on its own page."""
        results_dir = os.path.join(self.output_dir, "profile_likelihood")
        os.makedirs(results_dir, exist_ok=True)

        num_params = len(self.profile_results)
        cols = min(3, num_params)
        rows = (num_params + cols - 1) // cols
        all_costs = np.concatenate(
            [np.asarray(p["costs"], dtype=float) for p in self.profile_results.values()]
        )
        y_limits = self._y_limits(all_costs)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.flatten()
        for idx, profile in enumerate(self.profile_results.values()):
            self._draw_profile(axes_flat[idx], profile)
            if y_limits is not None:
                axes_flat[idx].set_ylim(*y_limits)
        for ax in axes_flat[num_params:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(
            os.path.join(results_dir, "profile_likelihood_plots.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        for param_idx, profile in self.profile_results.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            self._draw_profile(ax, profile)
            y_limits = self._y_limits(profile["costs"])
            if y_limits is not None:
                ax.set_ylim(*y_limits)
            fig.savefig(
                os.path.join(results_dir, _file_stem(param_idx, profile["param_name"]) + ".pdf"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    # ---- entry point --------------------------------------------------------------------------

    def run(self):
        """Profile every calibrated entry, then save and plot the results (rank 0).

        Collective: every MPI rank must call it, since each profile point is a calibration.
        """
        if self.best_param_vals is None:
            raise ValueError("Best parameter values must be set first - Run param_id script first")

        num_params = len(self.best_param_vals)
        saved_costs = self._use_mle_costs()
        try:
            # get_cost throughout: the cost the re-fits minimise (the CasADi or AADC one when
            # the gradient is AD), and on casadi_python the only one left once AD has run.
            self.best_cost = float(self.param_id.get_cost(self.best_param_vals))
            if self.rank == 0:
                print(f"Starting profile likelihood analysis for {num_params} parameters")

            for param_idx in range(num_params):
                if self.rank == 0:
                    print(
                        f"Profiling parameter {param_idx + 1}/{num_params}: "
                        f"{param_label(self.param_id.param_id_info['param_names'][param_idx])}"
                    )
                self.profile_results[param_idx] = self._profile_single_parameter(param_idx)
                # Leave the model at the best fit, not at the last point evaluated.
                self.param_id.get_cost(self.best_param_vals)
        finally:
            if saved_costs is not None:
                self.param_id.obs_info["cost_type"], self.param_id.cost_type = saved_costs

        if self.rank == 0:
            self._save_results()
            self._plot_profiles()
