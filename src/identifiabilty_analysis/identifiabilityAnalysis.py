'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
from matplotlib.ticker import FuncFormatter
try:
    import corner
except ImportError:
    corner = None
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../solver_wrappers'))
import math as math
try:
    import opencor as oc
    opencor_available = True
except:
    opencor_available = False
    pass
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import paperPlotSetup
from utility_funcs import calculate_hessian
paperPlotSetup.Setup_Plot(3)
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import math
# from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )

class IdentifiabilityAnalysis():
    """Identifiability analysis for a 0D model.

    Quantifies how well calibrated parameters can be identified around a best
    fit. Currently the Laplace approximation is implemented (computing a
    covariance matrix from the Hessian of the cost); profile likelihood is
    planned. Requires an existing inner ``param_id`` object (typically
    ``CVS0DParamID.param_id``); build conveniently with
    [`init_from_dict`][identifiabilty_analysis.identifiabilityAnalysis.IdentifiabilityAnalysis.init_from_dict].

    Args:
        model_path: Path to the generated model file.
        model_type: ``'cellml_only'``, ``'python'`` or ``'casadi_python'``.
        file_name_prefix: Model name prefix (names the saved result files).
        DEBUG: Enable debug behaviour.
        param_id_output_dir: Root output directory.
        resources_dir: Directory holding input resources.
        param_id: The inner param-id engine to analyse (required).

    Attributes:
        mean_Laplace: Mean (best-fit) parameter vector after Laplace approximation.
        covariance_matrix_Laplace: Posterior covariance matrix from the Laplace
            approximation.
    """
    def __init__(self, model_path, model_type, file_name_prefix, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, param_id=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.best_param_vals = None
        self.covariance_matrix_Laplace = None
        self.mean_Laplace = None
        self.param_id = param_id
        if self.param_id is None:
            # TODO intialise the param_id_object here
            raise ValueError("param_id object must be provided to IdentifiabilityAnalysis")
        

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    @classmethod
    def init_from_dict(cls, inp_data_dict, param_id):
        """Build an `IdentifiabilityAnalysis` from a config dict and a param-id object.

        Args:
            inp_data_dict: Configuration dict (e.g. user inputs).
            param_id: The inner param-id engine, e.g. ``CVS0DParamID.param_id``.

        Returns:
            IdentifiabilityAnalysis: A configured instance.
        """
        arg_options = [
            "model_path", "model_type", "DEBUG", "param_id_output_dir", "resources_dir",
        ]
        kwargs = {key: inp_data_dict[key] for key in arg_options if key in inp_data_dict}
        if "file_name_prefix" not in kwargs and "file_prefix" in inp_data_dict:
            kwargs["file_name_prefix"] = inp_data_dict["file_prefix"]
        if "file_name_prefix" not in kwargs:
            kwargs["file_name_prefix"] = "no_name"
        kwargs["param_id"] = param_id
        return cls(**kwargs)

    def set_best_param_vals(self, best_param_vals):
        """Supply the best-fit parameter vector to analyse around.

        Args:
            best_param_vals: Array of best-fit parameter values.
        """
        self.param_id.set_best_param_vals(best_param_vals)
        self.best_param_vals = best_param_vals

    def run(self, ia_options):
        """Run the identifiability analysis using the chosen method.

        Args:
            ia_options: Options dict; ``method`` selects ``'Laplace'`` or
                ``'profile_likelihood'`` (the latter is not yet implemented).
        """
        if ia_options['method'] == 'profile_likelihood':
            self.run_profile_likelihood(ia_options)
        elif ia_options['method'] == 'Laplace':
            if self.rank == 0:
                # currently Laplace is not parallelised, so only run on rank 0
                self.run_laplace_approximation(ia_options)
        return

    def run_profile_likelihood(self, ia_options):
        """Profile-likelihood identifiability analysis (not yet implemented)."""
        # TODO
        print("Profile Likelihood method not yet implemented")
        exit()
        pass

    # Above this condition number the precision (Hessian) matrix cannot be inverted to a
    # trustworthy covariance -- the result would be massively inflated, meaningless uncertainties
    # (issue #293). float64 has ~16 digits, so 1e12 keeps ~4 digits of margin.
    _LAPLACE_MAX_CONDITION = 1e12

    def _fisher_information_matrix(self, gradient_source):
        """Fisher information matrix ``J^T diag(1/std^2) J`` from analytic observable sensitivities.

        ``J[k, j] = d(observable feature k)/d(param j)`` at the best fit, from
        ``OpencorParamID.get_observable_sensitivities`` -- the CasADi jacobian for
        ``casadi_python`` ('AD') or the Myokit CVODES sensitivities for ``cellml_only`` +
        ``CVODE_myokit`` ('FSA'), i.e. the sources ``gradient_sources(model_type, solver)``
        advertises. At the MLE this Gauss-Newton matrix is the positive-definite negative Hessian
        of the Gaussian log-likelihood, so the Laplace covariance is its inverse. Scalar (const)
        observables only -- the same scope as ``get_observable_sensitivities``.
        """
        from parsers.PrimitiveParsers import gradient_sources

        pid = self.param_id
        solver = pid.solver_info.get('solver') if isinstance(pid.solver_info, dict) else None
        available = {s['value'] for s in gradient_sources(pid.model_type, solver)}
        if gradient_source not in available:
            raise ValueError(
                f"Laplace gradient_source '{gradient_source}' is not available for model_type "
                f"'{pid.model_type}' / solver '{solver}'. Available: {sorted(available)}. "
                "(gradient_sources(model_type, solver) lists the analytic sources per model; use "
                "'FD' for finite differences.)")

        sens = pid.get_observable_sensitivities(np.asarray(self.best_param_vals, dtype=float))
        param_names = [n[0] if isinstance(n, (list, tuple)) else n
                       for n in pid.param_id_info['param_names']]
        obs_info = pid.obs_info
        const_to_obs = obs_info['const_idx_to_obs_idx']
        stds = np.asarray(obs_info['std_const_vec'], dtype=float)

        n_par = len(param_names)
        fim = np.zeros((n_par, n_par))
        n_used = 0
        for k, obs_idx in enumerate(const_to_obs):
            row = sens.get(pid._observable_label(obs_idx))
            if row is None or not stds[k] > 0:
                continue
            j = np.array([float(row.get(p, 0.0)) for p in param_names])
            fim += np.outer(j, j) / (stds[k] ** 2)
            n_used += 1
        if n_used == 0:
            raise RuntimeError(
                "Laplace approximation: no scalar observable with a positive std was available "
                "to build the Fisher information matrix.")
        return fim

    def _invert_precision_normalised(self, precision, gradient_source):
        """Invert the precision (Hessian) matrix to a parameter covariance, using symmetric
        diagonal normalisation so that a wide spread of *parameter magnitudes* does not, by itself,
        make the matrix look non-invertible.

        The precision ``P`` is rescaled by ``1/sqrt(|P_ii|)`` on each side (unit-magnitude
        diagonal) before inverting; the covariance is transformed straight back,
        ``C[i,j] = C_norm[i,j]/sqrt(|P_ii| |P_jj|)``. This is a mathematical identity -- ``C``
        equals ``inv(P)`` exactly (in exact arithmetic) -- so it does not change the answer, it only
        removes the *scaling* part of the ill-conditioning. The condition number is then taken on
        the normalised matrix, where it reflects genuine parameter *correlations* rather than mere
        magnitude spread. Models whose parameters merely span a wide magnitude range (e.g.
        3compartment, ~1e-9..1e8) therefore succeed with a real covariance, while genuinely
        collinear (non-identifiable) parameters still trip the condition guard. Issue #293.

        ``abs`` (not ``sqrt(P_ii)``) is used so a *mildly indefinite* finite-difference Hessian --
        a slightly negative curvature from FD/optimiser noise at a rough optimum -- still yields a
        finite covariance (as the plain inverse always did) rather than aborting the whole run; a
        non-positive-definite result is warned about, not fatal. Only a parameter with *zero*
        curvature (literally no information, infinite variance) is a genuine abort.

        Returns ``(covariance_matrix, cond)`` where ``cond`` is the normalised precision's
        condition number. Raises ``RuntimeError`` for a zero/non-finite-curvature parameter or if
        the normalised matrix is still ill-conditioned (jointly non-identifiable parameters).
        """
        precision = np.asarray(precision, dtype=float)
        diag = np.diag(precision)
        # Only a *zero* (or non-finite) curvature is fatal: that parameter carries no information at
        # all, so its variance is genuinely undefined (infinite) and the normalisation would divide
        # by zero. A merely negative curvature is kept -- see the abs() below.
        if not np.all(np.isfinite(diag)) or np.any(diag == 0.0):
            bad = [i for i, d in enumerate(diag) if not (np.isfinite(d) and d != 0.0)]
            raise RuntimeError(
                f"Laplace approximation aborted: the {gradient_source} precision (Hessian) has a "
                f"zero (or non-finite) curvature on the diagonal at parameter index(es) {bad}, so "
                "those parameters carry no curvature (information) at the best fit and their "
                "variance is undefined -- they are not identifiable from the data. Issue #293.")

        scale = 1.0 / np.sqrt(np.abs(diag))  # symmetric normalisation to unit-magnitude diagonal
        outer_scale = np.outer(scale, scale)
        precision_norm = precision * outer_scale

        # After normalisation the condition number reflects genuine parameter correlation, not the
        # parameter magnitude spread. Refuse to invert if it is still ill-conditioned: the
        # covariance would be numerically meaningless (massively inflated uncertainties). Error
        # instead of masking it with a pseudo-inverse (issue #293).
        cond = np.linalg.cond(precision_norm)
        if not np.isfinite(cond) or cond > self._LAPLACE_MAX_CONDITION:
            raise RuntimeError(
                f"Laplace approximation aborted: the {gradient_source} precision (Hessian) matrix "
                f"is ill-conditioned after normalisation (condition number {cond:.3e} > "
                f"{self._LAPLACE_MAX_CONDITION:.0e}), so inverting it to a parameter covariance "
                "would give numerically meaningless, massively inflated uncertainties. Because this "
                "is measured on the magnitude-normalised matrix, it reflects genuinely correlated "
                "(jointly non-identifiable) parameters -- not merely a wide magnitude range, which "
                "the normalisation already handles. Tracked in issue #293.")

        covariance_norm = np.linalg.inv(precision_norm)
        covariance_matrix = covariance_norm * outer_scale  # transform back: C = inv(P), exactly

        # A non-positive-definite Hessian (some negative curvature) yields a covariance with a
        # non-positive variance somewhere. That is a real, finite result (the parameter is poorly
        # constrained / the best fit is not a clean minimum for it), so return it, but flag it
        # rather than pass it off as a trustworthy covariance.
        if np.any(np.diag(covariance_matrix) <= 0):
            neg = [i for i, v in enumerate(np.diag(covariance_matrix)) if v <= 0]
            print(f"warning: the {gradient_source} Laplace covariance has a non-positive variance "
                  f"at parameter index(es) {neg}; the Hessian was not positive definite at the best "
                  "fit (curvature poorly resolved / not a clean minimum), so treat those "
                  "uncertainties with caution.")
        return covariance_matrix, cond

    def run_laplace_approximation(self, ia_options):
        """Run the Laplace approximation around the best fit.

        Forms the precision (negative log-likelihood Hessian) via ``ia_options['gradient_source']``
        -- ``'AD'``/``'FSA'`` build the Fisher information matrix from the analytic observable
        sensitivities, ``'FD'`` (default) uses the finite-difference ``sub_method`` -- and inverts
        it (in Jacobi-normalised space, see ``_invert_precision_normalised``) to the parameter
        covariance, saving ``{prefix}_laplace_mean.npy`` and ``{prefix}_laplace_covariance.npy``.

        The normalisation makes wide-magnitude models (e.g. 3compartment, params ~1e-9..1e8)
        succeed with a real covariance where the raw inverse used to give meaningless, massively
        inflated uncertainties. Raises ``RuntimeError`` only when the *normalised* precision is
        still ill-conditioned (jointly non-identifiable parameters) or a parameter has literally
        zero curvature (no information). A mildly indefinite finite-difference Hessian -- a small
        negative curvature from FD/optimiser noise at a rough optimum -- still yields a finite
        covariance (with a logged warning), as the plain inverse always did (issue #293).

        Args:
            ia_options: Options dict; ``gradient_source`` ('FD'|'AD'|'FSA', default 'FD') and,
                for 'FD', ``sub_method`` (default ``'parabola_fit'``).
        """
        from param_id.differentiable import assert_mle_cost_for_bayesian

        assert_mle_cost_for_bayesian(
            self.param_id.cost_type,
            self.param_id.cost_funcs_dict,
            "Laplace approximation",
        )

        gradient_source = ia_options.get('gradient_source', 'FD')
        if gradient_source in ('AD', 'FSA'):
            precision = self._fisher_information_matrix(gradient_source)
        else:
            # FD: calculate_hessian returns the Hessian of the log-posterior; the precision
            # (negative log-likelihood Hessian) is its negative.
            hessian = calculate_hessian(
                self.param_id, method=ia_options.get('sub_method', 'parabola_fit'))
            precision = -np.asarray(hessian, dtype=float)

        # Invert in Jacobi-normalised space so a wide spread of parameter magnitudes does not, on
        # its own, make the matrix look non-invertible; only genuine parameter correlations do
        # (issue #293).
        covariance_matrix, cond = self._invert_precision_normalised(precision, gradient_source)
        mean = self.best_param_vals
        print("Laplace Approximation Results:")
        print(f"Gradient source: {gradient_source}; normalised precision condition number: "
              f"{cond:.3e}")
        print("Mean (Best Parameter Values):", mean)
        print("Covariance Matrix:\n", covariance_matrix)
        self.covariance_matrix_Laplace = covariance_matrix
        self.mean_Laplace = mean
        parent_dir = os.path.dirname(self.param_id_output_dir)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'), self.mean_Laplace)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'), self.covariance_matrix_Laplace)

    def plot_laplace_results(self, parameter_names, output_dir):
        """
        Plot the results of the Laplace approximation as corner plots.

        Args:
          parameter_names: List of parameter names corresponding to the best_param_vals.
          output_dir: Directory to save the plots.
        """
        if corner is None:
            raise ImportError("corner is required to plot Laplace results.")
          

        if self.covariance_matrix_Laplace is None or self.mean_Laplace is None:
            try:
                parent_dir = os.path.dirname(self.param_id_output_dir)
                self.mean_Laplace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'))
                self.covariance_matrix_Laplace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'))
                print("Loaded Laplace approximation results from files.")
            except Exception as e:
                print("Error loading Laplace approximation results:", e)
                print("Please run the Laplace approximation before plotting.")
                return

        samples = np.random.multivariate_normal(self.mean_Laplace, self.covariance_matrix_Laplace, size=100000)
        print(f'samples shape: {samples.shape}')
        figure = corner.corner(samples, labels=parameter_names, truths=self.mean_Laplace, bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95))
        plot_path = os.path.join(output_dir, f"{self.file_name_prefix}_laplace_corner_plot.pdf")
        
        axes = figure.get_axes()
        num_params = len(parameter_names)
        # for idx, ax in enumerate(axes):
        #     if idx >= num_params*(num_params - 1):

        #         ax.tick_params(axis='both', rotation=0)
        #         formatterx = matplotlib.ticker.ScalarFormatter()
        #         ax.xaxis.set_major_formatter(formatterx)
        #         ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        #     if idx%num_params == 0:

        #         ax.tick_params(axis='both', rotation=0)
        #         formattery = matplotlib.ticker.ScalarFormatter()
        #         ax.yaxis.set_major_formatter(formattery)
        #         ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
        # from matplotlib.ticker import FuncFormatter

        # sci_formatter = FuncFormatter(lambda x, _: f"{x:.2e}")

        # for idx, ax in enumerate(axes):
        #     if idx >= num_params * (num_params - 1):
        #         ax.xaxis.set_major_formatter(sci_formatter)
        #     if idx % num_params == 0:
        #         ax.yaxis.set_major_formatter(sci_formatter)

        


        def make_sci_label_formatter(exponent):
            def formatter(val, pos):
                if val == 0:
                    return "0"
                else:
                    # Divide by 10**exponent and format
                    return f"{val / (10 ** exponent):.2f}"
            return FuncFormatter(formatter)

        # We'll store the exponent per axis
        for idx, ax in enumerate(axes):
            if idx >= num_params * (num_params - 1):  # Bottom row → x-axis
                x_min, x_max = ax.get_xlim()
                if x_max == x_min:
                    continue
                # Use log10 of max abs value to determine exponent
                exponent = int(np.floor(np.log10(np.max(np.abs([x_min, x_max])))))
                
                # Apply formatter that divides by 10^exp
                ax.xaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label just outside the plot
                ax.text(1.0, 0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='bottom', ha='right',
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if idx % num_params == 0:  # Left column → y-axis
                y_min, y_max = ax.get_ylim()
                if y_max == y_min:
                    continue
                exponent = int(np.floor(np.log10(np.max(np.abs([y_min, y_max])))))
                
                ax.yaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label above the y-axis
                ax.text(0, 1.0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='top', ha='left',
                        rotation=0,
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


        plt.subplots_adjust(hspace=0.12, wspace=0.1)
        figure.savefig(plot_path)
        print(f"Laplace approximation corner plot saved to {plot_path}")

