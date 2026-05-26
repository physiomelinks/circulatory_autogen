# Uncertainty Quantification in Circulatory Autogen

Uncertainty Quantification (UQ) assesses the reliability of model parameters and predictions by quantifying how uncertainties in inputs propagate to outputs. This is critical for validating that calibrated parameters are physically meaningful and for understanding prediction confidence intervals.

## MCMC
Markov Chain Monte Carlo (MCMC) is a Bayesian inference method that samples from the posterior distribution of model parameters given observed data. It provides full uncertainty quantification by exploring the parameter space and generating samples that represent the probability distribution of parameters consistent with the data and prior information.

### UQ Options Configuration
Configure MCMC settings in your user_inputs.yaml using the UQ_options block:
```yaml
UQ_options:  
  library: 'pymc'  # 'emcee' or 'zeus' or 'pymc'  
  settings:  
    num_draws: 100  
    num_chains: 8  
    burn_in: 0.0  
    method: 'mcmc'  # or 'smc' if using SMC  
  cost_type: gaussian_MLE  
  cost_convergence: 0.001
```

### Library-Specific Settings
emcee/zeus:
```yaml
UQ_options:  
  library: 'emcee'  
  settings:  
    num_steps: 1000  
    num_walkers: 64  
    burn_in: 0.0  
    method: 'mcmc'
```

pymc:
```yaml
UQ_options:  
  library: 'pymc'  
  settings:  
    num_draws: 1000  
    num_chains: 4  
    burn_in: 0.0  
    num_tune: 500  
    method: 'mcmc'
```

### Entry Explanations

| Entry | Description | Required |
| :--- | :--- | :--- |
| **library** | MCMC backend library: `emcee`, `zeus`, or `pymc` | Yes |
| **settings** | Nested dictionary containing library-specific settings | Yes |
| **num_draws** | Number of posterior samples to draw (pymc only) | Yes (pymc) |
| **num_chains** | Number of independent MCMC chains (pymc only) | Yes (pymc) |
| **num_steps** | Number of MCMC steps to run (emcee/zeus only) | Yes (emcee/zeus) |
| **num_walkers** | Number of walkers/chains (emcee/zeus only) | Yes (emcee/zeus) |
| **burn_in** | Burn-in percentage or steps to discard from start | Optional |
| **num_tune** | Number of tuning steps for sampler adaptation (pymc only) | Optional |
| **method** | Sampling method: `mcmc` or `smc` | Optional |
| **cost_type** | Cost function type (must be MLE variant) | Yes |
| **cost_convergence** | Convergence tolerance threshold | Yes |

## SMC
Sequential Monte Carlo (SMC) is a particle-based sampling method that iteratively refines a set of particles to approximate the posterior distribution. It is particularly useful for multi-modal distributions and when MCMC mixing is poor.

To use SMC, configure:
```yaml
UQ_options:  
  library: 'pymc'  
  settings:  
    num_draws: 1000  
    num_chains: 4  
    burn_in: 0.0  
    method: 'smc'  # SMC sampling method
```

## Generated Outputs
After UQ completion, the following outputs are generated:

1. **Trace Plots**: Walker trajectories over time showing chain convergence and mixing behavior

    - **Interpretation**:
        Well-mixed, overlapping chains indicate good convergence.
        Chains stuck or diverging suggest poor mixing or initialization issues.

2. **Corner Plots**: Visualizations showing:
    - Marginal posterior distributions for each parameter (diagonal)
    - Pairwise correlations between parameters (off-diagonal)
    - Best-fit parameter values shown as vertical/horizontal lines
    - 5th, 50th, and 95th percentile quantiles

3. **Autocorrelation Plots**: hows the autocorrelation of each parameter’s MCMC chain as a function of lag.
    - **Interpretation**: 
        - Fast decay to zero indicates efficient sampling.
        - Slow decay suggests high correlation and poor mixing.

4. **Chain Average Plots**: Plots the running average of each parameter across chains.

5. **Boxplots & Violin Plots for Predictions**: Visualizes the distribution of model predictions (from posterior samples) and compares them to experimental data.

    - **NOTE**: To enable these plots, ensure that you set `yaml plot_predictions: True` in your YAML configuration file.

6. **Posterior Statistics**: Summarizes posterior mean, median, credible intervals, effective sample size (ESS), and R-hat statistics.