# Identifiability Analysis

Identifiability Analysis (IA) ensures that identified parameters can be trusted, i.e. have a small uncertainty.

## Prerequisites

- A completed parameter identification run (best-fit parameters computed).
- `param_id_output` directory available for the model and dataset.

## Prerequisites  
  
- A completed parameter identification run (best-fit parameters computed).  
- `param_id_output` directory available for the model and dataset.  

## Available Methods  

### The Laplace Approximation

The Laplace approximation makes an approximation of your parameter posterior distribution, assuming it is gaussian. This uses the Hessian of the log-likelihood with respect to the parameters. 

### Profile Likelihood Analysis  
  
Profile likelihood analysis assesses parameter identifiability by systematically varying each parameter around its best-fit value while re-optimizing all other parameters. This method:  
  
1. Fixes one parameter at values around its best fit  
2. Re-optimizes all other parameters for each fixed value  
3. Records the minimum cost at each fixed parameter value  
4. Generates cost vs parameter value plots for each parameter  
  
The resulting profiles reveal:  
- **Identifiable parameters**: Show clear minima with steep cost increases  
- **Non-identifiable parameters**: Flat profiles indicating multiple parameter combinations produce similar costs  

## IA in Circulatory_Autogen

IA is run following a parameter identification run.

### Configuration for `user_inputs.yaml`

To run IA, you need to set 
```
do_ia: True
```

and add a specific `ia_options` block to your `user_inputs.yaml` configuration file:
#### For laplace approximation
```
ia_options: 
    method: 'Laplace' 
```

#### For profile likelihood:
```
ia_options:  
  method: 'profile_likelihood'  
  num_points: 50              # Number of points in parameter sweep  
  range_factor: 0.2           # Range around best fit (fraction of parameter range)  
```

## Running identifiability analysis

You can run IA as part of parameter identification by setting `do_ia: True` and running:

```
./run_param_id.sh <NUM_CORES>
```

Or run it separately after parameter identification completes:

```
./run_identifiability_analysis.sh
```

## Expected outcome

Laplace approximation results are saved in your `param_id_output` directory alongside parameter identification outputs.

## Troubleshooting

- If IA fails with missing files, confirm that parameter identification finished successfully and produced `best_param_vals.npy` and related outputs.

