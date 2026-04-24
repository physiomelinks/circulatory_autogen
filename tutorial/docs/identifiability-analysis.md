# Identifiability Analysis

Identifiability Analysis (IA) ensures that identified parameters can be trusted, i.e. have a small uncertainty.

## Prerequisites

- A completed parameter identification run (best-fit parameters computed).
- `param_id_output` directory available for the model and dataset.

## The Laplace Approximation

The Laplace approximation makes an approximation of your parameter posterior distribution, assuming it is gaussian. This uses the Hessian of the log-likelihood with respect to the parameters. 

## IA in Circulatory_Autogen

IA is run following a parameter identification run.

### Configuration for `user_inputs.yaml`

To run IA, you need to set 
```
do_ia: True
```

and add a specific `ia_options` block to your `user_inputs.yaml` configuration file:
```
ia_options: 
    method: 'Laplace' 
```
Currently, the available options for the `method` are **`'Laplace'`**. 

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

