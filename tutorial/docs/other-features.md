## Other Useful Scripts

These scripts can be found in `src/scripts/`. Most are examples or utilities and may need adjustments for your specific model.

## Prerequisites

- A configured `user_inputs.yaml` and relevant resource files.
- The OpenCOR Python environment if scripts invoke OpenCOR.

### generate_modules_files.py

Converts a generic CellML file into the modular format used by Circulatory_Autogen.

- Currently works for models made from a single module.
- Generates `[file_prefix]_modules.cellml` and `[file_prefix]_module_config.json` in `module_config_user/`.

### generate_obs_json.py

Example script for automating creation of an `obs_data.json` file from input data.

- Use this as a template if you do not want to hand-edit JSON.
- Note: it uses legacy fields such as `sample_rate`; update to `obs_dt` for series entries.
- For a more structured helper, see `example_format_obs_data_json_file.py`.

### example_format_obs_data_json_file.py

Example script showing how to build an `obs_data.json` using the helper class in `src/utilities/obs_data_helpers.py`.

### read_and_insert_parameters.py

Updates a `parameters.csv` file with values from a JSON input file.

### convert_0d_to_1d.py

Utility for converting a 0D vessel array into a 1D representation (useful when preparing 1D-0D coupled models).

### run_multiple_param_id.py

Runs multiple parameter identification experiments over a list of observation files.

## Expected outcome

You should be able to locate and run auxiliary scripts for module conversion, data preparation, and batch workflows.