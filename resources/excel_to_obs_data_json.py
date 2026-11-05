import pandas as pd
import json
import numpy as np
from scipy import signal as sig

output_json_file_name = "/home/finbar/Documents/data/cardiohance_data/cardiohance_observables.json"

excel_paths = ['/home/finbar/Documents/data/cardiohance_data/CardiacFunction.xlsx']
column_name = 'Cardiohance_029'
mmHg_to_Pa = 133.332
ml_per_s_to_m3_per_s = 1e-6
ml_to_m3 = 1e-6
dt = 0.01
T = 1.0
nSteps = int(T/dt)


variable_info = [[('Pressure_raw', 'aortic_root/u', 'alg', 'J_per_m3', mmHg_to_Pa),
                  ('FlowRate_raw', 'aortic_root/v', 'state', 'm3_per_s', ml_per_s_to_m3_per_s),
                  ('Volume_lv_raw', 'heart/q_lv', 'state', 'm3', ml_to_m3),
                  ('Volume_rv_raw', 'heart/q_rv', 'state', 'm3', ml_to_m3)]]

# (sheet_name, variable_name, state_or_alg, conversion_rate)
column_infos = [[[(column_name, ['mean', 'min', 'max', 'series'])],
                 [(column_name, ['mean', 'min', 'max', 'series'])],
                 [(column_name, ['mean', 'min', 'max', 'series'])],
                 [(column_name, ['mean', 'min', 'max', 'series'])]]]

# (column name, list of operations to compute for that column)
#
# One column can yield several items -- its mean, its min, its max, the series itself -- and
# `data_item_name` is each item's **identity**, which must be unique (#466). So the name carries
# the operation and the shared spelling goes in `trace_name_for_plotting`, which may repeat.
# `operands` names the model variable and is required; `operation` is what reduces it (the
# deprecated spelling was `obs_type`, which also implied the operand).
#
# `state_or_alg` is not in the obs_data schema and never has been -- a file carrying it does not
# parse -- so it is not written.
entry_list = []
for file_idx, excel_path in enumerate(excel_paths):
    for sheet_idx, (sheet_name, variable_name, state_or_alg, unit, conversion) in enumerate(variable_info[file_idx]):
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        for column_info in column_infos[file_idx][sheet_idx]:
            column_name = column_info[0]
            operations_wanted = column_info[1]
            series = df[column_name]
            t_resampled = np.linspace(0, 2*T, 2*nSteps + 1)
            series_resampled = sig.resample(series, nSteps + 1)
            series_rs_2period = np.concatenate([series_resampled, series_resampled[1:]])

            if 'mean' in operations_wanted:
                mean_val = np.mean(series_resampled)
                entry = {'data_item_name': f'mean {variable_name}',
                         'operands': [variable_name],
                         'operation': 'mean',
                         'data_type': 'constant',
                         'trace_name_for_plotting': variable_name,
                         'unit': unit,
                         'weight': 1.0,
                         'value': mean_val*conversion}
                entry_list.append(entry)
            if 'min' in operations_wanted:
                min_val = np.min(series_resampled)
                entry = {'data_item_name': f'min {variable_name}',
                         'operands': [variable_name],
                         'operation': 'min',
                         'data_type': 'constant',
                         'trace_name_for_plotting': variable_name,
                         'unit': unit,
                         'weight': 1.0,
                         'value': min_val*conversion}
                entry_list.append(entry)
            if 'max' in operations_wanted:
                max_val = np.max(series_resampled)
                entry = {'data_item_name': f'max {variable_name}',
                         'operands': [variable_name],
                         'operation': 'max',
                         'data_type': 'constant',
                         'trace_name_for_plotting': variable_name,
                         'unit': unit,
                         'weight': 1.0,
                         'value': max_val*conversion}
                entry_list.append(entry)
            if 'series' in operations_wanted:
                entry = {'data_item_name': f'series {variable_name}',
                         'operands': [variable_name],
                         'operation': None,
                         'data_type': 'series',
                         'trace_name_for_plotting': variable_name,
                         'unit': unit,
                         'weight': 0.0,
                         'series': series_rs_2period*conversion,
                         't': t_resampled}
                entry_list.append(entry)

# Now create json file from entry_dict
json_df = pd.DataFrame(entry_list)
# json_df.to_json(output_json_file_name)
result = json_df.to_json(orient='records')
parsed = json.loads(result)
print(parsed)
with open(output_json_file_name, 'w') as wf:
    json.dump(parsed, wf, indent=2)
