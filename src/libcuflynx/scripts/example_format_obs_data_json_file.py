"""Worked example of building an ``obs_data.json`` in Python instead of by hand.

This file is meant to be **copied** and edited for your own calibration task -- it ships inside
the package (``libcuflynx.scripts``), so edits made in place are lost the next time the package
is upgraded. Copy it somewhere of your own::

    python -c "import libcuflynx.scripts.example_format_obs_data_json_file as m; print(m.__file__)"

then change the data loading, the protocol and the data items. Everything it uses is imported
from the ``libcuflynx`` namespace, so no directory has to be added to the import path; the
example CSV it reads is package data, located with ``importlib.resources``, not a repo-relative
path.

Run as shipped with::

    python -m libcuflynx.scripts.example_format_obs_data_json_file
"""
import numpy as np
import pandas as pd
import yaml
import json
import os, sys
from libcuflynx.utilities.paths import default_resources_dir
from libcuflynx.utilities.obs_data_helpers import ObsDataCreator
from libcuflynx.utilities.package_resources import package_data_file

def example_format_obs_data_json_file(output_path=None):
    """
    Example function to demonstrate how to use the ObsDataCreator class
    to create a JSON file for observation data.
    This function creates a JSON file with protocol information, prediction items,
    and data items based on example data.
    """
    # here is an exaple of using the above class with an example data file

    # Load the data that you want to use as ground truth
    # change this to the path of your data file
    # Shipped example data, so a package resource rather than a repo-relative path.
    data_file = package_data_file('libcuflynx.scripts', 'example_data',
                                  'example_data_for_conversion.csv')

    # output path for the JSON file
    # change this to the desired output path
    if output_path is None:
        output_path = os.path.join(default_resources_dir(), 'NKE_pump_obs_data.json')
    with data_file.open('r') as rf:
        data = pd.read_csv(rf)

    # access the data in the way you want it
    time = data['environment | t (second)'].values

    # create obs_data_creator instance
    obs_data_creator = ObsDataCreator()

    # first create protocol_info to define subexperiments and parameter changes for those subexperiments
    pre_times = [1]
    sim_times = [[100, float(time[-1])]]
    params_to_change = {}
    params_to_change['NKE_pump/flag_0'] = [[0.0, 1.0]]  # example parameter change
    experiment_labels = ['exp_0']  # example labels

    obs_data_creator.add_protocol_info(pre_times, sim_times, params_to_change, 
                                       experiment_labels=experiment_labels)

    # now create the prediction items
    obs_data_creator.add_prediction_item('NKE_pump/v_2', 'fmol_per_s', 0)
    obs_data_creator.add_prediction_item('NKE_pump/u_e', 'fmol_per_s', 0)  

    ## Now fill the data items with the data you want to use for the parameter estimation
    # get data vecs from loaded data
    dt = time[1] - time[0]
    V_R1 = data['environment | v_R1 (fmol_per_sec)'].values

    # and make an entry
    entry = {}
    entry['variable'] = 'NKE_pump/v_1'
    entry['name_for_plotting'] = 'v1_{01}'
    entry['data_type'] = 'series'
    entry['operation'] = None
    entry['operands'] = ['NKE_pump/v_1']
    entry['unit'] = 'fmol_per_s'
    entry['weight'] = 1.0
    entry['value'] = V_R1.tolist()
    entry['std'] = (0.1*V_R1).tolist()
    entry['obs_dt'] = dt
    entry['experiment_idx'] = 0
    entry['subexperiment_idx'] = 1
    obs_data_creator.add_data_item(entry)

    # a made up entry to test different dt
    dt = 0.5

    entry = {}
    entry['variable'] = 'NKE_pump/v_1'
    entry['name_for_plotting'] = 'v1_{00}'
    entry['data_type'] = 'series'
    entry['operation'] = None
    entry['operands'] = ['NKE_pump/v_1']
    entry['unit'] = 'fmol_per_s'
    entry['weight'] = 0.01

    data_vec = 13*np.ones((int(100/dt)))

    entry['value'] = data_vec.tolist()
    entry['std'] = (data_vec/10).tolist()
    entry['obs_dt'] = dt
    entry['experiment_idx'] = 0
    entry['subexperiment_idx'] = 0
    obs_data_creator.add_data_item(entry)

    # add more entries to data_items if you have more data to add

    obs_data_dict = obs_data_creator.get_obs_data_dict()
    obs_data_creator.dump_to_path(output_path)

if __name__ == "__main__":
    example_format_obs_data_json_file()
    print(f"Observation data JSON file created at: {os.path.join(default_resources_dir(), 'NKE_pump_obs_data.json')}")
    # This script demonstrates how to create a JSON file for observation data
