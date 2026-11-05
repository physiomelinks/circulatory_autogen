"""obs_data fixtures for tests that run a shipped obs_data with other experiment times."""
import json
import os


def retimed_obs_data(path, pre_time, sim_time, out_dir):
    """A copy of the obs_data at ``path``, with the same file name (output directories are
    named after it) under ``out_dir/obs_data``, whose one experiment is ``pre_time`` then
    ``sim_time``. Returns its path."""
    with open(path, encoding='utf-8-sig') as f:
        doc = json.load(f)
    doc['protocol_info'] = dict(doc['protocol_info'], pre_times=[float(pre_time)],
                                sim_times=[[float(sim_time)]])
    target_dir = os.path.join(out_dir, 'obs_data')
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, os.path.basename(path))
    # every MPI rank writes the same copy: each to its own file, moved into place atomically
    partial = f'{target}.{os.getpid()}'
    with open(partial, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2)
    os.replace(partial, target)
    return target
