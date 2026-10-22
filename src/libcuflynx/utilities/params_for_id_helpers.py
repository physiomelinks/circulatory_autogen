from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union



def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, list):
        if len(value) == 0:
            return None
        # Fall through to string conversion using first entry
        value = value[0]
    value_str = str(value).strip()
    if value_str == "":
        return None
    return float(value_str)


def _normalize_vessel_name(value: Any) -> Union[str, List[str]]:
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return value
    value_str = str(value).strip()
    if value_str == "":
        return ""
    parts = [part.strip() for part in value_str.split()]
    if len(parts) == 1:
        return parts[0]
    return parts


def load_params_for_id(params_for_id_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a params_for_id JSON file into the dict format used in generation_and_calibration.ipynb.

    Returns a list of dicts, each containing at least:
        vessel_name, param_name, min, max, name_for_plotting (if available).
    An entry with several targets gives vessel_name as a list of strings; its targets must share
    one variable name. A modifier entry has no such form and is refused.
    """
    from libcuflynx.utilities.params_for_id_file import read_params_for_id
    from libcuflynx.utilities.parameter_names import output_model_name

    params_for_id: List[Dict[str, Any]] = []
    for idx, raw in enumerate(read_params_for_id(params_for_id_path).get("params") or []):
        if raw.get("modifies"):
            raise ValueError(f"{params_for_id_path}: entry {idx} is a modifier, which the "
                             f"vessel_name/param_name form cannot hold.")
        targets = raw.get("targets")
        targets = [targets] if isinstance(targets, str) else list(targets or [])
        pairs = [output_model_name(t).rsplit("/", 1) for t in targets]
        names = {p for _, p in pairs}
        if len(names) != 1:
            raise ValueError(f"{params_for_id_path}: entry {idx} has targets of different "
                             f"variables {sorted(names)}, which the vessel_name/param_name "
                             f"form cannot hold.")
        entry: Dict[str, Any] = {
            "vessel_name": _normalize_vessel_name([v for v, _ in pairs]),
            "param_name": names.pop(),
            "min": _to_float(raw.get("min")),
            "max": _to_float(raw.get("max")),
        }
        for key in ("name_for_plotting", "param_type"):
            if raw.get(key):
                entry[key] = str(raw[key])
        params_for_id.append(entry)

    return params_for_id
