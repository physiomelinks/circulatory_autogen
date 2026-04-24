from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from parsers.PrimitiveParsers import CSVFileParser


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


def load_params_for_id_csv(params_for_id_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a params_for_id.csv into the dict format used in generation_and_calibration.ipynb.

    Returns a list of dicts, each containing at least:
        vessel_name, param_name, min, max, name_for_plotting (if available).
    If a row has multiple vessel names, vessel_name will be a list of strings.
    """
    csv_parser = CSVFileParser()
    df = csv_parser.get_data_as_dataframe_multistrings(str(params_for_id_path))

    params_for_id: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        vessel_name = _normalize_vessel_name(row.get("vessel_name", ""))
        param_name = str(row.get("param_name", "")).strip()

        # Skip empty rows
        if (vessel_name == "" or vessel_name == []) and param_name == "":
            continue

        entry: Dict[str, Any] = {
            "vessel_name": vessel_name,
            "param_name": param_name,
            "min": _to_float(row.get("min", None)),
            "max": _to_float(row.get("max", None)),
        }

        name_for_plotting = str(row.get("name_for_plotting", "")).strip()
        if name_for_plotting != "":
            entry["name_for_plotting"] = name_for_plotting

        param_type = str(row.get("param_type", "")).strip()
        if param_type != "":
            entry["param_type"] = param_type

        params_for_id.append(entry)

    return params_for_id

