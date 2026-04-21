"""
Helpers for reading minimal OMEX/COMBINE archives used by analysis pipelines.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import numpy as np

from utilities.utility_funcs import change_parameter_values_and_save


SEDML_NS = {"sedml": "http://sed-ml.org/sed-ml/level1/version4"}
CELLML_TARGET_RE = re.compile(
    r"/cellml:model/cellml:component\[@name='([^']+)'\]/cellml:variable\[@name='([^']+)'\]"
)


def _safe_name(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    return value or "omex_item"


def _extract_unit_from_axis_title(axis_title: str, fallback: str = "dimensionless") -> str:
    match = re.search(r"\(([^)]+)\)", axis_title or "")
    return match.group(1).strip() if match else fallback


def _split_expression(expression: str) -> List[str]:
    return [token.strip() for token in str(expression).split("+") if token.strip()]


def _normalize_series_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", str(text)).lower()


def _constant_std(values: List[float], scale: float = 0.05) -> List[float]:
    array = np.asarray(values, dtype=float)
    std = float(np.std(array) * scale)
    if not np.isfinite(std) or std <= 0.0:
        std = max(float(np.max(np.abs(array))) * 1e-3, 1e-6)
    return [std] * len(values)


@dataclass
class ObservableSpec:
    plot_index: int
    trace_index: int
    name: str
    model_expression: str
    model_operands: List[str]
    external_expression: str
    values: List[float]
    time_values: List[float]
    unit: str
    operation_name: Optional[str] = None


class OMEXArchiveParser:
    """Minimal OMEX parser for archives with a CellML model and JSON output data."""

    def __init__(self, archive_path: str):
        self.archive_path = os.path.abspath(archive_path)
        self.archive_name = os.path.basename(self.archive_path)
        self.archive_stem = os.path.splitext(self.archive_name)[0]

        with zipfile.ZipFile(self.archive_path) as archive:
            self.members = archive.namelist()
            self._simulation_json = json.loads(archive.read(self._find_member("simulation.json")))
            self._sedml_root = ET.fromstring(archive.read(self._find_sedml_member()))

        self._model_lookup = self._build_model_lookup()
        self._external_lookup = self._build_external_lookup()

    def _find_member(self, preferred_name: str) -> str:
        for member in self.members:
            if member.endswith(preferred_name):
                return member
        raise FileNotFoundError(f"{preferred_name} not found in {self.archive_path}")

    def _find_sedml_member(self) -> str:
        for member in self.members:
            if member.lower().endswith(".sedml"):
                return member
        raise FileNotFoundError(f"No SED-ML file found in {self.archive_path}")

    def get_model_member(self) -> str:
        for model in self._sedml_root.findall(".//sedml:model", SEDML_NS):
            source = model.attrib.get("source")
            if source:
                return source
        for member in self.members:
            if member.lower().endswith(".cellml"):
                return member
        raise FileNotFoundError(f"No CellML model found in {self.archive_path}")

    def get_default_file_prefix(self) -> str:
        return _safe_name(self.archive_stem)

    def get_sedml_changes(self) -> List[Dict[str, str]]:
        changes = []
        for change in self._sedml_root.findall(".//sedml:changeAttribute", SEDML_NS):
            target = change.attrib.get("target", "")
            new_value = change.attrib.get("newValue")
            match = CELLML_TARGET_RE.fullmatch(target)
            if not match or new_value is None:
                continue
            component_name, variable_name = match.groups()
            changes.append({
                "component_name": component_name,
                "variable_name": variable_name,
                "qualified_name": f"{component_name}/{variable_name}",
                "new_value": new_value,
            })
        return changes

    def _build_model_lookup(self) -> Dict[str, str]:
        output_data = self._simulation_json.get("output", {}).get("data", [])
        return {
            entry["id"]: entry["name"]
            for entry in output_data
            if isinstance(entry, dict) and "id" in entry and "name" in entry
        }

    def _build_model_alias_lookup(self) -> Dict[str, str]:
        alias_lookup: Dict[str, str] = {}
        output_data = self._simulation_json.get("output", {}).get("data", [])
        for entry in output_data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            model_name = entry.get("name")
            if not model_id or not model_name:
                continue
            alias_lookup[_normalize_series_name(model_id)] = model_name
            alias_lookup[_normalize_series_name(model_name.split("/")[-1])] = model_name
            alias_lookup[_normalize_series_name(model_name)] = model_name
        return alias_lookup

    def _build_external_lookup(self) -> Dict[str, Dict[str, object]]:
        output = self._simulation_json.get("output", {})
        external_lookup: Dict[str, Dict[str, object]] = {}
        for external_group in output.get("externalData", []):
            metadata_entries = external_group.get("data", [])
            series_entries = external_group.get("dataSeries", [])
            name_to_id = {
                entry["name"]: entry["id"]
                for entry in metadata_entries
                if isinstance(entry, dict) and "id" in entry and "name" in entry
            }
            time_values = [float(value) for value in external_group.get("voiValues", [])]
            for series in series_entries:
                series_name = series.get("name")
                series_id = name_to_id.get(series_name)
                if series_id is None:
                    continue
                external_lookup[series_id] = {
                    "id": series_id,
                    "name": series_name,
                    "values": [float(value) for value in series.get("values", [])],
                    "time_values": time_values,
                    "description": external_group.get("description"),
                }
        return external_lookup

    def _build_unit_lookup(self) -> Dict[str, str]:
        unit_lookup: Dict[str, str] = {}
        plots = self._simulation_json.get("output", {}).get("plots", [])
        for plot in plots:
            unit = _extract_unit_from_axis_title(str(plot.get("yAxisTitle", "")))
            for token in _split_expression(plot.get("yValue", "")):
                unit_lookup[_normalize_series_name(token)] = unit
        return unit_lookup

    def _build_all_direct_series_observable_specs(self) -> List[ObservableSpec]:
        alias_lookup = self._build_model_alias_lookup()
        unit_lookup = self._build_unit_lookup()
        observable_specs: List[ObservableSpec] = []
        trace_counts: Dict[str, int] = {}

        for spec_index, external_payload in enumerate(self._external_lookup.values()):
            external_name = str(external_payload["name"])
            model_operand = alias_lookup.get(_normalize_series_name(external_name))
            if model_operand is None:
                continue
            model_leaf_name = model_operand.split("/")[-1]
            unit = (
                unit_lookup.get(_normalize_series_name(model_leaf_name))
                or unit_lookup.get(_normalize_series_name(model_operand))
                or "dimensionless"
            )
            trace_index = trace_counts.get(external_name, 0)
            trace_counts[external_name] = trace_index + 1
            observable_specs.append(
                ObservableSpec(
                    plot_index=spec_index,
                    trace_index=trace_index,
                    name=external_name,
                    model_expression=model_leaf_name,
                    model_operands=[model_operand],
                    external_expression=str(external_payload["id"]),
                    values=list(external_payload["values"]),
                    time_values=list(external_payload["time_values"]),
                    unit=unit,
                    operation_name=None,
                )
            )
        return observable_specs

    def get_direct_series_selection_options(self) -> Dict[str, List[str]]:
        selection_options: Dict[str, List[str]] = {}
        for spec in self._build_all_direct_series_observable_specs():
            selection_options.setdefault(spec.name, []).append(spec.external_expression)
        return {
            name: options
            for name, options in selection_options.items()
            if len(options) > 1
        }

    def describe_archive(self) -> Dict[str, object]:
        output = self._simulation_json.get("output", {})
        return {
            "archive_path": self.archive_path,
            "members": list(self.members),
            "model_member": self.get_model_member(),
            "sedml_changes": self.get_sedml_changes(),
            "inputs": self._simulation_json.get("input", []),
            "parameters": self._simulation_json.get("parameters", []),
            "plots": output.get("plots", []),
            "external_series_ids": sorted(self._external_lookup.keys()),
        }

    def extract_model(self, output_dir: str, file_prefix: Optional[str] = None) -> str:
        file_prefix = file_prefix or self.get_default_file_prefix()
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        raw_model_path = os.path.join(output_dir, f"{file_prefix}_source.cellml")
        final_model_path = os.path.join(output_dir, f"{file_prefix}.cellml")

        with zipfile.ZipFile(self.archive_path) as archive:
            with archive.open(self.get_model_member()) as src, open(raw_model_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        changes = self.get_sedml_changes()
        if changes:
            change_parameter_values_and_save(
                raw_model_path,
                [change["qualified_name"] for change in changes],
                [change["new_value"] for change in changes],
                final_model_path,
            )
        else:
            shutil.copyfile(raw_model_path, final_model_path)

        return final_model_path

    def build_default_params_for_id(self) -> List[Dict[str, object]]:
        input_by_id = {
            entry["id"]: entry
            for entry in self._simulation_json.get("input", [])
            if isinstance(entry, dict) and "id" in entry
        }
        params_for_id: List[Dict[str, object]] = []
        for parameter in self._simulation_json.get("parameters", []):
            if not isinstance(parameter, dict):
                continue
            model_name = parameter.get("name", "")
            input_id = parameter.get("value")
            if "/" not in model_name or input_id not in input_by_id:
                continue
            component_name, variable_name = model_name.split("/", 1)
            input_meta = input_by_id[input_id]
            params_for_id.append({
                "vessel_name": component_name,
                "param_name": variable_name,
                "param_type": "const",
                "min": float(input_meta.get("minimumValue", input_meta.get("defaultValue", 0.0))),
                "max": float(input_meta.get("maximumValue", input_meta.get("defaultValue", 1.0))),
                "name_for_plotting": input_meta.get("name", variable_name),
                "default_value": float(input_meta.get("defaultValue", 0.0)),
            })
        return params_for_id

    def _resolve_model_expression(self, expression: str) -> Dict[str, object]:
        tokens = _split_expression(expression)
        operands = [self._model_lookup[token] for token in tokens if token in self._model_lookup]
        if not operands:
            raise ValueError(f"Could not resolve model expression '{expression}' from OMEX output metadata.")
        operation_name = "omex_series_sum" if len(operands) > 1 else None
        return {"operands": operands, "operation_name": operation_name}

    def _evaluate_external_expression(self, expression: str) -> Dict[str, object]:
        tokens = _split_expression(expression)
        if not tokens:
            raise ValueError(f"Empty external expression '{expression}'")
        missing = [token for token in tokens if token not in self._external_lookup]
        if missing:
            raise ValueError(f"Unknown external data ids in '{expression}': {missing}")
        time_values = self._external_lookup[tokens[0]]["time_values"]
        values = np.zeros(len(time_values), dtype=float)
        for token in tokens:
            values = values + np.asarray(self._external_lookup[token]["values"], dtype=float)
        return {
            "values": values.tolist(),
            "time_values": list(time_values),
        }

    def build_default_observable_specs(self) -> List[ObservableSpec]:
        observable_specs: List[ObservableSpec] = []
        plots = self._simulation_json.get("output", {}).get("plots", [])
        for plot_index, plot in enumerate(plots):
            additional_traces = plot.get("additionalTraces", [])
            if not additional_traces:
                continue
            trace = additional_traces[0]
            external_expression = trace.get("yValue")
            if not external_expression:
                continue
            resolved_model = self._resolve_model_expression(plot.get("yValue", ""))
            external_payload = self._evaluate_external_expression(external_expression)
            observable_specs.append(
                ObservableSpec(
                    plot_index=plot_index,
                    trace_index=0,
                    name=str(plot.get("name", f"plot_{plot_index}")),
                    model_expression=str(plot.get("yValue", "")),
                    model_operands=resolved_model["operands"],
                    external_expression=external_expression,
                    values=external_payload["values"],
                    time_values=external_payload["time_values"],
                    unit=_extract_unit_from_axis_title(str(plot.get("yAxisTitle", ""))),
                    operation_name=resolved_model["operation_name"],
                )
            )
        return observable_specs

    def build_direct_series_observable_specs(
        self,
        series_selection_indices: Optional[Dict[str, int]] = None,
    ) -> List[ObservableSpec]:
        series_selection_indices = series_selection_indices or {}
        grouped_specs: Dict[str, List[ObservableSpec]] = {}
        for spec in self._build_all_direct_series_observable_specs():
            grouped_specs.setdefault(spec.name, []).append(spec)

        observable_specs: List[ObservableSpec] = []
        for name, candidate_specs in grouped_specs.items():
            selected_idx = int(series_selection_indices.get(name, 0))
            if selected_idx < 0 or selected_idx >= len(candidate_specs):
                raise ValueError(
                    f"Requested dataset index {selected_idx} for '{name}', "
                    f"but only {len(candidate_specs)} dataset(s) are available."
                )
            observable_specs.append(candidate_specs[selected_idx])
        return observable_specs

    def build_obs_data_dict(self, observable_specs: Optional[List[ObservableSpec]] = None) -> Dict[str, object]:
        observable_specs = observable_specs or self.build_direct_series_observable_specs()
        if not observable_specs:
            raise ValueError(f"No observable specs found in {self.archive_path}")

        time_end = float(observable_specs[0].time_values[-1]) if observable_specs[0].time_values else 0.0
        if len(observable_specs[0].time_values) >= 2:
            obs_dt = float(observable_specs[0].time_values[1] - observable_specs[0].time_values[0])
        else:
            obs_dt = 1.0

        data_items = []
        prediction_items = []
        for spec in observable_specs:
            display_name = re.sub(r"<[^>]+>", "", spec.name).strip() or spec.external_expression
            data_items.append({
                "variable": display_name,
                "name_for_plotting": display_name,
                "data_type": "series",
                "unit": spec.unit,
                "weight": 1.0,
                "operands": spec.model_operands,
                "operation": spec.operation_name,
                "value": spec.values,
                "std": _constant_std(spec.values),
                "obs_dt": obs_dt,
                "experiment_idx": 0,
                "subexperiment_idx": 0,
            })
            prediction_items.append({
                "variable": spec.model_operands[0],
                "name_for_plotting": display_name,
                "unit": spec.unit,
                "experiment_idx": 0,
            })

        return {
            "protocol_info": {
                "pre_times": [0.0],
                "sim_times": [[time_end]],
                "params_to_change": {},
                "experiment_labels": [self.archive_stem],
            },
            "data_items": data_items,
            "prediction_items": prediction_items,
        }

