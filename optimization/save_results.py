"""
Save Optimization Results
=========================

Exports portfolio optimization results to JSON and CSV formats.

- JSON: Full structured results with config, per-target results, and best solution
- CSV: Flat summary table with one row per LCOE target
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_optimization_json(results, site_data, turbine, input_npz, output_path, region=None):
    """
    Save full optimization results to a JSON file.

    Handles both multi-target (min-variance) and single-result (enumeration)
    models.  For enumeration results the ``lcoe_target`` / ``lcoe_targets``
    fields are omitted.

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``, ``run_lcoe_optimization()``,
        or ``run_generation_optimization()``.
    site_data : dict
        Site data dict (latitudes, longitudes, capacity_factors, etc.).
    turbine : dict
        Turbine specification dict from ``load_turbine()``.
    input_npz : str or Path
        Path to the input ``.npz`` file used for this run.
    output_path : str or Path
        Destination JSON file path.
    region : str or None
        Region name (e.g. "North_Carolina").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-target results with site coordinates
    target_results = []
    for r in results["results"]:
        entry = {"feasible": r["feasible"]}
        if "lcoe_target" in r:
            entry["lcoe_target"] = r["lcoe_target"]
        if r["feasible"]:
            selected = r["selected_sites"]
            cp = r["collection_point"]
            entry.update({
                "lcoe": r["lcoe"],
                "variance": r["variance"],
                "total_energy": r["total_energy"],
                "total_cost": r["total_cost"],
                "cost_breakdown": r["cost_breakdown"],
                "transmission_mode": r["transmission_mode"],
                "collection_point": {
                    "latitude": float(site_data["latitudes"][cp]),
                    "longitude": float(site_data["longitudes"][cp]),
                },
                "selected_sites": [
                    {
                        "latitude": float(site_data["latitudes"][idx]),
                        "longitude": float(site_data["longitudes"][idx]),
                        "capacity_factor": float(site_data["capacity_factors"][idx]),
                        "dist_to_shore_km": float(site_data["dist_to_shore_km"][idx]),
                        "depth_m": float(site_data["depths_m"][idx]),
                    }
                    for idx in selected
                ],
            })
            if "solve_time" in r:
                entry["solve_time"] = r["solve_time"]
                entry["solver_used"] = r.get("solver_used", "gurobi")
        target_results.append(entry)

    # Best result — for enumeration models there is exactly one result
    feasible = [r for r in results["results"] if r["feasible"]]
    best_entry = None
    if feasible:
        best = min(feasible, key=lambda r: r["lcoe"])
        best_selected = best["selected_sites"]
        best_cp = best["collection_point"]
        best_entry = {
            "lcoe": best["lcoe"],
            "variance": best["variance"],
            "total_energy": best["total_energy"],
            "total_cost": best["total_cost"],
            "collection_point": {
                "latitude": float(site_data["latitudes"][best_cp]),
                "longitude": float(site_data["longitudes"][best_cp]),
            },
            "selected_sites": [
                {
                    "latitude": float(site_data["latitudes"][idx]),
                    "longitude": float(site_data["longitudes"][idx]),
                }
                for idx in best_selected
            ],
        }
        if "lcoe_target" in best:
            best_entry["lcoe_target"] = best["lcoe_target"]

    # Site selection frequency
    n_sites = site_data["n_sites"]
    frequency = [0] * n_sites
    for r in results["results"]:
        if r["feasible"]:
            for idx in r["selected_sites"]:
                frequency[int(idx)] += 1

    config_data = {
        "turbine": turbine,
        "num_arrays": results["num_arrays"],
        "turbines_per_array": results["turbines_per_array"],
        "project_capacity_mw": results["project_capacity_mw"],
        "wake_loss_factor": results["wake_loss_factor"],
        "cluster_radius_km": results["cluster_radius_km"],
        "fcr": results["fcr"],
        "current_mode": results["current_mode"],
        "n_candidate_sites": n_sites,
    }
    if "lcoe_targets" in results:
        config_data["lcoe_targets"] = results["lcoe_targets"]

    data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_npz": str(input_npz),
            "region": region,
            "turbine_name": turbine.get("name"),
            "model": results.get("model"),
        },
        "config": config_data,
        "results": target_results,
        "best_result": best_entry,
        "site_selection_frequency": frequency,
    }

    data = _make_json_serializable(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"      Saved JSON: {output_path}")


def save_optimization_csv(results, site_data, output_path):
    """
    Save a flat summary table to CSV (one row per LCOE target).

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``.
    site_data : dict
        Site data dict (latitudes, longitudes, etc.).
    output_path : str or Path
        Destination CSV file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_lcoe_target = any("lcoe_target" in r for r in results["results"])

    fieldnames = []
    if has_lcoe_target:
        fieldnames.append("lcoe_target")
    fieldnames += [
        "feasible",
        "lcoe",
        "variance",
        "total_energy",
        "total_cost",
        "cost_fixed",
        "cost_inter_array",
        "cost_transmission",
        "transmission_mode",
        "collection_point_lat",
        "collection_point_lon",
        "selected_site_coords",
    ]
    if any("solve_time" in r for r in results["results"]):
        fieldnames.append("solve_time")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results["results"]:
            row = {"feasible": r["feasible"]}
            if has_lcoe_target and "lcoe_target" in r:
                row["lcoe_target"] = r["lcoe_target"]
            if r["feasible"]:
                selected = r["selected_sites"]
                cp = r["collection_point"]
                coords = [
                    f"({site_data['latitudes'][idx]:.4f}, {site_data['longitudes'][idx]:.4f})"
                    for idx in (selected.tolist() if isinstance(selected, np.ndarray) else selected)
                ]
                row.update({
                    "lcoe": f"{r['lcoe']:.2f}",
                    "variance": f"{r['variance']:.6f}",
                    "total_energy": f"{r['total_energy']:.2f}",
                    "total_cost": f"{r['total_cost']:.2f}",
                    "cost_fixed": f"{r['cost_breakdown']['fixed']:.2f}",
                    "cost_inter_array": f"{r['cost_breakdown']['inter_array']:.2f}",
                    "cost_transmission": f"{r['cost_breakdown']['transmission']:.2f}",
                    "transmission_mode": r["transmission_mode"],
                    "collection_point_lat": f"{site_data['latitudes'][cp]:.4f}",
                    "collection_point_lon": f"{site_data['longitudes'][cp]:.4f}",
                    "selected_site_coords": "; ".join(coords),
                })
                if "solve_time" in r:
                    row["solve_time"] = f"{r['solve_time']:.3f}"
            writer.writerow(row)

    print(f"      Saved CSV:  {output_path}")


def save_optimization_results(
    results, site_data, turbine, input_npz, output_dir, turbine_name,
    region=None, model="min_variance",
):
    """
    Save both JSON and CSV optimization results.

    Files are written to ``output_dir`` with names derived from
    ``turbine_name`` and ``model``:

    - ``<turbine_name>_<model>_results.json``
    - ``<turbine_name>_<model>_summary.csv``

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``,
        ``run_lcoe_optimization()``, or ``run_generation_optimization()``.
    site_data : dict
        Site data dict from ``load_site_results()``.
    turbine : dict
        Turbine specification dict from ``load_turbine()``.
    input_npz : str or Path
        Path to the input ``.npz`` file.
    output_dir : str or Path
        Directory to write output files.
    turbine_name : str
        Turbine model name (used in filenames).
    region : str or None
        Region name (e.g. "North_Carolina").
    model : str
        Model identifier used in filenames (e.g. "min_variance", "min_lcoe",
        "max_generation").
    """
    output_dir = Path(output_dir)
    prefix = f"{turbine_name}_{model}"

    json_path = output_dir / f"{prefix}_results.json"
    csv_path = output_dir / f"{prefix}_summary.csv"

    save_optimization_json(results, site_data, turbine, input_npz, json_path, region=region)
    save_optimization_csv(results, site_data, csv_path)


def load_optimization_results(output_dir, turbine_name, model):
    """
    Load a saved JSON result file.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing saved results.
    turbine_name : str
        Turbine model name used when saving.
    model : str
        Model identifier (e.g. "min_variance", "min_lcoe", "max_generation").

    Returns
    -------
    dict
        Parsed JSON results.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist.
    """
    output_dir = Path(output_dir)
    json_path = output_dir / f"{turbine_name}_{model}_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No saved results at {json_path}")
    with open(json_path) as f:
        return json.load(f)
