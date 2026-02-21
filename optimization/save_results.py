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

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``.
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
        entry = {
            "lcoe_target": r["lcoe_target"],
            "feasible": r["feasible"],
        }
        if r["feasible"]:
            selected = r["selected_sites"]
            entry.update({
                "achieved_lcoe": r["achieved_lcoe"],
                "variance": r["variance"],
                "energy_gross": r["energy_gross"],
                "energy_net": r["energy_net"],
                "cost_total": r["cost_total"],
                "cost_fixed": r["cost_fixed"],
                "cost_inter_array": r["cost_inter_array"],
                "cost_transmission": r["cost_transmission"],
                "transmission_mode": r["transmission_mode"],
                "collection_point": r["collection_point"],
                "selected_sites": selected,
                "solve_time": r["solve_time"],
                "solver_used": r.get("solver_used", "gurobi"),
                "site_details": [
                    {
                        "site_index": int(idx),
                        "latitude": float(site_data["latitudes"][idx]),
                        "longitude": float(site_data["longitudes"][idx]),
                        "capacity_factor": float(site_data["capacity_factors"][idx]),
                        "dist_to_shore_km": float(site_data["dist_to_shore"][idx]),
                        "depth_m": float(site_data["depths_m"][idx]),
                    }
                    for idx in selected
                ],
            })
        target_results.append(entry)

    # Best result
    feasible = [r for r in results["results"] if r["feasible"]]
    best_entry = None
    if feasible:
        best = min(feasible, key=lambda r: r["achieved_lcoe"])
        best_entry = {
            "lcoe_target": best["lcoe_target"],
            "achieved_lcoe": best["achieved_lcoe"],
            "variance": best["variance"],
            "energy_net": best["energy_net"],
            "cost_total": best["cost_total"],
            "selected_sites": best["selected_sites"],
        }

    # Site selection frequency
    n_sites = site_data["n_sites"]
    frequency = [0] * n_sites
    for r in results["results"]:
        if r["feasible"]:
            for idx in r["selected_sites"]:
                frequency[int(idx)] += 1

    data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_npz": str(input_npz),
            "region": region,
            "turbine_name": turbine.get("name"),
        },
        "config": {
            "turbine": turbine,
            "num_arrays": results["num_arrays"],
            "turbines_per_array": results["turbines_per_array"],
            "project_capacity_mw": results["project_capacity_mw"],
            "wake_loss_factor": results["wake_loss_factor"],
            "cluster_radius_km": results["cluster_radius_km"],
            "fcr": results["fcr"],
            "current_mode": results["current_mode"],
            "lcoe_targets": results["lcoe_targets"],
            "n_candidate_sites": n_sites,
        },
        "results": target_results,
        "best_result": best_entry,
        "site_selection_frequency": frequency,
    }

    data = _make_json_serializable(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"      Saved JSON: {output_path}")


def save_optimization_csv(results, output_path):
    """
    Save a flat summary table to CSV (one row per LCOE target).

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``.
    output_path : str or Path
        Destination CSV file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "lcoe_target",
        "feasible",
        "achieved_lcoe",
        "variance",
        "energy_gross",
        "energy_net",
        "cost_total",
        "cost_fixed",
        "cost_inter_array",
        "cost_transmission",
        "transmission_mode",
        "collection_point",
        "selected_sites",
        "solve_time",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results["results"]:
            row = {"lcoe_target": r["lcoe_target"], "feasible": r["feasible"]}
            if r["feasible"]:
                selected = r["selected_sites"]
                if isinstance(selected, np.ndarray):
                    selected = selected.tolist()
                row.update({
                    "achieved_lcoe": f"{r['achieved_lcoe']:.2f}",
                    "variance": f"{r['variance']:.6f}",
                    "energy_gross": f"{r['energy_gross']:.2f}",
                    "energy_net": f"{r['energy_net']:.2f}",
                    "cost_total": f"{r['cost_total']:.2f}",
                    "cost_fixed": f"{r['cost_fixed']:.2f}",
                    "cost_inter_array": f"{r['cost_inter_array']:.2f}",
                    "cost_transmission": f"{r['cost_transmission']:.2f}",
                    "transmission_mode": r["transmission_mode"],
                    "collection_point": r["collection_point"],
                    "selected_sites": selected,
                    "solve_time": f"{r['solve_time']:.3f}",
                })
            writer.writerow(row)

    print(f"      Saved CSV:  {output_path}")


def save_optimization_results(
    results, site_data, turbine, input_npz, output_dir, turbine_name, region=None,
):
    """
    Save both JSON and CSV optimization results.

    Files are written to ``output_dir`` with names derived from
    ``turbine_name``:

    - ``<turbine_name>_optimization_results.json``
    - ``<turbine_name>_optimization_summary.csv``

    Parameters
    ----------
    results : dict
        Output from ``run_portfolio_optimization()``.
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
    """
    output_dir = Path(output_dir)
    prefix = f"{turbine_name}_optimization"

    json_path = output_dir / f"{prefix}_results.json"
    csv_path = output_dir / f"{prefix}_summary.csv"

    save_optimization_json(results, site_data, turbine, input_npz, json_path, region=region)
    save_optimization_csv(results, csv_path)
