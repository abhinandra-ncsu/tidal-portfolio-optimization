#!/usr/bin/env python3
"""
Model Comparison: Min-Variance vs Min-LCOE vs Max-Generation
=============================================================

Loads saved results from each optimization model and prints a side-by-side
comparison table.  Run each model's script first to generate the JSON files:

    python scripts/run_optimization.py            # min_variance
    python scripts/run_lcoe_optimization.py       # min_lcoe
    python scripts/run_generation_optimization.py  # max_generation

Usage:
    python scripts/run_model_comparison.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import load_optimization_results
from tidal_portfolio.config import get_region_paths

from config import REGION, TURBINE_NAME, NUM_ARRAYS

# =============================================================================
# CONFIGURATION
# =============================================================================

_region = get_region_paths(REGION)
RESULTS_DIR = Path(_region["output_dir"]) / "optimization"

MODEL_NAMES = [
    ("Min-Variance", "min_variance"),
    ("Min-LCOE", "min_lcoe"),
    ("Max-Generation", "max_generation"),
]


# =============================================================================
# COMPARISON OUTPUT
# =============================================================================

def print_model_comparison(loaded, num_arrays):
    """Print side-by-side comparison from loaded JSON dicts.

    Parameters
    ----------
    loaded : list of (display_name, json_dict_or_None)
    num_arrays : int
    """
    # Extract best_result from each
    models = []
    for display_name, data in loaded:
        if data is None:
            models.append((display_name, None))
        else:
            best = data.get("best_result")
            if best is None:
                models.append((display_name, None))
            else:
                # Merge in fields from the single result for enumeration models
                result_entry = data["results"][0] if data["results"] else {}
                merged = {**result_entry, **best, "feasible": True}
                models.append((display_name, merged))

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25}", end="")
    for name, _ in models:
        print(f"{name:>18}", end="")
    print()
    print("-" * 79)

    # Feasibility
    print(f"{'Feasible':<25}", end="")
    for _, r in models:
        val = "Yes" if r and r.get("feasible") else "No"
        print(f"{val:>18}", end="")
    print()

    # Metrics
    metrics = [
        ("LCOE ($/MWh)", "lcoe", "${:.0f}"),
        ("Variance (MW^2)", "variance", "{:.2f}"),
        ("Total Energy (MWh/yr)", "total_energy", "{:,.0f}"),
        ("Total Cost ($/yr)", "total_cost", "${:,.0f}"),
        ("Transmission", "transmission_mode", "{}"),
    ]

    for label, key, fmt in metrics:
        print(f"{label:<25}", end="")
        for _, r in models:
            if r and r.get("feasible") and key in r:
                val = fmt.format(r[key])
                print(f"{val:>18}", end="")
            else:
                print(f"{'--':>18}", end="")
        print()

    # Cost breakdown
    print(f"\n{'Cost Breakdown':<25}")
    cost_keys = [
        ("  Fixed", "fixed"),
        ("  Inter-array", "inter_array"),
        ("  Transmission", "transmission"),
    ]
    for label, ckey in cost_keys:
        print(f"{label:<25}", end="")
        for _, r in models:
            if r and r.get("feasible") and "cost_breakdown" in r:
                val = f"${r['cost_breakdown'][ckey]:,.0f}"
                print(f"{val:>18}", end="")
            else:
                print(f"{'--':>18}", end="")
        print()

    # Selected sites (coordinates from JSON)
    print(f"\n{'Selected Sites':<25}")
    for name, r in models:
        if r and r.get("feasible") and "selected_sites" in r:
            print(f"\n  {name}:")
            cp = r.get("collection_point", {})
            if "latitude" in cp:
                print(f"    CP: ({cp['latitude']:.4f}, {cp['longitude']:.4f})")
            sites = r["selected_sites"]
            for i, site in enumerate(sites):
                if isinstance(site, dict):
                    lat = site["latitude"]
                    lon = site["longitude"]
                    cf = site.get("capacity_factor")
                    cf_str = f" | CF={cf:.1%}" if cf is not None else ""
                    print(f"    Array {i+1}: ({lat:.4f}, {lon:.4f}){cf_str}")

    # Site overlap (using coordinate tuples)
    feasible_models = [(n, r) for n, r in models if r and r.get("feasible")]
    if len(feasible_models) >= 2:
        print(f"\n{'Site Overlap':<25}")
        for i in range(len(feasible_models)):
            for j in range(i + 1, len(feasible_models)):
                n1, r1 = feasible_models[i]
                n2, r2 = feasible_models[j]
                s1 = {(s["latitude"], s["longitude"]) for s in r1["selected_sites"]
                       if isinstance(s, dict)}
                s2 = {(s["latitude"], s["longitude"]) for s in r2["selected_sites"]
                       if isinstance(s, dict)}
                shared = s1 & s2
                print(f"  {n1} \u2229 {n2}: {len(shared)}/{num_arrays} sites shared")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TIDAL TURBINE MODEL COMPARISON")
    print("=" * 80)
    print(f"\nLoading saved results from: {RESULTS_DIR}")

    loaded = []
    for display_name, model_key in MODEL_NAMES:
        try:
            data = load_optimization_results(RESULTS_DIR, TURBINE_NAME, model_key)
            print(f"  Loaded {model_key}")
            loaded.append((display_name, data))
        except FileNotFoundError as e:
            print(f"  Missing {model_key}: {e}")
            loaded.append((display_name, None))

    if all(d is None for _, d in loaded):
        print("\nNo saved results found. Run the individual optimization scripts first.")
        sys.exit(1)

    print_model_comparison(loaded, NUM_ARRAYS)

    return loaded


if __name__ == "__main__":
    loaded = main()

    # -------------------------------------------------------------------------
    # Generate Comparison Plots
    # -------------------------------------------------------------------------
    from tidal_portfolio.visualization import (
        plot_comparison_cost_breakdown,
        plot_comparison_radar,
        plot_comparison_pareto_overlay,
        plot_comparison_cf_profile,
    )

    save_dir = Path(_region["plots_dir"]) / "model_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build model_results dict from loaded JSON best_results
    model_results = {}
    for display_name, data in loaded:
        if data is not None:
            best = data.get("best_result")
            if best is not None:
                result_entry = data["results"][0] if data["results"] else {}
                merged = {**result_entry, **best, "feasible": True}
                model_results[display_name] = merged
            else:
                model_results[display_name] = None
        else:
            model_results[display_name] = None

    print("\n[1/4] Cost breakdown comparison...")
    plot_comparison_cost_breakdown(
        model_results,
        save_path=str(save_dir / "comparison_cost_breakdown.png"),
    )

    print("[2/4] Radar chart...")
    plot_comparison_radar(
        model_results,
        save_path=str(save_dir / "comparison_radar.png"),
    )

    print("[3/4] Pareto overlay...")
    plot_comparison_pareto_overlay(
        loaded,
        save_path=str(save_dir / "comparison_pareto_overlay.png"),
    )

    print("[4/4] Capacity factor profile...")
    plot_comparison_cf_profile(
        model_results,
        save_path=str(save_dir / "comparison_cf_profile.png"),
    )

    print(f"\nPlots saved to: {save_dir}/")
