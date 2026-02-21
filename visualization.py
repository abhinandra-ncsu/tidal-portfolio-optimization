"""
Visualization Module for Portfolio Optimization Results
========================================================

Generates plots to analyze and communicate optimization results.

Two-level narrative:
  1. Single LCOE: deployment map (with shoreline) + cost breakdown
  2. Across LCOE targets: Pareto frontier + correlation evolution + site robustness map
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Any, Optional


def _overlay_shoreline(ax, shoreline_path: Optional[str]):
    """Load and plot shoreline shapefile as a background layer."""
    if shoreline_path is None:
        return
    try:
        import geopandas as gpd

        p = Path(shoreline_path)
        if p.exists():
            shoreline = gpd.read_file(p)
            shoreline.plot(
                ax=ax,
                color="tan",
                edgecolor="black",
                linewidth=1.0,
                alpha=0.6,
                zorder=1,
            )
    except ImportError:
        pass


def plot_lcoe_vs_variance(results: List[Dict], save_path: Optional[str] = None):
    """
    Plot LCOE vs Portfolio Variance (Pareto frontier).

    Shows the fundamental trade-off: lower LCOE targets require
    accepting higher portfolio variance (more correlated sites).

    Args:
        results: List of result dicts
        save_path: Optional path to save figure
    """
    feasible = [r for r in results if r['feasible']]

    if not feasible:
        print("No feasible results to plot.")
        return

    lcoe_targets = [r['lcoe_target'] for r in feasible]
    achieved_lcoe = [r['achieved_lcoe'] for r in feasible]
    variances = [r['variance'] for r in feasible]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient
    scatter = ax.scatter(achieved_lcoe, variances,
                         c=lcoe_targets, cmap='viridis',
                         s=150, edgecolors='black', linewidth=1.5, zorder=3)

    # Connect points to show frontier
    sorted_idx = np.argsort(achieved_lcoe)
    ax.plot([achieved_lcoe[i] for i in sorted_idx],
            [variances[i] for i in sorted_idx],
            'k--', alpha=0.5, linewidth=1.5, zorder=2)

    # Labels
    ax.set_xlabel('Achieved LCOE ($/MWh)', fontsize=12)
    ax.set_ylabel('Portfolio Variance (MW²)', fontsize=12)
    ax.set_title('LCOE vs Portfolio Variance\n(Pareto Frontier)', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='LCOE Target ($/MWh)')

    # Annotations
    for r in feasible:
        ax.annotate(f'${r["lcoe_target"]}',
                    (r['achieved_lcoe'], r['variance']),
                    textcoords="offset points",
                    xytext=(8, 5), fontsize=9, alpha=0.8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_site_map(site_data: Dict, results: List[Dict],
                  lcoe_target: Optional[float] = None,
                  shoreline_path: Optional[str] = None,
                  save_path: Optional[str] = None):
    """
    Plot geographic map of sites with selected sites highlighted.

    Args:
        site_data: Dict with latitudes, longitudes, capacity_factors
        results: List of result dicts
        lcoe_target: Specific LCOE target to show (default: best feasible)
        shoreline_path: Optional path to shoreline shapefile for overlay
        save_path: Optional path to save figure
    """
    lats = site_data['latitudes']
    lons = site_data['longitudes']
    cfs = site_data['capacity_factors']

    # Find the result to display
    feasible = [r for r in results if r['feasible']]
    if not feasible:
        print("No feasible results to plot.")
        return

    if lcoe_target:
        result = next((r for r in feasible if r['lcoe_target'] == lcoe_target), None)
        if not result:
            print(f"No feasible result for LCOE target ${lcoe_target}")
            return
    else:
        # Use best (lowest achieved LCOE)
        result = min(feasible, key=lambda r: r['achieved_lcoe'])

    selected = result['selected_sites']
    collection_point = result['collection_point']

    fig, ax = plt.subplots(figsize=(12, 10))

    # Shoreline background layer
    _overlay_shoreline(ax, shoreline_path)

    # All candidate sites (gray)
    non_selected_mask = np.ones(len(lats), dtype=bool)
    non_selected_mask[selected] = False

    scatter_all = ax.scatter(lons[non_selected_mask], lats[non_selected_mask],
                             c=cfs[non_selected_mask], cmap='Blues',
                             s=40, alpha=0.5, edgecolors='gray', linewidth=0.5,
                             vmin=0, vmax=cfs.max(), zorder=2)

    # Selected sites (colored by capacity factor, larger)
    scatter_selected = ax.scatter(lons[selected], lats[selected],
                                  c=cfs[selected], cmap='Reds',
                                  s=200, edgecolors='black', linewidth=2,
                                  marker='s', zorder=5,
                                  vmin=0, vmax=cfs.max())

    # Collection point (star)
    ax.scatter(lons[collection_point], lats[collection_point],
               c='gold', s=400, marker='*', edgecolors='black',
               linewidth=2, zorder=6, label='Collection Point')

    # Draw lines from selected sites to collection point
    for site_idx in selected:
        ax.plot([lons[site_idx], lons[collection_point]],
                [lats[site_idx], lats[collection_point]],
                'k-', alpha=0.3, linewidth=1, zorder=3)

    # Labels
    ax.set_xlabel('Longitude (\u00b0)', fontsize=12)
    ax.set_ylabel('Latitude (\u00b0)', fontsize=12)
    ax.set_title(f'Selected Sites for LCOE Target ${result["lcoe_target"]}/MWh\n'
                 f'Achieved: ${result["achieved_lcoe"]:.0f}/MWh | Variance: {result["variance"]:.1f} MW\u00b2',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(scatter_all, ax=ax, label='Capacity Factor')
    cbar.ax.set_ylabel('Capacity Factor', fontsize=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='gray', label='Candidate Sites'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Selected Sites'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=20, markeredgecolor='black', label='Collection Point'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_cost_breakdown(results: List[Dict], save_path: Optional[str] = None):
    """
    Plot stacked bar chart of cost breakdown by LCOE target.

    Args:
        results: List of result dicts
        save_path: Optional path to save figure
    """
    feasible = [r for r in results if r['feasible']]

    if not feasible:
        print("No feasible results to plot.")
        return

    # Sort by LCOE target
    feasible = sorted(feasible, key=lambda r: r['lcoe_target'])

    lcoe_targets = [f'${r["lcoe_target"]}' for r in feasible]
    cost_fixed = [r['cost_fixed'] / 1e6 for r in feasible]  # Convert to $M
    cost_inter = [r['cost_inter_array'] / 1e6 for r in feasible]
    cost_trans = [r['cost_transmission'] / 1e6 for r in feasible]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(lcoe_targets))
    width = 0.6

    # Stacked bars
    bars1 = ax.bar(x, cost_fixed, width, label='Device + Intra-Array', color='#2ecc71')
    bars2 = ax.bar(x, cost_inter, width, bottom=cost_fixed,
                   label='Inter-Array Cable', color='#3498db')
    bars3 = ax.bar(x, cost_trans, width,
                   bottom=[f + i for f, i in zip(cost_fixed, cost_inter)],
                   label='Transmission', color='#e74c3c')

    # Labels
    ax.set_xlabel('LCOE Target ($/MWh)', fontsize=12)
    ax.set_ylabel('Annual Cost ($M/year)', fontsize=12)
    ax.set_title('Cost Breakdown by LCOE Target', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lcoe_targets)
    ax.legend(loc='upper right')

    # Add total cost labels on top of bars
    for i, r in enumerate(feasible):
        total = r['cost_total'] / 1e6
        ax.annotate(f'${total:.2f}M', (i, total + 0.1),
                    ha='center', fontsize=9, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_correlation_evolution(site_data: Dict, results: List[Dict],
                               save_path: Optional[str] = None):
    """
    Plot correlation heatmaps of selected sites at each LCOE target.

    Shows how the optimizer trades off site correlation as the LCOE
    constraint relaxes: tighter targets force higher-energy (but more
    correlated) sites, while relaxed targets allow decorrelated portfolios.

    Args:
        site_data: Dict with power_timeseries (n_sites x n_timesteps)
        results: List of result dicts
        save_path: Optional path to save figure
    """
    feasible = sorted(
        [r for r in results if r['feasible']],
        key=lambda r: r['lcoe_target'],
    )

    if not feasible:
        print("No feasible results to plot.")
        return

    power_ts = site_data['power_timeseries']  # (n_sites, n_timesteps)
    n_panels = len(feasible)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                             squeeze=False)
    axes = axes[0]

    for idx, (r, ax) in enumerate(zip(feasible, axes)):
        selected = r['selected_sites']
        n_sel = len(selected)

        # Correlation matrix for selected sites
        ts_subset = power_ts[selected]  # (n_sel, n_timesteps)
        corr = np.corrcoef(ts_subset)   # (n_sel, n_sel)

        im = ax.imshow(corr, vmin=0, vmax=1, cmap='YlOrRd', aspect='equal')

        # Annotate each cell
        for i in range(n_sel):
            for j in range(n_sel):
                ax.text(j, i, f'{corr[i, j]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if corr[i, j] > 0.7 else 'black')

        ax.set_xticks(range(n_sel))
        ax.set_yticks(range(n_sel))
        ax.set_xticklabels([f'S{s}' for s in selected], fontsize=8)
        ax.set_yticklabels([f'S{s}' for s in selected], fontsize=8)
        ax.set_title(f'LCOE ${r["lcoe_target"]}/MWh\n'
                     f'Var = {r["variance"]:.1f} MW\u00b2',
                     fontsize=10, fontweight='bold')

    fig.suptitle('Correlation Evolution Across LCOE Targets',
                 fontsize=14, fontweight='bold', y=1.02)

    # Shared colorbar
    fig.colorbar(im, ax=axes, label='Pearson Correlation',
                 fraction=0.02, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_site_robustness_map(site_data: Dict, results: List[Dict],
                             shoreline_path: Optional[str] = None,
                             save_path: Optional[str] = None):
    """
    Plot geographic robustness map: marker size = selection frequency,
    marker color = capacity factor.

    Sites selected across many LCOE targets are "robust" — the optimizer
    consistently wants them regardless of how tight the cost constraint is.

    Args:
        site_data: Dict with latitudes, longitudes, capacity_factors
        results: List of result dicts
        shoreline_path: Optional path to shoreline shapefile for overlay
        save_path: Optional path to save figure
    """
    feasible = [r for r in results if r['feasible']]

    if not feasible:
        print("No feasible results to plot.")
        return

    lats = site_data['latitudes']
    lons = site_data['longitudes']
    cfs = site_data['capacity_factors']
    n_sites = len(lats)

    # Count selection frequency per site
    freq = np.zeros(n_sites)
    for r in feasible:
        freq[r['selected_sites']] += 1

    max_freq = freq.max()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Shoreline background
    _overlay_shoreline(ax, shoreline_path)

    # Never-selected sites: small, faded
    never_mask = freq == 0
    ax.scatter(lons[never_mask], lats[never_mask],
               c=cfs[never_mask], cmap='viridis',
               s=15, alpha=0.3, edgecolors='none',
               vmin=0, vmax=cfs.max(), zorder=2)

    # Selected sites: size proportional to frequency
    sel_mask = freq > 0
    sel_sizes = 60 + 200 * (freq[sel_mask] / max_freq)  # range ~60–260

    scatter = ax.scatter(lons[sel_mask], lats[sel_mask],
                         c=cfs[sel_mask], cmap='viridis',
                         s=sel_sizes, alpha=0.8,
                         edgecolors='black', linewidth=1,
                         vmin=0, vmax=cfs.max(), zorder=4)

    # Colorbar for capacity factor
    cbar = plt.colorbar(scatter, ax=ax, label='Capacity Factor')
    cbar.ax.set_ylabel('Capacity Factor', fontsize=10)

    # Size legend
    unique_counts = sorted(set(freq[sel_mask].astype(int)))
    legend_sizes = []
    legend_labels = []
    for count in unique_counts:
        sz = 60 + 200 * (count / max_freq)
        legend_sizes.append(
            plt.scatter([], [], s=sz, c='gray', edgecolors='black',
                        linewidth=1, alpha=0.8)
        )
        legend_labels.append(f'{count}x selected')

    # Add "never selected" to legend
    legend_sizes.append(
        plt.scatter([], [], s=15, c='gray', edgecolors='none', alpha=0.3)
    )
    legend_labels.append('Never selected')

    ax.legend(legend_sizes, legend_labels,
              loc='upper left', title='Selection Frequency',
              fontsize=9, title_fontsize=10, scatterpoints=1)

    ax.set_xlabel('Longitude (\u00b0)', fontsize=12)
    ax.set_ylabel('Latitude (\u00b0)', fontsize=12)
    ax.set_title(f'Site Robustness Map\n'
                 f'Selection frequency across {len(feasible)} LCOE targets',
                 fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_capacity_factor_comparison(site_data: Dict, results: List[Dict],
                                    save_path: Optional[str] = None):
    """
    Plot histogram comparing capacity factors of selected vs all sites.

    Args:
        site_data: Dict with capacity_factors
        results: List of result dicts
        save_path: Optional path to save figure
    """
    all_cfs = site_data['capacity_factors']

    feasible = [r for r in results if r['feasible']]
    if not feasible:
        print("No feasible results to plot.")
        return

    # Collect all selected site CFs across all LCOE targets
    selected_cfs_all = []
    for r in feasible:
        selected_cfs_all.extend(all_cfs[r['selected_sites']])

    # Best result
    best = min(feasible, key=lambda r: r['achieved_lcoe'])
    best_cfs = all_cfs[best['selected_sites']]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, max(all_cfs) * 1.1, 25)

    # All sites
    ax.hist(all_cfs, bins=bins, alpha=0.5, label='All Candidate Sites',
            color='gray', edgecolor='black')

    # Selected sites (best result)
    ax.hist(best_cfs, bins=bins, alpha=0.7,
            label=f'Selected Sites (${best["lcoe_target"]} target)',
            color='#e74c3c', edgecolor='black')

    # Vertical lines for means
    ax.axvline(np.mean(all_cfs), color='gray', linestyle='--', linewidth=2,
               label=f'All Sites Mean: {np.mean(all_cfs):.1%}')
    ax.axvline(np.mean(best_cfs), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Selected Mean: {np.mean(best_cfs):.1%}')

    ax.set_xlabel('Capacity Factor', fontsize=12)
    ax.set_ylabel('Number of Sites', fontsize=12)
    ax.set_title('Capacity Factor Distribution: Selected vs All Sites',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_site_selection_frequency(results: List[Dict], n_sites: int,
                                  save_path: Optional[str] = None):
    """
    Plot how often each site is selected across LCOE targets.

    Args:
        results: List of result dicts
        n_sites: Total number of candidate sites
        save_path: Optional path to save figure
    """
    feasible = [r for r in results if r['feasible']]

    if not feasible:
        print("No feasible results to plot.")
        return

    # Count selections
    selection_count = np.zeros(n_sites)
    for r in feasible:
        selection_count[r['selected_sites']] += 1

    # Get sites that were selected at least once
    selected_ever = np.where(selection_count > 0)[0]
    counts = selection_count[selected_ever]

    # Sort by count
    sort_idx = np.argsort(counts)[::-1]
    selected_ever = selected_ever[sort_idx]
    counts = counts[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(selected_ever))
    colors = plt.cm.RdYlGn(counts / counts.max())

    bars = ax.bar(x, counts, color=colors, edgecolor='black')

    ax.set_xlabel('Site Index', fontsize=12)
    ax.set_ylabel('Selection Count', fontsize=12)
    ax.set_title(f'Site Selection Frequency Across {len(feasible)} LCOE Targets',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in selected_ever], rotation=45, ha='right')

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.annotate(f'{int(count)}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_energy_vs_variance(results: List[Dict], save_path: Optional[str] = None):
    """
    Plot Energy vs Variance (efficient frontier).

    Args:
        results: List of result dicts
        save_path: Optional path to save figure
    """
    feasible = [r for r in results if r['feasible']]

    if not feasible:
        print("No feasible results to plot.")
        return

    energies = [r['energy_net'] / 1e3 for r in feasible]  # GWh
    variances = [r['variance'] for r in feasible]
    lcoe_targets = [r['lcoe_target'] for r in feasible]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(variances, energies,
                         c=lcoe_targets, cmap='viridis_r',
                         s=150, edgecolors='black', linewidth=1.5, zorder=3)

    # Connect points
    sorted_idx = np.argsort(variances)
    ax.plot([variances[i] for i in sorted_idx],
            [energies[i] for i in sorted_idx],
            'k--', alpha=0.5, linewidth=1.5, zorder=2)

    ax.set_xlabel('Portfolio Variance (MW\u00b2)', fontsize=12)
    ax.set_ylabel('Net Energy (GWh/year)', fontsize=12)
    ax.set_title('Energy vs Variance (Efficient Frontier)', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, label='LCOE Target ($/MWh)')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_all(site_data: Dict, results: Dict, save_dir: Optional[str] = None,
             shoreline_path: Optional[str] = None):
    """
    Generate all visualization plots in a two-level narrative.

    Level 1 — Per LCOE target: deployment map + cost breakdown
    Level 2 — Across targets: Pareto frontier + correlation evolution + robustness map

    Args:
        site_data: Dict with site information (must include power_timeseries)
        results: Dict from run_portfolio_optimization containing 'results' list
        save_dir: Optional directory to save all figures
        shoreline_path: Optional path to shoreline shapefile for map overlays
    """
    result_list = results['results']
    feasible = sorted(
        [r for r in result_list if r['feasible']],
        key=lambda r: r['lcoe_target'],
    )
    n_feasible = len(feasible)

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

    # Total plot count: n_feasible deployment maps + 4 cross-target plots
    n_total = n_feasible + 4
    step = 0

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    # ── Level 1: Per-LCOE deployment maps ──────────────────────────────
    print("\n── Level 1: Per-LCOE Deployment Maps ──")
    for r in feasible:
        step += 1
        lcoe = r['lcoe_target']
        print(f"\n[{step}/{n_total}] Deployment Map — LCOE ${lcoe}/MWh...")
        save_path = (f"{save_dir}/site_map_lcoe_{lcoe}.png"
                     if save_dir else None)
        plot_site_map(site_data, result_list, lcoe_target=lcoe,
                      shoreline_path=shoreline_path, save_path=save_path)

    # ── Level 2: Cross-target analysis ─────────────────────────────────
    print("\n── Level 2: Cross-Target Analysis ──")

    # Pareto Frontier
    step += 1
    print(f"\n[{step}/{n_total}] LCOE vs Portfolio Variance (Pareto Frontier)...")
    save_path = f"{save_dir}/lcoe_vs_variance.png" if save_dir else None
    plot_lcoe_vs_variance(result_list, save_path)

    # Cost Breakdown
    step += 1
    print(f"\n[{step}/{n_total}] Cost Breakdown...")
    save_path = f"{save_dir}/cost_breakdown.png" if save_dir else None
    plot_cost_breakdown(result_list, save_path)

    # Correlation Evolution
    step += 1
    print(f"\n[{step}/{n_total}] Correlation Evolution...")
    save_path = f"{save_dir}/correlation_evolution.png" if save_dir else None
    plot_correlation_evolution(site_data, result_list, save_path)

    # Site Robustness Map
    step += 1
    print(f"\n[{step}/{n_total}] Site Robustness Map...")
    save_path = f"{save_dir}/site_robustness.png" if save_dir else None
    plot_site_robustness_map(site_data, result_list,
                            shoreline_path=shoreline_path, save_path=save_path)

    print("\n" + "=" * 60)
    print("All plots generated!")
    if save_dir:
        print(f"Figures saved to: {save_dir}/")
    print("=" * 60)
