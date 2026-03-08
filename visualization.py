"""
Visualization Module for Portfolio Optimization Results
========================================================

Generates plots to analyze and communicate optimization results.

Two-level narrative:
  1. Single LCOE: deployment map (with shoreline) + cost breakdown
  2. Across LCOE targets: Pareto frontier + cost breakdown + correlation evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Any, Optional


def _active_cfs(site_data: Dict):
    """Return tidal capacity factors when present, else total."""
    if 'tidal_capacity_factors' in site_data:
        return site_data['tidal_capacity_factors']
    return site_data['capacity_factors']


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
    achieved_lcoe = [r['lcoe'] for r in feasible]
    variances = [r['variance'] for r in feasible]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient
    scatter = ax.scatter(achieved_lcoe, variances,
                         c=lcoe_targets, cmap='RdYlGn_r',
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
                    (r['lcoe'], r['variance']),
                    textcoords="offset points",
                    xytext=(8, 5), fontsize=9, alpha=0.8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


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
    cfs = _active_cfs(site_data)

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
        result = min(feasible, key=lambda r: r['lcoe'])

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
    if "lcoe_target" in result:
        title = (f'Selected Sites for LCOE Target ${result["lcoe_target"]}/MWh\n'
                 f'Achieved: ${result["lcoe"]:.0f}/MWh | Variance: {result["variance"]:.1f} MW\u00b2')
    else:
        title = (f'Selected Sites — LCOE: ${result["lcoe"]:.0f}/MWh | '
                 f'Variance: {result["variance"]:.1f} MW\u00b2')
    ax.set_title(title, fontsize=14, fontweight='bold')

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

    plt.close(fig)


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
    cost_fixed = [r['cost_breakdown']['fixed'] / 1e6 for r in feasible]  # Convert to $M
    cost_inter = [r['cost_breakdown']['inter_array'] / 1e6 for r in feasible]
    cost_trans = [r['cost_breakdown']['transmission'] / 1e6 for r in feasible]

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
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), ncol=3,
              fontsize=10, frameon=True)

    # Add total cost labels on top of bars
    for i, r in enumerate(feasible):
        total = r['total_cost'] / 1e6
        ax.annotate(f'${total:.2f}M', (i, total + 0.1),
                    ha='center', fontsize=9, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


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

    from math import ceil
    ncols = min(n_panels, 4)
    nrows = ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, (r, ax) in enumerate(zip(feasible, axes_flat)):
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

    # Hide unused subplots
    for idx in range(n_panels, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Correlation Evolution Across LCOE Targets',
                 fontsize=14, fontweight='bold')

    # Shared colorbar placed horizontally below the panels
    cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.025])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                 label='Pearson Correlation')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


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
    cfs = _active_cfs(site_data)
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
               c=cfs[never_mask], cmap='Blues',
               s=15, alpha=0.3, edgecolors='none',
               vmin=0, vmax=cfs.max(), zorder=2)

    # Selected sites: size proportional to frequency
    sel_mask = freq > 0
    sel_sizes = 60 + 200 * (freq[sel_mask] / max_freq)  # range ~60–260

    scatter = ax.scatter(lons[sel_mask], lats[sel_mask],
                         c=cfs[sel_mask], cmap='Blues',
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

    plt.close(fig)


def plot_capacity_factor_comparison(site_data: Dict, results: List[Dict],
                                    save_path: Optional[str] = None):
    """
    Plot histogram comparing capacity factors of selected vs all sites.

    Args:
        site_data: Dict with capacity_factors
        results: List of result dicts
        save_path: Optional path to save figure
    """
    all_cfs = _active_cfs(site_data)

    feasible = [r for r in results if r['feasible']]
    if not feasible:
        print("No feasible results to plot.")
        return

    # Collect all selected site CFs across all LCOE targets
    selected_cfs_all = []
    for r in feasible:
        selected_cfs_all.extend(all_cfs[r['selected_sites']])

    # Best result
    best = min(feasible, key=lambda r: r['lcoe'])
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

    plt.close(fig)


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

    plt.close(fig)


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

    energies = [r['total_energy'] / 1e3 for r in feasible]  # GWh
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

    plt.close(fig)


def plot_all(site_data: Dict, results: Dict, save_dir: Optional[str] = None,
             shoreline_path: Optional[str] = None):
    """
    Generate all visualization plots in a two-level narrative.

    Level 1 — Per LCOE target: deployment map + cost breakdown
    Level 2 — Across targets: Pareto frontier + cost breakdown + correlation evolution

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

    # Total plot count: n_feasible deployment maps + 3 cross-target plots
    n_total = n_feasible + 3
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

    print("\n" + "=" * 60)
    print("All plots generated!")
    if save_dir:
        print(f"Figures saved to: {save_dir}/")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Comparison Plots
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_COLORS = {
    'Min-Variance': '#2ecc71',
    'Min-LCOE': '#3498db',
    'Max-Generation': '#e74c3c',
}
_MODEL_MARKERS = {
    'Min-Variance': 's',
    'Min-LCOE': 'D',
    'Max-Generation': '^',
}


def plot_comparison_site_map(site_data: Dict, model_results: Dict[str, Dict],
                              shoreline_path: Optional[str] = None,
                              save_path: Optional[str] = None):
    """
    Plot geographic map with selected sites from each model overlaid.

    Each model's selections are shown in a distinct color/marker.
    Shared sites get a special highlight ring.

    Args:
        site_data: Dict with latitudes, longitudes, capacity_factors
        model_results: Dict mapping model name to its single result dict,
            e.g. {'Min-Variance': result, 'Min-LCOE': result, ...}
        shoreline_path: Optional path to shoreline shapefile
        save_path: Optional path to save figure
    """
    lats = site_data['latitudes']
    lons = site_data['longitudes']
    cfs = _active_cfs(site_data)

    fig, ax = plt.subplots(figsize=(13, 10))
    _overlay_shoreline(ax, shoreline_path)

    # Background: all candidate sites
    ax.scatter(lons, lats, c=cfs, cmap='Blues', s=25, alpha=0.35,
               edgecolors='gray', linewidth=0.3, vmin=0, vmax=cfs.max(),
               zorder=2, label='_nolegend_')

    # Overlay each model's selections
    offset_lon = [-0.012, 0.0, 0.012]  # slight jitter so overlapping markers are visible
    legend_handles = []

    for k, (name, result) in enumerate(model_results.items()):
        if result is None or not result.get('feasible'):
            continue
        sel = result['selected_sites']
        cp = result['collection_point']

        color = _MODEL_COLORS.get(name, f'C{k}')
        marker = _MODEL_MARKERS.get(name, 'o')

        # Selected sites
        h = ax.scatter(lons[sel] + offset_lon[k % 3], lats[sel],
                        s=180, marker=marker, c=color, edgecolors='black',
                        linewidth=1.5, zorder=5 + k)
        legend_handles.append(h)

        # Collection point
        ax.scatter(lons[cp] + offset_lon[k % 3], lats[cp],
                    s=350, marker='*', c=color, edgecolors='black',
                    linewidth=1.5, zorder=6 + k)

        # Lines from sites to CP
        for si in sel:
            ax.plot([lons[si] + offset_lon[k % 3], lons[cp] + offset_lon[k % 3]],
                    [lats[si], lats[cp]],
                    color=color, alpha=0.35, linewidth=1, zorder=3)

    ax.set_xlabel('Longitude (\u00b0)', fontsize=12)
    ax.set_ylabel('Latitude (\u00b0)', fontsize=12)
    ax.set_title('Model Comparison — Selected Sites', fontsize=14, fontweight='bold')

    ax.legend(legend_handles,
              [n for n, r in model_results.items() if r and r.get('feasible')],
              loc='upper left', fontsize=10, title='Model')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison_metrics(model_results: Dict[str, Dict],
                             save_path: Optional[str] = None):
    """
    Bar chart comparing LCOE, variance, and energy across models.

    Three side-by-side subplots with one bar per model.

    Args:
        model_results: Dict mapping model name to its single result dict
        save_path: Optional path to save figure
    """
    names = []
    lcoes, variances, energies = [], [], []

    for name, r in model_results.items():
        if r is None or not r.get('feasible'):
            continue
        names.append(name)
        lcoes.append(r['lcoe'])
        variances.append(r['variance'])
        energies.append(r['total_energy'] / 1e3)  # GWh

    if not names:
        print("No feasible results to plot.")
        return

    colors = [_MODEL_COLORS.get(n, 'gray') for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, ylabel, title, fmt in zip(
        axes,
        [lcoes, variances, energies],
        ['$/MWh', 'MW\u00b2', 'GWh/year'],
        ['LCOE', 'Portfolio Variance', 'Net Energy'],
        ['${:.0f}', '{:.1f}', '{:.1f}'],
    ):
        bars = ax.bar(x, vals, color=colors, edgecolor='black', width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Model Comparison — Key Metrics', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison_cost_breakdown(model_results: Dict[str, Dict],
                                    save_path: Optional[str] = None):
    """
    Stacked bar chart of cost breakdown for each model.

    Args:
        model_results: Dict mapping model name to its single result dict
        save_path: Optional path to save figure
    """
    names = []
    fixed, inter, trans = [], [], []

    for name, r in model_results.items():
        if r is None or not r.get('feasible'):
            continue
        names.append(name)
        cb = r['cost_breakdown']
        fixed.append(cb['fixed'] / 1e6)
        inter.append(cb['inter_array'] / 1e6)
        trans.append(cb['transmission'] / 1e6)

    if not names:
        print("No feasible results to plot.")
        return

    x = np.arange(len(names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_fixed = ax.bar(x, fixed, width, label='Device + Intra-Array', color='#2ecc71')
    bars_inter = ax.bar(x, inter, width, bottom=fixed, label='Inter-Array Cable', color='#3498db')
    bottoms_trans = [f + i for f, i in zip(fixed, inter)]
    bars_trans = ax.bar(x, trans, width, bottom=bottoms_trans,
                        label='Transmission', color='#e74c3c')

    # Dollar labels inside each stacked segment
    for bars, vals, bases in [
        (bars_fixed, fixed, [0] * len(names)),
        (bars_inter, inter, fixed),
        (bars_trans, trans, bottoms_trans),
    ]:
        for bar, v, base in zip(bars, vals, bases):
            if v > 0.5:  # skip tiny segments
                cy = base + v / 2
                ax.text(bar.get_x() + bar.get_width() / 2, cy,
                        f'${v:.1f}M', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    # Total labels on top
    for i in range(len(names)):
        total = fixed[i] + inter[i] + trans[i]
        ax.annotate(f'${total:.1f}M', (i, total + 0.2),
                    ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Annual Cost ($M/year)', fontsize=12)
    ax.set_title('Cost Breakdown by Model', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(0.5, -0.10), loc='upper center', ncol=3,
              fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison_radar(model_results: Dict[str, Dict],
                          save_path: Optional[str] = None):
    """
    Radar chart comparing normalized metrics across models.

    Metrics are normalized to [0, 1] where 1 = best.  LCOE and variance
    are inverted so outward = better on every axis.

    Args:
        model_results: Dict mapping model name to its single result dict
        save_path: Optional path to save figure
    """
    names = []
    raw = {'lcoe': [], 'variance': [], 'energy': [], 'avg_cf': []}

    for name, r in model_results.items():
        if r is None or not r.get('feasible'):
            continue
        names.append(name)
        raw['lcoe'].append(r['lcoe'])
        raw['variance'].append(r['variance'])
        raw['energy'].append(r['total_energy'] / 1e3)  # GWh

        sites = r.get('selected_sites', [])
        if sites and isinstance(sites[0], dict) and 'capacity_factor' in sites[0]:
            avg_cf = np.mean([s['capacity_factor'] for s in sites])
        else:
            avg_cf = 0.0
        raw['avg_cf'].append(avg_cf)

    if len(names) < 2:
        print("Need at least 2 feasible models for radar chart.")
        return

    # Normalize to [0, 1]; invert LCOE and variance (lower is better)
    metrics = ['LCOE', 'Variance', 'Energy', 'Avg CF']
    invert = [True, True, False, False]
    arrays = [raw['lcoe'], raw['variance'], raw['energy'], raw['avg_cf']]

    normalized = []
    for vals, inv in zip(arrays, invert):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            normalized.append([1.0] * len(vals))
        elif inv:
            normalized.append([(hi - v) / (hi - lo) for v in vals])
        else:
            normalized.append([(v - lo) / (hi - lo) for v in vals])

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, name in enumerate(names):
        values = [normalized[m][i] for m in range(n_metrics)]
        values += values[:1]
        color = _MODEL_COLORS.get(name, f'C{i}')
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8,
                        color='gray')
    ax.set_title('Model Comparison — Radar', fontsize=14, fontweight='bold',
                 pad=20)
    ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center',
              ncol=len(names), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison_pareto_overlay(loaded: List, save_path: Optional[str] = None):
    """
    Pareto frontier from min-variance results with min-LCOE and max-generation
    overlaid as distinct markers.

    Args:
        loaded: List of (display_name, json_dict_or_None) from the comparison
                script.  The min-variance entry has multiple ``results``
                (one per LCOE target); enumeration models have a single result.
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for display_name, data in loaded:
        if data is None:
            continue

        results = data.get('results', [])
        feasible = [r for r in results if r.get('feasible')]
        if not feasible:
            continue

        color = _MODEL_COLORS.get(display_name, 'gray')
        marker = _MODEL_MARKERS.get(display_name, 'o')

        if display_name == 'Min-Variance' and len(feasible) > 1:
            # Frontier curve
            feasible_sorted = sorted(feasible, key=lambda r: r['variance'])
            variances = [r['variance'] for r in feasible_sorted]
            lcoes = [r['lcoe'] for r in feasible_sorted]
            targets = [r.get('lcoe_target', 0) for r in feasible_sorted]

            scatter = ax.scatter(variances, lcoes, c=targets, cmap='RdYlGn_r',
                                 s=80, edgecolors='black', linewidth=1,
                                 zorder=4)
            ax.plot(variances, lcoes, '--', color='gray', linewidth=1.5,
                    alpha=0.6, zorder=3)
            plt.colorbar(scatter, ax=ax, label='LCOE Target ($/MWh)',
                         shrink=0.8)
        else:
            # Single-point overlay (enumeration model)
            best = data.get('best_result', {})
            v = best.get('variance') or feasible[0].get('variance')
            l = best.get('lcoe') or feasible[0].get('lcoe')
            ax.scatter([v], [l], s=200, marker=marker, c=color,
                       edgecolors='black', linewidth=2, zorder=6,
                       label=display_name)
            ax.annotate(display_name, (v, l),
                        textcoords='offset points', xytext=(10, 8),
                        fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Portfolio Variance (MW²)', fontsize=12)
    ax.set_ylabel('LCOE ($/MWh)', fontsize=12)
    ax.set_title('Pareto Frontier with Model Overlay', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_comparison_cf_profile(model_results: Dict[str, Dict],
                               save_path: Optional[str] = None):
    """
    Grouped bar chart of capacity factor per selected site, grouped by model.

    Args:
        model_results: Dict mapping model name to its single result dict.
            Each result must have ``selected_sites`` with ``capacity_factor``.
        save_path: Optional path to save figure
    """
    models = []
    for name, r in model_results.items():
        if r is None or not r.get('feasible'):
            continue
        sites = r.get('selected_sites', [])
        if sites and isinstance(sites[0], dict) and 'capacity_factor' in sites[0]:
            cfs = [s['capacity_factor'] for s in sites]
            models.append((name, cfs))

    if not models:
        print("No CF data available for profile plot.")
        return

    n_models = len(models)
    n_sites = max(len(cfs) for _, cfs in models)
    bar_width = 0.25
    x = np.arange(n_sites)

    fig, ax = plt.subplots(figsize=(10, 6))

    for k, (name, cfs) in enumerate(models):
        padded = cfs + [0] * (n_sites - len(cfs))
        offset = (k - (n_models - 1) / 2) * bar_width
        color = _MODEL_COLORS.get(name, f'C{k}')
        bars = ax.bar(x + offset, [c * 100 for c in padded], bar_width,
                       label=name, color=color, edgecolor='black',
                       linewidth=0.8)

        for bar, c in zip(bars, padded):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f'{c:.1%}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Array {i+1}' for i in range(n_sites)], fontsize=11)
    ax.set_ylabel('Capacity Factor (%)', fontsize=12)
    ax.set_title('Capacity Factor Profile by Model', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)
