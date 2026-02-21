# Tidal Energy Portfolio Optimization

Portfolio optimization framework for tidal energy sites along the North Carolina coast using mean-variance optimization and LCOE constraints.

## Overview

This project analyzes tidal energy potential across multiple sites using ocean current data (HYCOM) and applies portfolio theory to minimize generation variance while meeting levelized cost of energy (LCOE) targets.

## Structure

```
tidal_portfolio/
├── config.py                 # Configuration parameters
├── visualization.py          # Optimization results visualization
├── energy/                   # Energy generation calculations
│   ├── generation.py        # Power generation models
│   ├── covariance.py        # Covariance matrix calculations
│   ├── model.py             # Energy models
│   └── wake_losses.py       # Wake loss calculations
├── optimization/            # Portfolio optimization
│   ├── model.py            # Optimization model setup
│   ├── portfolio.py        # Portfolio optimization logic
│   ├── solver.py           # Solver configurations
│   └── save_results.py     # Results export utilities
├── costs/                  # Cost modeling
│   ├── device.py          # Turbine device costs
│   ├── aggregator.py      # Cost aggregation
│   └── electrical/        # Electrical infrastructure costs
├── site_processing/       # Site data processing
│   ├── loaders.py        # Data loading utilities
│   ├── process_sites.py  # Site processing pipeline
│   ├── turbine.py        # Turbine specifications
│   └── utide_bridge.py   # UTide harmonic analysis bridge
├── lib/                  # External libraries
│   └── utide/           # UTide tidal analysis toolbox
├── data/                # Reference data
│   ├── turbine_specifications.csv
│   ├── cables/         # HVAC/HVDC cable specifications
│   └── regions/        # HYCOM/GEBCO data (not in repo)
└── scripts/            # Main execution scripts
    ├── run_energy_pipeline.py
    ├── run_optimization.py
    ├── run_utide_analysis.m      # MATLAB harmonic analysis
    └── visualize_energy_pipeline.py
```

## Requirements

**Python:** 3.11+

Install Python dependencies:
```bash
pip install -r requirements.txt
```

**MATLAB:** Required for harmonic tidal analysis (`scripts/run_utide_analysis.m`)

## Usage

### 1. Energy Generation Analysis
```bash
python scripts/run_energy_pipeline.py
```

### 2. Portfolio Optimization
```bash
python scripts/run_optimization.py
```

### 3. Visualization
```bash
python scripts/visualize_energy_pipeline.py
```

## Data Requirements

### Required (not in repository - user must provide):
- HYCOM ocean current data (u, v velocity components) → place in `data/regions/`
- GEBCO bathymetry data → place in `data/regions/`

### Included in repository:
- Turbine specifications (`data/turbine_specifications.csv`)
- HVAC/HVDC cable specifications (`data/cables/`)

**Note:** Large oceanographic datasets are excluded from the repository. Download separately and place in `data/regions/`.

## Configuration

Edit `config.py` to adjust:
- Site locations and parameters
- Turbine specifications
- LCOE targets
- Optimization constraints

## Author

Abhinandra  
PhD Student, Operations Research  
NC State University

