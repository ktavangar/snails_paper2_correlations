# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Activate the local virtual environment before running any Python code:
```bash
source snails-env/bin/activate
```

All scripts are run from the `code/` directory or called from notebooks with paths relative to `notebooks/`. Key dependencies: `numpy`, `scipy`, `astropy`, `gala`, `pandas`, `matplotlib`, `cmasher`, and `pyEXP` (external BFE/mSSA library).

To run the preprocessing pipeline on the HPC cluster (Flatiron/CCA), use the SLURM script:
```bash
sbatch code/make_channels.sh
```
This runs `make_mssa_table.py` with MPI across 10 nodes × 5 tasks.

To run mSSA diagnostics locally:
```bash
cd code && python run_mssa.py
```

Notebooks are opened and run interactively with JupyterLab. There is no test suite.

## Analysis Pipeline

The workflow proceeds in this order:

1. **Load simulation FITS data** → transform to cylindrical + action-angle coordinates (`load_data.py`, `load_data_B2.py`)
2. **Compute BFE coefficients per spatial bin** using `LaguerreSnails` in `df_helpers.py` → extract m=0,1,2 amplitudes, pitch angles, phase angles
3. **Build coefficient tables** using `MSSATable` in `mssa_prep.py` → saves per-timestep FITS files in `data/tables/`
4. **Combine tables** and write `.dat` files (binary arrays of shape `[timesteps × spatial_bins]`) for mSSA input
5. **Run mSSA** via `MSSAOperations` in `run_mssa.py` using `pyEXP.mssa.expMSSA` → eigenvalues, PC time series, W-correlation matrices
6. **Visualize** with diagnostic plots and animations (`helper.py` `MakeAnimations` class)

Notebooks are numbered to reflect this sequence (1–32 for main analysis, 100–108 for combined live/test comparisons).

## Code Architecture

### `code/df_helpers.py` — `LaguerreSnails`
Core scientific class. Given a particle data DataFrame and a rectangular region in `(J_phi, theta_phi)` action-angle space, it:
- Fits a Laguerre-Fourier basis function expansion (BFE) to the vertical phase spiral
- Coefficients `[m, n]` are Fourier mode × Laguerre polynomial order; `m=0` is the axisymmetric background, `m=1` and `m=2` are the spiral modes
- `get_coeffs()` → populates `self.coeffs[m_max, n_max]` (complex)
- `get_pitch_phase_angles(m)` → fits a log-spiral to the peak ridge of mode `m` to extract pitch angle and phase angle; also estimates `time_since_int` (time since last perturbation)
- `make_spiral_residual()` → bins data into `(sqrt(J_z)cos(θ_z), sqrt(J_z)sin(θ_z))` space and subtracts annular background
- The Laguerre scale parameter `a` is fit to the exponential tail of the `J_z` distribution per region

### `code/mssa_prep.py` — `MSSATable`
Orchestrates the BFE computation across all spatial bins for a single timestep. The spatial grid defaults to `J_phi ∈ [1000, 3000]` kpc km/s in 20 bins × `theta_phi ∈ [0, 2π]` in 16 bins = 320 spatial regions per timestep. Produces an astropy `Table` with columns: `timestep, jphi_cen, tphi_cen, m0_amp, m1_amp, m2_amp, pitch_ang_m1/m2, phase_ang_m1/m2, pitch_phase_flag_m1/m2, time_since_int_m1/m2`.

The `pitch_phase_flag` encodes data quality: `0` = good, `1` = marginal (flagged), `2` = failed (pitch/phase set to 0/π).

### `code/run_mssa.py` — `MSSAOperations`
Wraps `pyEXP.mssa.expMSSA`. Takes a `.dat` coefficient file and a channel name, runs mSSA with window size = 50% of time series by default. Key outputs: eigenvalue spectrum (`ev+wcorr.png`), PC time series (`PCs.png`), F-G contribution matrices (`fg_matrices.png`), and optionally face-on animation movies.

The `create_datafile_list` / `gen_figure_directory_names` helpers generate matching lists of `.dat` input files and output figure directories for batch processing across channels (`amp`, `pitch`, `phase`, `int_time`) and modes (`m1`, `m2`).

### `code/load_data.py` / `code/load_data_B2.py`
Simulation data is in FITS format. Coordinates are stored in simulation units (`ro=8 kpc`, `vo=220 km/s`) and converted to physical units on load. Action-angle variables (`jr`, `jphi`, `jz`, `theta_r`, `theta_phi`, `theta_z`, `freq_r`, `freq_phi`, `freq_z`) are stored separately as pickled arrays and merged into a DataFrame by `load_data_actions()`.

## Data Layout

| Path | Contents |
|---|---|
| `data/*.dat` | mSSA input arrays (binary, shape: timesteps × channels) |
| `data/*.fits` | Aggregated coefficient tables (one row per spatial bin per timestep) |
| `data/mssa_channels_B2/` | Pre-split per-channel `.dat` files for B2 simulation |
| `data/mSSA_channels_live_t280-480/` | Live simulation channels for t=280–480 |
| `data/live-sim/` | Raw FITS tables from live simulation (first 600 timesteps) |
| `figures/<sim>_figures/` | Diagnostic plots and animations per simulation |

## Simulations

| Name | Description |
|---|---|
| `B2` | Binary interaction (satellite + disk) |
| `kiyan-live` | Full live N-body disk simulation |
| `kiyan-test` | Test particle simulation (same potential) |
| `kiyan-fast` | Faster variant of kiyan simulation |
| `no_interaction` | Control run without perturbation |

## Units

- Actions (`J_phi`, `J_z`, `J_r`): kpc km/s
- Angles (`theta_phi`, `theta_z`, `theta_r`): radians
- Frequencies (`freq_z` etc.): km/s/kpc (i.e., Gyr⁻¹ up to a factor)
- Simulation raw units: `ro=8 kpc`, `vo=220 km/s` → multiply by `8*220` for actions
