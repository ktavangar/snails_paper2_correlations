# Fiducial Run — Test Particle Simulation

## What this run does

Runs mSSA on the **one-armed (m=1) absolute spiral amplitude** from the test particle simulation, using default/fiducial parameters. This serves as the baseline analysis for the test particle case.

## Input data

- **Simulation**: `kiyan-test` (test particle simulation)
- **Data file**: `data/mSSA_channels_Kiyan_test_t40-300/m1_amp_bins_j25_t16.dat`
- **Time range**: t = 40–300 (≈ 2.6 Gyr)
- **Spatial grid**: 25 J_phi bins × 16 theta_phi bins = 400 channels

## mSSA parameters

| Parameter | Value |
|---|---|
| Channel | `snails_m1_amp` (one-armed absolute amplitude) |
| Window size | 1/2 total time series length |
| Number of PCs | 50 |

## Outputs (`figures/`)

| File | Description |
|---|---|
| `eigenvalues.pdf` | PC eigenvalue spectrum |
| `F_matrix.pdf` | PC contribution to each channel |
| `G_matrix.pdf` | Channel contribution to each PC |
| `wCorr.pdf` | W-correlation matrix (PC grouping guide) |
| `pc_time_series.pdf` | First 20 PC time series (normalized, offset) |
| `pc_grouping_summary.pdf` | Eigenvalues + wCorr + power spectra with PC group annotations |
| `face-on_plots/data.mp4` | Animation of raw m1 amplitude across the disk over time |
| `face-on_plots/pc*.mp4` | PC reconstruction animations (multiple groupings) |

## PC groupings in summary plot

Based on inspection of the eigenvalue spectrum and W-correlation matrix:
- **Group 1 (PCs 0–1, blue)**: Mean signal / slowly-varying background
- **Group 2 (PCs 2–17, red)**: Macro-spiral (the phase spiral signal of interest)
- **Group 3 (PCs 17+, green)**: Noise

These groupings should be revisited after inspecting the output figures.

## Notes

- vmin/vmax for all face-on animations are computed automatically from the data (1st/99th percentile) rather than hardcoded.
- The `linthresh` for SymLogNorm (mean-subtracted movies) is set to the 10th percentile of non-zero absolute values in each reconstruction.
