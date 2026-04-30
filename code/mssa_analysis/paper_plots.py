"""
Generate some specific plots for the more complicated paper figures
"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

import matplotlib as mpl

from mssa_viz import *

tstep = 100
data_file = "/Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/mSSA_channels_B2/m1_amp_bins_j30_t16.dat"
data_ = np.loadtxt(data_file)
data = data_[tstep, 1:]


if np.all(data > 0):
    norm = mpl.colors.LogNorm(vmin=np.min(data),
                                vmax=np.max(data))
    cmap = cmr.sunburst
else:
    vmax = np.max(np.abs(data))
    norm = mpl.colors.SymLogNorm(linthresh=vmax / 1e2, vmin=-vmax, vmax=vmax)
    cmap = cmr.holly

RMS = RewindMacroSpiral(data, None, 1000, 31, 'B2', 'one-armed amplitude', m=1)
data_reshaped = np.reshape(data, RMS.O.shape, 'F')

fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
pcm = ax.pcolormesh(RMS.O, RMS.T2, data_reshaped, cmap=cmap, norm=norm, alpha=1., antialiased=False)
pcm.set_edgecolor('lightgray')
pcm.set_linewidth(0.5)
print(pcm._coordinates[0,15,0])
print(RMS.O[0,15], RMS.O[0,16])
ax.axvline(pcm._coordinates[0,15,0], color='deepskyblue', lw=2)
ax.axvline(pcm._coordinates[0,16,0], color='deepskyblue', lw=2)
ax.plot(pcm._coordinates[0,15:17,0], [0,0], color='deepskyblue', lw=5)
ax.plot(pcm._coordinates[0,15:17,0], [2*np.pi,2*np.pi], color='deepskyblue', lw=5)
ax.set_xlabel(r'$\Omega_\phi$',  fontsize=14)
ax.set_ylabel(r'$\theta_{\phi}$', fontsize=14)
ax.set_title(r'Example simulation $C_1$ values in each $\theta_{\phi} - \Omega_{\phi}$ bin',
                fontsize=16)

fig.colorbar(pcm, ax=ax, label=r'$C_1$')
ax.legend()
plt.savefig('/Users/Tavangar/Work/EXP_Projects/paper2_correlations/figures/paper_schematics/data_macro_fitting_example.pdf', dpi=300)

# --- Fit a pure Fourier mode of order m to the selected J_phi bin.
#     This mirrors RewindMacroSpiral._fit_sine_maximum but keeps all three
#     coefficients so we can evaluate (and plot) the best-fit curve, not
#     just the angle of its maximum.
bin_idx    = 15
m          = RMS.m
theta_bin  = RMS.T2[:, bin_idx]
data_bin   = data_reshaped[:, bin_idx]

X = np.column_stack([np.ones(len(theta_bin)),
                     np.cos(m * theta_bin),
                     np.sin(m * theta_bin)])
c, _, _, _ = np.linalg.lstsq(X, data_bin, rcond=None)

theta_dense = np.linspace(0, 2 * np.pi, 400)
sine_fit    = (c[0]
               + c[1] * np.cos(m * theta_dense)
               + c[2] * np.sin(m * theta_dense))

phi_max = (np.arctan2(c[2], c[1]) / m) % (2 * np.pi / m)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
ax.bar(theta_bin, data_bin, width=np.pi/8,
       facecolor='white', edgecolor='deepskyblue',
       label=r'data in $\Omega_\phi$ bin', lw=3)
ax.plot(theta_dense, sine_fit, color='k', lw=2,
        label=rf'best-fit $m={m}$ sine')
ax.axvline(phi_max, color='k', ls='--', lw=1.5,
           label=r'$\theta_{\max}$')
ax.set_xlabel(r'$\theta_{\phi}$ [rad]', fontsize=14)
ax.set_ylabel(r'$C_1$ [(kpc km/s)$^{-1}$]', fontsize=14)
ax.set_xlim(0, 2 * np.pi)
ax.legend(fontsize=14, loc='best')
ax.set_title(r'Example sine fit to one $\Omega_\phi$ bin', fontsize=16)
plt.savefig('/Users/Tavangar/Work/EXP_Projects/paper2_correlations/figures/paper_schematics/sine_fit_one_bin.pdf', dpi=300)


threshold = np.pi/2
# tfit = RMS.derive_winding_time(tstep, threshold=threshold)

nbins = data_reshaped.shape[1]

# Choose the azimuthal order (m=1 or m=2) that wins more J_phi bins by RSS
m1_wins = sum(
    RMS._fit_sine_maximum(data_reshaped[:, j], 1)[1] <=
    RMS._fit_sine_maximum(data_reshaped[:, j], 2)[1]
    for j in range(nbins)
)
RMS.best_m = 1 if m1_wins >= nbins / 2 else 2

# Fit all bins with the winning mode.
# _fit_sine_maximum returns phi_max in (-π/m, π/m].  Wrap to [0, 2π/m)
# before np.unwrap so the unwrapping works on a monotone sequence, then
# scale back to recover the original (unwrapped) angle range.
max_angles_ = np.array([
    RMS._fit_sine_maximum(data_reshaped[:, j], RMS.best_m)[0]
    for j in range(nbins)
])
max_angles_        = max_angles_ % (2 * np.pi / RMS.best_m)
RMS.max_angles    = (1 / RMS.best_m) * np.unwrap(RMS.best_m * max_angles_)

ind1, ind2             = RMS.find_fitting_interval(threshold=threshold)       
RMS.jphi_c_fit       = RMS.jphi_c[ind1:ind2]
RMS.omega_phi_fit     = RMS.omega_phi[ind1:ind2]
max_angles_fit         = RMS.max_angles[ind1:ind2]

RMS.macro_spl        = make_splrep(RMS.jphi_c_fit, max_angles_fit,
                                        w=None, k=3, s=len(RMS.jphi_c_fit))
RMS.macro_line_coeff = np.polyfit(RMS.omega_phi_fit, max_angles_fit, 1)
RMS.macro_spl_omega  = np.poly1d(RMS.macro_line_coeff)


omega_fit = np.linspace(RMS.omega_phi_fit[0], RMS.omega_phi_fit[-1], 100)
theta_fit = RMS.macro_spl_omega(omega_fit)

omega1, omega2 = RMS.omega_phi_fit[0], RMS.omega_phi_fit[-1]
theta1, theta2 = RMS.macro_spl_omega(omega1), RMS.macro_spl_omega(omega2)
tfit = (theta2 - theta1) / (omega2 - omega1)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
ax.pcolormesh(RMS.O, RMS.T2, data_reshaped, cmap=cmap, norm=norm, alpha=0.3)
ax.plot(RMS.omega_phi, RMS.max_angles % (2 * np.pi), 'o', c='k', ms=5,
        label='Max Amplitude Angles')
ax.plot(omega_fit, theta_fit % (2 * np.pi), 'b-', label='Linear Fit')
ax.axvline(RMS.omega_phi_fit[0],  color='k', linestyle='--', label='Fitting bounds')
ax.axvline(RMS.omega_phi_fit[-1], color='k', linestyle='--')
ax.set_xlabel(r'$\Omega_\phi$',  fontsize=14)
ax.set_ylabel(r'$\theta_{max}$', fontsize=14)
ax.legend()
ax.set_title(f'Tfit = {tfit:.2f}', fontsize=16)
plt.savefig('/Users/Tavangar/Work/EXP_Projects/paper2_correlations/figures/paper_schematics/winding_fit_example.pdf', dpi=300)

