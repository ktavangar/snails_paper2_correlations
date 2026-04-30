"""
mssa_viz.py  —  Refactored diagnostics.py + macro_rewinding.py
================================================================

Key design changes
------------------
PhaseSpaceGrid (base)
    Holds the (J_phi, theta_phi) grid and tstep_diff that were duplicated
    in both MakeAnimations and RewindMacroSpiral.  Also owns the shared
    _format_pc_string static helper.

MakeAnimations(PhaseSpaceGrid)
    Private helpers (_reshape_frame, _style_polar_ax, _style_cartesian_ax,
    _add_colorbar, _run_animation) eliminate repeated per-frame boilerplate.
    Two core templates _make_polar_movie / _make_dual_movie collapse the
    original four make_*_mov / _animate_* method pairs into simple one-liners.

RewindMacroSpiral(PhaseSpaceGrid)
    Inherits the grid; pc_string computed once via the shared helper.
    DATA_DIR is a class variable so it can be overridden without subclassing.
    Bug fix: plt.close() in plot_macro_tfit_over_time is now conditional on
    whether the caller owned the figure.

All diagnostic helpers (suggest_pc_groups, compute_pc_limits, plot_*) are
retained unchanged but collected here so only one import is needed.
"""

import os
import pickle

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d, make_splrep

mpl.rcParams['axes.linewidth'] = 2


# ============================================================
#  Shared base class
# ============================================================

class PhaseSpaceGrid:
    """
    Builds the (J_phi, theta_phi) grid and Gyr time-step conversion shared by
    MakeAnimations and RewindMacroSpiral.
    """

    _TSTEP_DIFF = {'B2': 0.009778, 'live': 0.009778, 'test': 0.01}

    def __init__(self, jphi_min, jbins, sim_name):
        self.sim_name   = sim_name
        self.tstep_diff = self._TSTEP_DIFF[sim_name]

        # J_phi bin centres, evenly spaced by 100 kpc km/s
        self.jphi_c = np.linspace(jphi_min, jphi_min + (jbins - 1) * 100, jbins)

        # theta_phi bin centres: divide [0, 2π) into 16 equal bins
        tphi_edges     = np.linspace(0, 2 * np.pi, 17)          # 16 bins + closing edge
        bin_half_width = 0.5 * (tphi_edges[1] - tphi_edges[0])
        self.tphi_c    = tphi_edges[:-1] + bin_half_width

        # 2-D meshgrids: J[i,j] = jphi_c[j],  T[i,j] = tphi_c[i]
        # In polar plots  → T is the angular coordinate, J is the radial coordinate
        # In Cartesian plots → J is x, T is y
        self.J, self.T = np.meshgrid(self.jphi_c, self.tphi_c)

    @staticmethod
    def _format_pc_string(pcs):
        """Return a compact label for a list of PC indices."""
        if pcs is None:
            return 'data'
        if len(pcs) == 1:
            return f'pc{pcs[0]}'
        if np.any(np.diff(pcs) > 1):           # non-contiguous
            return f'pc{list(pcs)}'
        return f'pc{pcs[0]}-{pcs[-1]}'         # contiguous range


# ============================================================
#  MakeAnimations
# ============================================================

class MakeAnimations(PhaseSpaceGrid):
    """
    Generate face-on (polar) and dual-panel (polar + Cartesian) movies
    from raw coefficient data or mSSA PC reconstructions.
    """

    def __init__(self, mssa, sim_name, channel_name, times, jphi_min, jbins):
        super().__init__(jphi_min, jbins, sim_name)
        self.mssa         = mssa
        self.times        = times
        self.channel_name = channel_name

    # ----------------------------------------------------------
    # PC reconstruction
    # ----------------------------------------------------------

    def reconstruct_from_pcs(self, pcs):
        """
        Run mSSA reconstruction for the given PC list and store the result.
        Populates self.pc_rc (shape: n_channels × n_times), self.pcs, self.pc_string.
        """
        self.pcs       = pcs
        self.pc_string = self._format_pc_string(pcs)
        self.mssa.reconstruct(pcs)
        recon      = self.mssa.getReconstructed()
        self.pc_rc = recon[list(recon.keys())[0]].getAllCoefs()

    def create_pc_movie_filename(self, file_directory, subtract_mean=False):
        """Return the output path for a PC reconstruction movie."""
        suffix = '_mean_subtracted' if subtract_mean else ''
        return os.path.join(file_directory, f'{self.pc_string}{suffix}.mp4')

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _reshape_frame(self, raw_1d):
        """Reshape a flat data vector; flip sign for pitch channels."""
        dat = np.reshape(raw_1d, self.T.shape, 'F')
        return -dat if 'pitch' in self.channel_name else dat

    def _t_gyr(self, timestep):
        """Convert a frame index to simulation time in Gyr (rounded to 2 d.p.)."""
        return np.around((self.times[0] + timestep) * self.tstep_diff, 2)

    def _style_polar_ax(self, ax):
        ax.set_yticks(
            [np.min(self.jphi_c), np.max(self.jphi_c)],
            labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'],
            fontsize=18, color='k',
        )
        ax.set_rmax(np.max(self.jphi_c))
        ax.tick_params(left=False, right=False, labelleft=True,
                       labelbottom=False, bottom=False)
        ax.grid(visible=False)
        ax.set_rlabel_position(50)

    def _style_cartesian_ax(self, ax):
        ax.set_xlabel(r'$J_\phi$ [kpc km/s]', fontsize=14)
        ax.set_ylabel(r'$\theta_\phi$ [rad]',  fontsize=14)
        ax.set_yticks(
            [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
            fontsize=12,
        )

    def _add_colorbar(self, im, ax, fig=None, label_fontsize=16):
        """Create a colorbar and add channel-specific annotations."""
        cbar = (fig.colorbar(im, ax=ax, location='right') if fig is not None
                else plt.colorbar(im, ax=ax))
        cbar.set_label(self.channel_name, fontsize=label_fontsize)
        kw = dict(ha='center', transform=cbar.ax.transAxes)
        if 'ratio' in self.channel_name:
            cbar.ax.text(3.5, 0.8, 'one-armed spiral\ndominates',
                         va='bottom', fontsize=14, **kw)
            cbar.ax.text(3.5, 0.2, 'two-armed spiral\ndominates',
                         va='top',    fontsize=14, **kw)
            cbar.set_label('m1 Amp / m2 Amp', fontsize=18)
        elif 'pitch' in self.channel_name:
            cbar.ax.text(0.5,  1.02, 'Less wound up',
                         va='bottom', fontsize=15, **kw)
            cbar.ax.text(0.5, -0.02, 'More wound up',
                         va='top',    fontsize=15, **kw)
        return cbar

    def _run_animation(self, fig, animate_fn, filename, fps=10):
        anim = FuncAnimation(
            fig, animate_fn,
            frames=np.arange(len(self.times)),
            interval=20, blit=False,
        )
        anim.save(filename, writer=animation.FFMpegWriter(fps=fps))
        plt.close()

    # ----------------------------------------------------------
    # Core movie templates
    # ----------------------------------------------------------

    def _make_polar_movie(self, filename, get_frame_fn, norm_function, cmap,
                          title_fn=None, title_fontsize=15, **kwargs):
        """
        Single-panel polar movie template.

        Polar coordinate convention: T (theta_phi) is the angular axis,
        J (J_phi) is the radial axis.  The colorbar is created once from a
        dummy frame; the norm is fixed for the entire animation.

        Parameters
        ----------
        get_frame_fn  : callable(timestep) → 2-D array of shape self.T.shape
        title_fn      : callable(timestep) → str title, or None for no title
        """
        norm = norm_function(**kwargs)          # create once; reused every frame

        fig, ax = plt.subplots(1, 1, figsize=(7, 6), subplot_kw={'projection': 'polar'})
        im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap, norm=norm)
        self._add_colorbar(im, ax)
        fig.tight_layout()

        def animate(timestep):
            ax.clear()
            ax.pcolormesh(self.T, self.J, get_frame_fn(timestep), cmap=cmap, norm=norm)
            self._style_polar_ax(ax)
            ax.text(3 * np.pi / 2, 100, f'{self._t_gyr(timestep)} Gyr',
                    fontsize=20, ha='center', c='k')
            if title_fn is not None:
                ax.set_title(title_fn(timestep), fontsize=title_fontsize)
            plt.draw()

        self._run_animation(fig, animate, filename)

    def _make_dual_movie(self, filename, get_frame_fn, norm_function, cmap,
                         title_fn=None, title_fontsize=14, **kwargs):
        """
        Dual-panel movie: left = polar face-on, right = Cartesian (J_phi vs theta_phi).

        The same data array is plotted in both panels each frame.  The norm
        is pre-created once and shared.  title_fn is applied to the Cartesian axes.

        Coordinate convention (see PhaseSpaceGrid.__init__):
          polar   → pcolormesh(T, J, dat)  — T=angle,  J=radius
          Cartesian → pcolormesh(J, T, dat) — J=x-axis, T=y-axis
        """
        fig     = plt.figure(figsize=(14, 6))
        ax_pol  = fig.add_subplot(1, 2, 1, projection='polar')
        ax_cart = fig.add_subplot(1, 2, 2)

        norm = norm_function(**kwargs)
        ax_pol.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap, norm=norm)
        im = ax_cart.pcolormesh(self.J, self.T, np.ones(self.T.shape), cmap=cmap, norm=norm)
        self._add_colorbar(im, ax_cart, fig=fig)
        self._style_polar_ax(ax_pol)
        self._style_cartesian_ax(ax_cart)
        fig.tight_layout()

        def animate(timestep):
            ax_pol.clear()
            ax_cart.clear()
            dat = get_frame_fn(timestep)
            ax_pol.pcolormesh(self.T, self.J, dat, cmap=cmap, norm=norm)
            ax_cart.pcolormesh(self.J, self.T, dat, cmap=cmap, norm=norm)
            self._style_polar_ax(ax_pol)
            self._style_cartesian_ax(ax_cart)
            if title_fn is not None:
                ax_cart.set_title(title_fn(timestep), fontsize=title_fontsize)
            plt.draw()

        self._run_animation(fig, animate, filename)

    # ----------------------------------------------------------
    # Public movie methods
    # ----------------------------------------------------------

    def _make_pc_frame_fn(self, subtract_mean):
        """
        Return a frame-getter for PC reconstruction movies.
        Shared by the polar and dual variants to avoid duplicating this logic.
        """
        def get_frame(t):
            dat = self._reshape_frame(self.pc_rc[:, t])
            return dat - np.mean(dat, axis=0) if subtract_mean else dat
        return get_frame

    def make_data_mov(self, filename, data_tbl,
                      norm_function=mpl.colors.Normalize, cmap='Reds', **kwargs):
        """Polar animation of the raw data table."""
        self._make_polar_movie(
            filename,
            get_frame_fn=lambda t: self._reshape_frame(data_tbl[t, 1:]),
            norm_function=norm_function, cmap=cmap,
            title_fn=lambda t: self.channel_name, title_fontsize=20,
            **kwargs,
        )

    def make_pc_reconstruction_mov(self, file_directory, subtract_mean=False,
                                   norm_function=mpl.colors.Normalize,
                                   cmap=cmr.holly, **kwargs):
        """Polar animation of a PC reconstruction."""
        filename = self.create_pc_movie_filename(file_directory, subtract_mean)
        suffix   = ' (mean-subtracted)' if subtract_mean else ''
        self._make_polar_movie(
            filename, self._make_pc_frame_fn(subtract_mean),
            norm_function=norm_function, cmap=cmap,
            title_fn=lambda t: f'{self.channel_name}, {self.pc_string}{suffix}',
            **kwargs,
        )

    def make_dual_data_mov(self, filename, data_tbl,
                           norm_function=mpl.colors.Normalize, cmap='Reds', **kwargs):
        """Dual-panel animation of the raw data table."""
        self._make_dual_movie(
            filename,
            get_frame_fn=lambda t: self._reshape_frame(data_tbl[t, 1:]),
            norm_function=norm_function, cmap=cmap,
            title_fn=lambda t: f't = {self._t_gyr(t)} Gyr',
            title_fontsize=16,
            **kwargs,
        )

    def make_dual_pc_reconstruction_mov(self, file_directory, subtract_mean=False,
                                        norm_function=mpl.colors.Normalize,
                                        cmap=cmr.holly, **kwargs):
        """Dual-panel animation of a PC reconstruction."""
        filename = self.create_pc_movie_filename(file_directory, subtract_mean)
        filename = filename.replace('.mp4', '_dual.mp4')
        suffix   = ' (mean-subtracted)' if subtract_mean else ''
        self._make_dual_movie(
            filename, self._make_pc_frame_fn(subtract_mean),
            norm_function=norm_function, cmap=cmap,
            title_fn=lambda t: f't = {self._t_gyr(t)} Gyr  —  {self.pc_string}{suffix}',
            title_fontsize=13,
            **kwargs,
        )

    # ----------------------------------------------------------
    # Inject a pre-computed reconstruction (avoids re-running mSSA)
    # ----------------------------------------------------------

    def load_reconstruction(self, pc_rc, pcs):
        """
        Inject an already-computed PC reconstruction so that movie methods
        can be called without triggering another mssa.reconstruct() call.

        This is useful when the macro-spiral fitting loop has already called
        mssa.reconstruct(pc_list) and retrieved pc_rc — the same array can be
        handed directly to MakeAnimations rather than recomputing it.

        Parameters
        ----------
        pc_rc : ndarray, shape (n_channels, n_times)
            Reconstructed coefficient array from mssa.getReconstructed().
        pcs   : list of int
            PC indices this reconstruction corresponds to (used for filenames
            and titles via self.pc_string).
        """
        self.pcs       = pcs
        self.pc_string = self._format_pc_string(pcs)
        self.pc_rc     = pc_rc

    def make_pc_movie_pair(self, file_directory, data_vmin, data_vmax,
                           dual=True, cmap_log=None, cmap_sym=None):
        """
        Make the standard LogNorm + mean-subtracted SymLogNorm movie pair for
        the currently loaded PC reconstruction (set via load_reconstruction or
        reconstruct_from_pcs).

        Parameters
        ----------
        file_directory : str
            Output directory (passed to make_[dual_]pc_reconstruction_mov).
        data_vmin, data_vmax : float
            Fallback limits used when the reconstruction is near-zero.
        dual : bool
            If True (default) produce dual-panel movies; otherwise polar only.
        cmap_log, cmap_sym : matplotlib colormap, optional
            Colormaps for LogNorm and SymLogNorm movies. Defaults: cmr.sunburst
            and cmr.holly respectively.
        """
        cmap_log = cmap_log or cmr.sunburst
        cmap_sym = cmap_sym or cmr.holly

        vmin_pc, vmax_pc, linthresh = compute_pc_limits(
            self.pc_rc, fallback_vmin=data_vmin, fallback_vmax=data_vmax)
        print(f'    PC movie limits: [{vmin_pc:.2f}, {vmax_pc:.2f}]  '
              f'linthresh={linthresh:.2f}')

        make_mov     = (self.make_dual_pc_reconstruction_mov if dual
                        else self.make_pc_reconstruction_mov)
        make_mov(file_directory, subtract_mean=False,
                 norm_function=mpl.colors.LogNorm, cmap=cmap_log,
                 vmin=vmin_pc, vmax=vmax_pc)
        make_mov(file_directory, subtract_mean=True,
                 norm_function=mpl.colors.SymLogNorm, cmap=cmap_sym,
                 vmin=-vmax_pc, vmax=vmax_pc, linthresh=linthresh)


# ============================================================
#  RewindMacroSpiral
# ============================================================

class RewindMacroSpiral(PhaseSpaceGrid):
    """
    Derive the winding time of a macro spiral arm from an mSSA PC reconstruction
    and produce diagnostic plots.
    """

    DATA_DIR = '/Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/'

    _FREQ_FILE_NAMES = {
        'B2':   'B2_freq_phi_spl.pkl',
        'live': 'B2_freq_phi_spl.pkl',
        'test': 'tp_freq_phi_spl.pkl',
    }

    def __init__(self, pc_rc, pcs, jphi_min, jbins, sim_name,
                 channel_name='one-armed amplitude', m=1):
        super().__init__(jphi_min, jbins, sim_name)

        freq_file = os.path.join(self.DATA_DIR, self._FREQ_FILE_NAMES[sim_name])
        with open(freq_file, 'rb') as f:
            freq_spl = pickle.load(f)

        freqs          = freq_spl(self.jphi_c)
        self.omega_phi = freqs * 2 * np.pi
        self.O, self.T2 = np.meshgrid(self.omega_phi, self.tphi_c)

        self.channel_name = channel_name
        self.m         = m
        self.pc_rc     = pc_rc
        self.pcs       = pcs
        self.pc_string = self._format_pc_string(pcs)

    # ----------------------------------------------------------
    # Fitting helpers
    # ----------------------------------------------------------

    def _ensure_fit(self, tstep, threshold):
        """
        Run fit_macro_spiral only when the (tstep, threshold) pair differs from
        the last call. This prevents re-computing the same fit multiple times
        when plot_fit_and_dipole calls derive_winding_time, plot_fitting_tstep,
        and make_rewind_dipole_fig in sequence for the same timestep.
        """
        if getattr(self, '_fit_cache_key', None) != (tstep, threshold):
            self.fit_macro_spiral(tstep, threshold=threshold)
            self._fit_cache_key = (tstep, threshold)

    def _fit_sine_maximum(self, data_col, m):
        """
        Fit a pure Fourier mode of order m to data_col sampled at self.tphi_c.
        Returns (phi_max, rss): angle of maximum and residual sum of squares.
        phi_max = arctan2(A_sin, A_cos) / m, in (-pi/m, pi/m].
        """
        theta = self.tphi_c
        X     = np.column_stack([np.ones(len(theta)),
                                  np.cos(m * theta),
                                  np.sin(m * theta)])
        c, _, _, _ = np.linalg.lstsq(X, data_col, rcond=None)
        rss = np.sum((data_col - X @ c) ** 2)
        return np.arctan2(c[2], c[1]) / m, rss

    def fit_macro_spiral(self, tstep, threshold=np.pi):
        """
        Fit the macro spiral at a single timestep.  Sets self.tstep_data,
        self.best_m, self.max_angles, self.macro_spl, self.macro_spl_omega,
        self.jphi_c_fit, and self.omega_phi_fit for use by downstream methods.
        """
        self.tstep_data = np.reshape(self.pc_rc[:, tstep], self.T.shape, 'F')
        nbins = self.tstep_data.shape[1]

        # Choose the azimuthal order (m=1 or m=2) that wins more J_phi bins by RSS
        m1_wins = sum(
            self._fit_sine_maximum(self.tstep_data[:, j], 1)[1] <=
            self._fit_sine_maximum(self.tstep_data[:, j], 2)[1]
            for j in range(nbins)
        )
        self.best_m = 1 if m1_wins >= nbins / 2 else 2

        # Fit all bins with the winning mode.
        # _fit_sine_maximum returns phi_max in (-π/m, π/m].  Wrap to [0, 2π/m)
        # before np.unwrap so the unwrapping works on a monotone sequence, then
        # scale back to recover the original (unwrapped) angle range.
        max_angles_ = np.array([
            self._fit_sine_maximum(self.tstep_data[:, j], self.best_m)[0]
            for j in range(nbins)
        ])
        max_angles_        = max_angles_ % (2 * np.pi / self.best_m)
        self.max_angles    = (1 / self.best_m) * np.unwrap(self.best_m * max_angles_)

        ind1, ind2             = self.find_fitting_interval(threshold=threshold)
        self.jphi_c_fit        = self.jphi_c[ind1:ind2]
        self.omega_phi_fit     = self.omega_phi[ind1:ind2]
        max_angles_fit         = self.max_angles[ind1:ind2]

        self.macro_spl        = make_splrep(self.jphi_c_fit, max_angles_fit,
                                             w=None, k=3, s=len(self.jphi_c_fit))
        self.macro_line_coeff = np.polyfit(self.omega_phi_fit, max_angles_fit, 1)
        self.macro_spl_omega  = np.poly1d(self.macro_line_coeff)

    def find_fitting_interval(self, threshold=np.pi / 2):
        """
        Return (ind1, ind2) such that max_angles[ind1:ind2] is the longest
        sub-sequence where every adjacent step is in [-threshold, 0]
        (i.e. the spiral is winding up slowly and monotonically).

        Algorithm: build `good` = indices where the step condition holds, then
        scan for the longest run of *consecutive* indices in `good` (meaning
        back-to-back valid steps, not isolated ones).
        """
        diffs = np.diff(self.max_angles)
        good  = np.where((diffs <= 0) & (diffs >= -threshold))[0]

        max_length, best_start = 0, 0
        for i in range(len(good) - 1):
            if good[i + 1] == good[i] + 1:          # start or continuation of a run
                length = 2
                while i + length < len(good) and good[i + length] == good[i] + length:
                    length += 1
                if length > max_length:
                    max_length, best_start = length, good[i]

        return best_start, best_start + max_length

    # ----------------------------------------------------------
    # Winding-time derivation
    # ----------------------------------------------------------

    def derive_winding_time(self, tstep, threshold=np.pi):
        """
        Return the winding time (Gyr) from the slope of the linear fit of
        theta_max vs Omega_phi over the best fitting interval.

        Uses _ensure_fit to avoid redundant recomputation when this is called
        multiple times for the same (tstep, threshold).  Returns np.nan on failure.
        """
        try:
            self._ensure_fit(tstep, threshold)
        except Exception:
            print(f"Error fitting macro spiral at tstep {tstep}")
            return np.nan

        omega1, omega2 = self.omega_phi_fit[0], self.omega_phi_fit[-1]
        theta1, theta2 = self.macro_spl_omega(omega1), self.macro_spl_omega(omega2)
        return (theta2 - theta1) / (omega2 - omega1)

    # ----------------------------------------------------------
    # Diagnostic plots
    # ----------------------------------------------------------

    def plot_macro_tfit_over_time(self, threshold=np.pi,
                                  ax=None, savefig=False, fig_dir=None):
        ntimes = self.pc_rc.shape[-1]
        tfits  = np.array([self.derive_winding_time(t, threshold=threshold)
                           for t in range(ntimes)])

        created_fig = ax is None
        if created_fig:
            plt.figure(figsize=(6, 6), constrained_layout=True)
            ax = plt.gca()

        t_axis = np.arange(ntimes) * self.tstep_diff
        t_end  = ntimes * self.tstep_diff
        ax.plot(t_axis, tfits, 'o', c='blue', ms=2, label='Fitted Winding Time')
        ax.plot([0, 1.5*t_end], [0, 1.5*t_end], 'k--', label='"Correct" Winding Time')
        ax.set_xlabel('Simulation Time (Gyr)', fontsize=14)
        ax.set_ylabel('Winding Time (Gyr)',    fontsize=14)
        if self.sim_name == 'test' and self.pc_string == 'data':
            ax.set_title(f'Test Particle Macro-spiral Winding', fontsize=16)
        elif (self.sim_name == 'B2' or self.sim_name == 'live') and self.pc_string == 'data':
            ax.set_title(f'N-body Macro-spiral Winding', fontsize=16)
        else:
            ax.set_title(f'{self.sim_name} {self.channel_name} tfit for {self.pc_string}',
                        fontsize=16)
        ax.set_aspect('equal')
        ax.set_xlim(0, np.max([np.min([1.25*t_end, 1.05*np.nanmax(tfits)]), t_end]))
        ax.set_ylim(0, np.max([np.min([1.25*t_end, 1.05*np.nanmax(tfits)]), t_end]))
        ax.legend(loc='upper left')

        if savefig:
            plt.savefig(os.path.join(fig_dir, f'winding_times_{self.pc_string}.pdf'))
            print(f'Saved winding time fit figure for {self.pc_string}')
        if created_fig:
            plt.close()

    def plot_fitting_tstep(self, tstep, threshold=np.pi,
                           ax=None, savefig=False, fig_dir=None):
        """
        Plot the theta_max vs Omega_phi diagram for a single timestep, showing
        the pcolormesh background, the per-bin max-amplitude angles, the linear
        fit, and the fitting bounds.
        """
        try:
            tfit = self.derive_winding_time(tstep, threshold=threshold)
        except Exception:
            print(f"Error fitting macro spiral at tstep {tstep}")
            return

        omega_fit = np.linspace(self.omega_phi_fit[0], self.omega_phi_fit[-1], 100)
        theta_fit = self.macro_spl_omega(omega_fit)

        if np.all(self.tstep_data > 0):
            norm = mpl.colors.LogNorm(vmin=np.min(self.tstep_data),
                                      vmax=np.max(self.tstep_data))
            cmap = cmr.sunburst
        else:
            vmax = np.max(np.abs(self.tstep_data))
            norm = mpl.colors.SymLogNorm(linthresh=vmax / 1e2, vmin=-vmax, vmax=vmax)
            cmap = cmr.holly

        created_fig = ax is None
        if created_fig:
            plt.figure(figsize=(8, 5), constrained_layout=True)
            ax = plt.gca()

        ax.pcolormesh(self.O, self.T2, self.tstep_data, cmap=cmap, norm=norm, alpha=0.3)
        ax.plot(self.omega_phi, self.max_angles % (2 * np.pi), 'o', c='k', ms=5,
                label='Max Amplitude Angles')
        ax.plot(omega_fit, theta_fit % (2 * np.pi), 'b-', label='Linear Fit')
        ax.axvline(self.omega_phi_fit[0],  color='k', linestyle='--', label='Fitting bounds')
        ax.axvline(self.omega_phi_fit[-1], color='k', linestyle='--')
        ax.set_xlabel(r'$\Omega_\phi$',  fontsize=14)
        ax.set_ylabel(r'$\theta_{max}$', fontsize=14)
        ax.set_title(f'Fitting the m={self.best_m} Macro Spiral, tfit={tfit:.2f} Gyr',
                     fontsize=16)
        ax.legend()

        if created_fig:
            if savefig:
                plt.savefig(os.path.join(fig_dir,
                                          f'macro_fit_t{int(tstep)}_{self.pc_string}.pdf'))
            plt.close()


    def make_rewind_dipole_fig(self, tstep, axs=None, savefig=False, fig_dir=None):
        """
        Two-panel polar plot showing the phase spiral before and after rewinding.

        Left panel: the PC reconstruction at `tstep`, with the azimuthal mean
        subtracted so that the dipole (m=1 asymmetry) is visible.
        Right panel: the same data after rotating each radial shell backwards by
        tfit * Omega_phi(J_phi), effectively unwinding the phase spiral to its
        state at the time of the perturbation.

        If axs is None a standalone figure is created and closed; otherwise the
        two polar Axes from the caller are used (e.g. from plot_fit_and_dipole).
        """
        tfit = self.derive_winding_time(tstep, threshold=np.pi / 2)
        if np.isnan(tfit):
            print(f"Could not derive winding time for tstep {tstep}, "
                  "skipping rewind dipole figure.")
            return

        frame    = np.reshape(self.pc_rc[:, tstep], self.T.shape, 'F')
        rad_mean = np.mean(frame, axis=0)
        _, vmax, linthresh = compute_pc_limits(self.pc_rc)
        norm     = mpl.colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
        cmap     = cmr.holly

        created_fig = axs is None
        if created_fig:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5),
                                    subplot_kw={'projection': 'polar'})
        else:
            fig = axs[0].get_figure()

        d_rot  = tfit * self.omega_phi
        interp = interp1d(self.tphi_c,
                           self.pc_rc[:, tstep].reshape(self.T.T.shape),
                           axis=1, fill_value='extrapolate')
        rewound = np.diagonal(interp((self.T + d_rot) % (2 * np.pi)), axis1=0, axis2=2)

        polar_kw = dict(cmap=cmap, rasterized=True, shading='nearest', norm=norm)
        axs[0].pcolormesh(self.T, self.J, frame - rad_mean,   **polar_kw)
        im = axs[1].pcolormesh(self.T, self.J, rewound - rad_mean, **polar_kw)

        axs[0].set_title(f'Time = {tstep * self.tstep_diff:.2f} Gyr',
                         pad=10, fontsize=14)
        axs[1].set_title(f'Rewind {np.around(tfit, decimals=2)} Gyr',
                         pad=10, fontsize=14)

        for ax in axs:
            ax.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)],
                          labels=[r'$J_\phi=1000$', r'$J_\phi=3000$'])
            ax.set_rmax(np.max(self.jphi_c))
            ax.tick_params(left=False, right=False, labelleft=True,
                           labelbottom=False, bottom=False)
            ax.grid(visible=False)

        if created_fig:
            fig.suptitle(f'Rewinding to dipole, {self.pc_string}', fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.75])
            fig.colorbar(im, cax=cbar_ax, label='One Armed Phase Spiral Amplitude')
            if savefig:
                plt.savefig(os.path.join(fig_dir,
                                          f'rewind_t{int(tstep)}_{self.pc_string}.pdf'))
            plt.close()
        else:
            fig.colorbar(im, ax=axs[1], label='One Armed Phase Spiral Amplitude')

    def plot_fit_and_dipole(self, tstep, threshold=np.pi / 2,
                            savefig=False, fig_dir=None):
        """
        Combined diagnostic figure: top row = theta_max vs Omega_phi fit
        (plot_fitting_tstep), bottom row = before/after rewind dipole panels
        (make_rewind_dipole_fig).  All three sub-calls share the cached fit
        result so fit_macro_spiral is computed only once.
        """
        try:
            tfit = self.derive_winding_time(tstep, threshold=threshold)
            if np.isnan(tfit):
                print(f"Could not derive winding time for tstep {tstep}, "
                      "skipping rewind dipole figure.")
                return
        except Exception:
            print(f"Error fitting macro spiral at tstep {tstep}")
            return

        fig      = plt.figure(figsize=(11, 10))
        gs       = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
        ax_top   = fig.add_subplot(gs[0, :])
        ax_left  = fig.add_subplot(gs[1, 0], projection='polar')
        ax_right = fig.add_subplot(gs[1, 1], projection='polar')

        self.plot_fitting_tstep(tstep, threshold=threshold, ax=ax_top)
        self.make_rewind_dipole_fig(tstep, axs=[ax_left, ax_right])

        fig.suptitle(f'{self.sim_name} {self.channel_name}, {self.pc_string}',
                     fontsize=16)
        if savefig:
            plt.savefig(os.path.join(fig_dir,
                                      f'fit_and_dipole_t{int(tstep)}_{self.pc_string}.pdf'))
        plt.close()


# ============================================================
#  Module-level utility functions
# ============================================================

def suggest_pc_groups(mssa, n_pcs, threshold=0.5):
    """
    Suggest PC groupings by thresholding the W-correlation matrix and finding
    connected components.

    Parameters
    ----------
    mssa      : pyEXP mSSA object
    n_pcs     : int — number of PCs to consider
    threshold : float — wCorr value above which two PCs are grouped (default 0.5)

    Returns
    -------
    groups : list of lists sorted by first index
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    wcorr = mssa.wCorrAll()[:n_pcs, :n_pcs]
    adj   = (wcorr > threshold).astype(int)
    np.fill_diagonal(adj, 0)

    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    groups = [list(np.where(labels == i)[0]) for i in range(n_components)]
    groups.sort(key=lambda g: g[0])
    return groups


def print_pc_groups(groups, ev):
    """Print suggested PC groups with eigenvalue info."""
    print('\nSuggested PC groupings:')
    print(f'  {"Group":<8} {"PCs":<40} {"Eigenvalue range"}')
    print('  ' + '-' * 72)
    for i, group in enumerate(groups):
        pc_str   = (str(group) if len(group) <= 6
                    else f'[{group[0]}..{group[-1]}] ({len(group)} PCs)')
        ev_range = f'{ev[group[0]]:.3e} – {ev[group[-1]]:.3e}'
        print(f'  {i:<8} {pc_str:<40} {ev_range}')
    print()


def compute_pc_limits(pc_rc, fallback_vmin=None, fallback_vmax=None):
    """
    Compute appropriate vmin / vmax / linthresh for PC reconstruction movies.

    vmin_pc / vmax_pc are the 1st / 99th percentiles of positive values in
    pc_rc (falling back to the supplied fallback values when the reconstruction
    is near-zero).  linthresh for SymLogNorm is set to 1 % of vmax_pc.

    Parameters
    ----------
    pc_rc              : ndarray, shape (n_channels, n_times)
    fallback_vmin/vmax : used when fewer than 10 positive values exist in pc_rc

    Returns
    -------
    vmin_pc, vmax_pc, linthresh : floats
    """
    pos_vals = pc_rc[pc_rc > 0]
    if len(pos_vals) > 10:
        vmin_pc = np.percentile(pos_vals, 1)
        vmax_pc = np.percentile(pos_vals, 99)
    else:
        vmin_pc = fallback_vmin
        vmax_pc = fallback_vmax

    linthresh = vmax_pc * 0.01
    return vmin_pc, vmax_pc, linthresh


# ------------------------------------------------------------------
# Diagnostic plots
# ------------------------------------------------------------------

def plot_eigenvalues(ev, fig_dir):
    """Semilog plot of mSSA eigenvalues. Saves to fig_dir/eigenvalues.pdf."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(ev, '-o')
    ax.set_xlabel('index',      fontsize=20)
    ax.set_ylabel('eigenvalue', fontsize=20)
    ax.set_title('PC Eigenvalues', fontsize=20)
    plt.savefig(os.path.join(fig_dir, 'eigenvalues.pdf'))
    plt.close()
    print('Saved eigenvalues plot.')


def plot_fg_matrices(mssa, fig_dir):
    """F and G contribution matrices. Saves to fig_dir/F_matrix.pdf and G_matrix.pdf."""
    t1, t2 = mssa.contrib()
    for data, title, fname in [(t2, 'G Matrix', 'G_matrix.pdf'),
                                (t1, 'F Matrix', 'F_matrix.pdf')]:
        plt.figure(figsize=(20, 5))
        plt.imshow(data, aspect='auto', norm=mpl.colors.LogNorm(), cmap=cmr.freeze)
        plt.xlabel('Input Channel')
        plt.ylabel('Principal Component')
        plt.title(title)
        plt.savefig(os.path.join(fig_dir, fname))
        plt.close()
    print('Saved F and G matrix plots.')


def plot_wcorr(mssa, fig_dir):
    """W-correlation matrix. Saves to fig_dir/wCorr.pdf."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mssa.wCorrAll(), cmap='gray_r')
    plt.savefig(os.path.join(fig_dir, 'wCorr.pdf'))
    plt.close()
    print('Saved W-correlation matrix.')


def plot_pc_time_series(mssa, times, fig_dir, n_pcs=20):
    """
    Normalized, offset PC time series plot. Saves to fig_dir/pc_time_series.pdf.
    """
    pc        = mssa.getPC()
    lag_times = times[:pc.shape[0]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    for i in range(n_pcs):
        ax.plot(lag_times,
                pc[:, i] / np.max(np.abs(pc[:, i])) - 2 * i,
                label=str(i))
    ax.legend()
    ax.set_xlim(np.min(lag_times), 1.2 * np.max(lag_times))
    ax.set_xlabel('Time',                             fontsize=16)
    ax.set_ylabel('Normalized PC amplitude (offset)', fontsize=14)
    ax.set_title('PC Time Series',                    fontsize=18)
    plt.savefig(os.path.join(fig_dir, 'pc_time_series.pdf'))
    plt.close()
    print('Saved PC time series.')


def plot_diagnostics_summary(mssa, ev, times, fig_dir, n_pcs=20, savefig=True):
    """
    Combined diagnostic figure with all four standard mSSA diagnostics as
    sub-panels in a single figure.

    Layout (3 rows × 3 columns, PC time series spans the full right column):

        ┌──────────────┬──────────────┬──────────────────────┐
        │ Eigenvalues  │  W-corr      │                      │
        ├──────────────┴──────────────┤   PC time series     │
        │        G matrix             │   (all rows)         │
        ├─────────────────────────────┤                      │
        │        F matrix             │                      │
        └─────────────────────────────┴──────────────────────┘

    Parameters
    ----------
    mssa    : pyEXP mSSA object
        Must have already had mssa.reconstruct() called so that contrib() and
        wCorrAll() are available (i.e. called after the full [0..npc]
        reconstruction in the --diagnostics block).
    ev      : array-like
        Eigenvalue spectrum returned by mssa.eigenvalues().
    times   : array-like
        Simulation time array (from coefs.Times()).
    fig_dir : str
        Directory in which to save 'diagnostics_summary.pdf'.
    n_pcs   : int
        Number of PCs to show in the time-series panel (default 20).
    savefig : bool
        If True (default), save to fig_dir/diagnostics_summary.pdf.

    Returns
    -------
    fig : matplotlib Figure
        The composed figure object (useful for interactive inspection).
    """
    t_fmat, t_gmat = mssa.contrib()    # t1 = F matrix, t2 = G matrix
    wcorr          = mssa.wCorrAll()
    pc             = mssa.getPC()
    lag_times      = times[:pc.shape[0]]

    # ---- Figure layout -----------------------------------------------
    # height_ratios: rows 0 (eigenvalues/wcorr) taller, rows 1-2 shorter
    # for the wide-but-flat F/G matrices.
    # width_ratios: cols 0-1 equal; col 2 (PC time series) slightly wider.
    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(
        3, 3,
        height_ratios=[2, 1, 1],
        width_ratios=[1, 1, 1.1],
        hspace=0.2, wspace=0.3,
        top=0.93,
    )

    ax_ev    = fig.add_subplot(gs[0, 0])        # top-left:  eigenvalues
    ax_wcorr = fig.add_subplot(gs[0, 1])        # top-mid:   W-correlation
    ax_pcts  = fig.add_subplot(gs[:, 2])        # right col: PC time series (all rows)
    ax_gmat  = fig.add_subplot(gs[1, :2])       # mid row:   G matrix
    ax_fmat  = fig.add_subplot(gs[2, :2])       # bot row:   F matrix

    # ---- Eigenvalues -------------------------------------------------
    ax_ev.semilogy(ev, '-o', ms=4)
    ax_ev.set_xlabel('Index',      fontsize=14)
    ax_ev.set_ylabel('Eigenvalue', fontsize=14)
    ax_ev.set_title('PC Eigenvalues', fontsize=15)

    # ---- W-correlation matrix ----------------------------------------
    im_wc = ax_wcorr.imshow(wcorr, cmap='gray_r', aspect='auto')
    ax_wcorr.set_xlabel('PC index', fontsize=12)
    ax_wcorr.set_ylabel('PC index', fontsize=12)
    ax_wcorr.set_title('W-Correlation Matrix', fontsize=15)
    ax_wcorr.set_aspect('equal')
    fig.colorbar(im_wc, ax=ax_wcorr, fraction=0.046, pad=0.04)

    # ---- G and F contribution matrices --------------------------------
    # Both share the same LogNorm so the colour scale is comparable.
    # combined_max = max(t_gmat.max(), t_fmat.max())
    fg_norm      = mpl.colors.LogNorm()#vmin=combined_max * 1e-4, vmax=combined_max)

    im_g = ax_gmat.imshow(t_gmat, aspect='auto', norm=fg_norm, cmap=cmr.freeze)
    ax_gmat.set_xlabel('Input Channel',      fontsize=12)
    ax_gmat.set_ylabel('Principal Component', fontsize=12)
    ax_gmat.set_title('G Matrix (grouping contribution)', fontsize=13)
    fig.colorbar(im_g, ax=ax_gmat, fraction=0.03, pad=0.02)

    im_f = ax_fmat.imshow(t_fmat, aspect='auto', norm=fg_norm, cmap=cmr.freeze)
    ax_fmat.set_xlabel('Input Channel',      fontsize=12)
    ax_fmat.set_ylabel('Principal Component', fontsize=12)
    ax_fmat.set_title('F Matrix (forecasting contribution)', fontsize=13)
    fig.colorbar(im_f, ax=ax_fmat, fraction=0.03, pad=0.02)

    # ---- PC time series ----------------------------------------------
    for i in range(n_pcs):
        ax_pcts.plot(lag_times,
                     pc[:, i] / np.max(np.abs(pc[:, i])) - 2 * i,
                     lw=0.9, label=str(i))
    ax_pcts.set_xlim(np.min(lag_times), np.max(lag_times))
    ax_pcts.set_ylim(-2 * n_pcs, 6)
    ax_pcts.set_xlabel('Time',                              fontsize=14)
    ax_pcts.set_ylabel('Normalised PC amplitude (offset)',  fontsize=14)
    ax_pcts.set_title(f'PC Time Series (first {n_pcs})',    fontsize=15)
    ax_pcts.legend(loc='upper right', fontsize=12, ncol=4)

    # ---- Finalise ----------------------------------------------------
    fig.suptitle('mSSA Diagnostics Summary', fontsize=17, y=0.97)

    if savefig:
        out_path = os.path.join(fig_dir, 'diagnostics_summary.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved diagnostics summary to {out_path}')

    return fig


# ------------------------------------------------------------------
# Convenience wrappers: batch movie generation
# ------------------------------------------------------------------

def compute_data_limits(data_tbl):
    """
    Compute LogNorm vmin/vmax for the raw data table.

    Strips the leading time column (column 0) and returns the 1st and 99th
    percentile of all positive values — suitable as vmin/vmax for LogNorm
    in data movies and as fallback limits for PC movies.

    Parameters
    ----------
    data_tbl : ndarray, shape (n_times, 1 + n_channels)
        Raw data array as loaded by np.loadtxt; first column is time.

    Returns
    -------
    vmin, vmax : float
    """
    pos = data_tbl[:, 1:].flatten()
    pos = pos[pos > 0]
    return np.percentile(pos, 1), np.percentile(pos, 99)


def make_face_on_movies(mssa, data_file, times, face_on_dir, list_of_pc_lists,
                        sim_name, channel_name, jphi_min, jbins,
                        cmap_log=None, cmap_sym=None):
    """
    Generate all single-panel polar movies: raw data + one LogNorm / one
    mean-subtracted SymLogNorm pair per PC group.
    """
    cmap_log = cmap_log or cmr.sunburst
    cmap_sym = cmap_sym or cmr.holly

    data = np.loadtxt(data_file)
    MA   = MakeAnimations(mssa, sim_name=sim_name, channel_name=channel_name,
                          times=times, jphi_min=jphi_min, jbins=jbins)

    data_vmin, data_vmax = compute_data_limits(data)
    print(f'Data movie: vmin={data_vmin:.1f}, vmax={data_vmax:.1f}')
    MA.make_data_mov(os.path.join(face_on_dir, 'data.mp4'), data,
                     norm_function=mpl.colors.LogNorm, cmap=cmap_log,
                     vmin=data_vmin, vmax=data_vmax)

    for pc_list in list_of_pc_lists:
        print(f'  Creating movie for PCs {pc_list}')
        MA.reconstruct_from_pcs(pc_list)
        MA.make_pc_movie_pair(face_on_dir, data_vmin, data_vmax,
                              dual=False, cmap_log=cmap_log, cmap_sym=cmap_sym)


def make_dual_movies(mssa, data_file, times, face_on_dir, list_of_pc_lists,
                     sim_name, channel_name, jphi_min, jbins,
                     cmap_log=None, cmap_sym=None):
    """
    Generate dual-panel (polar + Cartesian) movies for the raw data and each
    PC group. Files are saved with a '_dual.mp4' suffix.

    Use this function when movies are the only goal.  If macro-spiral fitting
    is also being performed, call MA.load_reconstruction() + MA.make_pc_movie_pair()
    inside your own loop instead of calling this function, so that the shared
    mssa.reconstruct() call is not duplicated.
    """
    cmap_log = cmap_log or cmr.sunburst
    cmap_sym = cmap_sym or cmr.holly

    data = np.loadtxt(data_file)
    MA   = MakeAnimations(mssa, sim_name=sim_name, channel_name=channel_name,
                          times=times, jphi_min=jphi_min, jbins=jbins)

    data_vmin, data_vmax = compute_data_limits(data)
    print(f'Dual data movie: vmin={data_vmin:.1f}, vmax={data_vmax:.1f}')
    MA.make_dual_data_mov(os.path.join(face_on_dir, 'data_dual.mp4'), data,
                          norm_function=mpl.colors.LogNorm, cmap=cmap_log,
                          vmin=data_vmin, vmax=data_vmax)

    for pc_list in list_of_pc_lists:
        print(f'  Creating dual movie for PCs {pc_list}')
        MA.reconstruct_from_pcs(pc_list)
        MA.make_pc_movie_pair(face_on_dir, data_vmin, data_vmax,
                              dual=True, cmap_log=cmap_log, cmap_sym=cmap_sym)