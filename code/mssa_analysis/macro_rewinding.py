import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2 #set the value globally
import matplotlib.pyplot as plt
import cmasher as cmr
import scipy
import pickle
from scipy.interpolate import make_splrep, interp1d
import diagnostics

# use pc reconstruction at a given timestep to derive the time of the perturbation
class RewindMacroSpiral():
    def __init__(self, pc_rc, pcs, jphi_min, jbins, sim_name, 
                 channel_name='one-armed amplitude', m=1):

        jphi_c = np.linspace(jphi_min, jphi_min+((jbins-1)*100), jbins)
        self.jphi_c = jphi_c
        tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
        rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
        self.tphi_c = tphi_c_[:-1] + rad[1]
        self.J, self.T = np.meshgrid(jphi_c, self.tphi_c)

        self.sim_name = sim_name
        DATA_DIR = '/Users/Tavangar/Work/EXP_Projects/paper2_correlations/data/'
        if (sim_name == 'B2') | (sim_name == 'live'):
            self.tstep_diff = 0.009778
            freq_file = DATA_DIR + 'B2_freq_phi_spl.pkl'
            # freqs = np.load(DATA_DIR + 'live_frequency_array_j30.npy')
        elif sim_name == 'test':
            self.tstep_diff = 0.01
            freq_file = DATA_DIR + 'tp_freq_phi_spl.pkl'
            # freqs = np.load(DATA_DIR + 'test_frequency_array_j25.npy')
        with open(freq_file, 'rb') as f:
            freq_spl = pickle.load(f)

        freqs = freq_spl(self.jphi_c)
        self.omega_phi = freqs*2*np.pi
        self.O, self.T2 = np.meshgrid(self.omega_phi, self.tphi_c)
        self.channel_name = channel_name

        self.m = m

        self.pc_rc = pc_rc
        self.pcs = pcs

        if pcs is None:
            self.pc_string = 'data'
        elif len(self.pcs) == 1:
            self.pc_string = 'pc{}'.format(self.pcs[0])
        elif np.any(np.diff(self.pcs) > 1): # non-continuous pcs
            self.pc_string = 'pc{}'.format(self.pcs)
        else: # continuous pcs
            self.pc_string = 'pc{}-{}'.format(self.pcs[0], self.pcs[-1])

    def _fit_sine_maximum(self, data_col, m):
        """
        Fit a pure Fourier mode of order m to data_col sampled at self.tphi_c.
        Returns (phi_max, rss): angle of maximum and residual sum of squares.
        phi_max = arctan2(A_sin, A_cos) / m, in (-pi/m, pi/m].
        """
        theta = self.tphi_c
        n = len(theta)
        X = np.column_stack([np.ones(n), np.cos(m*theta), np.sin(m*theta)])
        c, _, _, _ = np.linalg.lstsq(X, data_col, rcond=None)
        rss = np.sum((data_col - X @ c)**2)
        return np.arctan2(c[2], c[1]) / m, rss

    def fit_macro_spiral(self, tstep, threshold=np.pi):
        # get the array of reconstructed amplitudes from the PCs
        self.tstep_data = np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F')

        # for each Jphi bin, determine which mode fits better
        nbins = self.tstep_data.shape[1]
        m1_wins = sum(
            self._fit_sine_maximum(self.tstep_data[:, j], 1)[1] <=
            self._fit_sine_maximum(self.tstep_data[:, j], 2)[1]
            for j in range(nbins)
        )
        self.best_m = 1 if m1_wins >= nbins / 2 else 2

        # fit all bins with the winning mode
        max_angles_ = np.array([self._fit_sine_maximum(self.tstep_data[:, j], self.best_m)[0]
                                 for j in range(nbins)])
        # wrap to [0, 2pi/m) then unwrap
        max_angles_ = max_angles_ % (2 * np.pi / self.best_m)
        self.max_angles = 1/self.best_m * np.unwrap(self.best_m * max_angles_)

        ind1, ind2 = self.find_fitting_interval(threshold=threshold)
        self.jphi_c_fit = self.jphi_c[ind1:ind2]
        self.omega_phi_fit = self.omega_phi[ind1:ind2]
        max_angles_fit = self.max_angles[ind1:ind2]

        # for Jphi, theta_phi fit
        self.macro_spl = make_splrep(self.jphi_c_fit, max_angles_fit, 
                        w=None, k=3, s=len(self.jphi_c_fit)) 
        
        # for Omega_phi, theta_phi fit
        self.macro_line_coeff = np.polyfit(self.omega_phi_fit, max_angles_fit, 1)
        self.macro_spl_omega = np.poly1d(self.macro_line_coeff)
        
    def find_fitting_interval(self, threshold=np.pi/2):
        '''
        Find the region of the array that we want to use for fitting
        the macro spiral. We do this by finding the longest consecutive 
        region where the array is relatively continuous, meaning that the 
        difference between adjacent values is less than a threshold.
        
        In the future we may want to:
        1) require a minimum number of points
        2) exclude phase mixed regions (would need to define this more rigorously)
        '''
        continuous_decrease = (np.diff(self.max_angles) <= 0) & (np.diff(self.max_angles) >= -threshold)
        # continuous_sel = np.abs(np.diff(self.max_angles)) <= threshold
        good_indices = np.where(continuous_decrease)[0]
        
        # Find the good interval with the longest length
        max_length = 0
        for i, good_ind in enumerate(good_indices[:-1]):
            if good_indices[i+1] == good_ind + 1:
                length = 2
                while i+length < len(good_indices) and good_indices[i+length] == good_ind + length:
                    length += 1
                if length > max_length:
                    max_length = length
                    best_start_ind = good_ind
        best_end_ind = best_start_ind + max_length
        return best_start_ind, best_end_ind

    def derive_winding_time(self, tstep, threshold=np.pi):
        try:
            self.fit_macro_spiral(tstep, threshold=threshold)
        except:
            print(f"Error occurred while fitting macro spiral at tstep {tstep}")
            return np.nan

        #get the frequencies at those radii
        omega1, omega2 = self.omega_phi_fit[0], self.omega_phi_fit[-1]
        theta1, theta2 = self.macro_spl_omega(omega1), self.macro_spl_omega(omega2)

        #calculate the winding time
        tfit = (theta2 - theta1) / (omega2 - omega1)
        return tfit
    
    def plot_macro_tfit_over_time(self, threshold=np.pi, 
                                  ax=None, savefig=False, fig_dir=None):
        ntimes = self.pc_rc.shape[-1]

        tfits = np.zeros(ntimes)
        for tstep in range(ntimes):
            tfits[tstep] = self.derive_winding_time(tstep, threshold=threshold)

        
        if ax is None:
            plt.figure(figsize=(5,5))
            ax = plt.gca()
        ax.plot(np.arange(ntimes)*self.tstep_diff, tfits, 'o', c='blue', ms=2, 
                label='Fitted Winding Time')
        ax.plot([0,ntimes*self.tstep_diff], [0,ntimes*self.tstep_diff], 'k--', 
                label='t_fit = t')
        ax.set_xlabel('Simulation Time (Gyr)', fontsize=14)
        ax.set_ylabel('Winding Time (Gyr)', fontsize=14)

        ax.set_title(f'{self.sim_name} {self.channel_name} tfit for {self.pc_string}', 
                     fontsize=16)
        ax.set_aspect('equal')
        ax.set_xlim(0,ntimes*self.tstep_diff)
        ax.set_ylim(0,3.5)
        ax.legend()
        if savefig:
            plt.savefig(fig_dir + f'/winding_times_{self.pc_string}.pdf')
        plt.close()
        print(f'Saved winding time fit figure for {self.pc_string}')

    def plot_fit_and_dipole(self, tstep, threshold=np.pi/2,
                            savefig=False, fig_dir=None):
        try:
            tfit = self.derive_winding_time(tstep, threshold=threshold)
            if np.isnan(tfit):
                print(f"Could not derive winding time for tstep {tstep}, \
                    skipping rewind dipole figure.")
                return
        except:
            print(f"Error occurred while fitting macro spiral at tstep {tstep}")
            return
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4)
        ax_top = fig.add_subplot(gs[0, :])
        ax_left = fig.add_subplot(gs[1, 0], projection='polar')
        ax_right = fig.add_subplot(gs[1, 1], projection='polar')

        self.plot_fitting_tstep(tstep, threshold=threshold, ax=ax_top)
        self.make_rewind_dipole_fig(tstep, axs=[ax_left, ax_right])

        fig.suptitle(f'{self.sim_name} {self.channel_name}, {self.pc_string}',
                     fontsize=16)
        if savefig:
            plt.savefig(fig_dir + f'/fit_and_dipole_t{int(tstep)}_{self.pc_string}.pdf')
        plt.close()
    
    def plot_fitting_tstep(self, tstep, threshold=np.pi,
                           ax=None, savefig=False, fig_dir=None):
        try:
            tfit = self.derive_winding_time(tstep, threshold=threshold)
        except:
            print(f"Error occurred while fitting macro spiral at tstep {tstep}")
            return

        omega_fit = np.linspace(self.omega_phi_fit[0], self.omega_phi_fit[-1], 100)
        theta_fit = self.macro_spl_omega(omega_fit)

        cmap = cmr.sunburst
        if np.all(self.tstep_data > 0):
            norm = mpl.colors.LogNorm(vmin=np.min(self.tstep_data),
                                      vmax=np.max(self.tstep_data))
        else:
            vmax = np.max(np.abs(self.tstep_data))
            norm = mpl.colors.SymLogNorm(linthresh=vmax/1e2, vmin=-vmax, vmax=vmax)
            cmap = cmr.holly

        created_fig = ax is None
        if created_fig:
            plt.figure(figsize=(8,5))
            ax = plt.gca()
        ax.pcolormesh(self.O, self.T2, self.tstep_data, cmap=cmap, norm=norm, alpha=0.3)
        ax.plot(self.omega_phi, self.max_angles % (2*np.pi), 'o', c='k', ms=5,
                label='Max Amplitude Angles')
        ax.plot(omega_fit, theta_fit % (2*np.pi), 'b-',
                label='Linear Fit')
        ax.axvline(self.omega_phi_fit[0], color='k', linestyle='--',
                   label=f'Fitting bounds')
        ax.axvline(self.omega_phi_fit[-1], color='k', linestyle='--')
        ax.set_xlabel(r'$\Omega_\phi$', fontsize=14)
        ax.set_ylabel(r'$\theta_{max}$', fontsize=14)
        ax.set_title(f'Fitting the m={self.best_m} Macro Spiral, tfit={tfit:.2f} Gyr',
                     fontsize=16)
        ax.legend()
        if created_fig:
            if savefig:
                plt.savefig(fig_dir + f'/macro_fit_t{int(tstep)}_{self.pc_string}.pdf')
            plt.close()

    def make_rewind_dipole_fig(self, tstep, axs=None, savefig=False, fig_dir=None):

        tfit = self.derive_winding_time(tstep, threshold=np.pi/2)
        if np.isnan(tfit):
            print(f"Could not derive winding time for tstep {tstep}, \
                  skipping rewind dipole figure.")
            return

        rad_mean_amp = np.mean(np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F'), axis=0)

        _, vmax, linthresh = diagnostics.compute_pc_limits(self.pc_rc, self.T.shape)

        cmap = cmr.holly
        created_fig = axs is None
        if created_fig:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5),
                                    subplot_kw={'projection': 'polar'})
        else:
            fig = axs[0].get_figure()

        d_rot = (tfit * self.omega_phi)

        #now interpolate at each radius so that I can use the same grid as before
        interp = interp1d(self.tphi_c, self.pc_rc[:,tstep].reshape(self.T.T.shape), axis=1,
                                            fill_value="extrapolate")
        all_interp = interp((self.T+d_rot)%(2*np.pi))
        future_grid = np.diagonal(all_interp, axis1=0, axis2=2)

        #plot what that timestep looks like after subtracting background
        axs[0].pcolormesh(self.T, self.J,
                          np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F') - rad_mean_amp,
                          cmap=cmap, rasterized=True, shading='nearest',
                          norm=mpl.colors.SymLogNorm(linthresh=linthresh, vmax=vmax, vmin=-vmax))

        im = axs[1].pcolormesh(self.T, self.J, future_grid - rad_mean_amp,
                               cmap=cmap, rasterized=True, shading='nearest',
                               norm=mpl.colors.SymLogNorm(linthresh=linthresh, vmax=vmax, vmin=-vmax))

        axs[0].set_title(f'Time = {tstep*self.tstep_diff:.2f} Gyr', pad=10, fontsize=14)
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
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('One Armed Phase Spiral Amplitude')
            if savefig:
                plt.savefig(fig_dir + f'/rewind_t{int(tstep)}_{self.pc_string}.pdf')
            plt.close()
        else:
            fig.colorbar(im, ax=axs[1], label='One Armed Phase Spiral Amplitude',
                         shrink=0.8)
