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

    def fit_macro_spiral(self, tstep, threshold=np.pi):
        # get the array of reconstructed amplitudes from the PCs
        self.tstep_data = np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F')

        #get angle of maximum amplitude at each Jphi for the given timestep
        max_inds = np.argmax(self.tstep_data, axis=0)
        
        max_angles_ = self.tphi_c[max_inds]
        self.max_angles = 1/self.m * np.unwrap(self.m*max_angles_)

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

        # jphi1, jphi2 = self.jphi_c_fit[0], self.jphi_c_fit[-1]
        #get the angle at two chosen radii
        # theta1, theta2 = self.macro_spl(jphi1), self.macro_spl(jphi2)

        #get the frequencies at those radii
        # omega_spline = make_splrep(self.jphi_c, self.omega_phi, w=None, k=3, s=0)
        # omega1, omega2 = omega_spline(jphi1), omega_spline(jphi2)
        omega1, omega2 = self.omega_phi_fit[0], self.omega_phi_fit[-1]
        theta1, theta2 = self.macro_spl_omega(omega1), self.macro_spl_omega(omega2)

        #calculate the winding time
        tfit = (theta2 - theta1) / (omega2 - omega1)
        return tfit

    def plot_fitting_tstep(self, tstep, threshold=np.pi, 
                           ax=None, savefig=False, fig_dir=None):
        try:
            self.fit_macro_spiral(tstep, threshold=threshold)
        except:
            print(f"Error occurred while fitting macro spiral at tstep {tstep}")
            return

        # j_fit = np.linspace(self.jphi_c_fit[0], self.jphi_c_fit[-1], 100)
        omega_fit = np.linspace(self.omega_phi_fit[0], self.omega_phi_fit[-1], 100)
        # theta_fit = self.macro_spl(j_fit)
        theta_fit = self.macro_spl_omega(omega_fit)

        cmap = cmr.sunburst
        if np.all(self.tstep_data > 0):
            norm = mpl.colors.LogNorm(vmin=np.min(self.tstep_data), 
                                      vmax=np.max(self.tstep_data))
        else:
            vmax = np.max(np.abs(self.tstep_data))
            norm = mpl.colors.SymLogNorm(linthresh=vmax/1e2, vmin=-vmax, vmax=vmax)
            cmap = cmr.holly

        if ax is None:
            plt.figure(figsize=(8,5))
            ax = plt.gca()
        # ax.pcolormesh(self.J, self.T, self.tstep_data, cmap=cmap, norm=norm, alpha=0.3)
        ax.pcolormesh(self.O, self.T2, self.tstep_data, cmap=cmap, norm=norm, alpha=0.3)
        # ax.plot(self.jphi_c, self.max_angles % (2*np.pi), 'o', c='k', ms=5, 
        #         label='Max Amplitude Angles')
        ax.plot(self.omega_phi, self.max_angles % (2*np.pi), 'o', c='k', ms=5, 
                label='Max Amplitude Angles')
        # ax.plot(j_fit, theta_fit % (2*np.pi), '-', 
        #         label='Cubic Spline Fit')
        ax.plot(omega_fit, theta_fit % (2*np.pi), '-', 
                label='Cubic Spline Fit')
        # ax.axvline(self.jphi_c_fit[0], color='r', linestyle='--', 
        #            label=f'j={self.jphi_c_fit[0]}')
        # ax.axvline(self.jphi_c_fit[-1], color='g', linestyle='--', 
        #            label=f'j={self.jphi_c_fit[-1]}')
        ax.axvline(self.omega_phi_fit[0], color='r', linestyle='--', 
                   label=f'j={self.jphi_c_fit[0]}')
        ax.axvline(self.omega_phi_fit[-1], color='g', linestyle='--', 
                   label=f'j={self.jphi_c_fit[-1]}')
        # ax.set_xlabel(r'$J_\phi$')
        ax.set_xlabel(r'$\Omega_\phi$')
        ax.set_ylabel(r'$\theta_{max}$')
        ax.set_title('Fitting the Macro Spiral')
        ax.legend()
        if savefig:
            plt.savefig(fig_dir + f'/macro_fit_t{int(tstep)}_{self.pc_string}.pdf')
        plt.close()

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
        ax.set_xlabel('Simulation Time (Gyr)')
        ax.set_ylabel('Winding Time (Gyr)')

        ax.set_title(f'{self.sim_name} {self.channel_name} tfit for {self.pc_string}')
        ax.set_aspect('equal')
        ax.set_xlim(0,ntimes*self.tstep_diff)
        ax.set_ylim(0,ntimes*self.tstep_diff)
        ax.legend()
        if savefig:
            plt.savefig(fig_dir + f'/winding_times_{self.pc_string}.pdf')
        plt.close()
        print(f'Saved winding time fit figure for {self.pc_string}')

    def make_rewind_dipole_fig(self, tstep, axs=None, savefig=False, fig_dir=None):

        tfit = self.derive_winding_time(tstep, threshold=np.pi/2)
        if tfit == np.nan:
            print(f"Could not derive winding time for tstep {tstep}, \
                  skipping rewind dipole figure.")
            return
        
        rad_mean_amp = np.mean(np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F'), axis=0)

        _, vmax, linthresh = diagnostics.compute_pc_limits(self.pc_rc, self.T.shape)

        cmap=cmr.holly
        if axs==None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), 
                                    subplot_kw={'projection': 'polar'})
        
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
        
        axs[0].set_title(f'Timestep {tstep}', pad=10, fontsize=14)
        axs[1].set_title(f'Rewind {tfit} Gyr', 
                         pad=10, fontsize=14)
        
        for ax in axs:
            ax.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)], 
                          labels=[r'$J_\phi=1000$', r'$J_\phi=3000$'])
            ax.set_rmax(np.max(self.jphi_c))
            ax.tick_params(left = False, right = False , labelleft = True ,
                                labelbottom = False, bottom = False)      
            ax.grid(visible=False)
        
        fig.suptitle(f'Rewinding to dipole, {self.pc_string}', fontsize=16)
        
        fig.tight_layout()
        
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('One Armed Phase Spiral Amplitude')
        if savefig:
            plt.savefig(fig_dir + f'/rewind_t{int(tstep)}_{self.pc_string}.pdf')
        plt.close()

    def fit_all_pc_macro_spirals(mssa, list_of_pc_lists, jphi_min, jbins, sim_name, channel_name, 
                             INDIVIDUAL_WINDING_DIR, DIPOLE_DIR, WINDING_DIR):
        for pc_list in list_of_pc_lists:

            mssa.reconstruct(pc_list)
            get_recon = mssa.getReconstructed()
            pc_rc = get_recon[list(get_recon.keys())[0]].getAllCoefs()
            MS = RewindMacroSpiral(pc_rc, pc_list, jphi_min, jbins, sim_name, channel_name, m=1)
            MS.plot_macro_tfit_over_time(threshold=np.pi/2, savefig=True, fig_dir=WINDING_DIR)

            PC_WINDING_DIR = os.path.join(INDIVIDUAL_WINDING_DIR, MS.pc_string)
            os.makedirs(PC_WINDING_DIR, exist_ok=True)
            
            PC_DIPOLE_DIR = os.path.join(DIPOLE_DIR, MS.pc_string)
            os.makedirs(PC_DIPOLE_DIR, exist_ok=True)

            for tstep in range(pc_rc.shape[1]):
                MS.plot_fitting_tstep(tstep, threshold=np.pi/2, savefig=True, fig_dir=PC_WINDING_DIR)
                MS.make_rewind_dipole_fig(tstep, savefig=True, fig_dir=PC_DIPOLE_DIR)
