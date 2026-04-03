import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import scipy
from scipy.interpolate import make_splrep
import diagnostics

# use pc reconstruction at a given timestep to derive the time of the perturbation
class RewindMacroSpiral():
    def __init__(self, pc_rc, pcs, jphi_min, jbins, sim_name, channel_name='one-armed amplitude', m=1):

        jphi_c = np.linspace(jphi_min, jphi_min+((jbins-1)*100), jbins)
        self.jphi_c = jphi_c
        tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
        rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
        self.tphi_c = tphi_c_[:-1] + rad[1]
        self.J, self.T = np.meshgrid(jphi_c, self.tphi_c)

        self.sim_name = sim_name
        if (sim_name == 'B2') | (sim_name == 'live'):
            self.tstep_diff = 0.009778
            freqs = np.load('fill in with correct file') # TODO: add correct frequency array for B2 and live runs
        elif sim_name == 'test':
            self.tstep_diff = 0.01
            freqs = np.load('test_frequency_array_j25.npy') # TODO: check filepath
        self.omega_phi = freqs*2*np.pi
        self.channel_name = channel_name

        self.m = m

        self.pc_rc = pc_rc
        self.pcs = pcs

        if len(self.pcs) == 1:
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

        fit_start_ind, fit_end_ind = find_fitting_interval(self.max_angles, threshold=threshold)
        self.jphi_c_fit = self.jphi_c[fit_start_ind:fit_end_ind]
        max_angles_fit = self.max_angles[fit_start_ind:fit_end_ind]

        self.macro_spl = make_splrep(self.jphi_c_fit, max_angles_fit, 
                        w=None, k=3, s=len(self.jphi_c_fit))

    def derive_perturbation_time(self, tstep, threshold=np.pi):
        self.fit_macro_spiral(tstep, threshold=threshold)
        jphi1, jphi2 = self.jphi_c_fit[0], self.jphi_c_fit[-1]
        #get the angle at two chosen radii
        theta1, theta2 = self.macro_spl(jphi1), self.macro_spl(jphi2)

        #get the frequencies at those radii
        omega_spline = make_splrep(self.jphi_c, self.omega_phi, w=None, k=3, s=0)
        omega1, omega2 = omega_spline(jphi1), omega_spline(jphi2)

        #calculate the time of the perturbation using the formula derived from the phase spiral winding
        t_perturb = (theta2 - theta1) / (omega2 - omega1)
        return t_perturb

    def plot_fitting_step(self, tstep, threshold=np.pi, ax=None, savefig=False, fig_dir=None): 
        self.fit_macro_spiral(tstep, threshold=threshold)
        
        j_fit = np.linspace(self.jphi_c_fit[0], self.jphi_c_fit[-1], 100)
        theta_fit = self.macro_spl(j_fit)

        cmap = cmr.holly
        norm = mpl.colors.LogNorm(vmin=np.min(self.tstep_data), vmax=np.max(self.tstep_data))
        
        if ax is None:
            plt.figure(figsize=(8,5))
            ax = plt.gca()
        ax.pcolormesh(self.J, self.T, self.tstep_data, cmap=cmap, norm=norm, alpha=0.3)
        ax.plot(self.jphi_c, self.max_angles % (2*np.pi), 'o', c='k', ms=5, label='Max Amplitude Angles')
        ax.plot(j_fit, theta_fit % (2*np.pi), '-', label='Cubic Spline Fit')
        ax.axvline(self.jphi_c_fit[0], color='r', linestyle='--', label=f'j={self.jphi_c_fit[0]}')
        ax.axvline(self.jphi_c_fit[-1], color='g', linestyle='--', label=f'j={self.jphi_c_fit[-1]}')
        ax.set_xlabel(r'$J_\phi$')
        ax.set_ylabel(r'$\theta_{max}$')
        ax.set_title('Fitting the Macro Spiral')
        ax.legend()
        if savefig:
            plt.savefig(fig_dir + f'macro_fit_t{int(tstep)}_{self.pc_string}.pdf')
        plt.show()

    def plot_macro_tfit_over_time(self, threshold=np.pi, ax=None, savefig=False, fig_dir=None):
        ntimes = self.pc_rc.shape[-1]

        tfits = np.zeros(ntimes)
        for tstep in range(ntimes):
            tfits[tstep] = self.derive_perturbation_time(tstep, threshold=threshold)

        
        if ax is None:
            plt.figure(figsize=(5,5))
            ax = plt.gca()
        ax.plot(np.arange(ntimes)*self.tstep_diff, tfits, 'o', c='blue', ms=2, label='Fitted Perturbation Time')
        ax.plot([0,ntimes*self.tstep_diff], [0,ntimes*self.tstep_diff], 'k--', label='t_fit = t')
        ax.set_xlabel('Simulation Time (Gyr)')
        ax.set_ylabel('Winding Time (Gyr)')

        ax.set_title(f'{self.sim_name} {self.channel_name} winding times for {self.pc_string}')
        ax.set_aspect('equal')
        ax.set_xlim(0,ntimes*self.tstep_diff)
        ax.set_ylim(0,ntimes*self.tstep_diff)
        ax.legend()
        if savefig:
            plt.savefig(fig_dir + f'winding_times_{self.pc_string}.pdf')
        plt.show()

    def make_rewind_dipole_fig(self, tstep, axs=None, savefig=False, fig_dir=None):

        rewind_time = self.derive_perturbation_time(tstep, threshold=np.pi/2)
        dipole_tstep = np.max([int(tstep - rewind_time*100), 0])
        rad_mean_amp_dipole = np.mean(np.reshape(self.pc_rc[:,dipole_tstep], self.T.shape, 'F'), axis=0)
        rad_mean_amp_future = np.mean(np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F'), axis=0)

        _, vmax, linthresh = diagnostics.compute_pc_limits(self.pc_rc, self.T.shape)

        cmap=cmr.holly
        if axs==None:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})

        im1 = axs[0].pcolormesh(self.T, self.J, np.reshape(self.pc_rc[:,dipole_tstep], self.T.shape, 'F') - rad_mean_amp_dipole, 
                            cmap=cmap, rasterized=True,
                            norm=mpl.colors.SymLogNorm(linthresh=linthresh, vmax=vmax, vmin=-vmax), shading='nearest')
        
        
        time_elapsed = (tstep - dipole_tstep) * self.tstep_diff
        d_rot = (time_elapsed * self.omega_phi)
        
        #now interpolate at each radius so that I can use the same grid as before
        interp = scipy.interpolate.interp1d(self.tphi_c, self.pc_rc[:,tstep].reshape(self.T.T.shape), axis=1, 
                                            fill_value="extrapolate")
        all_interp = interp((self.T+d_rot)%(2*np.pi))
        future_grid = np.diagonal(all_interp, axis1=0, axis2=2)
        
        #plot what that timestep looks like after subtracting background
        im2 = axs[1].pcolormesh(self.T, self.J, np.reshape(self.pc_rc[:,tstep], self.T.shape, 'F') - rad_mean_amp_future, 
                            cmap=cmap, rasterized=True,
                            norm=mpl.colors.SymLogNorm(linthresh=linthresh, vmax=vmax, vmin=-vmax), shading='nearest')

        im3 = axs[2].pcolormesh(self.T, self.J, future_grid - rad_mean_amp_future, cmap=cmap, rasterized=True,
                        norm=mpl.colors.SymLogNorm(linthresh=linthresh, vmax=vmax, vmin=-vmax), shading='nearest')
        
        axs[0].set_title(r'$T_{int}$', pad=10, fontsize=14)
        t_passed_str = str((tstep-dipole_tstep)*self.tstep_diff)
        axs[1].set_title(r'$T - T_{int}$' + r'$ = {}$ Gyr'.format(t_passed_str), pad=10, fontsize=14)
        axs[2].set_title(r'Rewind from $T - T_{int}$' + r'$ = {}$ Gyr to Interaction Time'.format(t_passed_str), 
                         pad=10, fontsize=14)
        
        for ax in axs:
            ax.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)], labels=[r'$J_\phi=1000$', r'$J_\phi=3000$'])
            ax.set_rmax(np.max(self.jphi_c))
            ax.tick_params(left = False, right = False , labelleft = True ,
                                labelbottom = False, bottom = False)      
            ax.grid(visible=False)
        
        fig.suptitle(f'Rewinding from t={tstep*self.tstep_diff:.2f} Gyr, {self.pc_string}', fontsize=16)
        
        fig.tight_layout()
        
        
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.75])
        cbar = fig.colorbar(im3, cax=cbar_ax)
        cbar.set_label('One Armed Phase Spiral Amplitude')
        if savefig:
            plt.savefig(fig_dir + f'rewind_t{int(tstep)}_{self.pc_string}.pdf')
        plt.show()


def find_fitting_interval(arr, threshold=np.pi/2):
    # Find the longest consecutive region where the array is relatively continuous
    good_indices = np.where(np.abs(np.diff(arr)) <= threshold)[0]
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