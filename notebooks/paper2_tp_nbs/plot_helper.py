import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import scipy

# To get colorbar
jphi_c = np.linspace(1000, 3500, 25+1)
tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
tphi_c = tphi_c_[:-1] + rad[1]
J, T = np.meshgrid(jphi_c, tphi_c)

freqs = np.load('../../data/test_frequency_array_j25.npy')
omega_phi = freqs*2*np.pi

def make_rewind_dipole_fig(dipole_timestep, future_timestep, recon_amp, ratio = 0.01, axs=None):

    
    rad_mean_amp_dipole = np.mean(np.reshape(recon_amp[:,dipole_timestep], T.shape, 'F'), axis=0)
    rad_mean_amp_future = np.mean(np.reshape(recon_amp[:,future_timestep], T.shape, 'F'), axis=0)
    
    cmap=cmr.holly
    if axs==None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})

    im1 = axs[0].pcolormesh(T, J, np.reshape(recon_amp[:,dipole_timestep], T.shape, 'F') - rad_mean_amp_dipole, 
                         cmap=cmap, rasterized=True,
                         norm=mpl.colors.SymLogNorm(linthresh=100, vmax=1e4, vmin=-1e4), shading='nearest')
    
    
    time_elapsed = (future_timestep - dipole_timestep) * ratio
    d_rot = (time_elapsed * omega_phi)
    
    #now interpolate at each radius so that I can use the same grid as before
    interp = scipy.interpolate.interp1d(tphi_c, recon_amp[:,future_timestep].reshape(T.T.shape), axis=1, 
                                        fill_value="extrapolate")
    all_interp = interp((T+d_rot)%(2*np.pi))
    future_grid = np.diagonal(all_interp, axis1=0, axis2=2)
    
    #plot what that timestep looks like after subtracting background
    im2 = axs[1].pcolormesh(T, J, np.reshape(recon_amp[:,future_timestep], T.shape, 'F') - rad_mean_amp_future, 
                         cmap=cmap, rasterized=True,
                         norm=mpl.colors.SymLogNorm(linthresh=100, vmax=1e4, vmin=-1e4), shading='nearest')

    im3 = axs[2].pcolormesh(T, J, future_grid - rad_mean_amp_future, cmap=cmap, rasterized=True,
                    norm=mpl.colors.SymLogNorm(linthresh=100, vmax=1e4, vmin=-1e4), shading='nearest')
    
    axs[0].set_title(r'$T_{int}$', pad=10, fontsize=14)
    axs[1].set_title(r'$T - T_{int} = 0.5$ Gyr', pad=10, fontsize=14)
    axs[2].set_title(r'Rewind from $T - T_{int} = 0.5$ Gyr to Interaction Time', pad=10, fontsize=14)
    
    for ax in axs:
        ax.set_yticks([np.min(jphi_c), np.max(jphi_c)], labels=[r'$J_\phi=1000$', r'$J_\phi=3000$'])
        ax.set_rmax(np.max(jphi_c))
        ax.tick_params(left = False, right = False , labelleft = True ,
                             labelbottom = False, bottom = False)      
        ax.grid(visible=False)
    
    fig.tight_layout()
    
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.75])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('One Armed Phase Spiral Amplitude')
    # plt.savefig(fig_dir + 'rewind_m1_amplitudes_paper.pdf')
    plt.show()