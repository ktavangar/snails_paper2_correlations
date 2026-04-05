import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2 #set the value globally
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import cmasher as cmr

import numpy as np
import os

from functools import partial


class MakeAnimations:
    def __init__(self, mssa, sim_name, channel_name, times, jphi_min, jbins):
        self.mssa = mssa
        self.times = times
        self.sim_name = sim_name
        self.channel_name = channel_name
        
        jphi_c = np.linspace(jphi_min, jphi_min+((jbins-1)*100), jbins)
        self.jphi_c = jphi_c
        tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
        rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
        self.tphi_c = tphi_c_[:-1] + rad[1]
        self.J, self.T = np.meshgrid(jphi_c, self.tphi_c)

        if sim_name == 'B2':
            self.tstep_diff = 0.009778
        elif sim_name == 'test':
            self.tstep_diff = 0.01

    def make_data_mov(self, filename, data_tbl, norm_function=mpl.colors.Normalize, cmap='Reds', **kwargs):

        fig, ax = plt.subplots(1, 1, figsize=(7, 6), subplot_kw={'projection':'polar'})
        im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap,
                           norm=norm_function(**kwargs))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.channel_name, fontsize=20)
        if 'ratio' in self.channel_name:
            cbar.ax.text(3.5, 0.8, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=14)
            cbar.ax.text(3.5, 0.2, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=14)
            
        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            partial(self.animate_data, data_tbl=data_tbl, ax=ax, norm_function=norm_function, cmap=cmap, **kwargs),
            frames=np.arange(0, int(len(self.times)), 1),
            interval=20,
            blit=False,
        )
        FFwriter = animation.FFMpegWriter(fps=10)
        anim.save(filename, writer = FFwriter)
        plt.close()

    def animate_data(self, timestep, data_tbl, ax, norm_function, cmap, **kwargs):
        ax.clear()
        
        first_timestep = self.times[0]
        
        if 'pitch' in self.channel_name:
            dat = -np.reshape(data_tbl[timestep,1:], self.T.shape, 'F')
        else:
            dat = np.reshape(data_tbl[timestep,1:], self.T.shape, 'F')
        
        im = ax.pcolormesh(self.T, self.J, dat, cmap=cmap,
                            norm=norm_function(**kwargs))
            
        ax.set_title(self.channel_name, fontsize=20)
        ax.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)], 
                      labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], 
                      fontsize=18, color='k')
        ax.set_rmax(np.max(self.jphi_c))
        ax.tick_params(left = False, right = False , labelleft = True ,
                             labelbottom = False, bottom = False)      
        ax.grid(visible=False)
        ax.set_rlabel_position(50)

        ax.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*self.tstep_diff, 2)), 
                 fontsize=20, ha="center", c='k')

        plt.draw()

    def reconstruct_from_pcs(self, pcs):
        self.pcs = pcs
        self.mssa.reconstruct(self.pcs)
        get_reconstructed = self.mssa.getReconstructed()

        #reconstructed principal components
        self.pc_rc = get_reconstructed[list(get_reconstructed.keys())[0]].getAllCoefs()

    def create_pc_movie_filename(self, file_directory, subtract_mean=False):
        if len(self.pcs) == 1:
            pc_string = 'pc{}'.format(self.pcs[0])
        elif np.any(np.diff(self.pcs) > 1): # non-continuous pcs
            pc_string = 'pc{}'.format(self.pcs)
        else: # continuous pcs
            pc_string = 'pc{}-{}'.format(self.pcs[0], self.pcs[-1])

        self.pc_string = pc_string

        if subtract_mean:
            filename = file_directory+pc_string+'_mean_subtracted.mp4'
        else:
            filename = file_directory+pc_string+'.mp4'
            
        return filename

    def make_pc_reconstruction_mov(self, file_directory, subtract_mean=False, norm_function=mpl.colors.Normalize, cmap=cmr.holly, **kwargs):

        filename = self.create_pc_movie_filename(file_directory, subtract_mean)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), subplot_kw={'projection': 'polar'})

        im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap,
                                      norm=norm_function(**kwargs))

        # Create the colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self.channel_name)
        
        if 'pitch' in self.channel_name:
            cbar.ax.text(0.5, 1.02, 'Less wound up', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=15)
            cbar.ax.text(0.5, -0.02, 'More wound up', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=15)
        elif 'ratio' in self.channel_name:
            cbar.ax.text(3.5, 0.8, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=14)
            cbar.ax.text(3.5, 0.2, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=14)
            cbar.set_label('m1 Amp / m2 Amp', fontsize=18)
        
        fig.tight_layout()
        
        anim = FuncAnimation(
            fig,
            partial(self.animate_pc_reconstruction, ax=ax, subtract_mean=subtract_mean, norm_function=norm_function, cmap=cmap, **kwargs),
            frames=np.arange(0, int(len(self.times)), 1),
            interval=20,
            blit=False,
        )
        FFwriter = animation.FFMpegWriter(fps=10)
            
        anim.save(filename, writer = FFwriter)
        plt.close()


    def animate_pc_reconstruction(self, timestep, ax, subtract_mean=False, norm_function=mpl.colors.Normalize, cmap=cmr.holly, **kwargs):
        ax.clear()
        first_timestep = self.times[0]
        if "pitch" in self.channel_name:
            dat = -np.reshape(self.pc_rc[:,timestep], self.T.shape, 'F')
        else:
            dat = np.reshape(self.pc_rc[:,timestep], self.T.shape, 'F')

        if subtract_mean:
            #get the mean at each radius
            radial_mean = np.mean(dat, axis=0)
            dat -= radial_mean
        
        im1 = ax.pcolormesh(self.T, self.J, dat, cmap=cmap, norm=norm_function(**kwargs))
        
        ax.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)], 
                      labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], 
                      fontsize=18, color='k')
        ax.set_rmax(np.max(self.jphi_c))
        ax.tick_params(left = False, right = False , labelleft = True ,
                             labelbottom = False, bottom = False)
        ax.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*self.tstep_diff, 2)), fontsize=20, 
            ha="center", c='k')
        ax.grid(False)
        ax.set_rlabel_position(50)

        if subtract_mean:
            ax.set_title(self.channel_name + ', ' + self.pc_string + '(mean-subtracted)', fontsize=15)
        else:
            ax.set_title(self.channel_name + ', ' + self.pc_string, fontsize=15)

        plt.draw()

    # ------------------------------------------------------------------
    # Dual-panel (polar + Cartesian) movies
    # ------------------------------------------------------------------

    def make_dual_data_mov(self, filename, data_tbl, norm_function=mpl.colors.Normalize, cmap='Reds', **kwargs):
        """Animate raw data: left = polar face-on, right = Cartesian (J_phi vs theta_phi)."""
        fig = plt.figure(figsize=(14, 6))
        ax_pol = fig.add_subplot(1, 2, 1, projection='polar')
        ax_cart = fig.add_subplot(1, 2, 2)

        norm = norm_function(**kwargs)
        ax_pol.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap, norm=norm)
        im = ax_cart.pcolormesh(self.J, self.T, np.ones(self.T.shape), cmap=cmap, norm=norm)

        cbar = fig.colorbar(im, ax=ax_cart, location='right')
        cbar.set_label(self.channel_name, fontsize=16)

        ax_pol.set_rmax(np.max(self.jphi_c))
        ax_pol.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)],
                          labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], fontsize=13, color='k')
        ax_pol.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
        ax_pol.grid(visible=False)
        ax_pol.set_rlabel_position(50)

        ax_cart.set_xlabel(r'$J_\phi$ [kpc km/s]', fontsize=14)
        ax_cart.set_ylabel(r'$\theta_\phi$ [rad]', fontsize=14)
        ax_cart.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                           labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)
        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            partial(self._animate_dual_data, data_tbl=data_tbl, ax_pol=ax_pol, ax_cart=ax_cart,
                    norm_function=norm_function, cmap=cmap, **kwargs),
            frames=np.arange(0, int(len(self.times)), 1),
            interval=20, blit=False,
        )
        anim.save(filename, writer=animation.FFMpegWriter(fps=10))
        plt.close()

    def _animate_dual_data(self, timestep, data_tbl, ax_pol, ax_cart, norm_function, cmap, **kwargs):
        ax_pol.clear()
        ax_cart.clear()
        if 'pitch' in self.channel_name:
            dat = -np.reshape(data_tbl[timestep, 1:], self.T.shape, 'F')
        else:
            dat = np.reshape(data_tbl[timestep, 1:], self.T.shape, 'F')

        norm = norm_function(**kwargs)
        ax_pol.pcolormesh(self.T, self.J, dat, cmap=cmap, norm=norm)
        ax_cart.pcolormesh(self.J, self.T, dat, cmap=cmap, norm=norm)

        t_gyr = np.around((self.times[0] + timestep) * self.tstep_diff, 2)
        ax_pol.set_rmax(np.max(self.jphi_c))
        ax_pol.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)],
                          labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], fontsize=13, color='k')
        ax_pol.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
        ax_pol.grid(visible=False)
        ax_pol.set_rlabel_position(50)

        ax_cart.set_xlabel(r'$J_\phi$ [kpc km/s]', fontsize=14)
        ax_cart.set_ylabel(r'$\theta_\phi$ [rad]', fontsize=14)
        ax_cart.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                           labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)

        ax_cart.set_title(r't = {} Gyr'.format(t_gyr), fontsize=16)
        plt.draw()

    def make_dual_pc_reconstruction_mov(self, file_directory, subtract_mean=False,
                                        norm_function=mpl.colors.Normalize,
                                        cmap=cmr.holly, **kwargs):
        """Dual-panel PC reconstruction movie: left = polar, right = Cartesian."""
        filename = self.create_pc_movie_filename(file_directory, subtract_mean)
        filename = filename.replace('.mp4', '_dual.mp4')

        fig = plt.figure(figsize=(14, 6))
        ax_pol = fig.add_subplot(1, 2, 1, projection='polar')
        ax_cart = fig.add_subplot(1, 2, 2)

        norm = norm_function(**kwargs)
        ax_pol.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmap, norm=norm)
        im = ax_cart.pcolormesh(self.J, self.T, np.ones(self.T.shape), cmap=cmap, norm=norm)

        cbar = fig.colorbar(im, ax=ax_cart, location='right')
        cbar.set_label(self.channel_name, fontsize=16)

        ax_pol.set_rmax(np.max(self.jphi_c))
        ax_pol.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)],
                          labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], fontsize=13, color='k')
        ax_pol.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
        ax_pol.grid(visible=False)
        ax_pol.set_rlabel_position(50)

        ax_cart.set_xlabel(r'$J_\phi$ [kpc km/s]', fontsize=14)
        ax_cart.set_ylabel(r'$\theta_\phi$ [rad]', fontsize=14)
        ax_cart.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                           labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)
        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            partial(self._animate_dual_pc_reconstruction, ax_pol=ax_pol, ax_cart=ax_cart,
                    subtract_mean=subtract_mean, norm_function=norm_function, cmap=cmap, **kwargs),
            frames=np.arange(0, int(len(self.times)), 1),
            interval=20, blit=False,
        )
        anim.save(filename, writer=animation.FFMpegWriter(fps=10))
        plt.close()

    def _animate_dual_pc_reconstruction(self, timestep, ax_pol, ax_cart, subtract_mean=False,
                                        norm_function=mpl.colors.Normalize,
                                        cmap=cmr.holly, **kwargs):
        ax_pol.clear()
        ax_cart.clear()
        if 'pitch' in self.channel_name:
            dat = -np.reshape(self.pc_rc[:, timestep], self.T.shape, 'F')
        else:
            dat = np.reshape(self.pc_rc[:, timestep], self.T.shape, 'F')

        if subtract_mean:
            dat -= np.mean(dat, axis=0)

        norm = norm_function(**kwargs)
        ax_pol.pcolormesh(self.T, self.J, dat, cmap=cmap, norm=norm)
        ax_cart.pcolormesh(self.J, self.T, dat, cmap=cmap, norm=norm)

        ax_pol.set_rmax(np.max(self.jphi_c))
        ax_pol.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)],
                          labels=[r'$J_\phi=1000$', r'$J_\phi=4000$'], fontsize=13, color='k')
        ax_pol.tick_params(left=False, right=False, labelleft=True, labelbottom=False, bottom=False)
        ax_pol.grid(False)
        ax_pol.set_rlabel_position(50)

        ax_cart.set_xlabel(r'$J_\phi$ [kpc km/s]', fontsize=14)
        ax_cart.set_ylabel(r'$\theta_\phi$ [rad]', fontsize=14)
        ax_cart.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                           labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)

        t_gyr = np.around((self.times[0] + timestep) * self.tstep_diff, 2)
        suffix = ' (mean-subtracted)' if subtract_mean else ''
        ax_cart.set_title(r't = {} Gyr  —  {}{}'.format(t_gyr, self.pc_string, suffix), fontsize=13)
        plt.draw()


def suggest_pc_groups(mssa, n_pcs, threshold=0.5):
    """
    Suggest PC groupings by thresholding the W-correlation matrix and finding
    connected components.

    Parameters
    ----------
    mssa      : pyEXP mSSA object
    n_pcs     : int — number of PCs to consider
    threshold : float — wCorr value above which two PCs are considered grouped (default 0.5)

    Returns
    -------
    groups : list of lists — each inner list is a suggested PC group, sorted by first index
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    wcorr = mssa.wCorrAll()[:n_pcs, :n_pcs]
    adj = (wcorr > threshold).astype(int)
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
        pc_str = str(group) if len(group) <= 6 else f'[{group[0]}..{group[-1]}] ({len(group)} PCs)'
        ev_range = f'{ev[group[0]]:.3e} – {ev[group[-1]]:.3e}'
        print(f'  {i:<8} {pc_str:<40} {ev_range}')
    print()


def compute_pc_limits(pc_rc, T_shape, fallback_vmin=None, fallback_vmax=None):
    """
    Compute appropriate vmin/vmax for face-on PC reconstruction movies.

    Parameters
    ----------
    pc_rc : ndarray, shape (n_channels, n_times)
        Reconstructed PC data from MakeAnimations.pc_rc.
    T_shape : tuple
        Shape of the theta/J meshgrid (MakeAnimations.T.shape), used for mean subtraction.
    fallback_vmin, fallback_vmax : float, optional
        Values to use for LogNorm limits if the reconstruction is near-zero (e.g. noise PCs).

    Returns
    -------
    vmin_pc, vmax_pc : float
        Limits for LogNorm (non-mean-subtracted movie).
    """
    # Non-mean-subtracted (LogNorm): percentiles of positive values
    pos_vals = pc_rc[pc_rc > 0]
    if len(pos_vals) > 10:
        vmin_pc = np.percentile(pos_vals, 1)
        vmax_pc = np.percentile(pos_vals, 99)
    else:
        vmin_pc = fallback_vmin
        vmax_pc = fallback_vmax

    # Mean-subtracted (SymLogNorm): percentiles of mean-subtracted values across all times
    ms_vals = []
    for t in range(pc_rc.shape[1]):
        dat = np.reshape(pc_rc[:, t], T_shape, 'F')
        ms_vals.append((dat - np.mean(dat, axis=0)).flatten())
    ms_all = np.concatenate(ms_vals)
    abs_ms = np.abs(ms_all[ms_all != 0])
    # linthresh_ms = np.percentile(abs_ms, 10) if len(abs_ms) > 0 else vmax_pc * 0.01
    linthresh_ms = vmax_pc * 0.01

    return vmin_pc, vmax_pc, linthresh_ms



# ---------------------------------------------------------------------------
# Diagnostic plot functions
# ---------------------------------------------------------------------------

def plot_eigenvalues(ev, fig_dir):
    """Semilog plot of mSSA eigenvalues. Saves to fig_dir/eigenvalues.pdf."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.semilogy(ev, '-o')
    ax.set_xlabel('index', fontsize=20)
    ax.set_ylabel('eigenvalue', fontsize=20)
    ax.set_title('PC Eigenvalues', fontsize=20)
    plt.savefig(os.path.join(fig_dir, 'eigenvalues.pdf'))
    plt.close()
    print('Saved eigenvalues plot.')


def plot_fg_matrices(mssa, fig_dir):
    """F and G contribution matrices. Saves to fig_dir/F_matrix.pdf and G_matrix.pdf."""
    import cmasher as cmr
    t1, t2 = mssa.contrib()

    plt.figure(figsize=(20, 5))
    plt.imshow(t2, aspect='auto', norm=mpl.colors.LogNorm(), cmap=cmr.freeze)
    plt.xlabel('Input Channel')
    plt.ylabel('Principal Component')
    plt.title('G Matrix')
    plt.savefig(os.path.join(fig_dir, 'G_matrix.pdf'))
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.imshow(t1, aspect='auto', norm=mpl.colors.LogNorm(), cmap=cmr.freeze)
    plt.xlabel('Input Channel')
    plt.ylabel('Principal Component')
    plt.title('F Matrix')
    plt.savefig(os.path.join(fig_dir, 'F_matrix.pdf'))
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

    Parameters
    ----------
    n_pcs : int
        Number of PCs to plot (default 20).
    """
    pc = mssa.getPC()
    lag_times = times[:pc.shape[0]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    for i in range(n_pcs):
        ax.plot(lag_times, pc[:, i] / np.max(np.abs(pc[:, i])) - 2 * i, label=str(i))
    ax.legend()
    ax.set_xlim(np.min(lag_times), 1.2 * np.max(lag_times))
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Normalized PC amplitude (offset)', fontsize=14)
    ax.set_title('PC Time Series', fontsize=18)
    plt.savefig(os.path.join(fig_dir, 'pc_time_series.pdf'))
    plt.close()
    print('Saved PC time series.')


def make_face_on_movies(mssa, data_file, times, face_on_dir, list_of_pc_lists,
                        sim_name, channel_name, jphi_min, jbins,
                        cmap_log=None, cmap_sym=None):
    """
    Generate all face-on movies: raw data + one pair (LogNorm, mean-subtracted SymLogNorm)
    per PC group. vmin/vmax are computed automatically from the data.

    Parameters
    ----------
    data_file      : str — path to the .dat coefficient file (for the data movie)
    list_of_pc_lists : list of lists — each inner list is a PC group to reconstruct
    cmap_log       : colormap for LogNorm movies (default cmr.sunburst)
    cmap_sym       : colormap for mean-subtracted SymLogNorm movies (default cmr.holly)
    """
    import cmasher as cmr
    if cmap_log is None:
        cmap_log = cmr.sunburst
    if cmap_sym is None:
        cmap_sym = cmr.holly

    data = np.loadtxt(data_file)
    MakeAnim = MakeAnimations(mssa, sim_name=sim_name, channel_name=channel_name,
                              times=times, jphi_min=jphi_min, jbins=jbins)

    # Data movie: vmin/vmax from percentiles of positive values (LogNorm requires positive)
    data_pos = data[:, 1:].flatten()
    data_pos = data_pos[data_pos > 0]
    data_vmin = np.percentile(data_pos, 1)
    data_vmax = np.percentile(data_pos, 99)
    print(f'Data movie: vmin={data_vmin:.1f}, vmax={data_vmax:.1f}')

    MakeAnim.make_data_mov(os.path.join(face_on_dir, 'data.mp4'), data,
                           norm_function=mpl.colors.LogNorm,
                           vmin=data_vmin, vmax=data_vmax, cmap=cmap_log)

    for pc_list in list_of_pc_lists:
        print(f'  Creating movie for PCs {pc_list}')
        MakeAnim.reconstruct_from_pcs(pcs=pc_list)

        vmin_pc, vmax_pc, linthresh_ms = compute_pc_limits(
            MakeAnim.pc_rc, MakeAnim.T.shape,
            fallback_vmin=data_vmin, fallback_vmax=data_vmax)
        print(f'    LogNorm: [{vmin_pc:.2f}, {vmax_pc:.2f}]  '
              f'SymLogNorm linthresh={linthresh_ms:.2f}')

        MakeAnim.make_pc_reconstruction_mov(
            file_directory=face_on_dir + '/', subtract_mean=False,
            norm_function=mpl.colors.LogNorm, cmap=cmap_log,
            vmin=vmin_pc, vmax=vmax_pc)
        MakeAnim.make_pc_reconstruction_mov(
            file_directory=face_on_dir + '/', subtract_mean=True,
            norm_function=mpl.colors.SymLogNorm, cmap=cmap_sym,
            vmin=-vmax_pc, vmax=vmax_pc, linthresh=linthresh_ms)


def make_dual_movies(mssa, data_file, times, face_on_dir, list_of_pc_lists,
                     sim_name, channel_name, jphi_min, jbins,
                     cmap_log=None, cmap_sym=None):
    """
    Generate dual-panel (polar + Cartesian) movies: raw data + one pair per PC group.
    Files are saved with a '_dual.mp4' suffix alongside the polar-only movies.
    """
    import cmasher as cmr
    if cmap_log is None:
        cmap_log = cmr.sunburst
    if cmap_sym is None:
        cmap_sym = cmr.holly

    data = np.loadtxt(data_file)
    MakeAnim = MakeAnimations(mssa, sim_name=sim_name, channel_name=channel_name,
                              times=times, jphi_min=jphi_min, jbins=jbins)

    data_pos = data[:, 1:].flatten()
    data_pos = data_pos[data_pos > 0]
    data_vmin = np.percentile(data_pos, 1)
    data_vmax = np.percentile(data_pos, 99)
    print(f'Dual data movie: vmin={data_vmin:.1f}, vmax={data_vmax:.1f}')

    MakeAnim.make_dual_data_mov(os.path.join(face_on_dir, 'data_dual.mp4'), data,
                                norm_function=mpl.colors.LogNorm,
                                vmin=data_vmin, vmax=data_vmax, cmap=cmap_log)

    for pc_list in list_of_pc_lists:
        print(f'  Creating dual movie for PCs {pc_list}')
        MakeAnim.reconstruct_from_pcs(pcs=pc_list)

        vmin_pc, vmax_pc, linthresh_ms = compute_pc_limits(
            MakeAnim.pc_rc, MakeAnim.T.shape,
            fallback_vmin=data_vmin, fallback_vmax=data_vmax)

        MakeAnim.make_dual_pc_reconstruction_mov(
            file_directory=face_on_dir + '/', subtract_mean=False,
            norm_function=mpl.colors.LogNorm, cmap=cmap_log,
            vmin=vmin_pc, vmax=vmax_pc)
        MakeAnim.make_dual_pc_reconstruction_mov(
            file_directory=face_on_dir + '/', subtract_mean=True,
            norm_function=mpl.colors.SymLogNorm, cmap=cmap_sym,
            vmin=-vmax_pc, vmax=vmax_pc, linthresh=linthresh_ms)