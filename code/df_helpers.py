import scipy
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gala
from itertools import product
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import curve_fit

# from helper import *

def select_action_region(data, center, radius):
    '''
    Select a squareregion in the MW 
    Data is a dataframe object containing at least Galactocentric x and y coordinates
    Center must be in the form [x_coord, y_coord]
    Radius in kpc around the center (not from cneter of MW)
    '''
    select = data[(data.jphi > center[0] - radius[0]) & (data.jphi < center[0] + radius[0]) &
                  (data.theta_phi > center[1] - radius[1]) & (data.theta_phi < center[1] + radius[1])]
    return select

class LaguerreSnails:
    '''
    Class to create and create the BFE reconstruction of a phase-spiral 
    '''
    def __init__(self, data, center, radius, Jz_grid, thetaz_grid, m_max, n_max, time=None):
        '''
        Initialize the class

        Parameters:
        -------------------------------
        data        : (pandas df) Input data
                                  Include at least columns ['jphi', 'jz', 'theta_phi', 'theta_z']
        center      : (tuple) Center of the disk region where we want a BFE reconstruction
                              [J_phi, theta_phi] in [kpc km/s, radians]
        radius      : (tuple) Half the side length of the RECTANGULAR disk region 
                              [J_phi radius, theta_phi radius] in [kpc km/s, radians]
        Jz_grid     : (array) Array of vertical actions at which to calculate the BFE
        thetaz_grid : (array) Array of vertical angles at which to calculate the BFE
        m_max       : (integer) The maximum mode of the Fourier component to generate
        n_max       : (integer) The maximum mode of the Laguerre component to generate
        time        : (integer) Timestep of the simulation
        '''
        self.data = data
        self.time = time
        self.center, self.radius = center, radius
        self.Jz_grid, self.thetaz_grid = Jz_grid, thetaz_grid

        # We will plot sqrt(J_z)sin(theta_z) vs sqrt(J_z)cos(theta_z) to make a spiral
        self.rootjzmax =  np.sqrt(np.max(self.Jz_grid))
        self.rootjzstep = np.sqrt(self.Jz_grid[1]) - np.sqrt(self.Jz_grid[0])
        
        # m_max, n_max are integers, n_maxs is an array of length m_max
        self.m_max, self.n_max = m_max, n_max 
        self.n_maxs = [n_max]*m_max
        
        # array of different Fourier modes included in BFE
        self.ms = np.arange(self.m_max)
        
#         if len(n_maxs) != m_max:
#             raise Exception("The length of n_max must match m_max")
       
        #print('Selecting Region...')
        ## Create a data table for just the region of interest
        self.sel = select_action_region(data, center, radius)

        #calculate best fit value of a for disk_laguerre
        # Laguerre polynomials include a variable factor "a"
        #  Choose the best factor such that the n=0 polynomial best matches 
        #  the vertical exponential profile of the disk
        _,self.a = scipy.stats.expon.fit(self.sel.jz[~np.isnan(self.sel.jz)], floc=0)
        
        
    def n_m(self,m):
        """deltam0 is 0 for all orders except m=0, when it is 1.
        this is the angular normalisation."""
        deltam0 = 0.
        if m==0: deltam0 = 1.
        return np.power( (deltam0+1)*np.pi/2. , -0.5)
    
    def disk_lag(self, n, Jzs, a):
        '''
        Taken from Weinberg and Petersen 2021 and Johnson+22 as good approximation for disk
        Parameters:
        --------------------------------------
        n   : order of the Laguerre polynomial
        Jzs : actions
        a   : scale length of the disk

        Returns:
        --------------------------------------
        Laguerre part of the coefficient evaluation for the BFE
        '''
        norm = 2/(a*np.sqrt(n+1)) #references diverge on whether there is factor of two --> no factor of 2 and a=3 gives the disk exponential density for the 0th order term

        lag = scipy.special.eval_genlaguerre(n,1,2*Jzs/a) #take alpha=1 following prior work

        return norm * np.exp(-Jzs/a) * lag
    
    # def log_spiral(m, Jzs, thetas):
    #     r = Jzs * np.exp(-k*thetas) + b

    
    def get_coeffs(self, std=False):
        '''
        Generate the coefficients for the BFE
        '''
        self.coeffs = np.zeros((self.m_max,self.n_max), dtype=np.complex128)      #coefficient array
        if std:
            self.coeffs_std = np.zeros((self.m_max, self.n_max), dtype=np.complex128) #coefficient error array
        
        #calculate coefficients
        #print('Calculating coefficients...')
        exp_m = [np.exp(1j * m * self.sel.theta_z) for m in range(self.m_max)]
        for m, n in product(range(self.m_max), range(self.n_max)):
            coeff = self.n_m(m) * np.sum(self.disk_lag(n, self.sel.jz, self.a) * exp_m[m] * self.sel.jz)
            self.coeffs[m, n] = coeff
            if std:
                self.coeffs_std[m, n] = np.sqrt(self.n_m(m) * np.sum((self.disk_lag(n, self.sel.jz, self.a) * \
                                                                np.exp(1j*m*self.sel.theta_z) * self.sel.jz - \
                                                                coeff/len(self.sel))**2))
        return self.coeffs
    
    def create_df(self, n_maxs): # have not updated notebook with this yet
        '''
        Creating the phase spiral reconstruction from the BFE at each point in Jz_grid, thetaz_grid
        '''
        self.df = np.zeros((len(self.Jz_grid), len(self.thetaz_grid))) # each row is one action, each column is one angle
        for act_ind, ang_ind in product(range(len(self.Jz_grid)), range(len(self.thetaz_grid))):
            act, ang = self.Jz_grid[act_ind], self.thetaz_grid[ang_ind]

            df_ = 0
            for m in self.ms:
                ns_ = np.arange(n_maxs[m]) # array of Laguerre modes included in BFE
                # Add the contribution to this action, angle location from each BFE coefficient
                contributions = [
                    self.n_m(m) * self.disk_lag(n, act, self.a) * np.exp(-1j * m * ang) * self.coeffs[m, n]
                    for n in ns_
                ]
                # Sum the contributions
                df_ += np.sum(contributions,axis=0)
            
#             df_ = np.sum(self.n_m(m) * self.disk_lag(n,act,self.a) * np.exp(-1j*m*ang) * self.coeffs[m,n] \
#                          for m,n in product(self.ms,self.ns))
            self.df[act_ind, ang_ind] = df_.real
    
    def make_spiral_residual(self, hwidth=0.1, sigma_jz = 1):
        '''
        Create the residual map (unsmoothed and smoothed) for the m=1+ spirals in the data
        (i.e. subtract out the mean background, which should correspond to m=0 term in the BFE reconstrcution)
        
        There is definitely a better way to do this which is to subtract out the bkg mean in J_z, theta_z space 
        before converting into the spiral shape
        
        Parameters:
        ----------------------
        hwidth    : halfwidth of the bins
        sigma_jz  : standard deviation for the gaussian kernel used for the smoothing
        '''
        self.jzbins = np.arange(-self.rootjzmax, self.rootjzmax+0.01, self.rootjzstep)

        # Go from (J_z, theta_z) to the spiral reconstruction (sqrt(J_z)sin(theta_z), sqrt(J_z)cos(theta_z))
        root_jz = np.sqrt(self.sel.jz)
        self.xdata, self.ydata = root_jz*np.cos(self.sel.theta_z), root_jz*np.sin(self.sel.theta_z)

        xrange = [-self.rootjzmax-self.rootjzstep/2, self.rootjzmax+self.rootjzstep/2]
        yrange = [-self.rootjzmax-self.rootjzstep/2, self.rootjzmax+self.rootjzstep/2]

        rdata = np.sqrt(self.xdata**2+self.ydata**2)

        # counts: how many stars are in each (sqrt(J_z)sin(theta_z), sqrt(J_z)cos(theta_z)) bin
        counts,xedges,yedges,_= scipy.stats.binned_statistic_2d(self.ydata,self.xdata,\
                                                 None, statistic='count',
                                                 bins = (self.jzbins.shape[0],self.jzbins.shape[0]),
                                                 range=[xrange,yrange])
        # Get some additional facts about the bins
        self.extent = [yedges[0],yedges[-1], xedges[0],xedges[-1]] # Should get this just from the grid no?
        x_centers, y_centers = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
        all_bins = np.array(np.meshgrid(x_centers, y_centers)).T.reshape(-1, 2)

        # Area of each bin (need this to get the average background scaling later)
        self.binsize = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

        bin_rad = np.sqrt(all_bins[:,0]**2 + all_bins[:,1]**2)
        lower_r, upper_r = bin_rad-hwidth, bin_rad+hwidth # lower and upper J_z bounds of our bin annulus
        # Sum of all stars in an annulus
        ann_sum = np.sum((np.array(rdata) < upper_r[:,None]) & (np.array(rdata) > lower_r[:,None]), axis=1)
        # Total annular area
        ann_area = np.pi*(upper_r**2 - lower_r**2)
        # Average star count in the bins in this annulus
        bkg_mean = self.binsize * ann_sum/ann_area


        # Get smoothed residual for the data - monopole calculated above
        resid = counts.flatten() - bkg_mean
        self.residual = np.reshape(resid, counts.shape)
        self.smoothed_resid = scipy.ndimage.filters.gaussian_filter(self.residual, sigma_jz, order=0,\
                                                       output=None, mode='reflect', cval=0.0, truncate=4.0)
        
        
    def make_spiral_recon(self, n_maxs):
        '''
        Reconstruct the phase spiral in (sqrt(J_z)sin(theta_z), sqrt(J_z)cos(theta_z)) from our BFE
        '''
        xgrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        ygrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        grid_combs = np.array(np.meshgrid(xgrid,ygrid)).T.reshape(-1,2)
        r_grid = np.sqrt(grid_combs[:,0]**2 + grid_combs[:,1]**2) # radius in every grid cell
        phi_grid = np.arctan2(grid_combs[:,1], grid_combs[:,0])   # phi in every grid cell
        
        #get value of distribution function at each point in the grid
        recon_df_ = 0
        for m in self.ms:
            ns_ = np.arange(n_maxs[m])
            contributions = [
                self.n_m(m) * self.disk_lag(n,r_grid**2,self.a) * np.exp(-1j*m*phi_grid) * self.coeffs[m,n]
                for n in ns_
            ]
            recon_df_ += np.sum(contributions, axis=0).real
        
        contributions2 = [self.n_m(0) * self.disk_lag(n,r_grid**2,self.a) * self.coeffs[0,n]
                          for n in np.arange(n_maxs[0]) ]
        recon_df_m0_ = recon_df_ - np.sum(contributions2, axis=0).real
        
        self.recon_df = np.reshape(recon_df_, (len(xgrid), len(ygrid)))
        self.recon_df_m0 = np.reshape(recon_df_m0_, (len(xgrid), len(ygrid)))
        
        self.noise = self.residual/self.binsize - self.recon_df_m0.T #using make_spiral_recon function output
        
        
    def plot_xy(self):
        fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))
        #ax1.scatter(self.sel.x, self.sel.y,
        #          s = 0.01, c = 'k')
        #ax1.set_xlim(-15, 15)
        #ax1.set_ylim(-15,15)

        ax2.scatter(self.sel.jphi * np.cos(self.sel.theta_phi)/220, # approximate conversion to kpc 
                    self.sel.jphi * np.sin(self.sel.theta_phi)/220, # approximate conversion to kpc
                    s = 0.01, c = 'k')
        ax2.set_xlim(-15, 15)
        ax2.set_ylim(-15,15)
        plt.show()
    
    def plot_ncoeffs(self, n_maxs): # have not updated notebook with this yet
        self.get_coeffs(std=True)

        if self.m_max > 2:
            ns_ = np.arange(n_maxs[2])
            fig, [ax1,ax2] = plt.subplots(1,2,figsize=(9,4), sharey = True)
            ax2.errorbar(ns_, np.abs(self.coeffs[2,:n_maxs[2]].real) / np.sqrt(ns_+1), 
                     self.coeffs_std[2,:n_maxs[2]].real / np.sqrt(ns_+1), c='b', linstyle=None)
            ax2.set_yscale('log')
            ax2.set_xlabel('n order')
            ax2.set_title('Amplitude of n coeffs (Laguerre), m=2',fontsize=15)
        else:
            fig, ax1 = plt.subplots(1,1,figsize=(4,4), sharey = True)

#         ns_ = np.arange(self.n_maxs[0])
#         ax1.errorbar(ns_, np.abs(self.coeffs[0,:self.n_maxs[0]]) / np.sqrt(ns_+1), 
#                      self.coeffs_std[0,:self.n_maxs[0]] / np.sqrt(ns_+1), c='r', linestyle=None)
#         ax1.set_yscale('log')
#         ax1.set_xlabel('n order')
#         ax1.set_ylabel('coeff amplitude')
#         ax1.set_title('Amplitude of n coeffs (Laguerre), m=0',fontsize=15)

        ns_ = np.arange(n_maxs[1])
        ax1.errorbar(ns_, np.abs(self.coeffs[1,:n_maxs[1]].real) / np.sqrt(ns_+1), 
                     self.coeffs_std[1,:n_maxs[1]].real / np.sqrt(ns_+1), c='b', linstyle=None)
        ax1.set_yscale('log')
        ax1.set_xlabel('n order')
        ax1.set_title('Amplitude of n coeffs (Laguerre), m=1',fontsize=15)
        
        fig.tight_layout()
        plt.show()
        
    def plot_spiral_data(self):
        fig, ax = plt.subplots(1,3, figsize=(17,5))
        im0 = ax[0].hist2d(self.xdata, self.ydata,bins=(self.jzbins, self.jzbins), norm=mpl.colors.LogNorm())
        ax[0].set_aspect('equal')
        
        max_res = np.max(self.residual)
        im1 = ax[1].imshow(self.residual, extent=self.extent, 
                           interpolation='gaussian',origin='lower', aspect='equal',rasterized=True, cmap='viridis',
                           norm=mpl.colors.SymLogNorm(linthresh=max_res/20, vmin=-max_res, vmax=max_res))

        
        max_smooth = np.max(self.smoothed_resid)
        im2 = ax[2].imshow(self.smoothed_resid, extent=self.extent,\
                           interpolation='gaussian',origin='lower', aspect='equal',rasterized=True, cmap='viridis',
                           norm=mpl.colors.SymLogNorm(linthresh=max_smooth/20, vmin=-max_smooth, vmax=max_smooth))

        ax[0].set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax[1].set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax[2].set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax[0].set_ylabel(r'$\sqrt{J_z} \times \sin{\theta_z}$', fontsize=15)
        ax[0].set_title('Data', fontsize=20)
        ax[1].set_title('Residual', fontsize=20)
        ax[2].set_title('Smoothed Residual', fontsize=20)
        plt.colorbar(im1, ax=ax[1])
        plt.colorbar(im2, ax=ax[2])
        fig.tight_layout()
        plt.show()
        
    def plot_data_recon(self):
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        H, yedges, xedges = np.histogram2d(self.sel.theta_z, self.sel.jz, 
                                           bins = [len(self.thetaz_grid), len(self.Jz_grid)], 
                                           range=[[0, 2*np.pi], [0,np.max(self.Jz_grid)]])
        vals = H/H.sum(axis=0, keepdims=True)
        im1 = ax1.pcolormesh(yedges,xedges,vals.T, 
                             cmap = 'viridis', norm=mpl.colors.LogNorm(vmin=5e-3,vmax=2e-1),
                            rasterized=True)
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(0, np.max(self.Jz_grid))
        ax1.set_xlabel(r'$\theta_z$', fontsize=15)
        ax1.set_ylabel(r'$J_z$', fontsize=15)
        ax1.set_title('Data', fontsize=20)
        #ax1.set_title('Data {}'.format(self.time), fontsize=15)
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.pcolormesh(self.thetaz_grid, self.Jz_grid, self.df/self.df.sum(axis=1, keepdims=True), 
                             cmap='viridis', norm=mpl.colors.LogNorm(vmin=5e-3,vmax=2e-1),
                            rasterized=True)
        #ax2.set_title('Reconstruction {}'.format(self.time), fontsize=15)
        ax2.set_xlabel(r'$\theta_z$', fontsize=15)
        ax2.set_title('BFE Reconstruction', fontsize=20)
        fig.colorbar(im2, ax=ax2)
        fig.tight_layout()
        # plt.savefig('../figures/data_reconstruction_for_talk.pdf')
        plt.show()
        
        return fig
        
    def plot_spiral_recon(self):
        xgrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        ygrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        fig, [ax1, ax2] = plt.subplots(1,2,figsize=(11,5), sharey=True)
        plot_lims = np.sqrt(np.max(self.Jz_grid))
        im1 = ax1.pcolormesh(xgrid, ygrid, self.recon_df, 
                             cmap='viridis', norm=mpl.colors.LogNorm())
        ax1.set_aspect('equal')
        ax1.set_xlim(-plot_lims, plot_lims)
        ax1.set_ylim(-plot_lims, plot_lims)
        ax1.set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax1.set_ylabel(r'$\sqrt{J_z} \times \sin{\theta_z}$', fontsize=15)
        ax1.set_title('Total DF', fontsize=20)

        max_m0 = np.max(self.recon_df_m0)
        im2 = ax2.pcolormesh(xgrid, ygrid, self.recon_df_m0.T,
                             norm=mpl.colors.SymLogNorm(linthresh=max_m0/20, vmin=-max_m0, vmax=max_m0),
                             cmap='viridis')
        ax2.set_xlim(-plot_lims, plot_lims)
        ax2.set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax2.set_title('DF w/o m=0', fontsize = 20)
        plt.colorbar(im2, ax=ax2)
        
        fig.tight_layout()
        
        plt.show()
        
    def summary_plots(self):
        xgrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        ygrid = np.arange(-self.rootjzmax,self.rootjzmax+1e-5,self.rootjzstep)
        fig, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(16,5), sharey=True)
        plot_lims = np.sqrt(np.max(self.Jz_grid))
        
        max_resid = np.max(self.residual)/self.binsize
        im1 = ax1.imshow(self.residual/self.binsize, extent=self.extent,\
                           interpolation='gaussian',origin='lower',rasterized=True, cmap='magma_r',
                           norm=mpl.colors.SymLogNorm(linthresh=max_resid/20, vmin=-max_resid, vmax=max_resid))
        ax1.set_aspect('equal')
        ax1.set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        ax1.set_ylabel(r'$\sqrt{J_z} \times \sin{\theta_z}$', fontsize=15)
        #ax1.set_title('Data w/o m=0', fontsize=20)
        #plt.colorbar(im1, ax=ax1)
        
        max_m0 = np.max(self.recon_df_m0)
        im2 = ax2.pcolormesh(xgrid, ygrid, self.recon_df_m0.T,
                             norm=mpl.colors.SymLogNorm(linthresh=max_m0/20, vmin=-max_resid, vmax=max_resid),
                             cmap='magma_r', rasterized=True)
        ax2.set_aspect('equal')
        ax2.set_xlim(-plot_lims, plot_lims)
        ax2.set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        #ax2.set_title('DF w/o m=0', fontsize = 20)
        #plt.colorbar(im2, ax=ax2)
        
        self.noise = self.residual/self.binsize - self.recon_df_m0.T
        max_resid2 = np.max(self.noise)
        im3 = ax3.pcolormesh(xgrid, ygrid, self.noise, cmap='magma_r',
                             norm=mpl.colors.SymLogNorm(linthresh=max_resid2/20, vmin=-max_resid2, vmax=max_resid2),
                             rasterized=True)
        ax3.set_aspect('equal')
        ax3.set_xlim(-plot_lims, plot_lims)
        ax3.set_xlabel(r'$\sqrt{J_z} \times \cos{\theta_z}$', fontsize=15)
        #ax3.set_title('Residual', fontsize = 20)
        
        fig.tight_layout()
        
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.12, 0.02, 0.85])
        fig.colorbar(im3, cax=cbar_ax, label=r'$N$')
        cbar_ax.set_label('N')
        
        # plt.savefig('../figures/spiral_reconstruction_for_talk.pdf')
        plt.show()
        
        return fig
    
    def get_pitch_phase_angles(self, m):
        
        ns_ = np.arange(self.n_maxs[m])

        peaks_ = np.zeros(len(self.Jz_grid))

        for i in range(len(self.Jz_grid)):
            act = self.Jz_grid[i]
            C = np.sum([self.coeffs[m, n] * self.disk_lag(n, act, self.a) for n in ns_], axis=0)
            def find_peak(thetaz, C=C):
                return -(C * np.exp(-1j * m * thetaz)).real
            peaks_[i] = scipy.optimize.fmin(find_peak, x0=np.pi, disp=False)[0]
        peaks = 1/m * np.unwrap(m*peaks_) # multiply and divide to correctly use the unwrap function
        
        spl = IUS(np.sqrt(self.Jz_grid), peaks)
        dspl = spl.derivative(n=1)
        
        hist, _ = np.histogram(np.sqrt(self.sel.jz), bins=len(self.Jz_grid), 
                                       range=[0,np.max(np.sqrt(self.Jz_grid))])
        inner = np.argmin(np.abs(self.Jz_grid - self.a))
        # cut off the calculation of pitch angle when at the Jz for which there is <1 star per (Jz, thetaz)bin
        if np.min(hist) >= len(self.thetaz_grid):
            outer = len(self.Jz_grid)-1
        else:
            # outer = np.where(hist < len(self.thetaz_grid))[0][0]
            low_numbers = np.where(hist < len(self.thetaz_grid))[0]
            if len(low_numbers[low_numbers > inner]) > 0:
                outer = np.min(low_numbers[low_numbers > inner])
            else:
                outer=inner


        cut_jz = np.sqrt(self.Jz_grid[inner:outer])
        cut_peaks = peaks[inner:outer]

        ind_for_fit = dspl(cut_jz)<0
        
        if len(np.where(ind_for_fit)[0]) >= (outer-inner)/2:  
            self.pitch_phase_flag = 0
        else: #if the calculations are not well defined, do them anyway but flag the region    
            self.pitch_phase_flag = 1
            
        def get_values(cut_jz, pitch_angle, phase_angle):
            return 1/np.tan(pitch_angle) * np.log(cut_jz) + phase_angle
        
        try:
            (self.pitch_angle, phase), _ = curve_fit(get_values, cut_jz[ind_for_fit], cut_peaks[ind_for_fit], 
                                                p0=[-0.1, 20], bounds=[[-np.pi/2, -100], [0, 1000]], 
                                                loss='cauchy')[0]
            self.phase_angle = phase % (2*np.pi)
            
        except: # if calculation doesn't work, set pitch angle=0 and phase angle=pi
            self.pitch_angle = 0
            self.phase_angle = np.pi

            self.pitch_phase_flag = 2
            
        outer_mean_freq = self.get_mean_freq(self.Jz_grid[outer])
        inner_mean_freq = self.get_mean_freq(self.a) # more accurate than just using "inner"
        self.time_since_int = np.log(self.Jz_grid[outer]/self.a) / (2*np.tan(self.pitch_angle)*(outer_mean_freq - inner_mean_freq))[0]
        return self.pitch_angle, self.phase_angle, self.pitch_phase_flag, self.time_since_int

    def get_mean_freq(self, jz):
        if type(jz) != np.ndarray:
            jz=np.array([jz])
        mean_freqs = np.zeros(len(jz))
        for i in range(len(jz)):
            mean_freqs[i] = np.median(self.sel.freq_z[(self.sel.jz < jz[i]+0.0325)&(self.sel.jz > jz[i]-0.0325)])
            # print(jz[i], len(self.sel.freq_z[(self.sel.jz < jz[i]+0.0325)&(self.sel.jz > jz[i]-0.0325)]))
        return mean_freqs
    
    def get_mean_vels(self):
        self.mean_vr = np.mean(self.sel.v_R)
        self.mean_vphi = np.mean(self.sel.v_phi)
        self.mean_vz = np.mean(self.sel.v_z)
        
        return self.mean_vr, self.mean_vphi, self.mean_vz
        
    def make_snails(self, n_maxs):
        print('Calculating distribution function...')
        self.get_coeffs()
        self.create_df(n_maxs)
        print('Making Spiral Residual in Data...')
        self.make_spiral_residual()
        print('Making Spiral Reconstruction...')
        # self.transform2polar(self.df)
        self.make_spiral_recon(n_maxs)
        
        print('Plotting...')
        self.plot_ncoeffs(n_maxs)
        self.plot_data_recon()
        self.plot_spiral_data()
        self.plot_spiral_recon()
        self.summary_plots()
