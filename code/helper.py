import scipy
from scipy import stats
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2 #set the value globally
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import cmasher as cmr
from astropy.table import Table
import numpy as np
import os
import sys

from load_data_B2 import setup_B2
from df_helpers import *
from matplotlib import animation
from matplotlib.animation import FuncAnimation
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
        tphi_c = tphi_c_[:-1] + rad[1]
        self.J, self.T = np.meshgrid(jphi_c, tphi_c)

        if sim_name == 'B2':
            self.time_ratio = 0.009778
        elif sim_name == 'test':
            self.time_ratio = 0.01

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
                      fontsize=18, color='w')
        ax.set_rmax(np.max(self.jphi_c))
        ax.tick_params(left = False, right = False , labelleft = True ,
                             labelbottom = False, bottom = False)      
        ax.grid(visible=False)
        ax.set_rlabel_position(50)

        ax.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*self.time_ratio, 2)), 
                 fontsize=20, ha="center", c='w')

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

    def make_pc_reconstruction_mov(self, file_directory, subtract_mean=False, norm_function=mpl.colors.Normalize, cmap=cmr.prinsenvlag, **kwargs):

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


    def animate_pc_reconstruction(self, timestep, ax, subtract_mean=False, norm_function=mpl.colors.Normalize, cmap=cmr.prinsenvlag, **kwargs):
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
                      fontsize=18, color='w')
        ax.set_rmax(np.max(self.jphi_c))
        ax.tick_params(left = False, right = False , labelleft = True ,
                             labelbottom = False, bottom = False)
        ax.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*self.time_ratio, 2)), fontsize=20, 
            ha="center", c='w')
        ax.grid(False)
        ax.set_rlabel_position(50)

        if subtract_mean:
            ax.set_title(self.channel_name + ', ' + self.pc_string + '(mean-subtracted)', fontsize=15)
        else:
            ax.set_title(self.channel_name + ', ' + self.pc_string, fontsize=15)

        plt.draw()

























          
# class MakeAnimations_archived:
    
#     def __init__(self, mssa, shortname, times, n_channels, pc_number, jbins):
#         self.mssa = mssa
#         self.times = times
#         self.pc_number = pc_number
#         self.shortname = shortname
#         self.n_channels = n_channels
        
#         # For the reconstruction:
#         self.mssa.reconstruct(self.pc_number)
#         get_recon = self.mssa.getReconstructed()
#         #recon_all_coefs = get_recon[list(get_recon.keys())[0]].getAllCoefs()
        
#         if shortname == 'both':
#             recon_m1_amp_coefs = get_recon[list(get_recon.keys())[0]].getAllCoefs()
#             recon_m1_pitch_coefs = get_recon[list(get_recon.keys())[1]].getAllCoefs()
#             self.pc_rc = np.concatenate([recon_m1_amp_coefs, recon_m1_pitch_coefs])
#         else:
#             recon_all_coefs = get_recon[list(get_recon.keys())[0]].getAllCoefs()
#             self.pc_rc = recon_all_coefs
#             print(len(self.pc_rc))
        
#         # To get colorbar
#         jphi_min = 1000
#         jphi_c = np.linspace(jphi_min, jphi_min+(jbins*100), jbins+1)
#         self.jphi_c = jphi_c
#         tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
#         rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
#         tphi_c = tphi_c_[:-1] + rad[1]
#         self.J, self.T = np.meshgrid(jphi_c, tphi_c)
    
#     def make_recon_mov_disk_plot(self, filename, polar=True):
        
#         self.polar = polar
#         if polar:
#             if self.shortname=='both':
#                 fig, [self.ax1, self.ax2] = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': 'polar'})
#             else:
#                 fig, self.ax1 = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'})
#         else:
#             fig, [self.ax1, self.ax2] = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

        
#         if len(self.pc_rc) > self.n_channels:
#             print('Making movies for both Amplitude and Pitch Angle')
            
#             self.amp_vmin = 1e-3 #1e2 #np.min(self.pc_rc[:self.n_channels]) 
#             self.amp_vmax = 5e-1#np.max(self.pc_rc[:self.n_channels])
            
#             # if 0 in self.pc_number:
#             #     im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds',
#             #                               norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50,
#             #                                                          vmin=-self.amp_vmax, 
#             #                                                          vmax=self.amp_vmax))
#             #     im2 = self.ax2.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds_r',
#             #                               norm=mpl.colors.LogNorm(vmin=np.pi/32, vmax=np.pi/2))
                
#             # else:
#                 # im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                 #                           norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                 #                                                      vmin=-self.amp_vmax, 
#                 #                                                      vmax=self.amp_vmax))
#                 # im2 = self.ax2.pcolormesh(self.T, self.J, np.zeros(self.T.shape), cmap='RdBu',
#                 #                     norm=mpl.colors.SymLogNorm(linthresh=np.pi/512, vmin = -np.pi/2, vmax=np.pi/2))
        
#         else:
#             print('Making movies for just Amplitude or Pitch Angle')
            
#             self.amp_vmin = 1e-3 #1e2 #np.min(self.pc_rc[:self.n_channels]) 
#             self.amp_vmax = 5e-1#np.max(self.pc_rc[:self.n_channels])
            
#             if self.shortname == 'rel_amp':
#                 # if -1 in self.pc_number: # changing from 0
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                 #                               norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                 #                                                       vmin=-self.amp_vmax, 
#                 #                                                       vmax=self.amp_vmax))
#                 # else:
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds',
#                                           norm=mpl.colors.LogNorm(vmin=max(np.min(self.pc_rc), 1e-3), vmax=np.max(self.pc_rc)))
#             elif self.shortname == 'amp':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds',
#                                           norm=mpl.colors.LogNorm(vmin=1e2, vmax=1e4))
#             elif self.shortname == 'pitch':
#                 # if -1 in self.pc_number:
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds_r',
#                 #                               norm=mpl.colors.LogNorm(vmin=np.pi/128, vmax=np.pi/2))
#                 # else:
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                 #                               norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))
#                  im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds_r',
#                                               vmin=0.1, vmax=np.pi/2)
#             elif self.shortname == 'phase':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmr.fusion,
#                                           vmin = 0, vmax=2*np.pi)
#             elif self.shortname == 'ratio':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                                           norm=mpl.colors.LogNorm(vmin=0.1, vmax=10))
#             elif self.shortname == 'int_time':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu',
#                                           vmin=0, vmax=8)

#         # Create the colorbar
#         cbar1 = plt.colorbar(im1, ax=self.ax1)
#         if self.shortname=='both':
#             cbar1.set_label('Amplitude', fontsize=18)
#             cbar2 = plt.colorbar(im2, ax=self.ax2)
#             cbar2.set_label('Pitch Angle', fontsize=18)
#             cbar2.ax.text(0.5, 1.02, 'Less wound up', ha='center', va='bottom', transform=cbar2.ax.transAxes, fontsize=15)
#             cbar2.ax.text(0.5, -0.02, 'More wound up', ha='center', va='top', transform=cbar2.ax.transAxes, fontsize=15)
#         elif self.shortname=='rel_amp':
#             cbar1.set_label('Amplitude', fontsize=18)
#         elif self.shortname=='amp':
#             cbar1.set_label('Amplitude', fontsize=18)
#         elif self.shortname=='pitch':
#             cbar1.set_label('Pitch Angle', fontsize=18)
#             cbar1.ax.text(0.5, 1.02, 'Less wound up', ha='center', va='bottom', transform=cbar1.ax.transAxes, fontsize=15)
#             cbar1.ax.text(0.5, -0.02, 'More wound up', ha='center', va='top', transform=cbar1.ax.transAxes, fontsize=15)
#         elif self.shortname=='phase':
#             cbar1.set_label('Phase', fontsize=18)
#         elif self.shortname=='ratio':
#             cbar1.ax.text(3.5, 0.8, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar1.ax.transAxes, fontsize=14)
#             cbar1.ax.text(3.5, 0.2, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar1.ax.transAxes, fontsize=14)
#             cbar1.set_label('m1 Amp / m2 Amp', fontsize=18)
#         elif self.shortname=='int_time':
#             cbar1.set_label('Interaction Time', fontsize=18)
#             # cbar.ax.text(6, 0.7, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=14)
#             # cbar.ax.text(6, 0.3, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=14)
        
#         fig.tight_layout()
        
#         anim = FuncAnimation(
#             fig,
#             self.animate_pc_recon_disk, #animate_pc_recon_mean_subtracted_disk
#             frames=np.arange(0, int(len(self.times)), 1),
#             interval=20,
#             blit=False,
#         )
#         f = filename
#         FFwriter = animation.FFMpegWriter(fps=10)
#         anim.save(f, writer = FFwriter)
        
        
        
#     def animate_pc_recon_disk(self, timestep):

#         self.ax1.clear() 
#         if self.shortname=='both':
#             self.ax2.clear()
        
        
#         if len(self.pc_rc) > self.n_channels:
#             pc_rc_amp = self.pc_rc[:self.n_channels]
#             pc_rc_pitch = self.pc_rc[self.n_channels:]
#             if 0 not in self.pc_number:
#                 pc_rc_amp = pc_rc_amp - np.mean(pc_rc_amp, axis=0)
#                 pc_rc_pitch = pc_rc_pitch - np.mean(pc_rc_pitch, axis=0)
#             if 0 in self.pc_number:
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), 
#                                           cmap='RdBu_r',
#                                           norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                      vmin=-self.amp_vmax, 
#                                                                      vmax=self.amp_vmax))
#                 im2 = self.ax2.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), 
#                                           cmap='Reds_r', 
#                                           norm=mpl.colors.LogNorm(vmin=np.pi/32, vmax=np.pi/2))
#             else:
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), 
#                                           cmap='RdBu_r',
#                                           norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                      vmin=-self.amp_vmax, 
#                                                                      vmax=self.amp_vmax))
#                 im2 = self.ax2.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), 
#                                           cmap='RdBu', 
#                                           norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, 
#                                                                      vmin = -np.pi/2, 
#                                                                      vmax=np.pi/2))
                
#         else:
#             pc_rc = self.pc_rc[:self.n_channels]
#             if self.shortname == 'rel_amp':
#                 # if -1 in self.pc_number:
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), 
#                 #                               cmap='Reds',
#                 #                               norm=mpl.colors.LogNorm(vmin=self.amp_vmin, 
#                 #                                                       vmax=self.amp_vmax))
#                 # else:
#                     # pc_rc = self.pc_rc - np.mean(self.pc_rc, axis=0)
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'),
#                                           cmap='Reds', 
#                                           norm=mpl.colors.LogNorm(vmin=max(np.min(pc_rc), 1e-3), vmax=np.max(pc_rc)))
#             elif self.shortname == 'amp':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'),
#                                           cmap='Reds', 
#                                           norm=mpl.colors.LogNorm(vmin=1e2, vmax=1e4))
#             elif self.shortname == 'pitch':
#                 # if 0 in self.pc_number:
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, -np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap='Reds_r',
#                 #                               norm=mpl.colors.LogNorm(vmin=np.pi/128, vmax=np.pi/2))
#                 # else:
#                 #     im1 = self.ax1.pcolormesh(self.T, self.J, -np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap='RdBu_r',
#                 #                               norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))
#                 im1 = self.ax1.pcolormesh(self.T, self.J, -np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap='Reds_r',
#                                           vmin=0.1, vmax=np.pi/2)
#             elif self.shortname =='phase':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap=cmr.fusion,
#                                           vmin = 0, vmax=2*np.pi)
#             elif self.shortname == 'ratio':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap='RdBu_r',
#                                           norm=mpl.colors.LogNorm(vmin=0.1, vmax=10))
#             elif self.shortname == 'int_time':
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc[:,timestep], self.T.shape, 'F'), cmap='RdBu',
#                               vmin=0, vmax=8)


                
#         first_timestep = self.times[0]
        
#         if self.polar:
#             self.ax1.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#             self.ax1.set_rmax(np.max(self.jphi_c))
#             self.ax1.tick_params(left = False, right = False , labelleft = True ,
#                                  labelbottom = False, bottom = False)
#             self.ax1.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=20, 
#                 ha="center", c='k')
            
#             if self.shortname=='both':
#                 self.ax2.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#                 self.ax2.set_rmax(np.max(self.jphi_c))
#                 self.ax2.tick_params(left = False, right = False , labelleft = True ,
#                                      labelbottom = False, bottom = False)
#                 self.ax2.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=20, 
#                     ha="center", c='k')
            
            
            
#         else:
#             self.ax1.set_xlabel(r'$\theta_{\phi}$')
#             if self.shortname=='both':
#                 self.ax2.set_xlabel(r'$\theta_{\phi}$')
#             self.ax1.set_ylabel(r'$J_{\phi}$')
#             plt.suptitle(r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=25)
            
#         if self.shortname=='rel_amp':    
#             self.ax1.set_title('One-armed Amplitude, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname=='pitch':
#             self.ax1.set_title('One-Armed Pitch Angle, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname == 'phase':
#             self.ax1.set_title('One-Armed Phase Angle, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname == 'ratio':
#             self.ax1.set_title('one-Armed amp / two-armed amp, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname == 'int_time':
#             self.ax1.set_title('Time of Interaction, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname=='both':
#             self.ax1.set_title('One-armed Amplitude, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#             self.ax2.set_title('One-Armed Pitch Angle, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)

#         plt.draw()
        
        
#     def animate_pc_recon_mean_subtracted_disk(self, timestep):

#         self.ax1.clear() 
#         if self.shortname=='both':
#             self.ax2.clear()
        
        
#         if len(self.pc_rc) > self.n_channels:
#             pc_rc_amp = self.pc_rc[:self.n_channels]
#             pc_rc_pitch = self.pc_rc[self.n_channels:]
#             if 0 not in self.pc_number:
#                 pc_rc_amp = pc_rc_amp - np.mean(pc_rc_amp, axis=0)
#                 pc_rc_pitch = pc_rc_pitch - np.mean(pc_rc_pitch, axis=0)
                
#             # get the mean at each radius
#             rad_mean_amp = np.mean(np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), axis=0)
#             rad_mean_pitch = np.mean(np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), axis=0)
            
#             if 0 in self.pc_number:
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), 
#                                           cmap='RdBu_r',
#                                           norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                      vmin=-self.amp_vmax, 
#                                                                      vmax=self.amp_vmax))
#                 im2 = self.ax2.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), 
#                                           cmap='Reds_r', 
#                                           norm=mpl.colors.LogNorm(vmin = np.pi/32, vmax=np.pi/2))
#             else:
#                 im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F') - rad_mean_amp, 
#                                           cmap='RdBu_r',
#                                           norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                      vmin=-self.amp_vmax, 
#                                                                      vmax=self.amp_vmax))
#                 im2 = self.ax2.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F') + rad_mean_pitch,
#                                           cmap='RdBu', 
#                                           norm=mpl.colors.SymLogNorm(linthresh=np.pi/512, vmin = -np.pi/2, vmax=np.pi/2))
                
#         else:
#             if self.shortname == 'amp':
#                 pc_rc_amp = self.pc_rc[:self.n_channels]
#                 if -1 in self.pc_number: #change from 0
#                     im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), 
#                                               cmap='RdBu_r',
#                                               norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                          vmin=-self.amp_vmax, 
#                                                                          vmax=self.amp_vmax))
#                 else:    
#                     pc_rc_amp = self.pc_rc - np.mean(self.pc_rc, axis=0)
#                     rad_mean_amp = np.mean(np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F'), axis=0)
#                     im1 = self.ax1.pcolormesh(self.T, self.J, np.reshape(pc_rc_amp[:,timestep], self.T.shape, 'F') - rad_mean_amp, cmap='RdBu_r',
#                                               norm=mpl.colors.SymLogNorm(linthresh=self.amp_vmax/50, 
#                                                                          vmin=-self.amp_vmax, 
#                                                                          vmax=self.amp_vmax))
#             elif self.shortname == 'pitch':
                
#                 pc_rc_pitch = self.pc_rc[:self.n_channels]
                
#                 rad_mean_pitch = np.mean(np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), axis=0)
                
#                 if 0 in self.pc_number:
#                     im1 = self.ax1.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F'), cmap='Reds_r',
#                                               norm=mpl.colors.LogNorm(vmin=np.pi/128, vmax=np.pi/2))
#                 else:
#                     im1 = self.ax1.pcolormesh(self.T, self.J, -np.reshape(pc_rc_pitch[:,timestep], self.T.shape, 'F') + rad_mean_pitch, cmap='RdBu_r',
#                                               norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))
        
        
#         first_timestep = self.times[0]
        
#         if self.polar:
#             self.ax1.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#             self.ax1.set_rmax(np.max(self.jphi_c))
#             self.ax1.tick_params(left = False, right = False , labelleft = True ,
#                                  labelbottom = False, bottom = False)
#             self.ax1.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=20, 
#                 ha="center", c='k')
            
#             if self.shortname=='both':
#                 self.ax2.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#                 self.ax2.set_rmax(np.max(self.jphi_c))
#                 self.ax2.tick_params(left = False, right = False , labelleft = True ,
#                                      labelbottom = False, bottom = False)
#                 self.ax2.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=20, 
#                 ha="center", c='k')
            
            
            
#         else:
#             self.ax1.set_xlabel(r'$\theta_{\phi}$', fontsize=15)
#             if self.shortname=='both':
#                 self.ax2.set_xlabel(r'$\theta_{\phi}$', fontsize=15)
#             self.ax1.set_ylabel(r'$J_{\phi}$', fontsize=15)
#             plt.suptitle(r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), fontsize=25)
        
#         if self.shortname=='amp':    
#             self.ax1.set_title('One-armed Amplitude, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname=='pitch':
#             self.ax1.set_title('One-Armed Pitch Angle, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#         elif self.shortname=='both':
#             self.ax1.set_title('One-armed Amplitude, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)
#             self.ax2.set_title('One-Armed Pitch Angle, PC{}-{} Contribution'.format(self.pc_number[0], self.pc_number[-1]), fontsize=15)

#         plt.draw()
        
        
        
#     def make_pre_mssa_face_on(self, filename, table_dict):

#         if len(list(table_dict)) == 1:
#             fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection':'polar'})
#             if list(table_dict)[0] == 'rel_amp':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds',
#                                    norm=mpl.colors.LogNorm(vmin=5e-3, vmax=5e-1))
#             if list(table_dict)[0] == 'amp':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds',
#                                    norm=mpl.colors.LogNorm(vmin=1e2, vmax=1e4))
#             elif list(table_dict)[0] == 'pitch':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds_r',
#                                    vmin=0.5, vmax=np.pi/2)
#             elif list(table_dict)[0] == 'phase':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap=cmr.fusion,
#                                    vmin=0, vmax=2*np.pi)
#             elif list(table_dict)[0] == 'ratio':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                                    norm=mpl.colors.LogNorm(vmin=0.1, vmax=10))
#             elif list(table_dict)[0] == 'int_time':
#                 im = ax.pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu',
#                                    vmin=0, vmax=8)
#             cbar = plt.colorbar(im, ax=ax)
#             if list(table_dict)[0] == 'ratio':
#                 cbar.ax.text(3.5, 0.8, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=14)
#                 cbar.ax.text(3.5, 0.2, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=14)
#                 cbar.set_label('m1 Amp / m2 Amp', fontsize=18)
#             else:
#                 cbar.set_label(list(table_dict)[0], fontsize=18)
#             # cbar.ax.text(6, 0.7, 'one-armed spiral \n dominates', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=14)
#             # cbar.ax.text(6, 0.3, 'two-armed spiral \n dominates', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=14)
#             axs=[ax]
#         else:
#             fig, axs = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection':'polar'})
    
#             im1 = axs[0].pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='RdBu_r',
#                                  norm=mpl.colors.SymLogNorm(linthresh=5e-1 / 50, vmin=-5e-1, vmax=5e-1))
    
#             im2 = axs[1].pcolormesh(self.T, self.J, np.ones(self.T.shape), cmap='Reds_r',
#                                  norm=mpl.colors.LogNorm(vmin=np.pi/32, vmax=np.pi/2))

#             # Create the colorbar
#             cbar1 = plt.colorbar(im1, ax=axs[0])
#             cbar2 = plt.colorbar(im2, ax=axs[1])
#             cbar1.set_label('Relative Amplitude', fontsize=18)
#             cbar2.set_label('Pitch Angle', fontsize=18)
#             cbar2.ax.text(0.5, 1.02, 'Less wound up', ha='center', va='bottom', transform=cbar2.ax.transAxes, fontsize=15)
#             cbar2.ax.text(0.5, -0.02, 'More wound up', ha='center', va='top', transform=cbar2.ax.transAxes, fontsize=15)
        
#         fig.tight_layout()

#         anim = FuncAnimation(
#             fig,
#             partial(self.animate_pre_mssa_face_on, table_dict=table_dict, axs=axs),
#             frames=np.arange(0, int(len(self.times)), 1),
#             interval=20,
#             blit=False,
#         )
#         f = filename
#         FFwriter = animation.FFMpegWriter(fps=10)
#         anim.save(f, writer = FFwriter)

#     def animate_pre_mssa_face_on(self, timestep, table_dict, axs):
#         first_timestep = self.times[0]
        
#         if len(list(table_dict)) == 1:
#             ax0 = axs[0]
#             ax0.clear()

#             sname = list(table_dict)[0]
#             dat = np.reshape(table_dict[sname][timestep,1:], self.T.shape, 'F')
#             if sname == 'rel_amp':
#                 im = ax0.pcolormesh(self.T, self.J, dat, cmap='Reds',
#                                    norm=mpl.colors.LogNorm(vmin=5e-3, vmax=5e-1))
#             if sname == 'amp':
#                 im = ax0.pcolormesh(self.T, self.J, dat, cmap='Reds',
#                                    norm=mpl.colors.LogNorm(vmin=1e2, vmax=1e4))
#             elif sname == 'pitch':
#                 im = ax0.pcolormesh(self.T, self.J, -dat, cmap='Reds_r',
#                                    vmin=0.5, vmax=np.pi/2)
#             elif sname == 'phase':
#                 im = ax0.pcolormesh(self.T, self.J, dat, cmap=cmr.fusion, vmin=0, vmax=2*np.pi)
#             elif sname == 'ratio':
#                 im = ax0.pcolormesh(self.T, self.J, dat, cmap='RdBu_r',
#                                    norm=mpl.colors.LogNorm(vmin=0.1, vmax=10))
#             elif list(table_dict)[0] == 'int_time':
#                 im = ax0.pcolormesh(self.T, self.J, dat, cmap='RdBu',
#                                    vmin=0, vmax=8)
                
#             ax0.set_title(list(table_dict)[0], fontsize=20)

#             ax0.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#             ax0.set_rmax(np.max(self.jphi_c))

#             ax0.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), 
#                      fontsize=25, ha="center", c='k')
                
#         else:
#             ax1, ax2 = axs
    
#             ax1.clear() ; ax2.clear()
    
#             if self.shortname=='amp':
#                 im1 = ax1.pcolormesh(self.T, self.J, np.reshape(tbl_rel_amp[timestep,1:], self.T.shape, 'F'), cmap='RdBu_r',
#                                      norm=mpl.colors.SymLogNorm(linthresh=5e-1 / 50, vmin=-5e-1, vmax=5e-1))
                
#             elif self.shortname=='pitch':
#                 im2 = ax2.pcolormesh(self.T, self.J, np.reshape(-tbl_pitch[timestep,1:], self.T.shape, 'F'), cmap='Reds_r',
#                                      norm=mpl.colors.LogNorm(vmin=np.pi/32, vmax=np.pi/2))
                
#             elif self.shortname=='both':
#                 im1 = ax1.pcolormesh(self.T, self.J, np.reshape(tbl_rel_amp[timestep,1:], self.T.shape, 'F'), cmap='RdBu_r',
#                                      norm=mpl.colors.SymLogNorm(linthresh=5e-1 / 50, vmin=-5e-1, vmax=5e-1))
    
#                 im2 = ax2.pcolormesh(self.T, self.J, np.reshape(-tbl_pitch[timestep,1:], self.T.shape, 'F'), cmap='Reds_r',
#                                      norm=mpl.colors.LogNorm(vmin=np.pi/32, vmax=np.pi/2))
#                 ax1.set_title('One-armed Relative Amplitude', fontsize=20)
#                 ax2.set_title('One-armed Pitch Angle', fontsize=20)
            

#         plt.draw()

#     def make_pre_mssa_face_on_subtracted(self, filename, tbl_rel_amp, tbl_pitch):

#         fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection':'polar'})

#         im1 = ax1.pcolormesh(self.T, self.J, np.ones(self.T.shape),
#                              cmap='RdBu_r',
#                              norm=mpl.colors.SymLogNorm(linthresh=5e-1 / 50, 
#                                                         vmin=-5e-1, 
#                                                         vmax=5e-1))

#         im2 = ax2.pcolormesh(self.T, self.J, np.ones(self.T.shape),
#                              cmap='RdBu', 
#                              norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))

#         # Create the colorbar
#         cbar1 = plt.colorbar(im1, ax=ax1)
#         cbar2 = plt.colorbar(im2, ax=ax2)
#         cbar1.set_label('Relative Amplitude', fontsize=18)
#         cbar2.set_label('Pitch Angle', fontsize=18)
#         cbar2.ax.text(0.5, 1.02, 'Less wound up', ha='center', va='bottom', transform=cbar2.ax.transAxes, fontsize=15)
#         cbar2.ax.text(0.5, -0.02, 'More wound up', ha='center', va='top', transform=cbar2.ax.transAxes, fontsize=15)
#         fig.tight_layout()

#         anim = FuncAnimation(
#             fig,
#             partial(self.animate_pre_mssa_face_on_subtracted, tbl_rel_amp=tbl_rel_amp, tbl_pitch=tbl_pitch, axs=[ax1, ax2]),
#             frames=np.arange(0, int(len(self.times)), 1),
#             interval=20,
#             blit=False,
#         )
#         f = filename
#         FFwriter = animation.FFMpegWriter(fps=10)
#         anim.save(f, writer = FFwriter)

#     def animate_pre_mssa_face_on_subtracted(self, timestep, tbl_rel_amp, tbl_pitch, axs):

#         ax1, ax2 = axs

#         ax1.clear() ; ax2.clear()

#         data_amp = np.reshape(tbl_rel_amp[timestep,1:], self.T.shape, 'F') 
#         data_pitch = np.reshape(-tbl_pitch[timestep,1:], self.T.shape, 'F')
#         if self.shortname == 'amp':
#             bkg_amp = np.reshape(self.pc_rc[:self.n_channels][:,timestep], self.T.shape, 'F')
            
#             im1 = ax1.pcolormesh(self.T, self.J, data_amp-bkg_amp,
#                              cmap='RdBu_r',
#                              norm=mpl.colors.SymLogNorm(linthresh=5e-1 /50, 
#                                                          vmin=-5e-1, 
#                                                          vmax=5e-1))
            
#         elif self.shortname == 'pitch':
#             bkg_pitch = np.reshape(self.pc_rc[:self.n_channels][:,timestep], self.T.shape, 'F')
            
#             im2 = ax2.pcolormesh(self.T, self.J, data_pitch+bkg_pitch, 
#                              cmap='RdBu', 
#                              norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))

            
#         elif self.shortname =='both':
#             bkg_amp = np.reshape(self.pc_rc[:self.n_channels][:,timestep], self.T.shape, 'F')
#             bkg_pitch = np.reshape(self.pc_rc[self.n_channels:][:,timestep], self.T.shape, 'F')
            
#             im1 = ax1.pcolormesh(self.T, self.J, data_amp-bkg_amp,
#                              cmap='RdBu_r',
#                              norm=mpl.colors.SymLogNorm(linthresh=5e-1 /50, 
#                                                          vmin=-5e-1, 
#                                                          vmax=5e-1))

#             im2 = ax2.pcolormesh(self.T, self.J, data_pitch+bkg_pitch, 
#                                  cmap='RdBu', 
#                                  norm=mpl.colors.SymLogNorm(linthresh=np.pi/100, vmin = -np.pi/2, vmax=np.pi/2))

#         ax1.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#         ax2.set_yticks([np.min(self.jphi_c), np.max(self.jphi_c)])
#         ax1.set_rmax(np.max(self.jphi_c))
#         ax2.set_rmax(np.max(self.jphi_c))


#         first_timestep = self.times[0]

#         ax1.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), 
#                  fontsize=25, ha="center", c='k')
#         ax2.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around((first_timestep+timestep)*0.009778, 2)), 
#                  fontsize=25, ha="center", c='k')

#         ax1.set_title('One-armed Relative Amplitude', fontsize=20)
#         ax2.set_title('One-armed Pitch Angle', fontsize=20)

#         plt.draw()













def create_coeff_arrays(timestep):
    
    radius = [50, np.pi/16]
    fname = '../data/coeffs/coeffs_t{}_m3_n20_Jphi{}_thetaphi{}.npy'.format(int(timestep), 
                                                                int(2*radius[0]), 
                                                                int(np.pi/(2*radius[1])))
    
    if not os.path.exists(fname):
        print('Timestep = {}'.format(timestep))
        print(fname)
        data = setup_B2(timestep)

        print('Creating all_coeff array...')

        Jphi_centers = np.arange(1000, 3000, 2*radius[0])
        thetaphi_centers = np.arange(0, 2*np.pi, 2*radius[1])

        all_centers = np.array(np.meshgrid(Jphi_centers, thetaphi_centers)).T.reshape(-1,2)

        Jz_grid = np.arange(0, 64, .5)
        thetaz_grid = np.arange(0, 2*np.pi, np.pi/48)

        m_max, n_max = 3,20

        all_coeff_array = np.zeros((len(all_centers), m_max, n_max), dtype = np.complex_)
        for i in range(len(all_centers)):
            center = all_centers[i]

            LS = LaguerreSnails(data, center, radius, Jz_grid, thetaz_grid, m_max, n_max, timestep)
            coeffs = LS.get_coeffs()

            all_coeff_array[i] = coeffs

        print('Saving coefficient array...')
        np.save(fname, all_coeff_array)

        return all_coeff_array
    
def make_coeff_ratio_table(radius):
    t = Table(names=('timestep', 'Jphi_center', 'thetaphi_center', 'm1', 'm2'))

    for timestep in np.arange(848):
        filename = 'coeffs/coeffs_t{}_m3_n20_Jphi100_thetaphi8.npy'.format(timestep)
        if os.path.exists(filename):
            coeff_array = np.load(filename)

            Jphi_centers, thetaphi_centers = np.arange(1000, 3000, 2*radius[0]), np.arange(0, 2*np.pi, 2*radius[1])
            centers = np.array(np.meshgrid(Jphi_centers, thetaphi_centers)).T.reshape(-1,2)

            m1s = np.linalg.norm(np.abs(coeff_array[:, 1, :]), axis=1)
            m2s = np.linalg.norm(np.abs(coeff_array[:, 2, :]), axis=1)

            for j in range(len(centers)):
                t.add_row((timestep, centers[j][0], centers[j][1], m1s[j], m2s[j]))
                
    Jphi_centers = np.arange(1000, 3000, 2*radius[0])
    thetaphi_centers = np.arange(0, 2*np.pi, 2*radius[1])
    J, T = np.meshgrid(Jphi_centers, thetaphi_centers)

    return t
    
def animate_m1m2(timestep, tbl, axs):
    #tbl is the sorted table
    ax1, ax2, ax3 = axs
    
    Jphi_centers = np.unique(tbl['jphi_cen'])
    thetaphi_centers = np.unique(tbl['tphi_cen'])
    J, T = np.meshgrid(Jphi_centers, thetaphi_centers)
    
    t_sub = tbl[tbl['timestep'] == timestep]

    ax1.clear() ; ax2.clear() ; ax3.clear()

    im1 = ax1.pcolormesh(T, J, np.reshape(t_sub['m1_amp'], T.shape, 'F'), 
                             norm=mpl.colors.LogNorm(vmax=1e5), cmap='Blues')
    ax1.set_yticks([1000, 2000, 3000])
    ax1.set_rmax(3000)
    #ax1.text(3*np.pi/2, 100, r'{} Myr'.format(np.around(timestep)), fontsize=17, 
    #        ha="center", c='r')

    im2 = ax2.pcolormesh(T, J, np.reshape(t_sub['m2_amp'], T.shape, 'F'), 
                         norm=mpl.colors.LogNorm(vmax=1e5), cmap='Reds')
    ax2.set_yticks([1000, 2000, 3000])
    ax2.set_rmax(3000)
    ax1.text(3*np.pi/2, 100, r'{} Gyr'.format(np.around(timestep/100, 2)), fontsize=16, 
            ha="center", c='r')

    im3 = ax3.pcolormesh(T, J, np.reshape(t_sub['m1_amp']/t_sub['m2_amp'], T.shape, 'F'), 
                         norm=mpl.colors.LogNorm(vmin=0.05, vmax=20), cmap='RdBu')
    ax3.set_yticks([1000, 2000, 3000])
    ax3.set_rmax(3000)
    #ax3.text(3*np.pi/2, 100, r'{} Myr'.format(np.around(timestep)), fontsize=17, 
    #        ha="center", c='k')
    ax1.set_title('Single-Arm Amp')
    ax2.set_title('Double-Arm Amp')
    ax3.set_title('Single Amp / Double Amp')

    plt.draw()
    
    
        
    
def animate_pc_recon_m1m2(timestep, pc_number, coefs, n_chan,
                           ax1_val, ax2_val, ax3_val, axs):
    
    mssa_m1_amp, mssa_m2_amp, mssa_m1_m2_ratio = ax1_val, ax2_val, ax3_val
    
    jphi_c = np.linspace(1000, 3000, 20+1)
    tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
    rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
    tphi_c = tphi_c_[:-1] + rad[1]
    J, T = np.meshgrid(jphi_c, tphi_c)

    axs[0,0].clear() ; axs[0,1].clear() ; axs[0,2].clear()
    axs[1,0].clear() ; axs[1,1].clear() ; axs[1,2].clear()
    
    axs[0, 0] = plt.subplot(2, 3, 1, projection='polar')
    axs[0, 1] = plt.subplot(2, 3, 2, projection='polar')
    axs[0, 2] = plt.subplot(2, 3, 3, projection='polar')
    
    pc_rc_m1,pc_rc_m2,pc_rc_m1_m2 = np.zeros(n_chan),np.zeros(n_chan),np.zeros(n_chan)
    for i in range(n_chan):
        recon_m1_amp = mssa_m1_amp.getRC([i, 0])
        pc_rc_m1[i] = np.sum(recon_m1_amp[timestep,pc_number])
        
        recon_m2_amp = mssa_m2_amp.getRC([n_chan+i, 0])
        pc_rc_m2[i] = np.sum(recon_m2_amp[timestep,pc_number])
        
        recon_m1_m2_ratio = mssa_m1_m2_ratio.getRC([2*n_chan+i, 0])
        pc_rc_m1_m2[i] = np.sum(recon_m1_m2_ratio[timestep,pc_number])
        
        axs[1,0].plot(coefs.Times(), np.sum(recon_m1_amp[:,pc_number], axis=1), c='gray', alpha=0.2,)
        axs[1,1].plot(coefs.Times(), np.sum(recon_m2_amp[:,pc_number], axis=1), c='gray', alpha=0.2,)
        axs[1,2].plot(coefs.Times(), np.sum(recon_m1_m2_ratio[:,pc_number], axis=1), c='gray', alpha=0.2,)

    im1 = axs[0,0].pcolormesh(T, J, np.reshape(pc_rc_m1, T.shape, 'F'), cmap='RdBu', 
                         norm=mpl.colors.SymLogNorm(vmin=-2, vmax=2, linthresh=0.01))
    axs[0,0].set_yticks([1000, 2000, 3000])
    axs[0,0].set_rmax(3000)
    axs[0,0].tick_params(left = False, right = False , labelleft = True ,
                    labelbottom = False, bottom = False)
    axs[0,0].set_title('One-Armed Amp')

    im2 = axs[0,1].pcolormesh(T, J, np.reshape(pc_rc_m2, T.shape, 'F'), cmap='RdBu', 
                         norm=mpl.colors.SymLogNorm(vmin=-2, vmax=2, linthresh=0.01))
    axs[0,1].set_yticks([1000, 2000, 3000])
    axs[0,1].set_rmax(3000)
    # axs[0,1].text(3*np.pi/2, 100, r'{} Gyr'.format(np.around(timestep)/100), fontsize=15, 
    #         ha="center", c='k')
    axs[0,1].tick_params(left = False, right = False , labelleft = True ,
                    labelbottom = False, bottom = False)
    axs[0,1].set_title('Two-Armed Amp')

    im3 = axs[0,2].pcolormesh(T, J, np.reshape(pc_rc_m1_m2, T.shape, 'F'), cmap='RdBu', 
                         norm=mpl.colors.SymLogNorm(vmin=-2, vmax=2, linthresh=0.01))
    axs[0,2].set_yticks([1000, 2000, 3000])
    axs[0,2].set_rmax(3000)
    axs[0,2].tick_params(left = False, right = False , labelleft = True ,
                    labelbottom = False, bottom = False)
    #axs[0,2].text(3*np.pi/2, 100, r'{} Myr'.format(np.around(timestep)), fontsize=17, 
    #        ha="center", c='k')
    axs[0,2].set_title('One-Arm / Two-Arm')
    
    axs[1,0].axvline(timestep, ls='--', color='r')
    axs[1,1].axvline(timestep, ls='--', color='r')
    axs[1,2].axvline(timestep, ls='--', color='r')
    
    plt.subplots_adjust(hspace=0.2, wspace=0.1)
    #plt.suptitle('Time = {} Gyr'.format(np.around(timestep)/100), fontsize=25, y=0.5, x=0.5, va='center', ha='center')
    plt.draw()
    
    
def animate_pc_recon_one_plot(timestep, pc_number, mssa_amp, n_chan, axs, vert):
    
    jphi_c = np.linspace(1000, 3000, 20+1)
    tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
    rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
    tphi_c = tphi_c_[:-1] + rad[1]
    J, T = np.meshgrid(jphi_c, tphi_c)

    ax1 = axs[0, 1]
    ylim = ax1.get_ylim()
    if np.abs(ylim[0]) > np.abs(ylim[1]):
        vmin, vmax = ylim[0], -ylim[0]
    else:
        vmin, vmax = -ylim[1], ylim[1]
    ax2 = axs[1, 1]
    
    axs[0,0].clear() 
    
    pc_rc = np.zeros(n_chan)
    for i in range(n_chan):
        recon_amp = mssa_amp.getRC([i, 0])
        pc_rc[i] = np.sum(recon_amp[timestep,pc_number])
    
    axs[0,0] = plt.subplot(1, 2, 1, projection='polar')
    
    im1 = axs[0,0].pcolormesh(T, J, np.reshape(pc_rc, T.shape, 'F'), cmap='RdBu', 
                              norm=mpl.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=vmax/4))
    axs[0,0].set_yticks([1000, 2000, 3000])
    axs[0,0].set_rmax(3000)
    axs[0,0].tick_params(left = False, right = False , labelleft = True ,
                    labelbottom = False, bottom = False)
    axs[0,0].set_title('One-Armed Amp')
    
    vert1, vert2 = vert
    vert1.set_xdata(timestep*0.009778)
    vert2.set_xdata(timestep*0.009778)
    
    plt.subplots_adjust(hspace=0, wspace=0.3)
    plt.tight_layout()
    plt.draw()
    
    
def make_recon_mov_one_plot(animate, pc_number, coefs, n_chan, mssa_amp, filename, sgr_data):
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={'height_ratios': [5, 2]}, sharex='col')
    
    fig.suptitle('PC = {} Contribution'.format(pc_number), fontsize=20)

    time_array = np.array(coefs.Times())*0.009778
    # plot the parts that don't change from timestep to timestep
    ax1 = axs[0, 1]
    ax2 = axs[1, 1]
    
    radii = 21 ; thetas = 16
    color = iter(cm.rainbow(np.linspace(0, 1, radii+1)))

    # just for labelling:
    c = next(color)
    recon_amp = mssa_amp.getRC([0, 0])
    ax1.plot(time_array, np.sum(recon_amp[:,pc_number], axis=1), 
                    c=c, alpha=0.8, label='Jphi = {}'.format(1000))

    for r in range(radii): 
        recon_amp = mssa_amp.getRC([16*r, 0])
        for t in range(thetas):
            i = 16*r+t
            recon_amp = mssa_amp.getRC([i, 0])
            ax1.plot(time_array, np.sum(recon_amp[:,pc_number], axis=1), c=c, alpha=0.05)

        c = next(color)
    recon_amp = mssa_amp.getRC([n_chan-1, 0])
    ax1.plot(time_array, np.sum(recon_amp[:,pc_number], axis=1), 
                    c=c, alpha=0.8, label='Jphi = {}'.format(100*r+1000))
    ax1.set_ylabel('Amplitude')
    ax1.legend()


    ax2.plot(time_array, sgr_data.R, c='r', label='Sgr. R')
    ax2.plot(time_array, np.abs(sgr_data.z), c='b', label='Sgr. |z|')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time [Gyr]')
    ax2.set_ylabel('Distance [kpc]')
    ax1.set_xlim(0, 6)
    ymin, ymax = ax1.get_ylim()
    if np.abs(ymin) < np.abs(ymax):
        ax1.set_ylim(-ymax, ymax)
    else:
        ax1.set_ylim(ymin, -ymin)
    #ax1.set_ylim(-2, 2)
    ax2.legend()
    
    vert1 = ax1.axvline(0, ls='--', color='r')
    vert2 = ax2.axvline(0, ls='--', color='r') 
    vert = [vert1, vert2]
    
    anim = FuncAnimation(
        fig,
        partial(animate, pc_number=pc_number, mssa_amp=mssa_amp, n_chan=n_chan, axs=axs, vert=vert),
        frames=np.arange(0, int(len(coefs.Times())), 1),
        interval=2,
        blit=False,
    )
    f = filename
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f, writer = FFwriter)
    
def make_m1m2_recon_gif(animate, pc_number, coefs, n_chan, mssa_m1_amp, mssa_m2_amp, mssa_m1_m2_ratio, filename):
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharey='row')
    
    fig.suptitle('PC = {} Contribution'.format(pc_number), fontsize=20)
    
    anim = FuncAnimation(
        fig,
        partial(animate, pc_number=pc_number, coefs = coefs, n_chan=n_chan,
                         ax1_val = mssa_m1_amp, ax2_val = mssa_m2_amp, ax3_val = mssa_m1_m2_ratio,
                         axs=axs),
        frames=np.arange(0, int(len(coefs.Times())), 1),
        interval=2,
        blit=False,
    )
    f = filename
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f, writer = FFwriter)
     
    
def plot_pc_amp_mean_dispersion(pc, mssa, shortname, n_chan, coefs, sgr_data, radii=21, angles=16):
    #for each principal component, I want the mean and dispersion overall, 
    #  for different radii, for different angles
    #pc_amp = np.zeros((n_chan, len(coefs.Times())))
    fig, axs = plt.subplots(3, 3, figsize=(20, 8), sharex=True, 
                            gridspec_kw={'height_ratios': [2, 1, 2]} )
    
    mssa.reconstruct(pc)
    get_recon = mssa.getReconstructed()
    recon_all_coefs = get_recon[list(get_recon.keys())[0]].getAllCoefs()

    if shortname == 'm0_amp':
        n_data = 0
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm1_amp':
        n_data = 1
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm2_amp':
        n_data = 2
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm1_pitch':
        n_data = 3
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm2_pitch':
        n_data = 4
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm1_phase':
        n_data = 5
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm2_phase':
        n_data = 6
        pc_rc = recon_all_coefs[n_data*n_chan : (1+n_data)*n_chan, :]
    elif shortname == 'm1_combined':
        recon_m1_amp_coefs = get_recon[list(get_recon.keys())[0]].getAllCoefs()
        recon_m1_pitch_coefs = get_recon[list(get_recon.keys())[1]].getAllCoefs()
        pc_rc = np.concatenate([recon_m1_amp_coefs, recon_m1_pitch_coefs])
        #n_chan=n_chan*2
    
    for i in range(n_chan):
        axs[0, 0].plot(coefs.Times(), pc_rc[i,:], c='k', alpha=0.2,)
        
        
    #plot some summary stats of Sgr orbit
    for i in range(3):
        axs[1,i].plot(coefs.Times(), sgr_data.R, c='r', label='Sgr. R')
        axs[1,i].plot(coefs.Times(), np.abs(sgr_data.z), c='b', label='Sgr. |z|')
        axs[1,i].set_yscale('log')
        axs[1,i].legend()


    color = iter(cm.rainbow(np.linspace(0, 1, radii)))
    for r in range(radii): 
        c = next(color)
        axs[0, 1].plot(coefs.Times(), np.mean(pc_rc[336+angles*r:336+angles*(r+1),:], axis=0), 
                        c=c, alpha=0.5, label='Mean for jphi = {}'.format(100*r+1000))
        axs[2, 1].plot(coefs.Times(), np.std(pc_rc[angles*r:angles*(r+1)], axis=0), 
                        c=c, alpha=0.5, label='std for jphi = {}'.format(100*r+1000))

    color = iter(cm.rainbow(np.linspace(0, 1, angles)))
    for t in range(angles):
        c = next(color)
        axs[0, 2].plot(coefs.Times(), np.mean(pc_rc[t::angles, :], axis=0), c=c, alpha=0.5, 
                       label='mean for theta_phi = {}'.format(np.around(t*np.pi/8+np.pi/16, 2)))
        axs[2, 2].plot(coefs.Times(), np.std(pc_rc[t::angles, :], axis=0), c=c, alpha=0.5, 
                       label='std for theta_phi = {}'.format(np.around(t*np.pi/8+np.pi/16, 2)))

    #plot mean amplitude
    axs[0, 1].plot(coefs.Times(), np.mean(pc_rc[336:, :], axis=0), 
                   c='k', alpha=1, lw=3, ls='-.',label='Mean all channels')
    axs[0, 2].plot(coefs.Times(), np.mean(pc_rc[:336,:], axis=0), 
                   c='k', alpha=1, lw=3, ls='-.', label='Mean all channels')

    axs[2, 0].plot(coefs.Times(), np.std(pc_rc[:336,:], axis=0), 
                   c='k', alpha=1, lw=3, label='std all channels')
    axs[2, 1].plot(coefs.Times(), np.std(pc_rc[:336,:], axis=0), 
                   c='k', alpha=1, lw=3, ls='-.', label='std all channels')
    axs[2, 2].plot(coefs.Times(), np.std(pc_rc[:336,:], axis=0), 
                   c='k', alpha=1, lw=3, ls='-.', label='std all channels')

    for ax in [axs[0, 1], axs[0,2], axs[2,1], axs[2,2]]:
        handles, labels = ax.get_legend_handles_labels()
        # Only show the first and last handles and labels
        handles = [handles[-1], handles[0], handles[-2]]
        labels = [labels[-1], labels[0], labels[-2]]
        ax.legend(handles, labels, fontsize=15)
    axs[2,0].legend(fontsize=15)
    
    axs[2,0].set_xlabel('Time')
    axs[2,1].set_xlabel('Time')
    axs[2,2].set_xlabel('Time')
    axs[0,0].set_ylabel('PC{} Contribution'.format(pc))
    axs[2,0].set_ylabel('Std PC{} Contribution'.format(pc))
    
    plt.suptitle('PC{} Amp Mean and Dispersion'.format(pc), fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.1)
    #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/pc{}_amp_mean_disp_{}.pdf'.format(pc, mssa_short))
    #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/pc{}_amp_mean_disp_{}.png'.format(pc, mssa_short))
    plt.show()
    
    
    
def plot_frequency_mean_disp(mssa, mssa_short, radii=21, angles=16):
    #for each principal component, I want the mean and dispersion overall, 
    #  for different radii, for different angles
    
    freq_amp, power_amp = mssa.channelDFT()
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey='row')
            
    for i in range(power_amp.shape[1]):
        axs[0, 0].plot(freq_amp[0:40], power_amp[0:40,i], c='k', alpha=0.2,)#, label=str(i))


    color = iter(cm.rainbow(np.linspace(0, 1, radii)))
    for r in range(radii): 
        c = next(color)
        axs[0, 1].plot(freq_amp[0:40], np.mean(power_amp[0:40 , angles*r:angles*(r+1)], axis=1), 
                        c=c, alpha=0.5, label='Mean for jphi = {}'.format(100*r+1000))
        axs[1, 1].plot(freq_amp[0:40], np.std(power_amp[0:40 , angles*r:angles*(r+1)], axis=1), 
                        c=c, alpha=0.5, label='std for jphi = {}'.format(100*r+1000))

    color = iter(cm.rainbow(np.linspace(0, 1, angles)))
    for t in range(angles):
        c = next(color)
        axs[0, 2].plot(freq_amp[0:40], np.mean(power_amp[0:40 , t::angles], axis=1), c=c, alpha=0.5, 
                       label='mean for theta_phi = {}'.format(np.around(t*np.pi/8+np.pi/16, 2)))
        axs[1, 2].plot(freq_amp[0:40], np.std(power_amp[0:40 , t::angles], axis=1), c=c, alpha=0.5, 
                       label='std for theta_phi = {}'.format(np.around(t*np.pi/8+np.pi/16, 2)))

    #plot mean amplitude
    axs[0, 1].plot(freq_amp[0:40], np.mean(power_amp[0:40], axis=1), 
                   c='k', alpha=1, lw=3, ls='-.',label='Mean all channels')
    axs[0, 2].plot(freq_amp[0:40], np.mean(power_amp[0:40], axis=1), 
                   c='k', alpha=1, lw=3, ls='-.', label='Mean all channels')

    axs[1, 0].plot(freq_amp[0:40], np.std(power_amp[0:40], axis=1), 
                   c='k', alpha=1, lw=3, label='std all channels')
    axs[1, 1].plot(freq_amp[0:40], np.std(power_amp[0:40], axis=1), 
                   c='k', alpha=1, lw=3, ls='-.', label='std all channels')
    axs[1, 2].plot(freq_amp[0:40], np.std(power_amp[0:40], axis=1), 
                   c='k', alpha=1, lw=3, ls='-.', label='std all channels')

    for ax in axs[:, 1:].flat:
        handles, labels = ax.get_legend_handles_labels()
        # Only show the first and last handles and labels
        handles = [handles[-1], handles[0], handles[-2]]
        labels = [labels[-1], labels[0], labels[-2]]
        ax.legend(handles, labels, fontsize=15)
    axs[1,0].legend(fontsize=15)
    
    axs[1,0].set_xlabel('Time')
    axs[1,1].set_xlabel('Time')
    axs[1,2].set_xlabel('Time')
    axs[0,0].set_ylabel('Contribution')
    axs[1,0].set_ylabel('Std Contribution')
    
    axs[0,0].set_ylim(1, 1e6)
    axs[1,0].set_ylim(1e2, 1e6)
    
    axs[0,0].set_yscale('log')
    axs[1,0].set_yscale('log')
    plt.suptitle('Amp Mean and Dispersion', fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.1)
    #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/freq_mean_disp_{}.pdf'.format(mssa_short))
    #plt.savefig('/mnt/home/ktavangar/projects/MSSA_Snails/figures/shorter_sim/freq_mean_disp_{}.png'.format(mssa_short))
    plt.show()
