import numpy as np
import pyEXP

import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

import sys
import os
import helper

class MSSAOperations:
    def __init__(self, data_file, channel_name, figure_directory, window_frac=0.5, number_of_pcs=30):
        self.data_file = data_file
        self.channel_name = channel_name
        self.n_pc = number_of_pcs
        
        self.figure_directory = figure_directory
        if not os.path.exists(self.figure_directory):
            os.mkdir(self.figure_directory)
        else:
            print('figure directory exists - adding to it')
        
        #creates mssa object
        self.coefs = pyEXP.coefs.Coefs.factory(data_file)

        n_channels = int(len(self.coefs.getAllCoefs()))
        self.times = self.coefs.Times()
        
        keylst = [[i] for i in range(n_channels)]
        
        config = {channel_name: (self.coefs, keylst, [])}
        
        # Window size is half the time series (a good default choice if it's practical)
        window_size = int(window_frac * len(self.times))
        
        self.mssa = pyEXP.mssa.expMSSA(config, window_size, self.n_pc)
        
        # start this as nothing for later
        self.MakeAnim = None

    def save_mssa(self, file):
        self.mssa.saveState(file)

    def load_mssa(self, file):
        self.mssa.restoreState() #??? How does this work?

    def make_diagnostic_plots(self):
    
        self.ev = self.mssa.eigenvalues()
        self.coefs.zerodata()
        self.mssa.reconstruct([*range(self.n_pc)])
    
        self.make_ev_and_wCorr_plots(figure_directory=self.figure_directory)
    
        self.make_pc_plots(self.figure_directory)

        self.make_fg_matrix_plots(self.figure_directory)

    def make_data_movie(self, norm_function, cmap, 
                        sim_name='B2', jphi_min=1000, jbins=31, **kwargs):
        # kwargs are for the norm_function arguments

        data_tbl = np.loadtxt(self.data_file)

        
        face_on_plots_dir = self.figure_directory+'/face-on_plots/'
        if not os.path.exists(face_on_plots_dir):
            os.mkdir(face_on_plots_dir)
        filename = face_on_plots_dir+'data.mp4'
        
        self.MakeAnim = helper.MakeAnimations(mssa=self.mssa, sim_name=sim_name, channel_name=self.channel_name, 
                                  times=self.times, jphi_min=jphi_min, jbins=jbins)
        
        if not os.path.exists(filename):
            self.MakeAnim.make_data_mov(filename, data_tbl, norm_function=norm_function, cmap=cmap,  **kwargs)
        else: # at some point may want to skip if file already created
            self.MakeAnim.make_data_mov(filename, data_tbl, norm_function=norm_function, cmap=cmap,  **kwargs)

        return self.MakeAnim

    def make_pc_movies(self, list_of_pc_lists, norm_function, cmap, 
                       sim_name='B2', jphi_min=1000, jbins=31, **kwargs):
        # both norm_function and cmap should be for the non-mean-subtracted versions
        # figure directory should be previous directory 
        face_on_plots_dir = self.figure_directory + '/face-on_plots/'
        if not os.path.exists(face_on_plots_dir):
            print('creating face-on plots directory...')
            os.mkdir(face_on_plots_dir)

        self.MakeAnim = helper.MakeAnimations(mssa=self.mssa, sim_name=sim_name, channel_name=self.channel_name, 
                                  times=self.times, jphi_min=jphi_min, jbins=jbins)
        
        for pc_list in list_of_pc_lists:
            print('Creating movie for {}'.format(pc_list))
            self.MakeAnim.reconstruct_from_pcs(pcs=pc_list)
            self.MakeAnim.make_pc_reconstruction_mov(file_directory=face_on_plots_dir, subtract_mean=False, 
                                                norm_function=norm_function, cmap=cmap, **kwargs)

            if norm_function == mpl.colors.LogNorm:
                norm_function_mean_subtracted = mpl.colors.SymLogNorm
                kwargs2 = {}
                kwargs2['linthresh'] = kwargs['vmin']
                kwargs2['vmin'] = -kwargs['vmax']
                kwargs2['vmax'] = kwargs['vmax']
            elif 'int_time' in self.channel_name:
                norm_function_mean_subtracted = mpl.colors.SymLogNorm
                kwargs2 = {}
                kwargs2['linthresh'] = 0.1
                kwargs2['vmin'] = -2
                kwargs2['vmax'] = 2
            elif 'phase' in self.channel_name:
                norm_function_mean_subtracted = mpl.colors.Normalize
                kwargs2 = {}
                kwargs2['vmin'] = -np.pi/2
                kwargs2['vmax'] = np.pi/2
            elif norm_function == mpl.colors.Normalize:
                norm_function_mean_subtracted = mpl.colors.Normalize
                kwargs2 = {}
                kwargs2['vmin'] = kwargs['vmin'] - 0.5*(kwargs['vmin']+kwargs['vmax'])
                kwargs2['vmax'] = kwargs['vmax'] - 0.5*(kwargs['vmin']+kwargs['vmax'])
            self.MakeAnim.make_pc_reconstruction_mov(file_directory=face_on_plots_dir, subtract_mean=True, 
                                                norm_function=norm_function_mean_subtracted, cmap=cmr.prinsenvlag_r, **kwargs2)

    def make_ev_and_wCorr_plots(self, figure_directory, ev_n_pcs=50, wCorr_n_pcs = 30):
        fig, [ax1, ax2] = plt.subplots(1,2,figsize=(12,5))
        ax1.semilogy(self.ev[:ev_n_pcs], '-o')
        ax1.set_xlabel('index', fontsize=20)
        ax1.set_ylabel('eigenvalue', fontsize=20)
        ax1.set_title('PC Eigenvalues', fontsize=20)
    
        ax2.imshow(self.mssa.wCorrAll()[:wCorr_n_pcs, :wCorr_n_pcs], cmap='gray_r')
    
        plt.tight_layout(pad=0)
        plt.savefig(figure_directory+'/ev+wcorr.png')
        plt.close()

    def make_fg_matrix_plots(self, figure_directory):
        t1, t2 = self.mssa.contrib()
        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(15, 8), sharex=True, sharey=True)
        ax1.imshow(t1, aspect='auto', norm=mpl.colors.LogNorm())
        ax1.set_title('PC contributions to channels', fontsize=16)
        ax2.imshow(t2, aspect='auto', norm=mpl.colors.LogNorm())
        ax2.set_title('Channel contribution to PCs', fontsize=16)
        ax2.set_xlabel('Channel number')
        ax1.set_ylabel('PC number')
        ax2.set_ylabel('PC_number')
        plt.tight_layout()
        plt.savefig(figure_directory+'/fg_matrices.png')
        plt.close()
        
    def make_pc_plots(self, figure_directory):
        pc = self.mssa.getPC()
    
        nt = pc.shape[0]
        lag_times = self.times[:nt]
        
        n_plots = 6
        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 1.5*n_plots), sharex=True)
        
        for i in range(0,2):
            axs[0].plot(lag_times, pc[:,i], label=str(i)) 
        
        for i in range(2,4):
            axs[1].plot(lag_times, pc[:,i], label=str(i))    
            
        for i in range(4,6):
            axs[2].plot(lag_times, pc[:,i], label=str(i))    
            
        for i in range(6,10):
            axs[3].plot(lag_times, pc[:,i], label=str(i))
            
        for i in range(10,14):
            axs[4].plot(lag_times, pc[:,i], label=str(i))  
            
        for i in range(14,18):
            axs[5].plot(lag_times, pc[:,i], label=str(i))  
        
        axs[2].set_ylabel('PC Amplitude', fontsize=20)
        for i in range(n_plots):
            axs[i].legend(fontsize=8, loc='lower left')
        axs[-1].set_xlabel('Lag Time', fontsize=20)
                
        fig.tight_layout(pad=0)
        plt.savefig(figure_directory+'/PCs.png')
        plt.close()
    
def create_datafile_list(data_dir, channel_name_list=['amp', 'pitch', 'phase', 'int_time'], 
                         specification_list=['', '_first_passage', '_second_passage'], 
                         section_list=['', '_inner', '_outer'],
                         jbins=31, tbins=16):
    '''
    split_disk   : create additional filenames for just the inner or outer disks
    '''
    
    datafile_suffix = '_bins_j{}_t{}.dat'.format(int(jbins-1), int(tbins))
    
    datafile_list = []
    for channel in channel_name_list:
        datafile_prefix_m1 = data_dir+'m1_'+channel
        datafile_prefix_m2 = data_dir+'m2_'+channel

        for spec in specification_list:
            for sect in section_list:
                if len(sect) > 0:
                    datafile_list.append(datafile_prefix_m1+spec+sect+'.dat')
                    datafile_list.append(datafile_prefix_m2+spec+sect+'.dat')
                else:
                    datafile_list.append(datafile_prefix_m1+spec+datafile_suffix)
                    datafile_list.append(datafile_prefix_m2+spec+datafile_suffix)
            

    return datafile_list

def gen_figure_directory_names(channel_name_list=['amp', 'pitch', 'phase', 'int_time'],
                               specification_list=['', '_first_passage', '_second_passage'],
                               section_list=['', '_outer', '_inner']):
    
    fig_dir_names = []
    for channel in channel_name_list:
        figdir_prefix_m1 = 'm1_'+channel
        figdir_prefix_m2 = 'm2_'+channel

        for spec in specification_list:

            for sect in section_list:
                if (spec=='') & (sect==''):
                    fig_dir_names.append(figdir_prefix_m1+'_fiducial')
                    fig_dir_names.append(figdir_prefix_m2+'_fiducial')
                else:
                    fig_dir_names.append(figdir_prefix_m1+spec+sect)
                    fig_dir_names.append(figdir_prefix_m2+spec+sect)
    return fig_dir_names

def run_all_diagnostic_plots(datafile_names, fig_dir_names, short_window=False):
    print('Please be sure that the datafile name and the figure directory names are in the same order')
    assert len(datafile_names) == len(fig_dir_names), 'data and figure directory lists are not the same size'

    for i, data_fname in enumerate(datafile_names):
        print('Creating diagnostic plots for '+data_fname)
        if 'm1' in data_fname:
            if short_window:
                MSSA = MSSAOperations(datafile_names[i], 'one-armed '+channel, '../figures/B2_figures/'+fig_dir_names[i]+'_window0p25')
            else:
                MSSA = MSSAOperations(datafile_names[i], 'one-armed '+channel, '../figures/B2_figures/'+fig_dir_names[i])
        elif 'm2' in data_fname:
            if short_window:
                MSSA = MSSAOperations(datafile_names[i], 'two-armed '+channel, '../figures/B2_figures/'+fig_dir_names[i]+'_window0p25')
            else:
                MSSA = MSSAOperations(datafile_names[i], 'two-armed '+channel, '../figures/B2_figures/'+fig_dir_names[i])
        MSSA.make_diagnostic_plots()
        print('')


if __name__ == '__main__': 

    chan_name_list=['amp', 'pitch', 'phase', 'int_time']
    spec_list = ['', '_first_passage', '_second_passage']
    data_fnames = create_datafile_list(data_dir='../data/mssa_channels_B2/', 
                                       channel_name_list=chan_name_list, 
                                       specification_list=spec_list, 
                                       jbins=31, tbins=16, split_disk=False)

    fig_dir_names = gen_figure_directory_names(chan_name_list, spec_list)

    run_all_diagnostic_plots(data_fnames, fig_dir_names, short_window=False)
    
