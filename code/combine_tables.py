import os

from astropy.table import Table, vstack
import glob
import numpy as np


def save_coefs(dat_file, save_dir, start_timestep=40, end_timestep=None):
    file = np.loadtxt(dat_file)
    n_times = file.shape[0]
    if end_timestep==None:
        times = np.reshape(file[start_timestep:,0], (n_times-start_timestep,1))
    else:
        times = np.reshape(file[start_timestep:end_timestep,0], (end_timestep-start_timestep,1))
    
    n_channels = file.shape[1]//9

    m0_amp_      = file[start_timestep:end_timestep,1:1*n_channels+1]
    m1_amp_      = file[start_timestep:end_timestep,1*n_channels+1:2*n_channels+1]
    m2_amp_      = file[start_timestep:end_timestep,2*n_channels+1:3*n_channels+1]
    m1_pitch_    = file[start_timestep:end_timestep,3*n_channels+1:4*n_channels+1]
    m2_pitch_    = file[start_timestep:end_timestep,4*n_channels+1:5*n_channels+1]
    m1_phase_    = file[start_timestep:end_timestep,5*n_channels+1:6*n_channels+1]
    m2_phase_    = file[start_timestep:end_timestep,6*n_channels+1:7*n_channels+1]
    m1_int_time_ = file[start_timestep:end_timestep,7*n_channels+1:8*n_channels+1]
    m2_int_time_ = file[start_timestep:end_timestep,8*n_channels+1:9*n_channels+1]
    
    m0_amp = np.concatenate([times, m0_amp_], axis=1)
    m1_amp = np.concatenate([times, m1_amp_], axis=1)
    m2_amp = np.concatenate([times, m2_amp_], axis=1)
    m1_pitch = np.concatenate([times, m1_pitch_], axis=1)
    m2_pitch = np.concatenate([times, m2_pitch_], axis=1)
    m1_phase = np.concatenate([times, m1_phase_], axis=1)
    m2_phase = np.concatenate([times, m2_phase_], axis=1)
    m1_int_time = np.concatenate([times, m1_int_time_], axis=1)
    m2_int_time = np.concatenate([times, m2_int_time_], axis=1)
    
    np.savetxt(save_dir + 'm0_amp.dat', m0_amp)
    np.savetxt(save_dir + 'm1_amp.dat', m1_amp)
    np.savetxt(save_dir + 'm2_amp.dat', m2_amp)
    np.savetxt(save_dir + 'm1_pitch.dat', m1_pitch)
    np.savetxt(save_dir + 'm2_pitch.dat', m2_pitch)
    np.savetxt(save_dir + 'm1_phase.dat', m1_phase)
    np.savetxt(save_dir + 'm2_phase.dat', m2_phase)
    np.savetxt(save_dir + 'm1_int_time.dat', m1_int_time)
    np.savetxt(save_dir + 'm2_int_time.dat', m2_int_time)


if __name__ == '__main__':
    files = glob.glob('../data/B2_lagrangian_cache/*')

    all_table = vstack([Table.read(file) for file in files])

    all_table.write('../data/lagrangian_B2_mssa_prep_table.fits', format='fits', overwrite=True)

    ## Going from fits to the .dat files that mSSA expects
    all_table.sort(["timestep", "jphi_cen", "tphi_cen"])


    jphi_c = np.arange(1000, 4000+1, 100)
    tphi_c_ = np.linspace(0, 2*np.pi, 16+1)
    rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
    tphi_c = tphi_c_[:-1] + rad[1]
    centers = np.array(np.meshgrid(jphi_c, tphi_c)).T.reshape(-1,2)

    cfs_m0_amp    = ['m0_amp_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m1_amp    = ['m1_amp_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m2_amp    = ['m2_amp_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m1_pitch  = ['m1_pitch_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m2_pitch  = ['m2_pitch_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m1_phase  = ['m1_phase_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m2_phase  = ['m2_phase_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m1_int_time = ['m1_int_time_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]
    cfs_m2_int_time = ['m2_int_time_{}_{}pi16'.format(int(cf[0]), int(16*cf[1]/np.pi)) for cf in centers]

    colnames = ['timestep']+cfs_m0_amp+cfs_m1_amp+cfs_m2_amp+ \
                            cfs_m1_pitch+cfs_m2_pitch+ \
                            cfs_m1_phase+cfs_m2_phase+ \
                            cfs_m1_int_time+cfs_m2_int_time

    t = Table(names=colnames)

    ratio = 0.009778
    for time in np.unique(all_table['timestep']):
        subset = all_table[all_table['timestep'] == time]
        channels = np.concatenate([np.array(subset['m0_amp']), np.array(subset['m1_amp']), np.array(subset['m2_amp']),
                                np.array(subset['pitch_ang_m1']), np.array(subset['pitch_ang_m2']), 
                                np.array(subset['phase_ang_m1']), np.array(subset['phase_ang_m2']),
                                np.array(time*ratio - subset['time_since_int_m1']),
                                np.array(time*ratio - subset['time_since_int_m2'])])
        t.add_row(np.append([time], channels))

    dat_file = '../data/lagrangian_B2_particle.dat'
    np.savetxt(dat_file, t)

    save_dir1 = '../data/mssa_channels_lagrangian_B2/'
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    save_coefs(dat_file, save_dir1, start_timestep=0, end_timestep=None)

    save_dir2 = '../data/mssa_channels_lagrangian_B2_first_passage/'
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    save_coefs(dat_file, save_dir2, start_timestep=0, end_timestep=200)

    save_dir3 = '../data/mssa_channels_lagrangian_B2_second_passage/'
    if not os.path.exists(save_dir3):
        os.makedirs(save_dir3)
    save_coefs(dat_file, save_dir3, start_timestep=200, end_timestep=None)