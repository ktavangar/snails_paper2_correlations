import numpy as np
from astropy.table import Table

from load_data_B2 import setup_B2
from load_data import load_data_actions
from df_helpers import LaguerreSnails

'''
General Idea:
1) Take timestep as an input and import relevant data from that timestep
    a) This will allow parallelizing of the code
2) Split data into J_phi, theta_phi regions based on inputs (or hardcode)
3) Generate the BFE for the phase space spiral in each region
    a) Use 15 m=1 coefficients and 15 m=2 coefficients (can change later)
4) Create a table with the following columns:
    a) Timestep
    b) J_phi center
    c) theta_phi center
    d) 30 columns of individual coefficients
    e) m=1 combined amplitude
    f) m=2 combined amplitude
    g) Statistic(s) for pitch angle
  If we are using the full data, also add:
    h) Mean v_r
    i) Mean v_z
    j) Mean v_phi
5) Save table
6) Do for all timesteps
7) Combine all tables in master table (how large will this be)
'''

class MSSATable:
    
    def __init__(self, jphi_bounds=[1000,3000], nphi_bins=[20, 16], m_max=3, n_max=20):
        
        # GETTING ALL REGION CENTERS
        jphi_c = np.linspace(jphi_bounds[0], jphi_bounds[1], nphi_bins[0]+1)
        tphi_c_ = np.linspace(0, 2*np.pi, nphi_bins[1]+1)
        self.rad = [0.5*(jphi_c[1] - jphi_c[0]), 0.5*(tphi_c_[1] - tphi_c_[0])]
        tphi_c = tphi_c_[:-1] + self.rad[1]
        self.centers = np.array(np.meshgrid(jphi_c, tphi_c)).T.reshape(-1,2)

        self.m_max, self.n_max = m_max, n_max
        self.ms, self.ns = np.arange(m_max), np.arange(n_max)

        # SETTING UP TABLE
        # self.cs = np.array(np.meshgrid(self.ms, self.ns)).T.reshape(-1,2)
        # cfs = ['m{}n{}'.format(cf[0], cf[1]) for cf in self.cs[self.n_max:]]
        other_cols = ['timestep', 'jphi_cen', 'tphi_cen',
                  'm0_amp', 'm1_amp', 'm2_amp', 
                  'pitch_ang_m1', 'pitch_ang_m2',
                  'phase_ang_m1', 'phase_ang_m2',
                  'pitch_phase_flag_m1', 'pitch_phase_flag_m2',
                  'time_since_int_m1', 'time_since_int_m2']#,
                  #'mean_vr', 'mean_vphi', 'mean_vz']
            
        self.colnames = other_cols# + cfs

        self.jz_grid = (np.arange(0, 10, 0.1))**2
        self.tz_grid = np.arange(0, 2*np.pi, np.pi/48)


    def create_empty_table(self):
        empty_t = Table(names=self.colnames)
        return empty_t

    def create_table(self):
        empty_table = np.zeros((len(self.centers), len(self.colnames)))
        t = Table(empty_table, names=self.colnames)
        return t

    def fill_table(self, timestep, sim, data_file, action_file):
    
        self.timestep = int(timestep)

        # LOADING DATA
        if sim=='live':
            data = setup_B2(self.timestep)
        elif sim=='test':
            data = load_data_actions(data_file=data_file, 
                                     action_file=action_file)
        
        ######################
        ## Get coefficients ##
        ######################
        print('Getting Coefficients... \n')

        all_coeff_array = np.zeros((len(self.centers), self.m_max, self.n_max), dtype = np.complex_)
        pitch_angle_m1 = np.zeros(len(self.centers))
        phase_angle_m1 = np.zeros(len(self.centers))
        pitch_angle_m2 = np.zeros(len(self.centers))
        phase_angle_m2 = np.zeros(len(self.centers))
        pitch_phase_flag_m1 = np.zeros(len(self.centers))
        pitch_phase_flag_m2 = np.zeros(len(self.centers))
        time_since_int_m1 = np.zeros(len(self.centers))
        time_since_int_m2 = np.zeros(len(self.centers))
        # mean_vrs = np.zeros(len(self.centers))
        # mean_vphis = np.zeros(len(self.centers))
        # mean_vzs = np.zeros(len(self.centers))
        
        for i in range(len(self.centers)):
            cen = self.centers[i]
            print('Center: {}'.format(cen))

            LS = LaguerreSnails(data, cen, self.rad, 
                                self.jz_grid, self.tz_grid, 
                                self.m_max, self.n_max, 
                                self.timestep)
            coeffs = LS.get_coeffs()

            all_coeff_array[i] = coeffs
            
            pitch_angle_m1[i], phase_angle_m1[i], pitch_phase_flag_m1[i], time_since_int_m1[i] = LS.get_pitch_phase_angles(m=1)
            pitch_angle_m2[i], phase_angle_m2[i], pitch_phase_flag_m2[i], time_since_int_m2[i] = LS.get_pitch_phase_angles(m=2)
            
            # mean_vrs[i], mean_vphis[i], mean_vzs[i] = LS.get_mean_vels()

        t = self.create_table()

        ################
        ## Fill Table ##
        ################

        timestep = self.timestep * np.ones(len(self.centers))
        t['timestep'] = timestep

        jphi_cen, tphi_cen = self.centers[:,0], self.centers[:,1]
        t['jphi_cen'], t['tphi_cen'] = jphi_cen, tphi_cen

        m0_amp = np.linalg.norm(np.abs(all_coeff_array[:, 0, :]), axis=1)
        m1_amp = np.linalg.norm(np.abs(all_coeff_array[:, 1, :]), axis=1)
        m2_amp = np.linalg.norm(np.abs(all_coeff_array[:, 2, :]), axis=1)
        t['m0_amp'], t['m1_amp'], t['m2_amp'] = m0_amp, m1_amp, m2_amp

        t['pitch_ang_m1'] = pitch_angle_m1
        t['pitch_ang_m2'] = pitch_angle_m2
        t['phase_ang_m1'] = phase_angle_m1
        t['phase_ang_m2'] = phase_angle_m2
        t['pitch_phase_flag_m1'] = pitch_phase_flag_m1
        t['pitch_phase_flag_m2'] = pitch_phase_flag_m1
        t['time_since_int_m1'] = time_since_int_m1
        t['time_since_int_m2'] = time_since_int_m2

#         Doing mean velocities will have to be within LS,
#          since that is where I split into regions
#         t['mean_vr'] = mean_vrs
#         t['mean_vphi'] = mean_vphis
#         t['mean_vz'] = mean_vzs

#         for cf in self.cs[self.n_max:]:
#             t['m{}n{}'.format(cf[0], cf[1])] = all_coeff_array[:, cf[0], cf[1]]

        self.t = t
        return self.t

    def save_table(self):
        fname = 'data/tables/t{}.fits'.format(self.timestep)
        self.t.write(fname, overwrite=True)
