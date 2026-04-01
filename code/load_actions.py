import numpy as np
import pandas as pd
import pickle

################################################
## TEST PARTICLE SIMULATION LOADING FUNCTIONS ##
################################################

def load_actions_from_file(action_file):
    with open(action_file, 'rb') as file1:
        _,jphi1,jz1,_,_,_,_,theta_phi,theta_z=pickle.load(file1)

    jz = jz1*8*220 # convert to kpc km/s
    jphi = jphi1*8*220 # convert to kpc km/s
    
    for_tbl = np.array([jphi, jz, theta_phi, theta_z])
    
    act_tbl = pd.DataFrame(for_tbl.T, 
                                columns=['jphi', 'jz','theta_phi', 'theta_z'])
    
    return act_tbl

def load_test_actions(tstep, actions_dir=''):
    action_file = actions_dir + f'Actions{int(tstep)}.p'
    act_tbl = load_actions_from_file(action_file)
    
    return act_tbl

#####################################
## B2 SIMULATION LOADING FUNCTIONS ##
#####################################

def load_B2_actions(tstep, 
                    actions_dir = '/mnt/ceph/users/jhunt/Bonsai/r2/B2/FlattenedDiscActions/'):
    actions=np.load(actions_dir + f'MedianFlattenedDiskActions{int(tstep)}.npy')

    data = {"jphi": actions[1],
            "jz": actions[2],
            "theta_phi": actions[4],
            "theta_z": actions[5]}

    act_tbl = pd.DataFrame(data, index=actions[9].astype('int'))

    return act_tbl
    