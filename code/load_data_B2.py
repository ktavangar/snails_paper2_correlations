import matplotlib
matplotlib.use('pdf')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from galpy.util import coords
from scipy import stats
import scipy
from scipy.ndimage import gaussian_filter
#import shapely.geometry as geom
import pandas as pd

def select_region(data, center, radius):
    '''
    Select a squareregion in the MW 
    Data is a dataframe object containing at least Galactocentric x and y coordinates
    Center must be in the form [x_coord, y_coord]
    Radius in kpc around the center (not from cneter of MW)
    '''
    select = data[(data.x > center[0] - radius) & (data.x < center[0] + radius) &
                  (data.y > center[1] - radius) & (data.y < center[1] + radius)]
    return select

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


def rotation_axis_angle(axis):
    #function rotation_axis_angle, axis
    unitz=[0.0,0.0,1.0]
    angle=np.pi+(-1.0)*np.arccos(axis[2]/np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2))
    u = np.zeros(3)
    u[0]=-unitz[2]*axis[1]
    u[1]=unitz[2]*axis[0]
    u[2]=0.0
    n=np.sqrt(u[0]**2+u[1]**2)
    u=u/n
    kronecker=np.zeros(shape=(3,3))
    kronecker[0,0]=1.0
    kronecker[0,1]=0.0
    kronecker[0,2]=0.0
    kronecker[1,0]=0.0
    kronecker[1,1]=1.0
    kronecker[1,2]=0.0
    kronecker[2,0]=0.0
    kronecker[2,1]=0.0
    kronecker[2,2]=1.0
    ucrossu=np.zeros(shape=(3,3))
    for i in range (0,3) :
        for j in range (0,3):
            ucrossu[i,j]=u[i]*u[j]
    u_x=np.zeros(shape=(3,3))
    u_x[0,0]=0.0
    u_x[0,1]=(-1.0)*u[2]
    u_x[0,2]=u[1]
    u_x[1,0]=u[2]
    u_x[1,1]=0.0
    u_x[1,2]=(-1.0)*u[0]
    u_x[2,0]=(-1.0)*u[1]
    u_x[2,1]=u[0]
    u_x[2,2]=0.0
    rotation=np.zeros(shape=(3,3))
    rotation = ucrossu + np.cos(angle)*(kronecker-ucrossu)+ np.sin(angle)*u_x
    return rotation

# Open the smooth snapshot, sort and transform to cyl.

def load_step_data(timestep, z_only=False):
    print('Loading Data...')
    file = open('/mnt/ceph/users/jhunt/Bonsai/r2/B2/step{}.p'.format(timestep), 'rb')
    idd0,x0,y0,z0,vx0,vy0,vz0,mass0=pickle.load(file)

    print('Manipulating Data...')
    order=np.argsort(idd0)
    idd0=idd0[order]

    z0=z0[order]
    vz0=vz0[order]

    bulgeindx0=(mass0>1e-7)*(mass0<1e-6)
    mz=np.median(z0[bulgeindx0])
    mvz=np.median(vz0[bulgeindx0])
    z0=z0-mz
    vz0=vz0-mvz
    
    
    
    if not z_only:
        x0=x0[order]
        y0=y0[order]
        vx0=vx0[order]
        vy0=vy0[order]
    
        mass0=mass0[order]

        discindx0=(mass0<1e-7)
        sgrindx0=(mass0>1e-6)

        mx=np.median(x0[bulgeindx0])
        my=np.median(y0[bulgeindx0])
        mvx=np.median(vx0[bulgeindx0])
        mvy=np.median(vy0[bulgeindx0])
    
        x0=x0-mx
        y0=y0-my
        vx0=vx0-mvx
        vy0=vy0-mvy

    rr=np.sqrt(x0[discindx0]**2+y0[discindx0]**2+z0[discindx0]**2)#distance of stars
    Jdisk = np.zeros((3, len((x0[discindx0]))))
    Jdisk[0,:] = (y0[discindx0])*(vz0[discindx0])-(z0[discindx0])*(vy0[discindx0])
    Jdisk[1,:] = (z0[discindx0])*(vx0[discindx0])-(x0[discindx0])*(vz0[discindx0])
    Jdisk[2,:] = (x0[discindx0])*(vy0[discindx0])-(y0[discindx0])*(vx0[discindx0])
    rcyl=5.      #choose 30 kpc to select particles, 4 kpc turned out not to be sufficient enough
    wr4kpc= np.where((rr < rcyl))
    jtot=np.zeros(3)
    jtot[0]=np.sum(Jdisk[0,wr4kpc])
    jtot[1]=np.sum(Jdisk[1,wr4kpc])
    jtot[2]=np.sum(Jdisk[2,wr4kpc])
    magjtot=np.sqrt(jtot[0]**2+jtot[1]**2+jtot[2]**2)
    jtot_normalised=jtot/magjtot
    rot_align_J=rotation_axis_angle(jtot_normalised)

    post=np.vstack((x0,y0,z0))
    velt=np.vstack((vx0,vy0,vz0))

    post=np.matmul(rot_align_J,post).T
    velt=np.matmul(rot_align_J,velt).T
    post[:,2]*=-1.
    post[:,0]*=-1.
    velt[:,2]*=-1.
    velt[:,0]*=-1.

    fxx=np.argmax(x0)
    oxx=x0[fxx]
    oyy=y0[fxx]

    xxx=post[:,0][fxx]
    yyy=post[:,1][fxx]

    dummy,ang1,dummy2=coords.rect_to_cyl(oxx,oyy,0.)
    dummy,ang2,dummy2=coords.rect_to_cyl(xxx,yyy,0.)

    ang=ang1-ang2

    x0=post[:,0]*np.cos(ang)-post[:,1]*np.sin(ang)
    y0=post[:,0]*np.sin(ang)+post[:,1]*np.cos(ang)
    vx0=velt[:,0]*np.cos(ang)-velt[:,1]*np.sin(ang)
    vy0=velt[:,0]*np.sin(ang)+velt[:,1]*np.cos(ang)
    z0=post[:,2]
    vz0=velt[:,2]

    r0,phi0,z0 = coords.rect_to_cyl(x0,y0,z0)
    vr0,vphi0,vz0 = coords.rect_to_cyl_vec(vx0,vy0,vz0,x0,y0,z0)
    
    return np.array([idd0, r0, phi0, z0, vr0, vphi0, vz0, x0, y0, vx0, vy0])

# Load the actions for comparison with what agama gets
def load_step_actions(timestep, extras=False):
    print('Loading Actions...')
    actions=np.load('/mnt/ceph/users/jhunt/Bonsai/r2/B2/FlattenedDiscActions/MedianFlattenedDiskActions{}.npy'.format(timestep))
    actions=actions.T
    actsort=np.argsort(actions[:,9])
    idact0=actions[:,9][actsort]
    jphi0=actions[:,1][actsort]
    jz0=actions[:,2][actsort]
    theta_phi0=actions[:,4][actsort]
    theta_z0=actions[:,5][actsort]
    if extras:
        jr0=actions[:,0][actsort]
        theta_r0=actions[:,3][actsort]
        O_R0=actions[:,6][actsort]
        O_phi0=actions[:,7][actsort]
        O_z0=actions[:,8][actsort]
        return np.array([idact0, jr0, jphi0, jz0, theta_r0, theta_phi0, theta_z0, O_R0, O_phi0, O_z0])
    else:
        return np.array([idact0, jphi0, jz0, theta_phi0, theta_z0])
    
def load_step_actions_subset(timestep, indices, extras=False):
    print('Loading Actions...')
    actions=np.load('/mnt/ceph/users/jhunt/Bonsai/r2/B2/FlattenedDiscActions/MedianFlattenedDiskActions{}.npy'.format(timestep))
    actions=actions.T
    actsort=np.argsort(actions[:,9])
    idact0=actions[:,9][actsort]
    subset=np.isin(idact0, indices)
    idact0=idact0[subset]
    jphi0=actions[:,1][actsort][subset]
    jz0=actions[:,2][actsort][subset]
    theta_phi0=actions[:,4][actsort][subset]
    theta_z0=actions[:,5][actsort][subset]
    if extras:
        jr0=actions[:,0][actsort][subset]
        theta_r0=actions[:,3][actsort][subset]
        O_R0=actions[:,6][actsort][subset]
        O_phi0=actions[:,7][actsort][subset]
        O_z0=actions[:,8][actsort][subset]
        return np.array([idact0, jr0, jphi0, jz0, theta_r0, theta_phi0, theta_z0, O_R0, O_phi0, O_z0])
    else:
        return np.array([idact0, jphi0, jz0, theta_phi0, theta_z0])
    
def setup_B2(step, actions_only=True, extras=False):
    actions = load_step_actions(step, extras=extras)
    if actions_only:
        if extras:
            all_data_tbl = pd.DataFrame(actions[1:].T, 
                                        columns=['jr', 'jphi', 'jz', 'theta_r', 'theta_phi', 'theta_z', 'freq_r', 'freq_phi', 'freq_z'], 
                                        index = actions[0].astype('int'))
        else:
            all_data_tbl = pd.DataFrame(actions[1:].T, 
                                        columns=['jphi', 'jz', 'theta_phi', 'theta_z'], 
                                        index = actions[0].astype('int'))
    else:
        if extras:
            id_act, _, _, _, _, _, _, _, _, _ = actions # get ids that we have actions for
        else:
            id_act, _, _, _, _ = actions

        data = load_step_data(step)
        print('Creating Table...')
        data = data[:,id_act.astype('int')] # only data for which we have actions
        all_data_arr = np.concatenate([data[1:], actions[1:]])
        
        if extras:
            all_data_tbl = pd.DataFrame(all_data_arr.T, 
                                        columns=['r', 'phi', 'z', 'vr', 'vphi', 'vz',
                                                 'x', 'y', 'vx', 'vy',
                                                 'jr', 'jphi', 'jz',
                                                 'theta_r', 'theta_phi', 'theta_z',
                                                 'freq_r', 'freq_phi', 'freq_z'], 
                                        index = data[0].astype('int'))
        else:
            all_data_tbl = pd.DataFrame(all_data_arr.T, 
                                        columns=['r', 'phi', 'z', 'vr', 'vphi', 'vz',
                                                 'x', 'y', 'vx', 'vy',
                                                 'jphi', 'jz',
                                                 'theta_phi', 'theta_z'], 
                                        index = data[0].astype('int'))
            
            
    return all_data_tbl

def setup_B2_subset(step, indices, actions_only=True):
    actions = load_step_actions_subset(step, indices)
    if actions_only:
        all_data_tbl = pd.DataFrame(actions[1:].T, 
                                    columns=['jphi', 'jz', 'theta_phi', 'theta_z'], 
                                    index = actions[0].astype('int'))
    else:
        id_act, _, _, _, _, _, _ = actions # get ids that we have actions for

        data = load_step_data(step)
        print('Creating Table...')
        data = data[:,id_act.astype('int')] # only data for which we have actions
        all_data_arr = np.concatenate([data[1:], actions[1:]])
        all_data_tbl = pd.DataFrame(all_data_arr.T, 
                                    columns=['r', 'phi', 'z', 'vr', 'vphi', 'vz',
                                             'x', 'y', 'vx', 'vy',
                                             'jr', 'jphi', 'jz',
                                             'theta_r', 'theta_phi', 'theta_z'], 
                                    index = data[0].astype('int'))
    return all_data_tbl

def setup_B2_region(step, center, radius, actions_only=True, extras=False):
    #this won't speed up the time but will help with memory space in the notebook
    actions = load_step_actions(step, extras=extras)
    if extras:
        actions = actions # FIX THIS
    else:
        id_act, jphi, _, theta_phi, _ = actions # get ids that we have actions for
        sel = id_act[(jphi > center[0] - radius[0]) & #acts[2] is jphi
                      (jphi < center[0] + radius[0]) &
                      (theta_phi > center[1] - radius[1]) & # acts[5] is theta_phi
                      (theta_phi < center[1] + radius[1])]
        print(sel, flush=True)
        region_id = id_act[sel]
    if actions_only:
        if extras:
            region_data_tbl = pd.DataFrame(actions[1:][region_id].T, 
                    columns=['jr', 'jphi', 'jz', 'theta_r', 'theta_phi', 'theta_z', 'freq_r', 'freq_phi', 'freq_z'],
                    index = actions[0][region_id].astype('int'))
        else:
            region_data_tbl = pd.DataFrame(actions[1:][region_id].T,
                                        columns=['jphi', 'jz', 'theta_phi', 'theta_z'],
                                        index = actions[0][region_id].astype('int'))
    else:
    
        data = load_step_data(step)
        print('Creating Table...')
        data = data[:,region_id.astype('int')] # only data for which we have actions
        acts = actions[:,sel.astype('int')]
        region_data_arr = np.concatenate([data[1:], acts[1:]])
        region_data_tbl = pd.DataFrame(region_data_arr, 
                                    columns=['r', 'phi', 'z', 'vr', 'vphi', 'vz',
                                          'x', 'y', 'vx', 'vy',
                                             'jr', 'jphi', 'jz',
                                             'theta_r', 'theta_phi', 'theta_z'], 
                                    index = data[0].astype('int'))
        return region_data_tbl
