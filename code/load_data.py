import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd
import numpy as np
import pandas as pd
#import fitsio
from astropy.io import fits
from astropy.table import Table
import pickle

coord.galactocentric_frame_defaults.set('v4.0')
galcen_frame = coord.Galactocentric()

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
    Radius in kpc around the center (not from center of MW)
    '''
    select = data[(data.jphi > center[0] - radius[0]) & (data.jphi < center[0] + radius[0]) &
                  (data.theta_phi > center[1] - radius[1]) & (data.theta_phi < center[1] + radius[1])]
    return select

def load_data_actions(data_file, action_file):
    '''
    This loads the data and actions from the simulation files and 
    returns a table with the actions, angles, and frequencies.
    MAKE SURE THE FILENAMES ARE FOR THE SAME TIMESTEP
    '''
    print("Loading Data...")
    data = load_data(data_file)[1]
    #data['phi'][data['phi'] > np.pi] = data['phi'][data['phi'] > np.pi] - 2*np.pi
    
    print("Loading Actions...")
    file1 = open(action_file, 'rb')
    jr1,jphi1,jz1,O_R1,O_phi1,O_z1,theta_R1,theta_phi1,theta_z1=pickle.load(file1)

    jz = jz1*8*220 # convert to kpc km/s
    jphi = jphi1*8*220 # convert to kpc km/s
    jr = jr1*8*220 # convert to kpc km/s

    O_R = O_R1*27.5 # convert
    O_phi = O_phi1*27.5 # convert
    O_z = O_z1*27.5 # convert

    theta_R = theta_R1
    theta_phi = theta_phi1
    theta_z = theta_z1
    
    print("Creating Combined Table...")
    jr_tbl = jr[data.index]
    jphi_tbl = jphi[data.index]
    jz_tbl = jz[data.index]
    
    theta_R_tbl = theta_R[data.index]
    theta_phi_tbl = theta_phi[data.index]
    theta_z_tbl = theta_z[data.index]
    
    O_R_tbl = O_R[data.index]
    O_phi_tbl = O_phi[data.index]
    O_z_tbl = O_z[data.index]

#     data['jr'] = jr[data.index]
#     data['jphi'] = jphi[data.index]
#     data['jz'] = jz[data.index]
    
#     data['theta_R'] = theta_R[data.index]
#     data['theta_phi'] = theta_phi[data.index]
#     data['theta_z'] = theta_z[data.index]
    
#     data['O_R'] = O_R[data.index]
#     data['O_phi'] = O_phi[data.index]
#     data['O_z'] = O_z[data.index]
    
    for_tbl = np.array([jr_tbl, jphi_tbl, jz_tbl, 
                        theta_R_tbl, theta_phi_tbl, theta_z_tbl, 
                        O_R_tbl, O_phi_tbl, O_z_tbl])
    
    all_data_tbl = pd.DataFrame(for_tbl.T, 
                                columns=['jr', 'jphi', 'jz', 'theta_r', 'theta_phi', 'theta_z', 'freq_r', 'freq_phi', 'freq_z'], 
                                index = data.index)
    
    return all_data_tbl


def load_data(filename, timestep=None, ext=1, cols = 'subset'):

    hdu = fits.open(filename)
    hdr = hdu[ext].header
    N = hdr['NAXIS2']
    
    if timestep is None:
        # stay in simulation units
        timestep = u.one

    columns = ['R', 'phi', 'z', 'vR', 'vT', 'vz']
    data = Table.read(filename)[columns]

    tmp_t = Table.read(filename)['t']
    
    # simulation units
    ro = 8. * u.kpc
    vo = 220. * u.km/u.s

    rep = coord.CylindricalRepresentation(
        rho=np.array(data['R']) * ro, 
        phi=(np.array(data['phi']) + np.pi) * u.rad,
        z=np.array(data['z']) * ro)
    dif = coord.CylindricalDifferential(
        d_rho=np.array(data['vR']) * vo,
        d_phi=((-np.array(data['vT'])*vo) / (np.array(data['R'])*ro)).to(u.rad/u.Gyr, u.dimensionless_angles()),
        d_z=np.array(data['vz']) * vo
    )
    rep = rep.with_differentials(dif)
    #t = tmp_t['t'][0] * timestep
    t = tmp_t[0] * timestep
    
    w = gd.PhaseSpacePosition(rep).cartesian
    icrs = w.to_coord_frame(coord.ICRS(), galactocentric_frame=galcen_frame)
    gal = w.to_coord_frame(coord.Galactic(), galactocentric_frame=galcen_frame)
    
    
    if cols == 'all':
        df = pd.DataFrame({
            # Cylindrical
            'R': rep.rho.to_value(u.kpc),
            'phi': rep.phi.to_value(u.rad),
            'z': rep.z.to_value(u.kpc),
            'v_R': dif.d_rho.to_value(u.km/u.s),
            'v_phi': (-np.array(data['vT'])*vo).to_value(u.km/u.s),
            'v_z': dif.d_z.to_value(u.km/u.s),
            # Cartesian
            'x': w.x.to_value(u.kpc),
            'y': w.y.to_value(u.kpc),
            'v_x': w.v_x.to_value(u.km/u.s),
            'v_y': w.v_y.to_value(u.km/u.s),
            # Observed
            'ra': icrs.ra.degree,
            'dec': icrs.dec.degree,
            'distance': icrs.distance.to_value(u.kpc),
            'l': gal.l.wrap_at(360*u.deg).degree,
            'b': gal.b.degree
        })
    else: 
        df = pd.DataFrame({
            # Cylindrical
            #'R': rep.rho.to_value(u.kpc),
            #'phi': rep.phi.to_value(u.rad),
            #'z': rep.z.to_value(u.kpc),
            #'v_R': dif.d_rho.to_value(u.km/u.s),
            #'v_phi': (-np.array(data['vT'])*vo).to_value(u.km/u.s),
            #'v_z': dif.d_z.to_value(u.km/u.s),
            # Cartesian
            #'x': w.x.to_value(u.kpc),
            #'y': w.y.to_value(u.kpc),
            #'v_x': w.v_x.to_value(u.km/u.s),
            #'v_y': w.v_y.to_value(u.km/u.s),
            # Observed
            'ra': icrs.ra.degree,
        })
    
    return t, df.loc[np.isfinite(df.ra) & (df.ra != 0.)]
