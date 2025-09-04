import numpy as np
import pandas as pd
import os

def fault_slip_analysis(
    pres_folder_path: str,
    coor_fault_file_path: str,
    parameter_file_path: str,
    save_folder_path: str,
    case_name: str, 
    ):
    
    print(f"Processing {case_name}...")
    # check if the save folder exists
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # load parameter csv
    parameters = pd.read_csv(parameter_file_path)

    # input stress state
    # SH_grad = parameters.loc[parameters["case_num"] == case_name, 'SH_MPa/km'].iloc[0] * (-1)
    # Sh_grad = parameters.loc[parameters["case_num"] == case_name, 'Sh_MPa/km'].iloc[0] * (-1)
    # Sv_grad = parameters.loc[parameters["case_num"] == case_name, 'Sv_MPa/km'].iloc[0] * (-1)
    # SH_azi = parameters.loc[parameters["case_num"] == case_name, 'SH_azi_deg'].iloc[0]
    row = parameters.loc[parameters["case_num"] == case_name].iloc[0]
    SH_grad, Sh_grad, Sv_grad = -row['SH_MPa/km'], -row['Sh_MPa/km'], -row['Sv_MPa/km'] #change the sign of the gradient
    SH_azi = row['SH_azi_deg']

    mu = 0.6 #coefficient of friction
    cohesion = 1 # fault cohesion in MPa

    fault_strike = 10
    fault_dip = 90

    # load data
    coor_fault = np.load(coor_fault_file_path)
    pres = np.load(f'{pres_folder_path}/{case_name}_PRES.npy')/1000 # convert to MPa

    # extract depth
    z_coor = coor_fault[:,:,:,2]
    # compute principal stresses
    SH_stress = z_coor /1000 * SH_grad
    Sh_stress = z_coor /1000 * Sh_grad
    Sv_stress = z_coor /1000 * Sv_grad
    # compute rotation angles
    phi = SH_azi - fault_strike
    theta = fault_dip

    # fault_slip = np.full_like(pres, np.nan)
    # for year in range(pres.shape[3]):
    #     # extract pressure
    #     pres_slice = pres[:,:,:,year]
    #     # transform principla stresses to fault planes
    #     sigma, tau = StressTransform3D(pres_slice, SH_stress, Sh_stress, Sv_stress, phi, theta)
    #     # compute fault slip indicator, where 1 indicates slip and 0 indicates stability
    #     fault_slip[:,:,:,year] = ((tau - cohesion) / sigma >= mu).astype(int)
     # compute fault slip indicator (vectorized over all years)
    sigma, tau = StressTransform3D(pres, SH_stress, Sh_stress, Sv_stress, phi, theta)
    fault_slip = ((tau - cohesion) / sigma >= mu).astype(np.int8)

    # save fault slip indicator
    np.save(f'{save_folder_path}/{case_name}_fault_slip.npy', fault_slip)


def StressTransform3D(Pf, SH, Sh, Sv, phi, theta):
    # Pf: (..., time), SH/ Sh/ Sv: (...)   (broadcast together)
    # expand SH/ Sh/ Sv to match Pf shape
    SH, Sh, Sv = np.broadcast_arrays(SH[...,None], Sh[...,None], Sv[...,None])
    Pf = Pf  # already shaped (..., time)

    # stress tensor: shape (...,time,3,3)
    s_xx, s_yy, s_zz = SH - Pf, Sh - Pf, Sv - Pf
    s = np.stack([
        np.stack([s_xx, np.zeros_like(Pf), np.zeros_like(Pf)], axis=-1),
        np.stack([np.zeros_like(Pf), s_yy, np.zeros_like(Pf)], axis=-1),
        np.stack([np.zeros_like(Pf), np.zeros_like(Pf), s_zz], axis=-1)
    ], axis=-2)  # (...,time,3,3)

    # pre-calculate trigonometric values
    cos_phi, sin_phi = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    cos_theta, sin_theta = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    # perform stress tranformation
    # first rotate around z axis (vertical stress direction)
    Rz = np.array([[cos_phi,-sin_phi,0],[sin_phi,cos_phi,0],[0,0,1]])
    # next rotate around x axis (maximum horizotnal stress direction)
    Rx = np.array([[1,0,0],[0,cos_theta,-sin_theta],[0,sin_theta,cos_theta]])
    # compute rotation matrix using rotation components in reverse order
    R = Rx @ Rz
    # perform rotation
    # einsum does the batched matrix multiplication
    result = np.einsum("ab,...bc,cd->...ad", R, s, R.T)
    # retrieve stresses from matrices
    tau1 = result[...,2,0] # shear stress component in the strike direction
    tau2 = result[...,2,1] # shear stress component in the dip direction
    tau  = np.sqrt(tau1**2 + tau2**2) # shear stress
    sigma = result[...,2,2] # normal stress
    
    return sigma, tau


def fault_slip_analysis_3Darray(
    pres_folder_path: str,
    coor_fault_file_path: str,
    parameter_file_path: str,
    save_folder_path: str,
    year: int,
    year_list: list[int],
    case_name: str, 
    ):
    

    # load parameter csv
    parameters = pd.read_csv(parameter_file_path)

    # input stress state
    SH_grad = parameters.loc[parameters["case_num"] == case_name, 'SH_MPa/km'].iloc[0] * (-1)
    Sh_grad = parameters.loc[parameters["case_num"] == case_name, 'Sh_MPa/km'].iloc[0] * (-1)
    Sv_grad = parameters.loc[parameters["case_num"] == case_name, 'Sv_MPa/km'].iloc[0] * (-1)
    SH_azi = parameters.loc[parameters["case_num"] == case_name, 'SH_azi_deg'].iloc[0]

    mu = 0.6 #coefficient of friction
    cohesion = 1 # fault cohesion in MPa

    fault_strike = 10
    fault_dip = 90

    # load data
    coor_fault = np.load(coor_fault_file_path)
    pres = np.load(f'{pres_folder_path}/{case_name}_PRES.npy')

    # extract depth
    z_coor = coor_fault[:,:,:,2]
    # compute principal stresses
    SH_stress = z_coor /1000 * SH_grad
    Sh_stress = z_coor /1000 * Sh_grad
    Sv_stress = z_coor /1000 * Sv_grad
    # extract pressure
    pres_slice = pres[:,:,:,year_list.index(year)]/1000 # convert to MPa
    # compute rotation angles
    phi = SH_azi - fault_strike
    theta = fault_dip
    # transform principla stresses to fault planes
    sigma, tau = StressTransform3D_3Darray(pres_slice, SH_stress, Sh_stress, Sv_stress, phi, theta)
    # compute fault slip indicator, where 1 indicates slip and 0 indicates stability
    fault_slip = ((tau - cohesion) / sigma >= mu).astype(int)
    
    # check if the save folder exists
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    # save fault slip indicator
    np.save(f'{save_folder_path}/{case_name}_{year}.npy', fault_slip)


def StressTransform3D_3Darray(Pf, SH, Sh, Sv, phi, theta):
    # construct stress tensor field: shape (...,3,3)
    s = np.zeros(Pf.shape + (3,3))
    s[...,0,0] = SH - Pf
    s[...,1,1] = Sh - Pf
    s[...,2,2] = Sv - Pf
    # pre-calculate trigonometric values
    cos_phi, sin_phi = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    cos_theta, sin_theta = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    # perform stress tranformation
    # first rotate around z axis (vertical stress direction)
    Rz = np.array([[cos_phi,-sin_phi,0],
                   [sin_phi, cos_phi,0],
                   [0,0,1]])
    # next rotate around x axis (maximum horizotnal stress direction)
    Rx = np.array([[1,0,0],
                   [0,cos_theta,-sin_theta],
                   [0,sin_theta, cos_theta]])
    # compute rotation matrix using rotation components in reverse order
    R = Rx @ Rz
    # perform rotation
    # einsum does the batched matrix multiplication
    result = np.einsum("ab,...bc,cd->...ad", R, s, R.T)
    # retrieve stresses from matrices
    tau1 = result[...,2,0] # shear stress component in the strike direction
    tau2 = result[...,2,1] # shear stress component in the dip direction
    tau  = np.sqrt(tau1**2 + tau2**2) # shear stress
    sigma = result[...,2,2] # normal stress
    
    return sigma, tau