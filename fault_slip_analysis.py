import numpy as np
import pandas as pd
import os
from pathlib import Path

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
    # change the sign of the stress gradient and add back hydrostatic gradient to make total stress gradient
    row = parameters.loc[parameters["case_num"] == case_name].iloc[0]
    SH_grad, Sh_grad, Sv_grad = -row['SH_MPa/km']+10, -row['Sh_MPa/km']+10, -row['Sv_MPa/km']+10 
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

    sigma, tau = StressTransform3D(pres, SH_stress, Sh_stress, Sv_stress, phi, theta)
    fault_slip = ((tau - cohesion) / sigma >= mu).astype(np.int8)

    # save fault slip indicator
    np.save(f'{save_folder_path}/{case_name}_fault_slip.npy', fault_slip)


def StressTransform3D(Pf, SH, Sh, Sv, phi, theta):
    # Pf: (i,j,k,time); SH, Sh, Sv: single value
    # expand SH Sh Sv to match Pf shape
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


def FSA_stress_based(
    stress_folder_path: str,
    parameter_file_path: str,
    fault_cell_file_path: str,
    save_folder_path: str,
    case_name: str, 
    fault_id: int,
    fault_strike: float,
    fault_dip: float
    ):
    
    print(f"Processing {case_name}...")

    # create save folder if it does not exist
    save_path = Path(save_folder_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # load principal stress arrays
    SH = np.load(f'{stress_folder_path}/{case_name}_STRESMXP.npy')
    Sh = np.load(f'{stress_folder_path}/{case_name}_STRESMNP.npy')
    Sv = np.load(f'{stress_folder_path}/{case_name}_STRESINT.npy')
    
    # # select desired fault cells
    # coor_fault = np.load(fault_cell_file_path)
    # fault_id_mask = (coor_fault[:,:,:,3] == fault_id)
    # SH = SH[fault_id_mask]
    # Sh = Sh[fault_id_mask]
    # Sv = Sv[fault_id_mask]
    
    # remove zeros in stress arrays to avoid division by zeros
    SH[SH == 0] = np.nan
    Sh[Sh == 0] = np.nan
    Sv[Sv == 0] = np.nan


    # load parameter csv
    parameters = pd.read_csv(parameter_file_path)
    # extract the azimuth of the maximum horizontal stress
    row = parameters.loc[parameters["case_num"] == case_name].iloc[0]
    SH_azi = row['SH_azi_deg']

    mu = 0.6 #coefficient of friction
    cohesion = 1 # fault cohesion in MPa

    # compute rotation angles
    phi = SH_azi - fault_strike
    theta = fault_dip

    sigma, tau = StressTransform3D_stress_arrays(0, SH, Sh, Sv, phi, theta)
    fault_slip = ((tau - cohesion) / sigma >= mu).astype(np.int8)

    # save fault slip indicator
    np.save(f'{save_folder_path}/{case_name}_fault_slip.npy', fault_slip)

    return fault_slip



def StressTransform3D_stress_arrays(Pf, SH, Sh, Sv, phi, theta):
    """
    Pf    : scalar pore pressure
    SH,Sh,Sv : numpy arrays with shape (i,j,k,time)
    phi,theta : scalar angles in degrees
    Returns:
        sigma, tau : arrays with shape (i,j,k,time)
    """
    # subtract pore pressure
    s11 = SH - Pf
    s22 = Sh - Pf
    s33 = Sv - Pf

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

    # construct stress tensor components (diagonal only, symmetric)
    # shape (...,3,3)
    s = np.zeros(SH.shape + (3,3))
    s[...,0,0] = s11
    s[...,1,1] = s22
    s[...,2,2] = s33

    # perform rotation,rotate all tensors: result = R @ s @ R^T
    # einsum does the batched matrix multiplication
    result = np.einsum("ab,...bc,cd->...ad", R, s, R.T)
    # retrieve stresses from matrices
    tau1 = result[...,2,0] # shear stress component in the strike direction
    tau2 = result[...,2,1] # shear stress component in the dip direction
    tau  = np.sqrt(tau1**2 + tau2**2) # shear stress
    sigma = result[...,2,2] # normal stress
    
    return sigma, tau
