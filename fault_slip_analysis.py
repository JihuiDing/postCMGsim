import numpy as np
import pandas as pd
import os

def fault_slip_analysis(
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
    z_coor = coor_fault[:,:,40:79,2]
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
    sigma, tau = StressTransform3D(pres_slice, SH_stress, Sh_stress, Sv_stress, phi, theta)
    # compute fault slip indicator, where 1 indicates slip and 0 indicates stability
    fault_slip = ((tau - cohesion) / sigma >= mu).astype(int)
    
    # check if the save folder exists
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    # save fault slip indicator
    np.save(f'{save_folder_path}/{case_name}_fault_slip.npy', fault_slip)


def StressTransform3D(Pf, SH, Sh, Sv, phi, theta):
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