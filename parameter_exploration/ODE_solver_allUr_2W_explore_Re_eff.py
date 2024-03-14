"""
This code seeing the amplitude response of the cylinder with different values of drag coefficient, to see how it behaves
when we change that value and if a different value of C_D allows to get a more accurate prediction of the maximum 
vibration amplitudes.
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from time import time
import os
from keras import backend as K
import pandas as pd
from pathlib import Path
from scipy.signal import periodogram
import tikzplotlib as tpl


from scipy.integrate import solve_ivp
from tqdm import tqdm

os.chdir("C:\\Users\\Student\\Documents\\Project\\Endgame\\Unified_PODVM_Galerkin\\")

from parameters import Parameters
nCond = Parameters()

from projection_matrices_2W import *
from auxiliary_functions import *


import matplotlib as mpl
# Uncomment this line to plot on an extrenal renderer
# %matplotlib 
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.figsize'] = (40, 30)
plt.rc('font',size=28)

# from numba import jit

colors_geodataviz = ["#FF1F5B","#00CD6C","#009ADE","#AF58BA","#FFC61E","#F28522"]

# Extraction of the data ===============================================================================================
f = open("POD/modes/points.pl",'rb')
close_points = pickle.load(f)

f = open("POD/modes/true_time_coeffs.pl",'rb')
true_time_coeffs = pickle.load(f)


x_sp, y_sp = tf.transpose(close_points)

# Definition of the parameters =========================================================================================

solver_method = 'RK45'

# Number of time points
N_validation = 201

# Initial condition ----------------------------------------------------------------------------------------------------
# Initial cylinder displacement
eta_0 = 0.
xi_0 = 0.
# Initial cylinder velocity
eta_dot_0 = 0.0
xi_dot_0 = 0.0

# Arange distribution of Ur 
all_Ur = np.arange(3.0,12.0,0.2)


# all_Ur = np.array([6.0])
# Normal distribution of Ur
# mean_ur, std_ur, size = 5, 3, 50
# all_Ur = np.random.normal(mean_ur, std_ur, size)
# Filter the values for 2< Ur < 15
# all_Ur = np.sort(all_Ur[(all_Ur>2)&(all_Ur<15)])

# Extrapolation parameters 
N_ext = nCond.n_ext # Number of points for the extrapolation
t_max_ext = nCond.t_max_ext
t_ext = np.linspace(nCond.t_min,t_max_ext,N_ext)

# Options ==============================================================================================================
extrapol_test = True
save_coeffs = False
compute_matrices = False
postprocess = True
# Initializing parameters --------- ------------------------------------------------------------------------------------

N_modes = nCond.N_modes

print("Number of modes :", N_modes)

mode_path = "POD/ModePINN/saved/"

matrices_path = "saved\\{}_modes\\matrices\\".format(nCond.N_modes)


Path(matrices_path).mkdir(parents=True, exist_ok=True)

t_validation = tf.linspace(nCond.t_min, nCond.t_max,N_validation)

mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")

# Initializing the modes list, containing the models for all modes except the mean mode --------------------------------
modes = []
# N_modes counts the POD modes and the mean mode
for i in range(1, N_modes+1):
    modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))


# Computing the matrices needed for the projected equations ============================================================

# First let's compute the mode vectors needed to get the matrices 
velocity_modes = all_uv_modes(modes, x_sp, y_sp)


# Now let's get the matrices
if (not os.path.isfile(matrices_path + 'q_mat.p')) or compute_matrices:
    print("Computing the Q matrix.....")
    Q_mat = Q_matrix(modes, x_sp, y_sp)
    f = open(matrices_path + 'q_mat.p','wb')
    pickle.dump(Q_mat,f)
else:
    f = open(matrices_path + 'q_mat.p','rb')
    Q_mat = pickle.load(f)
    print("Q was already computed")

    
if (not os.path.isfile(matrices_path + 'chi_mat.p')) or compute_matrices:
    print("Computing the Chi matrix.....")
    Chi_mat = Chi_matrix(modes, x_sp, y_sp)
    f = open(matrices_path + 'chi_mat.p','wb')
    pickle.dump(Chi_mat,f)
else:
    f = open(matrices_path + 'chi_mat.p','rb')
    Chi_mat = pickle.load(f)
    print("Chi was already computed")

if (not os.path.isfile(matrices_path + 'd_mat.p')) or compute_matrices:
    print("Computing the D matrix.....")
    D_mat = D_matrix(modes, x_sp, y_sp).numpy()
    f = open(matrices_path + 'd_mat.p','wb')
    pickle.dump(D_mat,f)
else:
    f = open(matrices_path + 'd_mat.p','rb')
    D_mat = pickle.load(f)
    print("D was already computed")

if (not os.path.isfile(matrices_path + 'gamma_mat.p')) or compute_matrices:
    print("Computing the Gamma matrix.....")
    Gamma_mat = Gamma_matrix(modes, x_sp,y_sp)
    with  open(matrices_path + 'gamma_mat.p','wb') as f:
        pickle.dump(Gamma_mat,f)
else:
    with open(matrices_path + "gamma_mat.p",'rb') as f:
        Gamma_mat = pickle.load(f)
    print("Gamma was already computed")


# Alternative way of computing the forces !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Using Least-Square Method to get the force coefficients 
forces_CFD = pd.read_csv("Data\\Forces\\forces_CFD.csv").to_numpy()
Fx_cfd, Fy_cfd = np.transpose(forces_CFD[:,1:])

def force_coefficients(A_matrix, F_vector):
    res_ls = np.linalg.lstsq(A_matrix, F_vector, rcond=None)
    return res_ls[0]


Fx_mean_CFD = np.mean(Fx_cfd)
Fy_mean_CFD = np.mean(Fy_cfd)

Fx_osc_CFD = Fx_cfd - Fx_mean_CFD
Fy_osc_CFD = Fy_cfd - Fy_mean_CFD

time_steps_forces_CFD = pd.read_csv("Data\\Forces\\timesteps_forces_CFD.csv").to_numpy()[:,1]
# A of dimension (N_t,N_modes)    
time_coeffs_matrix = pd.read_csv("Data\\Forces\\time_coeffs_in_CFD_forces_time_domain.csv").to_numpy()
fx_i_coefficients = force_coefficients(time_coeffs_matrix, Fx_osc_CFD)
fy_i_coefficients = force_coefficients(time_coeffs_matrix, Fy_osc_CFD)


def sigmas_xy_MC_forall_Ur_Zeta(x, y, Ur, Zeta):
    M = nCond.Mass_number
    sigma_x_vect = np.zeros(N_modes+4)
    sigma_y_vect = np.zeros(N_modes+4)
  
    sigma_x_vect[:N_modes] = fx_i_coefficients * (M/(1+M))
    sigma_y_vect[:N_modes] = fy_i_coefficients * (M/(1+M))
    
    sigma_x_vect[N_modes] = - M/(1+M)* (2*np.pi/Ur)**2
    sigma_x_vect[N_modes+1] = - M/(1+M) *Zeta * 2 *(2*np.pi/Ur) 

    sigma_y_vect[N_modes+2] = - M/(1+M)* (2*np.pi/Ur)**2
    sigma_y_vect[N_modes+3] = - M/(1+M) * Zeta * 2 *(2*np.pi/Ur) 
    return (sigma_x_vect, sigma_y_vect)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def Lambda_matrix_MC_forall_Ur_Zeta(mean_mode, modes, x, y, Ur, Zeta):
    inv_Re = 1/nCond.Re
    lambda_matrix = np.zeros((N_modes+4,N_modes+4))
    delta_mat = delta_matrix(mean_mode, modes, x, y)
    # For P matrix 
    grad_p_modes = all_grad_p_modes(modes, x, y)
    vel_modes = all_uv_modes(modes, x, y)

    L_mat = L_matrix(modes, x, y) 
    M_mat = M_matrix(mean_mode, modes, x, y)
    P_mat = P_matrix(grad_p_modes, vel_modes)

    lambda_matrix[:N_modes,:] = inv_Re * L_mat - M_mat - P_mat + delta_mat
    
    lambda_matrix[N_modes,N_modes+1] = 1.0
    lambda_matrix[N_modes+2,N_modes+3] = 1.0


    lambda_matrix[N_modes +1,:], lambda_matrix[N_modes +3,:] = sigmas_xy_MC_forall_Ur_Zeta(x, y, Ur, Zeta)    
    
    return lambda_matrix



x_np, y_np = x_sp.numpy(), y_sp.numpy()

Lambda_mat = Lambda_matrix_MC_forall_Ur_Zeta(mean_mode,modes, x_np, y_np, all_Ur[0], nCond.Zeta)

with open("POD\\modes\\p\\p_modes.pl",'rb') as f:
    p_POD_modes = pickle.load(f) 

C_D = Fx_mean_CFD * np.pi / 2 * nCond.include_added_damping_force

# Exploration of the drag coefficient values *************************************************************************** 
Re_eff_0 = 100
Re_eff_min, Re_eff_max = 1/100, 1.
Re_eff_step = 0.1  # step of 10% to explore the range of values

# all_Re_eff = np.arange(Re_eff_min, Re_eff_max, Re_eff_step ) * Re_eff_0
all_Re_eff = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]) * Re_eff_0

# **********************************************************************************************************************

eta_max_all_Re_eff = []
xi_max_all_Re_eff = []
frequencies_all_Re_eff = []

for i_Re in tqdm(range(len(all_Re_eff))):
    # Drag force coefficient -------------------------------------------------------------------------------------------
    Re_eff = all_Re_eff[i_Re]
    print("Solving the ROM for an effective Reynolds = {:.2f}".format(Re_eff))

    eta_amplitudes_max = []
    xi_amplitudes_max = []

    late_frequencies = []

    psd_all_ur = []

    psd_eta_all_ur = []
    psd_xi_all_ur = [] 

    for i_ur in range(len(all_Ur)):

        Ur_val = all_Ur[i_ur]
        if nCond.verbose:
            print("\nSolving for Ur =", Ur_val,"================================")

        # Lambda_mat = Lambda_matrix_MC_forall_Ur(mean_mode,modes, x_np, y_np, Ur_val)
        Lambda_mat[N_modes +1,:], Lambda_mat[N_modes +3,:] = sigmas_xy_MC_forall_Ur_Zeta(x_np, y_np, Ur_val, nCond.Zeta)    
        

        # # Testing that we can compute all matrices *****************************************************************************

        # Show matrix heatmap **************************************************************************************************    
        # heatmap_matrix(Gamma_np,"$\\Gamma$")
        # ODE solver ===========================================================================================================

        Q_np = Q_mat.numpy()
        Chi_np = Chi_mat

        Gamma_np = Gamma_mat
        Lambda_np = Lambda_mat

        Gamma_no_DOF = Gamma_mat[:N_modes,:N_modes]
        Lambda_no_DOF = Lambda_mat[:N_modes, :N_modes]

        Gamma_inv = np.linalg.inv(Gamma_np)
        Gamma_inv_no_DOF = np.linalg.inv(Gamma_no_DOF)

         
        # ------------------------------------------------------------------------------------------------------------------

        def Fq_AM_ND(a):
            Fq_list = []
            M = nCond.Mass_number
            eta_dot = a[N_modes+1]
            xi_dot = a[N_modes +3]
            added_damping_fx = C_D * (-2*eta_dot + eta_dot**2 + xi_dot**2 )*np.cos(np.arctan2(-xi_dot , (1 - eta_dot))) *4 / (np.pi * nCond.Mass_number) 
            added_damping_fy = C_D * (-2*eta_dot + eta_dot**2 + xi_dot**2 )*np.sin(np.arctan2(-xi_dot , (1 - eta_dot)))  *4 / (np.pi * nCond.Mass_number) 
            

            for i in range(N_modes):
                Fq_i = -np.matmul(a,np.matmul(Q_np[i]+Chi_np[i]-D_mat[i],a))
                Fq_list.append(Fq_i)
            Fq_list.append(0.0)
            Fq_list.append(M/((1+M)) *Fx_mean_CFD +  M / (1 + M) * added_damping_fx)
            Fq_list.append(0.0)
            Fq_list.append(M/(1+M) * Fy_mean_CFD + M / (1 + M) * added_damping_fy)
            Fq_list = np.array(Fq_list)
            return Fq_list


        def solve_projected_ODE(t,a):
            """
            Method created to define the projected ODE for each timestep.
            input: timestep t, time coeffs. vector a(t) 
            output: dadt 
            """
            Fq_a_AM = Fq_AM_ND(a)
            dadt = np.matmul(np.matmul(Gamma_inv,Lambda_np),a)+ np.matmul(Gamma_inv,Fq_a_AM)    
            return dadt

        # Definition of the initial conditions of the time coeffs. vector
        a0 = np.zeros(N_modes+4)
        a0[:N_modes] = true_time_coeffs[:N_modes,0]
        a0_no_DOF = true_time_coeffs[:N_modes,0]
        a0[N_modes:] = np.array([eta_0,eta_dot_0, xi_0, xi_dot_0])

        if nCond.verbose:
            print("\nAll matrices have been computed. We are now going to solve the ODE.\n")

        # Time points
        t_solve = np.linspace(nCond.t_min,nCond.t_max,N_validation)

        if nCond.verbose:
            print("Solver in progress... Please wait........................")
        t_ODE_i = time()

        sol = solve_ivp(solve_projected_ODE,t_span=[nCond.t_min,nCond.t_max],\
                        y0=a0,t_eval=t_validation, method=solver_method)

        t_ODE_f = time()
        if nCond.verbose:
            print('{0} took {1:.8f}s to compute'.format("ODE solution", t_ODE_f - t_ODE_i))
    
        # Extrapolation results ================================================================================================

        if extrapol_test :
            # print("\nTrial on extrapolation :\n")
            # time points

            sol_ext = solve_ivp(solve_projected_ODE,t_span=[nCond.t_min,t_max_ext],y0=a0,t_eval=t_ext, method=solver_method)

            n_final = nCond.n_final
            t_late = t_ext[-n_final:]
            eta_late = sol_ext.y[N_modes][-n_final:]
            xi_late =  sol_ext.y[N_modes+2][-n_final:]


            eta_amplitudes_max.append( get_oscillation_amplitude(t_late, eta_late))
            xi_amplitudes_max.append(get_oscillation_amplitude(t_late, xi_late))
            # eta_amplitudes_max.append(get_oscillation_A_10(t_late, eta_late))
            # xi_amplitudes_max.append(get_oscillation_A_10(t_late, xi_late)) 

            late_frequencies_i = []
            psd_all_modes = []
            for i_mode in range(N_modes):
                a_i_late = sol_ext.y[i_mode][-N_ext//4:]
                sampling_rate = 1/(t_late[1]-t_late[0])
                freq_late_i = get_main_frequencies(a_i_late, sampling_rate, num_frequencies=1)[0]
                
                late_frequencies_i.append(freq_late_i)
                freqs_psd, psd_signal = periodogram(a_i_late, sampling_rate,scaling='spectrum')
                psd_all_modes.append(psd_signal)
            
            psd_all_ur.append(psd_all_modes)
            
            freqs_cyl, psd_eta = periodogram(eta_late, sampling_rate,scaling = 'spectrum')

            freqs_cyl, psd_xi = periodogram(xi_late, sampling_rate,scaling = 'spectrum')

            psd_eta_all_ur.append(psd_eta)
            psd_xi_all_ur.append(psd_xi)

            late_frequencies.append(late_frequencies_i)



    late_frequencies = np.array(late_frequencies)

    columns = {"Ur":all_Ur, "eta max":eta_amplitudes_max,"xi max":xi_amplitudes_max}
    amplitude_results = pd.DataFrame(columns)

    amplitude_results.head()

    columns_freqs = {"Ur":all_Ur}
    for i_mode in range(N_modes):
        columns_freqs.update({"a_{}".format(i_mode+1):late_frequencies[:,i_mode]})
    freq_results = pd.DataFrame(columns_freqs)

    if save_coeffs:

        amplitude_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_amplitudes_test_damping_coeff.csv".format(N_modes, nCond.Zeta))
        freq_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_frequencies_late.csv".format(N_modes, nCond.Zeta))

    eta_max_all_CD.append(eta_amplitudes_max)    
    xi_max_all_CD.append(xi_amplitudes_max)
    frequencies_all_CD.append(late_frequencies)

    # Comparison with FSI results
    FSI_sim_Ur_x = np.array([1.,2.5,3.,3.5,4.,4.5,5.5,6.5,7.,8.,9.,10.,11.,12.])

    FSI_sim_Ur_y = np.array([1.,2.,4.,4.5,5.,6.,7.,8.,9.,10.,11.,12.,13.])

    true_xi_max = np.array([0.006897,0.124138,0.636207,0.632759,0.615517,0.548276,0.448276,0.301724,0.093103,0.089655,0.082759,0.075862, 0.075862])
    true_eta_max =  np.array([0.002619,0.004549,0.026328,0.045075,0.046729,0.04218,0.032256,0.021779,0.010338,0.003308,0.003308,0.003446,0.003033,0.003308])
    # plt.plot(FSI_sim_Ur_x, true_eta_max, 'k--', marker='o', markerfacecolor='none', markeredgecolor='k')


    a_1_late = sol_ext.y[0][-N_ext//4:]
    a_2_late = sol_ext.y[1][-N_ext//4:]
    freqs_psd, psd_signal_1 = periodogram(a_1_late, sampling_rate,scaling='spectrum')
    freqs_psd, psd_signal_2 = periodogram(a_2_late, sampling_rate,scaling='spectrum') 



    added_mass_eq_coeff = nCond.Mass_number / (1+nCond.Mass_number) 

    savefig_path = "C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\two_way_coupling\\"

cmap_eta = plt.cm.Blues
cmap_xi = plt.cm.Oranges

# norm = colors.BoundaryNorm(np.arange(0, len(all_Zeta), 1), cmap_xi.N)

# Plot the cross-flow amplitudes of oscillation for all values of CD
plt.figure(figsize=(16,23))
plt.title("Variation of the cross-flow oscillation amplitude with the value of $C_D$", pad=50)
# Plot the initial value for reference
plt.plot(FSI_sim_Ur_y, true_xi_max, 'k--', marker='o', markerfacecolor='none', markeredgecolor='k', label = "FSI sim. results" )
plt.plot(all_Ur, xi_max_all_CD[0], marker='s',color='k', label="$C_D = {}$ (Initial value)".format("{:.2f}".format(all_CD[0])))
for i_CD in range(1,len(all_CD),2):
    plt.scatter(all_Ur,xi_max_all_CD[i_CD], marker = 's', label = "$C_D = {}$".format("{:.2f}".format(all_CD[i_CD])), color = cmap_xi(i_CD*5+60))
plt.xlim(0.0,17)
plt.xlabel("$U_r$")
plt.ylabel("$\\xi_{max}$")
plt.legend(fontsize=18, loc='upper right')
plt.show()
# tpl.save(savefig_path + "displacement_xi_max_vs_Ur.tex")


# def F_total(time_coeffs):
plt.figure(figsize=(16,23))
plt.title("Variation of the in-line oscillation amplitude with the value of $C_D$", pad = 50)
# Plot the initial value for reference
plt.plot(FSI_sim_Ur_x, true_eta_max, 'k--', marker='o', markerfacecolor='none', markeredgecolor='k',  label = "FSI sim. results" )
plt.plot(all_Ur, eta_max_all_CD[0], marker='s',color='k', label="$C_D = {}$ (Initial value)".format("{:.2f}".format(all_CD[0])))
for i_CD in range(1,len(all_CD),2):
    plt.scatter(all_Ur,eta_max_all_CD[i_CD],  marker = 's', label = "$C_D = {}$".format("{:.2f}".format(all_CD[i_CD])), color = cmap_eta((i_CD*5+60)))

plt.xlim(0.0,17)
plt.xlabel("$U_r$")
plt.ylabel("$\eta_{max}$")
plt.legend(fontsize=18, loc='upper right')
plt.show()

frequencies_all_CD = np.array(frequencies_all_CD)
frequencies_all_CD_main = frequencies_all_CD[:,:,0]

cmap_f = plt.cm.BuGn
plt.figure(figsize=(10,12))
for i_CD in range(len(all_CD)):
    plt.scatter(all_Ur,frequencies_all_CD[i_CD,:,0]*all_Ur* np.sqrt(2), marker = 's', color=cmap_f(i_CD*15+70), label = "$C_D = {}$".format("{:.2f}".format(all_CD[i_CD])))
plt.hlines(1.,0,all_Ur[-1]+10, linestyle='--',color='k')
plt.plot(all_Ur, 0.2*all_Ur*2**0.5, linestyle='--',color='k')
plt.gca().set_aspect('auto')
plt.xlim(0,all_Ur[-1]+1)
plt.ylim(0,3.5)
plt.xlabel("$U_r$")
plt.ylabel("Frequency ratio $f^*$")
plt.legend()
plt.show()