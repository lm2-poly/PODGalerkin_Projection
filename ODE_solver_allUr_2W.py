import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from time import time

from keras import backend as K

from datetime import datetime
import os
import shutil

from scipy.integrate import solve_ivp
import tensorflow as tf

from tqdm import tqdm
from parameters import Parameters

nCond = Parameters()

from projection_matrices_2W import *
from auxiliary_functions import *

import pandas as pd
from pathlib import Path
from scipy.signal import periodogram
import tikzplotlib as tpl

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
# all_Ur = np.arange(1.0,16.0,1)
all_Ur = np.array([3.0])
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

mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")

t_validation = tf.linspace(nCond.t_min, nCond.t_max,N_validation)


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

def sigmas_xy_MC_forall_Ur(x, y, p_POD_modes, Ur):
    M = nCond.Mass_number
    sigma_x_vect = np.zeros(N_modes+4)
    sigma_y_vect = np.zeros(N_modes+4)
    contour_indices = isolate_contour_points(x, y, epsilon=nCond.epsilon_contour)
    
    x_contour, y_contour = x[contour_indices], y[contour_indices]

    for i in range(N_modes):
        gamma_xi, gamma_yi = gammas_POD_mode_MC(x_contour, y_contour, p_POD_modes[contour_indices, i])
        
        sigma_x_vect[i] = - gamma_xi * 2 / (np.pi * (1+nCond.Mass_number))
        sigma_y_vect[i] = - gamma_yi * 2 / (np.pi * (1+nCond.Mass_number))

        if nCond.include_viscosity:
            mu_visc_xi, mu_visc_yi = mu_visc_i_integration(modes[i], x_contour, y_contour)
            # print(mu_visc_i_integration(modes[i], x_contour, y_contour)) 
            sigma_x_vect[i] +=  mu_visc_xi * 2 / (np.pi * (1+nCond.Mass_number))
            sigma_y_vect[i] +=  mu_visc_yi * 2 / (np.pi * (1+nCond.Mass_number))

    sigma_x_vect[N_modes] = - M/(1+M)* (2*np.pi/Ur)**2
    sigma_x_vect[N_modes+1] = - M/(1+M) * nCond.Zeta

    sigma_y_vect[N_modes+2] = - M/(1+M)* (2*np.pi/Ur)**2
    sigma_y_vect[N_modes+3] = - M/(1+M) * nCond.Zeta
    return (sigma_x_vect, sigma_y_vect)

def Lambda_matrix_MC_forall_Ur(mean_mode, modes, x, y, Ur):
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

    with open("POD\\modes\\p\\p_modes.pl",'rb') as f:
        p_POD_modes = pickle.load(f) 
    lambda_matrix[N_modes +1,:], lambda_matrix[N_modes +3,:] = sigmas_xy_MC_forall_Ur(x, y,p_POD_modes, Ur)    
    
    return lambda_matrix

eta_amplitudes_max = []
xi_amplitudes_max = []

late_frequencies = []

psd_all_ur = []

psd_eta_all_ur = []
psd_xi_all_ur = [] 

x_np, y_np = x_sp.numpy(), y_sp.numpy()

Lambda_mat = Lambda_matrix_MC_forall_Ur(mean_mode,modes, x_np, y_np, all_Ur[0])

with open("POD\\modes\\p\\p_modes.pl",'rb') as f:
    p_POD_modes = pickle.load(f) 

for i_ur in tqdm(range(len(all_Ur))):

    Ur_val = all_Ur[i_ur]
    print("\nSolving for Ur =", Ur_val,"================================")

    # Lambda_mat = Lambda_matrix_MC_forall_Ur(mean_mode,modes, x_np, y_np, Ur_val)
    Lambda_mat[N_modes +1,:], Lambda_mat[N_modes +3,:] = sigmas_xy_MC_forall_Ur(x_np, y_np,p_POD_modes, Ur_val)    
    

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

    with open("POD\\modes\\p\\p_mean.pl", 'rb') as f:
        p_mean = pickle.load(f)

    gamma_bar_x_value, gamma_bar_y_value  = gammas_POD_mean_MC(x_np, y_np, p_mean)
    
    mu_bar_x, mu_bar_y = 0, 0
    if nCond.include_viscosity:
        contour_indices = isolate_contour_points(x_sp, y_sp, epsilon=nCond.epsilon_contour)    
        x_contour, y_contour = x_np[contour_indices], y_np[contour_indices]
        mu_bar_x, mu_bar_y = mu_visc_i_integration(mean_mode, x_contour, y_contour)

    # Drag force coefficient -------------------------------------------------------------------------------------------
    C_D =  -gamma_bar_x_value*np.pi*nCond.Mass_number/2 * 0
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
        Fq_list.append( -2/(np.pi*(1+M)) *gamma_bar_x_value +  M / (1 + M) * added_damping_fx + 
                       nCond.include_viscosity * 2/(np.pi*(1+M))*mu_bar_x)
        Fq_list.append(0.0)
        Fq_list.append(-2/(np.pi*(1+M)) *gamma_bar_y_value  + M / (1 + M) * added_damping_fy + 
                       nCond.include_viscosity * 2/(np.pi*(1+M))*mu_bar_y)
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

    print("\nAll matrices have been computed. We are now going to solve the ODE.\n")

    # Time points
    t_solve = np.linspace(nCond.t_min,nCond.t_max,N_validation)

    print("Solver in progress... Please wait........................")
    t_ODE_i = time()

    sol = solve_ivp(solve_projected_ODE,t_span=[nCond.t_min,nCond.t_max],\
                    y0=a0,t_eval=t_validation, method=solver_method)

    t_ODE_f = time()
    print('{0} took {1:.8f}s to compute'.format("ODE solution", t_ODE_f - t_ODE_i))


    # Forces ===============================================================================================================
    print("Computing the forces ................................")

    def F_p_xy_MC(time_coeffs):

        with open("POD\\modes\\p\\p_modes.pl",'rb') as f:
            p_POD_modes = pickle.load(f) 
        # The gammas are already multiplied by - 2 /(pi * (1+M))
        sigmas_x, sigmas_y = sigmas_xy_MC(x_np,y_np, p_POD_modes) 
        sigmas_x = sigmas_x[:N_modes]* (1+nCond.Mass_number)/nCond.Mass_number
        sigmas_y = sigmas_y[:N_modes]* (1+nCond.Mass_number)/nCond.Mass_number

        f_tot_x = np.ones(len(time_coeffs[0]))*gamma_bar_x_value * 2/ (np.pi* nCond.Mass_number) 
        f_tot_y = np.ones(len(time_coeffs[0]))*gamma_bar_y_value * 2/ (np.pi* nCond.Mass_number) 
        for i in range(N_modes):
            gamma_xi, gamma_yi = sigmas_x[i],sigmas_y[i] 
            f_tot_x += gamma_xi * time_coeffs[i]
            f_tot_y += gamma_yi * time_coeffs[i]
        return (f_tot_x, f_tot_y)
            
    F_p_x, F_p_y = F_p_xy_MC(sol.y[:N_modes])
 
    # Extrapolation results ================================================================================================

    if extrapol_test :
        print("\nTrial on extrapolation :\n")
        # time points

        sol_ext = solve_ivp(solve_projected_ODE,t_span=[nCond.t_min,t_max_ext],y0=a0,t_eval=t_ext, method=solver_method)

        n_final = nCond.n_final
        t_late = t_ext[-n_final:]
        eta_late = sol_ext.y[N_modes][-n_final:]
        xi_late =  sol_ext.y[N_modes+2][-n_final:]


        eta_amplitudes_max.append( get_oscillation_amplitude(t_late, eta_late))
        xi_amplitudes_max.append(get_oscillation_amplitude(t_late, xi_late)) 

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


        # Plot only the late trajectory ------------------------------------------------------------------------------------
        Path("saved\\{}_modes\\zeta_{:.2f}\\trajectories".format(N_modes, nCond.Zeta)).mkdir(exist_ok=True, parents=True)
        plt.figure(figsize=(20,20))
        plt.title("Cylinder trajectory - Extrapolated")
        plt.plot(sol_ext.y[N_modes][-N_ext//4:]-np.mean(sol_ext.y[N_modes][-N_ext//4:]),
                 sol_ext.y[N_modes+2][-N_ext//4:],
                   '-k',linewidth = 3.0)
        plt.grid()
        plt.xlabel("$\eta$")
        plt.ylabel("$\\xi$")
        plt.xlim(-0.05,0.05)
        plt.ylim(-1.,1.)
        # plt.savefig("saved\\{}_modes\\zeta_{:.2f}\\trajectories\\traj_final_Ur_{:.2f}_test_damping_coeff.svg".format(N_modes,nCond.Zeta, Ur_val))
        plt.savefig("saved\\{}_modes\\zeta_{:.2f}\\trajectories\\traj_final_Ur_{:.2f}.svg".format(N_modes,nCond.Zeta, Ur_val))

        two_way_coupling_path ='C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\two_way_coupling\\trajectories\\'
        plt.figure(figsize=(15,15))
        plt.title("Trajectory of the cylinder at the end of extrapolation - $U_r={:.2f}$".format(Ur_val))
        plt.plot(eta_late,xi_late,c='black')
        plt.xlabel("$\\bar{x}$ [-]")
        plt.ylabel("$\\bar{y}$ [-]")
        plt.xlim(np.mean(eta_late)-0.08,np.mean(eta_late)+0.08)
        plt.ylim(np.mean(xi_late)-1.2,np.mean(xi_late)+1.2)
        plt.gca().set_aspect(1/5)
        plt.savefig(two_way_coupling_path + "end_traj_Ur_{:.2f}.svg".format(Ur_val),transparent=True)
        plt.close()

late_frequencies = np.array(late_frequencies)

columns = {"Ur":all_Ur, "eta max":eta_amplitudes_max,"xi max":xi_amplitudes_max}
amplitude_results = pd.DataFrame(columns)

amplitude_results.head()

columns_freqs = {"Ur":all_Ur}
for i_mode in range(N_modes):
    columns_freqs.update({"a_{}".format(i_mode+1):late_frequencies[:,i_mode]})
freq_results = pd.DataFrame(columns_freqs)

if save_coeffs:
    # amplitude_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_amplitudes.csv".format(N_modes, nCond.Zeta))  
    amplitude_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_amplitudes_test_damping_coeff.csv".format(N_modes, nCond.Zeta))
    freq_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_frequencies_late.csv".format(N_modes, nCond.Zeta))

# Comparison with FSI results
FSI_sim_Ur_x = np.array([1.,2.5,3.,3.5,4.,4.5,5.5,6.5,7.,8.,9.,10.,11.,12.])

FSI_sim_Ur_y = np.array([1.,2.,4.,4.5,5.,6.,7.,8.,9.,10.,11.,12.,13.])

true_xi_max = np.array([0.006897,0.124138,0.636207,0.632759,0.615517,0.548276,0.448276,0.301724,0.093103,0.089655,0.082759,0.075862, 0.075862])
true_eta_max =  np.array([0.002619,0.004549,0.026328,0.045075,0.046729,0.04218,0.032256,0.021779,0.010338,0.003308,0.003308,0.003446,0.003033,0.003308])
# plt.plot(FSI_sim_Ur_x, true_eta_max, 'k--', marker='o', markerfacecolor='none', markeredgecolor='k')

psd_all_ur = np.array(psd_all_ur)
psd_all_ur_signals = psd_all_ur

# Plot PSD for all Urs -------------------------------------------------------------------------------------------------
plt.figure(figsize=(7,7))
plt.imshow(psd_all_ur[:,1], extent= (freqs_psd[0],freqs_psd[-1],all_Ur[0],all_Ur[-1]),origin='lower')
# plt.gca().set_aspect('equal')
plt.xlabel("Frequency $f$")
plt.ylabel("$U_r$ ")
plt.colorbar()
plt.tight_layout()
# plt.plot()
plt.show()

with open("saved\\{}_modes\\zeta_{:.2f}\\data\\psd_urs.pl".format(N_modes, nCond.Zeta),'wb')as f:
    pickle.dump([all_Ur,freqs_psd, psd_all_ur_signals],f) 

psd_transpose = np.transpose(psd_all_ur)

psd_i_mode = 0 

# Plot log PSD for all Urs ---------------------------------------------------------------------------------------------
plt.figure(figsize=(7,7))
plt.imshow(np.log(psd_transpose[:,psd_i_mode,:]), extent= (all_Ur[0],all_Ur[-1],freqs_psd[0],freqs_psd[-1]),origin='lower',cmap="plasma",aspect='auto',vmin=-50)
plt.title("PSD for the time coefficients of mode {}".format(psd_i_mode))
plt.hlines(0.2, all_Ur[0],all_Ur[-1],linestyle='--', color='black', linewidth=1.)

# Plot dominant frequency for all Urs ----------------------------------------------------------------------------------
plt.plot(all_Ur,1/all_Ur,linestyle='--', color='black', linewidth=1.)
plt.xlabel("$U_r$ [-]")
plt.ylabel("Frequency [-]")
plt.colorbar()
plt.tight_layout()
plt.plot()
plt.show()

dom_freqs = late_frequencies[:,0]


x_urs = np.linspace(0,all_Ur[-1],100)
# Plot non dimensional (i.e. rescaled) frequencies for all Urs ---------------------------------------------------------
plt.figure(figsize=(8,8))
plt.plot(all_Ur,dom_freqs*all_Ur, marker = '+', color='b', linestyle='none')
plt.hlines(1.,0,all_Ur[-1], linestyle='--',color='k')
plt.plot(x_urs, 0.2*x_urs, linestyle='--',color='k')
# plt.gca().set_aspect('auto')
plt.xlim(0,15)
plt.ylim(0,2)
plt.show()

freqs_matrix = np.transpose(np.array([freqs_psd for i in range(len(all_Ur))]))
rescaled_freqs_matrix = freqs_matrix*all_Ur
urs_matrix = np.array([all_Ur for i in range(len(freqs_psd))])
rescaled_freqs_matrix_AM = freqs_matrix*all_Ur * np.sqrt(2)

# Plot PSD for all Urs, with rescaled frequencies ----------------------------------------------------------------------
plt.figure(figsize=(20,20))
plt.pcolormesh(urs_matrix,rescaled_freqs_matrix_AM,np.log(psd_transpose[:,psd_i_mode,:]),vmin=-40,cmap='magma')
plt.hlines(1.0,0,all_Ur[-1], linestyle='dotted',color='k',label='$\\bar{f_v}=\\bar{f_c^{AM}}$')
plt.plot(x_urs,0.2*x_urs, linestyle='dashed',color='k',label="$S_T U_R$")
plt.colorbar()
plt.xlim(all_Ur[0],10)
plt.ylim(0,5)
plt.xlabel("$U_R$ [-]")
plt.ylabel("$\\bar{f_v}/\\bar{f_c^{AM}}$ [-]")
plt.title("PSD of the extrapolated time coefficient $a_{}(t)$ - Mode {}".format(psd_i_mode+1,psd_i_mode+1))
plt.legend()
plt.savefig("C:\\Users\\Student\\Documents\\Suivi\\Biweekly meetings\\Suivi PdR - 2023_06_15\\figures\\PSD_mode_{}_AM_plus_lines.svg".format(psd_i_mode+1))
plt.show()



a_1_late = sol_ext.y[0][-N_ext//4:]
a_2_late = sol_ext.y[1][-N_ext//4:]
freqs_psd, psd_signal_1 = periodogram(a_1_late, sampling_rate,scaling='spectrum')
freqs_psd, psd_signal_2 = periodogram(a_2_late, sampling_rate,scaling='spectrum') 

# Plot first time coefficients -----------------------------------------------------------------------------------------
plt.figure(figsize=(25,16))
plt.plot(t_ext,sol_ext.y[0],color="#FF1F5B", label="$a_1(t)$")
plt.plot(t_ext,sol_ext.y[1],color=colors_geodataviz[1], label="$a_2(t)$")
plt.xlabel("Time [-]")
plt.ylabel("Time coefficients")
plt.title("First time coeffs. - Extrapolated")
plt.vlines(x=t_ext[-N_ext//4],ymin=-600,ymax=600, color='r',linewidth=2.)
plt.ylim(-500,500)
plt.legend()
plt.savefig("C:\\Users\\Student\\Documents\\Suivi\\Biweekly meetings\\Suivi PdR - 2023_06_15\\figures\\first_two_modes_Ur5_extrapolated.svg" )

# Plot the PSDs of the first time coefficients -------------------------------------------------------------------------
plt.figure(figsize=(25,16))
plt.semilogy(freqs_psd[1:], psd_signal_1[1:], color = colors_geodataviz[0], label = "$a_1(t)$")
plt.semilogy(freqs_psd[1:], psd_signal_2[1:], color = colors_geodataviz[1], label = '$a_2(t)$')
plt.legend()
plt.xlabel("Frequencies [-]")
plt.ylabel("PSD of the time coefficients [-]")
plt.title("Power Spectral Densities of the POD coefficients")
plt.savefig("C:\\Users\\Student\\Documents\\Suivi\\Biweekly meetings\\Suivi PdR - 2023_06_15\\figures\\PSD_first_two_modes_Ur5.svg" )


added_mass_eq_coeff = nCond.Mass_number / (1+nCond.Mass_number) 

savefig_path = "C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\two_way_coupling\\"

# Plot in-line maximum vibration amplitudes eta ------------------------------------------------------------------------
plt.figure()
plt.plot(all_Ur,eta_amplitudes_max, color="#004e7c", marker = 's')
plt.plot(FSI_sim_Ur_x, true_eta_max, 'k-', marker='o', markerfacecolor='none', markeredgecolor='k' )
plt.xlabel("$U_r$")
plt.ylabel("$\eta_{max}$")
# tpl.save(savefig_path + "displacement_eta_max_vs_Ur.tex")

# Plot cross-flow maximum vibration amplitudes xi ------------------------------------------------------------------------
plt.figure()
plt.plot(all_Ur,xi_amplitudes_max, color="#FF7F00", marker = 's')
plt.plot(FSI_sim_Ur_y, true_xi_max, 'k-', marker='o', markerfacecolor='none', markeredgecolor='k' )
plt.xlabel("$U_r$")
plt.ylabel("$\\xi_{max}$")
# tpl.save(savefig_path + "displacement_xi_max_vs_Ur.tex")


# def F_total(time_coeffs):
plt.figure(figsize=(10,15))
plt.plot(all_Ur[:],eta_amplitudes_max[:], color="#004e7c", marker = 's')
plt.plot(FSI_sim_Ur_x, true_eta_max, 'k-', marker='o', markerfacecolor='none', markeredgecolor='k' )
plt.xlabel("$U_r$")
plt.ylabel("$\eta_{max}$")
# tpl.save(savefig_path + "displacement_eta_max_vs_Ur.tex")

plt.figure(figsize=(10,15))
plt.plot(all_Ur[:],xi_amplitudes_max[:], color="#FF7F00", marker = 's')
plt.plot(FSI_sim_Ur_y, true_xi_max, 'k-', marker='o', markerfacecolor='none', markeredgecolor='k' )
plt.xlabel("$U_r$")
plt.ylabel("$\\xi_{max}$")