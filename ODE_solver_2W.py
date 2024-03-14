import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from time import time
from keras import backend as K
import os
import string
import pandas as pd
from pathlib import Path
from scipy.signal import periodogram
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.integrate import solve_ivp
from tqdm import tqdm
from auxiliary_functions import *
from parameters import Parameters
nCond = Parameters()
from projection_matrices_2W import *

# Parameters for plotting the figures ==================================================================================
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
# Colormap to plot the coefficients of the POD modes -------------------------------------------------------------------
custom_cmap = ["#004e7c","#FF7F00", "#00a878","#FF0000","#A742FF", "#004d00","#1ABC9C","#ec4373","#f0b729" ]
# from numba import jit

# Extraction of the data ===============================================================================================
f = open("POD/modes/points.pl",'rb')
close_points = pickle.load(f)

f = open("POD/modes/true_time_coeffs.pl",'rb')
true_time_coeffs = pickle.load(f)


x_sp, y_sp = tf.transpose(close_points)

# Simultation options ==================================================================================================
extrapol_test = True
save_coeffs = False
compute_matrices = True
postprocess = False

t_start = time()

# Definition of the parameters =========================================================================================
solver_method = 'RK45'
# solver_method = 'RK23'
# solver_method = 'Radau'

# number of time points
N_validation = 201

# Initial conditions ---------------------------------------------------------------------------------------------------
# Initial cylinder displacement
eta_0 = 0.
xi_0 = 0.
# Initial cylinder velocity
eta_dot_0 = 0.0
xi_dot_0 = 0.0

# Initializing parameters --------- ------------------------------------------------------------------------------------
N_modes = nCond.N_modes

print("Number of modes :", N_modes)

mode_path = "POD/ModePINN/saved/"

save_path = nCond.saved_path
figures_path = nCond.figures_path
matrices_path = "saved\\{}_modes\\matrices\\".format(nCond.N_modes)

saved_path_dir = Path(save_path)
figures_path_dir = Path(figures_path)


saved_path_dir.mkdir(parents=True, exist_ok=True)
figures_path_dir.mkdir( parents=True, exist_ok=True)
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

# Lambda_mat = Lambda_matrix(mean_mode,modes, x_sp, y_sp)
# Lambda_mat = Lambda_matrix_update(mean_mode,modes, x_sp, y_sp)
Lambda_mat = Lambda_matrix_MC(mean_mode,modes, x_sp.numpy(), y_sp.numpy())

# # Testing that we can compute all matrices *****************************************************************************
# print("T")
# print(T_matrix(modes, x_sp,y_sp))
# print("M")
# print(M_matrix(mean_mode,modes, x_sp,y_sp))
# print("L")
# print(L_matrix(modes, x_sp,y_sp))
# # print("P")
# # print(P_matrix(modes, x_sp,y_sp))
# print("Q")
# print(Q_matrix(modes, x_sp,y_sp))
# print("Chi")
# print(Chi_matrix(modes, x_sp,y_sp))
# print("Beta")
# print(Beta_matrix(modes, x_sp,y_sp))

# print("Gamma")
# print(Gamma_matrix(modes, x_sp,y_sp))
# print("Lambda")
# print(Lambda_matrix(mean_mode, modes, x_sp,y_sp))

# Show matrix heatmap **************************************************************************************************    
# heatmap_matrix(Gamma_np,"$\\Gamma$")
# ODE solver ===========================================================================================================

Q_np = Q_mat.numpy()
Chi_np = Chi_mat

Gamma_np = Gamma_mat
Lambda_np = Lambda_mat

# Gamma_no_DOF = Gamma_mat[:N_modes,:N_modes]
# Lambda_no_DOF = Lambda_mat[:N_modes, :N_modes]

Gamma_inv = np.linalg.inv(Gamma_np)
# Gamma_inv_no_DOF = np.linalg.inv(Gamma_no_DOF)

# gamma_bar_x_value = gamma_bar_x(mean_mode)
# gamma_bar_y_value = gamma_bar_y(mean_mode)
with open("POD\\modes\\p\\p_mean.pl", 'rb') as f:
    p_mean = pickle.load(f)

gamma_bar_x_value, gamma_bar_y_value  = gammas_POD_mean_MC(x_sp.numpy(), y_sp.numpy(), p_mean)

C_D =  -gamma_bar_x_value*np.pi*nCond.Mass_number/2


def Fq_AM_ND(a):
    """"
    Function that computes the force matrix F_NL (non-linear terms that don't depend on the POD time coefficients a_i(t)) in the projected Navier-Stokes equations.
    input: extended time coefficients vector for a given timestep t_0 (a_1 ... a_N eta eta_dot xi xi_dot)
    output: Force matrix of dimension N (number of modes).
    """
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
    Fq_list.append( -2/(np.pi*(1+M)) *gamma_bar_x_value +  M / (1 + M) * added_damping_fx)
    Fq_list.append(0.0)
    Fq_list.append(-2/(np.pi*(1+M)) *gamma_bar_y_value  + M / (1 + M) * added_damping_fy)
    Fq_list = np.array(Fq_list)
    return Fq_list


# @jit(forceobj=True)
def ODE_N_2DOF(t,a):
    """
    Function implemented to solve the ODE resulting from the projection of the NS equations on the POD modes. It establishes a link between the da/dt and a(t), a being the extended time coeff. matrix.
    """
    Fq_a_AM = Fq_AM_ND(a)
    dadt = np.matmul(np.matmul(Gamma_inv,Lambda_np),a)+ np.matmul(Gamma_inv,Fq_a_AM)    
    return dadt

# Initialization of the values of the time coefficients at the beginning of the simulation -----------------------------
a0 = np.zeros(N_modes+4)
a0[:N_modes] = true_time_coeffs[:N_modes,0]
a0[N_modes:] = np.array([eta_0,eta_dot_0, xi_0, xi_dot_0])
# a0_no_DOF = true_time_coeffs[:N_modes,0]

t_end_matrices = time()
print("Computing all matrices took: {}s".format(t_end_matrices - t_start))
print("\nAll matrices have been computed. We are now going to solve the ODE")

# Time steps of the simulation -----------------------------------------------------------------------------------------
t_solve = np.linspace(nCond.t_min,nCond.t_max,N_validation)

# Solving the ODE ======================================================================================================
print("\nSolver in progress... Please wait........................")
t_ODE_i = time()

sol = solve_ivp(ODE_N_2DOF,t_span=[nCond.t_min,nCond.t_max],y0=a0,t_eval=t_validation, method=solver_method)

t_ODE_f = time()
print('{0} took {1:.8f}s to compute'.format("ODE solution", t_ODE_f - t_ODE_i))
# ======================================================================================================================
# Plot comparison between real POD coeffs and predicted POD coeffs -----------------------------------------------------
plt.figure()
# for i in range(N_modes):
#     plt.plot(t_validation,true_time_coeffs[i,:],linestyle='dashed', label = 'True $a_{}(t)$ (fixed cylinder)'.format(i+1))
for i in range(N_modes):
    plt.plot(t_validation, sol.y[i], label = '$a_{}(t)$ '.format(i+1))
plt.xlabel("Time (s)")
plt.ylabel("Time coefficients")
plt.title("Comparing the true coefficients with the outputs of an ODE solver")
plt.legend()
plt.grid()
# plt.show()
plt.savefig(figures_path +"POD_time_coeffs_{}_modes.png".format(nCond.N_modes),transparent=True)
plt.close()

# Plot cylinder displacements ------------------------------------------------------------------------------------------
plt.figure()
plt.plot(t_validation, sol.y[N_modes], label = '$\eta(t)$ ')
plt.plot(t_validation, sol.y[N_modes+2], label = '$\\xi(t)$ ')
plt.xlabel("Time (s)")
plt.title("Cylinder displacements in $x$ and $y$")
plt.legend()
plt.grid()
# plt.show()
plt.savefig(figures_path +"displacement_{}_modes.png".format(nCond.N_modes),transparent=True)
plt.close()

# Plot early trajectory ------------------------------------------------------------------------------------------------
plt.figure()
plt.title("Cylinder trajectory")
plt.plot(sol.y[N_modes],sol.y[N_modes+2], '-',color = 'black',linewidth = 1.0)
plt.scatter(sol.y[N_modes][0],sol.y[N_modes+2][0], color = 'green', s=10.0, label = "Start of the trajectory")
plt.scatter(sol.y[N_modes][-1],sol.y[N_modes+2][-1], color = 'red', s=10.0, label = "End of the trajectory")
plt.grid()
plt.xlabel("$\eta$")
plt.ylabel("$\\xi$")
plt.legend()
# plt.show()
plt.savefig(figures_path +"cyl_trajectory_{}_modes.png".format(nCond.N_modes),transparent=True)
plt.close()

# beep()

# Plot POD coefficients only -------------------------------------------------------------------------------------------
plt.figure()
for i in range(N_modes):
    plt.plot(t_validation, sol.y[i], label = '$a_{{{}}}(t)$'.format(i+1))
plt.xlabel("Time (s)")
plt.ylabel("Time coefficients")
plt.title("Time coefficients - ODE solver")
plt.legend()
plt.grid()
# plt.show()
plt.close()

cmap = plt.cm.tab10.colors
t_test = np.linspace(400.,420.,201)
plt.figure()
for i in range(N_modes):
    plt.plot(t_validation, sol.y[i], label = '$a_{{{}}}(t)$'.format(i+1), c=cmap[i])
    plt.plot(t_test, true_time_coeffs[i], '--', c=cmap[i])
plt.xlabel("Time (s)")
plt.ylabel("Time coefficients")
plt.title("Time coefficients - ODE solver")
plt.legend()
plt.grid()
# plt.show()
# Plot cylinder displacements and velocities ---------------------------------------------------------------------------
plt.figure()
plt.plot(t_validation, sol.y[N_modes], label ='$\eta $')
plt.plot(t_validation, sol.y[N_modes+1], label ='$\dot{\eta}$')
plt.plot(t_validation, sol.y[N_modes+2], label ='$\\xi $')
plt.plot(t_validation, sol.y[N_modes+3], label ='$\dot{\\xi} $')
plt.xlabel("Time (s)")
plt.ylabel("Displacement coefficients of the cylinder")
plt.title("Displacement coefficients of the cylinder - ODE solver")
plt.legend()
plt.grid()
# plt.show()
plt.savefig(figures_path +"cyl_disp_vel_{}_modes.png".format(nCond.N_modes),transparent=True)
plt.close()


print("Number of modes : {}".format(N_modes))
print("Ur :", nCond.Ur)
print("\nMass number : {:.2f}".format(nCond.Mass_number))
# print("Gamma bar x : {:.5f}".format(gamma_bar_x_value) )
# print("Gamma bar y : {:.5f}".format(gamma_bar_y_value) )


# Saving the coeffs ----------------------------------------------------------------------------------------------------
if save_coeffs:
    sol_df = pd.DataFrame(sol.y)
    sol_df.to_csv(save_path + "time_coeffs_{}_modes.csv".format(N_modes),header=False, index=False)
    print("Predicted coefficients have been saved in " + save_path + "time_coeffs_{}_modes.csv".format(N_modes))
    

if not extrapol_test:
    beep()

# Forces ===============================================================================================================
print("Computing the forces ................................")

def F_p_xy_MC(time_coeffs):
    """
    Function to compute the total pressure force in x and y using the Monte-Carlo integration method.
    """
    with open("POD\\modes\\p\\p_modes.pl",'rb') as f:
        p_POD_modes = pickle.load(f) 
    # The gammas are already multiplied by - 2 /(pi * (1+M))
    sigmas_x, sigmas_y = sigmas_xy_MC(x_sp.numpy(),y_sp.numpy(), p_POD_modes) 
    sigmas_x = sigmas_x[:N_modes]* (1+nCond.Mass_number)/nCond.Mass_number
    sigmas_y = sigmas_y[:N_modes]* (1+nCond.Mass_number)/nCond.Mass_number

    f_tot_x = np.ones(len(time_coeffs[0]))*gamma_bar_x_value * 2/ (np.pi* nCond.Mass_number) 
    f_tot_y = np.ones(len(time_coeffs[0]))*gamma_bar_y_value * 2/ (np.pi* nCond.Mass_number) 
    for i in range(N_modes):
        gamma_xi, gamma_yi = sigmas_x[i],sigmas_y[i] 
        f_tot_x += gamma_xi * time_coeffs[i]
        f_tot_y += gamma_yi * time_coeffs[i]
    return (f_tot_x, f_tot_y)
        
Fx, Fy = F_p_xy_MC(sol.y[:N_modes])

# Plot the pressure forces in x and y ----------------------------------------------------------------------------------
plt.figure()
plt.title("Force $F_x(t)$")
plt.grid()
plt.xlabel("Time t")
plt.plot(t_validation,Fx)
# plt.show()
plt.savefig(figures_path +"Fx_force_{}_modes.png".format(nCond.N_modes),transparent=True)


plt.figure()
plt.title("Force $F_y(t)$")
plt.grid()
plt.xlabel("Time t")
plt.plot(t_validation,Fy)
# plt.show()
plt.savefig(figures_path +"Fy_force_{}_modes.png".format(nCond.N_modes),transparent=True)

if save_coeffs:
    pd.DataFrame(Fx).to_csv(save_path + "Fx_{}_modes.csv".format(nCond.N_modes),header=False, index = False)
    pd.DataFrame(Fy).to_csv(save_path + "Fy_{}_modes.csv".format(nCond.N_modes),header=False, index = False)

# Extrapolation results ================================================================================================
# Number of extrapolation points
N_ext = nCond.n_ext

if extrapol_test :
    print("\nTrial on extrapolation :\n")
    # Time points
    t_max_ext = nCond.t_max_ext
    t_ext = np.linspace(nCond.t_min,t_max_ext,N_ext)

    t_start_extODE = time()

    sol_ext = solve_ivp(ODE_N_2DOF,t_span=[nCond.t_min,t_max_ext],y0=a0,t_eval=t_ext, method=solver_method)
    t_end_extODE = time()
    print("Solving ext ODE took: {}s".format(t_end_extODE-t_start_extODE ))
    # Plot comparison between real POD coeffs and predicted POD coeffs -------------------------------------------------
    plt.figure(figsize=(30,20))
    for i in range(N_modes):
        plt.plot(t_ext, sol_ext.y[i], label = '$a_{}(t)$ '.format(i+1))
    plt.xlabel("Time (s)")
    plt.ylabel("Time coefficients")
    plt.title("Extrapolation of the time coefficients outside of the simulated time domain",fontsize=29)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(figures_path +"POD_time_coeffs_{}_modes_ext.png".format(nCond.N_modes),transparent=True)

    # Plot cylinder displacements --------------------------------------------------------------------------------------
    plt.figure()
    plt.plot(t_ext, sol_ext.y[N_modes], label = '$\eta(t)$ ')
    plt.plot(t_ext, sol_ext.y[N_modes+2], label = '$\\xi(t)$ ')
    plt.xlabel("Time (s)")
    plt.title("Extrapolation of the cylinder displacements in $x$ and $y$")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(figures_path +"displacement_coeffs_{}_modes_ext.png".format(nCond.N_modes),transparent=True)

    # Plot cylinder trajectory ----------------------------------------------------------------------------------------------
    plt.figure()
    plt.title("Cylinder trajectory - Extrapolated")
    plt.plot(sol_ext.y[N_modes],sol_ext.y[N_modes+2], '--k',linewidth = 2.0)
    plt.scatter(sol_ext.y[N_modes][0],sol_ext.y[N_modes+2][0], color = 'green', s=10.0, label = "Start of the trajectory")
    plt.scatter(sol_ext.y[N_modes][-1],sol_ext.y[N_modes+2][-1], color = 'red', s=10.0, label = "End of the trajectory")
    plt.grid()
    plt.xlabel("$\eta$")
    plt.ylabel("$\\xi$")
    plt.legend()
    plt.savefig(figures_path +"cyl_trajectory_ext_{}_modes.png".format(nCond.N_modes),transparent=True)
    # plt.show()

    # Plot only the late trajectory ------------------------------------------------------------------------------------
    plt.figure()
    plt.title("Cylinder trajectory - Extrapolated")
    plt.plot(sol_ext.y[N_modes][-200:-1],sol_ext.y[N_modes+2][-200:-1], '-k',linewidth = 1.0)
    plt.grid()
    plt.xlabel("$\eta$")
    plt.ylabel("$\\xi$")
    # plt.show()
    plt.savefig(figures_path +"cyl_late_trajectory_ext_{}_modes.png".format(nCond.N_modes),transparent=True)

    # Saving the extrapolation coeffs ----------------------------------------------------------------------------------
    if save_coeffs :
        sol_ext_df = pd.DataFrame(sol_ext.y)
        sol_ext_df.to_csv(save_path + "time_coeffs_ext_{}_modes.csv".format(N_modes),header=False, index=False)
        print("Predicted coefficients have been saved in " + save_path + "time_coeffs_ext_{}_modes.csv".format(N_modes))
        fig = plt.figure(figsize=(20,10))
    
    

n_final = nCond.n_final

sol_late = sol_ext.y[:,-n_final:]
t_late = t_ext[-n_final:]
sampling_rate = 1/(t_late[1]-t_late[0])

alphabet = list(string.ascii_lowercase)


savefig_path = "C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\two_way_coupling\\"

f,ax = plt.subplots(N_modes//2,2,figsize=(20,20),gridspec_kw={'width_ratios': [2, 1]})
for i in range(N_modes//2):
    ax[i,0].plot(t_late, sol_late[2*i], color=custom_cmap[2*i],label="$a_{}$".format(2*i+1))
    ax[i,0].plot(t_late, sol_late[2*i+1], color=custom_cmap[2*i+1],label="$a_{}$".format(2*i+1+1))
    ax[i,0].legend(loc='upper right')
    ax[i,0].set_xlabel("Time $t$")
    ax[i,0].set_ylabel("$a_i(t)$")
    ax[i,0].set_title("({})".format(alphabet[2*i]),loc="center",size=24)

    freqs, psd_1 =periodogram(sol_late[2*i], sampling_rate,scaling='density')
    freqs,psd_2 = periodogram(sol_late[2*i+1], sampling_rate,scaling='density')

    ax[i,1].plot(freqs, psd_1/np.max(psd_1), color=custom_cmap[2*i],label="$a_{}$".format(2*i+1))
    ax[i,1].plot(freqs, psd_2/np.max(psd_2), color=custom_cmap[2*i+1],label="$a_{}$".format(2*i+1+1))
    ax[i,1].legend(loc='upper right')
    ax[i,1].set_xlabel("Frequency $f$")  
    ax[i,1].set_ylabel("Normalized PSD")
    ax[i,1].set_xlim(0,1.5)
    ax[i,1].set_title("({})".format(alphabet[2*i+1]),loc="center",size=24)
plt.tight_layout()
plt.savefig(savefig_path + "extrapolated_POD_coefficients_PSD_legend_Ur2.svg")
    # beep()
# if postprocess:
#     import postprocessing_ODE

t_final = t_late
eta_final = sol_late[N_modes]
xi_final = sol_late[N_modes+2]

freqs, psd_eta_final =periodogram(eta_final, sampling_rate,scaling='density')
freqs, psd_xi_final = periodogram(xi_final, sampling_rate,scaling='density')

fig = plt.figure(figsize=(20,10))
plt.rc('font',size=17)
gs = GridSpec(2, 3, width_ratios=[3, 2, 2], height_ratios=[1,1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[:,2])
ax4 = fig.add_subplot(gs[3])
ax5 = fig.add_subplot(gs[4])

# Eta 
ax1.plot(t_final, eta_final,c=custom_cmap[0])
ax1.set_xlabel("Time $t$")
ax1.set_ylabel("$\\eta(t)$")
ax1.set_title("In-line displacement of the cylinder")

ax2.plot(freqs, psd_eta_final/np.max(psd_eta_final),c=custom_cmap[0])
ax2.set_xlabel("Frequency f")
ax2.set_ylabel("Normalized PSD")
ax2.set_xlim(0,2)
ax2.set_title("PSD of $\\eta(t)$")    

# Trajectory    
ax3.plot(eta_final, xi_final, c= 'k')
ax3.set_xlabel("$\\eta$")
ax3.set_ylabel("$\\xi$")
ax3.set_title("Trajectory of the cylinder")

# Xi   
ax4.plot(t_final, xi_final,c=custom_cmap[1])
ax4.set_xlabel("Time $t$")
ax4.set_ylabel("$\\xi(t)$")
ax4.set_title("Cross-flow displacement of the cylinder")

ax5.plot(freqs, psd_xi_final/np.max(psd_xi_final),c=custom_cmap[1])
ax5.set_xlabel("Frequency f")
ax5.set_ylabel("Normalized PSD")
ax5.set_xlim(0,2)
ax5.set_title("PSD of $\\xi(t)$")

plt.tight_layout()
plt.show()