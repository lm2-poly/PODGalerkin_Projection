import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from time import time
import os
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from projection_matrices_1W_stokes import *

os.chdir("c:\\Users\\Student\\Documents\\Project\\Endgame\\Unified_PODVM_Galerkin")
from parameters import Parameters
nCond = Parameters()


from auxiliary_functions import *
import tikzplotlib  as tpl
import string
from scipy.signal import periodogram
from matplotlib.gridspec import GridSpec

# Plotting options -----------------------------------------------------------------------------------------------------
# COLOR = 'black'
# mpl.rcParams['text.color'] = COLOR
# mpl.rcParams['axes.labelcolor'] = COLOR
# mpl.rcParams['xtick.color'] = COLOR
# mpl.rcParams['ytick.color'] = COLOR
# mpl.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (40, 30)
plt.rc('font',size=28)

custom_cmap = ["#004e7c","#FF7F00", "#00a878","#FF0000","#A742FF", "#004d00","#1ABC9C","#ec4373","#f0b729" ]
alphabet = list(string.ascii_lowercase)
# ----------------------------------------------------------------------------------------------------------------------

# Extraction of the data ===============================================================================================
# Points with limited amounts around the contour
f = open("POD/modes/points_limited_contour.pl",'rb')
close_points = pickle.load(f)

# All points around the contour
f = open("POD/modes/points.pl",'rb')
close_points_all = pickle.load(f)

# close_points = close_points_all 

f = open("POD/modes/true_time_coeffs.pl",'rb')
true_time_coeffs = pickle.load(f)


x_sp, y_sp = tf.transpose(close_points)

t_start = time()

# Definition of the parameters =========================================================================================

solver_method = 'RK45'
# solver_method = 'RK23'
# solver_method = 'Radau'

# number of time points
n_sol = 200
# Number of points in the extrapolation
n_ext = nCond.n_ext
# Number of points to define the final regime
n_final = nCond.n_final

# ROM options ==========================================================================================================
extrapol_test = True
save_coeffs = False
cylinder_motion = True
use_old_matrices = False 
cover_all_ur = True
reconstruct_flow_fields = False
# ======================================================================================================================

# Initializing parameters ----------------------------------------------------------------------------------------------

N_modes = param.N_modes

print("Number of modes :", N_modes)

mode_path = "POD/ModePINN/saved/"

save_path = "saved\\{}_modes\\fixed\\data\\".format(param.N_modes)
figures_path = "saved\\{}_modes\\fixed\\figures\\".format(param.N_modes)
matrices_path = "saved\\{}_modes\\matrices_fixed\\".format(param.N_modes)

saved_path_dir = Path(save_path)
figures_path_dir = Path(figures_path)

saved_path_dir.mkdir(parents=True, exist_ok=True)
figures_path_dir.mkdir( parents=True, exist_ok=True)

Path(matrices_path).mkdir(parents=True, exist_ok=True)

mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")

t_validation = tf.linspace(param.t_min, param.t_max,np.shape(true_time_coeffs)[1])

# Initializing the modes list, containing the models for all modes except the mean mode --------------------------------
modes = []
# N_modes counts the POD modes and the mean mode
for i in range(1, N_modes+1):
    modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))

# Computing the matrices needed for the projected equations ============================================================
# Now let's get the matrices
if not os.path.isfile(matrices_path + 'q_mat.p') or not use_old_matrices:
    Q_mat = Q_matrix(modes, x_sp, y_sp)
    f = open(matrices_path + 'q_mat.p','wb')
    pickle.dump(Q_mat,f)
else:
    f = open(matrices_path + 'q_mat.p','rb')
    Q_mat = pickle.load(f)
    print("Q was already computed")

if not os.path.isfile(matrices_path + 'gamma_mat.p') or not use_old_matrices:
    Gamma_mat = Gamma_matrix(modes, x_sp,y_sp)
    with open(matrices_path + 'gamma_mat.p','wb') as f:
        pickle.dump(Gamma_mat,f)
else:
    with open(matrices_path + "gamma_mat.p",'rb') as f:
        Gamma_mat = pickle.load(f)
    print("Gamma was already computed")

Lambda_mat = Lambda_matrix(mean_mode,modes, x_sp, y_sp)


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
# print("Gamma")
# print(Gamma_matrix(modes, x_sp,y_sp))
# print("Lambda")
# print(Lambda_matrix(mean_mode, modes, x_sp,y_sp))

# heatmap_matrix(Gamma_np,"$\\Gamma$")
# ODE solver ===========================================================================================================

Q_np = Q_mat.numpy()

Gamma_np = Gamma_mat
Lambda_np = Lambda_mat

Gamma_inv = np.linalg.inv(Gamma_np)


def Fq_AM_fixed(a):
    Fq_list = []
    for i in range(N_modes):
        Fq_i = -np.matmul(a,np.matmul(Q_np[i],a))
        Fq_list.append(Fq_i)
    return Fq_list

def solve_projected_ODE(t,a):
    Fq_a_AM = Fq_AM_fixed(a)
    dadt = np.matmul(np.matmul(Gamma_inv,Lambda_np),a)+ np.matmul(Gamma_inv,Fq_a_AM)    
    return dadt
    
# Define initial conditions of the time coefficients vector
a0 = np.ones(N_modes)
a0[:N_modes] = true_time_coeffs[:N_modes,0]

t_end_matrices = time()
print("Duration of matrix computation: {}\n".format(t_end_matrices - t_start))
print("All matrices have been computed. We are now going to solve the ODE.\n")

# time points
t_solve = np.linspace(param.t_min,param.t_max,n_sol)

print("Solver in progress... Please wait........................")
t_ODE_i = time()

sol = solve_ivp(solve_projected_ODE,t_span=[param.t_min,param.t_max],y0=a0,\
                t_eval=t_validation, method=solver_method)

t_ODE_f = time()
print('{0} took {1:.8f}s to compute'.format("ODE solution", t_ODE_f - t_ODE_i))

# Plot POD coefficients only -------------------------------------------------------------------------------------------
plt.figure(figsize=(30,20))
for i in range(N_modes):
    plt.plot(t_validation, true_time_coeffs[i],'--')
for i in range(N_modes):
    plt.plot(t_validation, sol.y[i], label = '$a_{{{}}}(t)$'.format(i+1))
plt.xlabel("Time (s)")
plt.ylabel("Time coefficients")
plt.title("Time coefficients - ODE solver")
plt.legend()
plt.grid()
plt.savefig(figures_path +"POD_time_coeffs_{}_modes.png".format(param.N_modes),transparent=True)

print("Number of modes : {}".format(N_modes))
print("Ur :", param.Ur)
print("\nMass number : {:.2f}".format(param.Mass_number))

# Saving the coeffs ----------------------------------------------------------------------------------------------------
if save_coeffs:
    sol_df = pd.DataFrame(sol.y)
    sol_df.to_csv(save_path + "time_coeffs_{}_modes.csv".format(N_modes),header=False, index=False)
    print("Predicted coefficients have been saved in " + save_path + "time_coeffs_{}_modes.csv".format(N_modes))
    

if not extrapol_test:
    beep()

if reconstruct_flow_fields :
    
    def reshape_tf(x):
        return tf.cast(tf.reshape(x,(1,len(x))),tf.float32)

    def reshape_np(x):
        return np.reshape(x,(1,len(x)))
    
    def reconstruct_all_fields(N_modes, time_coeffs):
        fields = tf.cast(mean_mode(close_points),tf.float32)
        u,v,p = tf.transpose(fields) 
        N_t = len(time_coeffs[0])

        u = tf.ones((N_t,1),dtype=tf.float32)*reshape_tf(u)
        v = tf.ones((N_t,1),dtype=tf.float32)*reshape_tf(v)
        p = tf.ones((N_t,1),dtype=tf.float32)*reshape_tf(p)

        for i_mode in range(N_modes):
            t_coeff_i = tf.cast(tf.transpose(tf.reshape((time_coeffs[i_mode,:]),(1,N_t))),tf.float32)
            u_i, v_i, p_i = tf.transpose(modes[i_mode](close_points))
            u_i = reshape_tf(u_i)
            v_i = reshape_tf(v_i)
            p_i = reshape_tf(p_i)
            
            u = u + t_coeff_i*u_i 
            v = v + t_coeff_i * v_i 
            p = p + t_coeff_i * p_i 
            
            # mode_i = tf.reshape(modes[i_mode](close_points), (1,3*N_points))
            # mode_i = tf.cast(mode_i, dtype=tf.float32)
            # fields = fields + tf.matmul(t_coeff_i , mode_i)
        return u,v,p
    
    Nx,Ny = 200,200 
    grid_x, grid_y = grid_x, grid_y = np.meshgrid(np.linspace(param.Lxmin, param.Lxmax, Nx), np.linspace(param.Lymin, param.Lymax, Ny))

    utest,vtest,ptest = reconstruct_all_fields(param.N_modes,sol.y)



    filename_data = '..\\..\\Data\\fixed_cylinder_atRe100' 

    file_data = open(filename_data, 'r')

    from text_flow import read_flow
    data = read_flow(filename_data)

    # Re, Ur = data[0], data[1]
    t_mes = tf.constant(data[2],dtype = tf.float32)

    nodes_x, nodes_y = data[3], data[4]
    u_mes, v_mes, p_mes = data[5], data[6], data[7]

    N_points = len(nodes_x[0])  # Nb of points in the data

    geom = param.get_geom()

    Lxmin, Lxmax, Lymin, Lymax, x_c, y_c, r_c = geom

    # Generating a list of points - for the inteprolation later
    points = []
    for i in range(N_points):
        points.append([nodes_x[0][i], nodes_y[0][i]])

    point_list = np.arange(N_points)
    close_points_id = []

    for i in range(N_points):
        if is_in_domain(points[i], geom=geom):
            close_points_id.append(i)

    close_points_id = np.array(close_points_id)
    N_points = len(close_points)
    xp = close_points[:,0]
    yp = close_points[:,1]

    u_data = u_mes[:,close_points_id]
    v_data = v_mes[:,close_points_id]
    p_data = p_mes[:,close_points_id]

    total_loss = tf.reduce_mean(tf.square(utest-u_data)) +\
          tf.reduce_mean(tf.square(vtest-v_data))+ \
            tf.reduce_mean(tf.square(ptest-p_data))
    
    print("Total loss for {} modes : {}".format(N_modes, total_loss.numpy()))

# Extrapolation results ================================================================================================

if extrapol_test :
    print("\nTrial on extrapolation :\n")
    # time points
    t_max_ext = nCond.t_max_ext
    t_ext = np.linspace(param.t_min,t_max_ext,n_ext)

    t_start_ODE_ext = time()
    sol_ext = solve_ivp(solve_projected_ODE,t_span=[param.t_min,t_max_ext],y0=a0,\
                        t_eval=t_ext, method=solver_method)
    t_end_ODE_ext = time()
    print("Time to solve ext. ODE: {}".format(t_end_ODE_ext - t_start_ODE_ext))

    # Plot comparison between real POD coeffs and predicted POD coeffs -------------------------------------------------
    plt.figure(figsize=(30,20))
    for i in range(N_modes):
        plt.plot(t_ext, sol_ext.y[i], label = '$a_{}(t)$ '.format(i+1))
    plt.xlabel("Time (s)")
    plt.ylabel("Time coefficients")
    plt.title("Extrapolation of the time coefficients outside of the simulated time domain")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(figures_path +"POD_time_coeffs_{}_modes_ext.png".format(param.N_modes),transparent=True)

    # Saving the extrapolation coeffs ----------------------------------------------------------------------------------
    if save_coeffs :
        sol_ext_df = pd.DataFrame(sol_ext.y)
        sol_ext_df.to_csv(save_path + "time_coeffs_ext_{}_modes.csv".format(N_modes),header=False, index=False)
        print("Predicted coefficients have been saved in " + save_path + "time_coeffs_ext_{}_modes.csv".format(N_modes))


if cylinder_motion:
    # Models the vibrations of the cylinder with one-way coupling
    # Get the contour points ===========================================================================================
    print("Now computing the resulting pressure forces on the cylinder")
    x_all, y_all = np.transpose(close_points_all)
    epsilon = nCond.epsilon_contour
    contour_indices = isolate_contour_points(x_all, y_all, epsilon)
    contour_points = close_points_all[contour_indices]
    x_contour, y_contour = np.transpose(contour_points)

    plt.figure()
    plt.title("Filtered contour points")
    plt.scatter(x_contour, y_contour, s=0.5)
    plt.gca().set_aspect("equal")
    plt.show()

    # Get the exact pressure modes =====================================================================================
    POD_p_modes_path = "POD\\modes\\p\\"
    with open(POD_p_modes_path + "p_mean.pl", 'rb') as f:
        p_mean_POD = pickle.load(f)[contour_indices]
    with open(POD_p_modes_path + "p_modes.pl", 'rb') as f:
        p_modes_POD = pickle.load(f)[contour_indices]

    N_c = len(contour_indices)
    print("Number of points isolated in the contour :", N_c)

    def F_pressure(time_coeffs):
        N_modes, N_t = np.shape(time_coeffs)
        gamma_bar_x, gamma_bar_y = gammas_POD_mean(x_contour, y_contour, p_mean_POD) 
        Fx_tot = gamma_bar_x * np.ones(N_t)
        Fy_tot = gamma_bar_y * np.ones(N_t)
        for i_mode in range(N_modes):
            gamma_i_x, gamma_i_y = gammas_POD_mode_i(i_mode=i_mode,x_contour=x_contour,y_contour=y_contour, p_modes_POD=p_modes_POD )
            Fx_tot += time_coeffs[i_mode] * gamma_i_x
            Fy_tot += time_coeffs[i_mode] * gamma_i_y
        return (Fx_tot, Fy_tot)
    

    def F_viscosity(time_coeffs):
        N_modes, N_t = np.shape(time_coeffs)
        mu_visc_x_bar, mu_visc_y_bar = mu_visc_i_integration(mean_mode, x_contour, y_contour)
        Fx_tot = mu_visc_x_bar * np.ones(N_t)
        Fy_tot = mu_visc_y_bar * np.ones(N_t)

        for i_mode in range(0,N_modes):
            mu_i_x, mu_i_y = mu_visc_i_integration(modes[i], x_contour, y_contour)
            Fx_tot += time_coeffs[i_mode] * mu_i_x
            Fy_tot += time_coeffs[i_mode] * mu_i_y
        return (Fx_tot, Fy_tot)
        

# Alternative way of computing the forces !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Using Least-Square Method to get the force coefficients 
    forces_CFD = pd.read_csv("Data\\Forces\\forces_CFD.csv").to_numpy()
    Fx_cfd, Fy_cfd = np.transpose(forces_CFD[:,1:])

    def force_coefficients(A_matrix, F_vector):
        res_ls = np.linalg.lstsq(A_matrix, F_vector)
        return res_ls[0]
    

    Fx_mean_CFD = np.mean(Fx_cfd)
    Fy_mean_CFD = np.mean(Fy_cfd)

    Fx_osc_CFD = Fx_cfd - Fx_mean_CFD
    Fy_osc_CFD = Fy_cfd - Fy_mean_CFD

    time_steps_forces_CFD = pd.read_csv("Data\\Forces\\timesteps_forces_CFD.csv").to_numpy()[:,1]
    sol_forces = solve_ivp(solve_projected_ODE,t_span=[param.t_min,param.t_max],y0=a0,\
                t_eval=time_steps_forces_CFD, method=solver_method)
    # A of dimension (N_t,N_modes)    
    time_coeffs_matrix = np.transpose(sol_forces.y)
    fx_i_coefficients = force_coefficients(time_coeffs_matrix, Fx_osc_CFD)
    fy_i_coefficients = force_coefficients(time_coeffs_matrix, Fy_osc_CFD)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Fx_POD_ext, Fy_POD_ext = F_pressure(sol_ext.y)
    Fx_visc_ext, Fy_visc_ext = F_viscosity(sol_ext.y)

    Fx_POD_truecoeffs, Fy_POD_truecoeffs = F_pressure(true_time_coeffs)
    Fx_POD_romcoeffs, Fy_POD_romcoeffs = F_pressure(sol.y)

    plt.figure()
    plt.title("Pressure force $F_x(t)$- extrapolated - Monte Carlo integration ")
    plt.plot(t_ext, Fx_POD_ext, c='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Force $F_x$")
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Pressure force $F_y(t)$- extrapolated - Monte Carlo integration ")
    plt.plot(t_ext, Fy_POD_ext, c='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Force $F_y$")
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Pressure force $F_x(t)$- extrapolated - Monte Carlo integration ")
    plt.plot(t_ext[:3000], Fx_POD_ext[:3000], c='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Force $F_x$")
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Pressure force $F_y(t)$- extrapolated - Monte Carlo integration ")
    plt.plot(t_ext[:3000], Fy_POD_ext[:3000], c='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Force $F_y$")
    plt.grid()
    plt.show()

    # One-way coupling =================================================================================================
    # We are going to use the forces computed by the ROM for the fixed cylinder to solve the cylinder motion equation
    intial_conditions_cyl = np.array([0.0,0.0, 0.0,0.0])
    
    # Non dimensional numbers
    Mass_number, Ur, Zeta = param.Mass_number, param.Ur, param.Zeta
    
    # Defining the matrices involved in the final structural equations -------------------------------------------------
    

    # Additional drag and lift forces that are function of the cylinder velocity
    # Drag coefficient for a sphere at Re = 100 (experimental): C_D = 1.5
    # Computed from the mean pressure force
    C_D = np.mean(Fx_POD_romcoeffs)*np.pi*Mass_number/2 * nCond.include_added_damping_force
    print("Computed drag coefficient : {:.2f}".format(C_D))
    
    # Lambda : linear terms ____________________________________________________________________________________________
    Lambda = np.zeros((4,4))
    
    # Line 1
    Lambda[0,1] = 1
    # Line 2
    Lambda[1,0] = - (2*np.pi/Ur)**2 * Mass_number/(1+Mass_number)
    Lambda[1,1] = - (Zeta + 6 / (nCond.Re * Mass_number))* Mass_number/(1+Mass_number)
    
    # Line 3
    Lambda[2,3] = 1
    # Line 4
    Lambda[3,2] = - (2*np.pi/Ur)**2 * Mass_number/(1+Mass_number)
    Lambda[3,3] = - (Zeta + 6 / (nCond.Re * Mass_number))* Mass_number/(1+Mass_number)

    # Non linear terms =================================================================================================
    # Interpolation of the ROM forces 
    Fx_rom = interp1d(t_ext, Fx_POD_ext + param.include_viscosity * Fx_visc_ext)
    Fy_rom = interp1d(t_ext, Fy_POD_ext + param.include_viscosity * Fy_visc_ext)

    # Correction to the added damping forces :
    def F_NL(t, a):
        f = np.zeros(4)
        eta_dot = a[1]
        xi_dot = a[3]
        added_damping_fx = C_D * (-2*eta_dot + eta_dot**2 + xi_dot**2 )*np.cos(np.arctan2(-xi_dot , (1 - eta_dot)))
        added_damping_fy = C_D * (-2*eta_dot + eta_dot**2 + xi_dot**2 )*np.sin(np.arctan2(-xi_dot , (1 - eta_dot))) 

        f[1] =  Fx_rom(t) + (added_damping_fx) *2 / (np.pi*Mass_number)
        f[3] =  Fy_rom(t) + (added_damping_fy)*2 / (np.pi*Mass_number)

        return f

    def ODE_cyl(t,a):
        dadt = np.matmul(Lambda,a) + F_NL(t,a)    
        return dadt

    # Solving the one-way coupled ODE ==================================================================================
    # Solving for the CFD time domain:
    sol = solve_ivp(ODE_cyl,t_span=[param.t_min,param.t_max],y0=intial_conditions_cyl,\
                    t_eval=t_validation, method='RK45')


    eta_eval = sol.y[0]
    xi_eval = sol.y[2] 

    # Solving for the extended time domain:
    sol_disp_ext = solve_ivp(ODE_cyl,t_span=[param.t_min,t_max_ext],y0=intial_conditions_cyl,\
                             t_eval=t_ext, method='RK45')

    eta_ext = sol_disp_ext.y[0]
    xi_ext = sol_disp_ext.y[2]

    eta_dot_ext = sol_disp_ext.y[1]
    xi_dot_ext = sol_disp_ext.y[3]

    # Plotting the results =============================================================================================
    plt.figure()
    plt.title("Trajectory of the cylinder - end of the extrapolation time domain")
    plt.plot(eta_ext[-n_final:],xi_ext[-n_final:])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Displacement of the cylinder in $x$")
    plt.plot(t_ext,eta_ext)
    plt.xlabel("Time [-]")
    plt.ylabel("$\\eta(\\Bar{t})$")
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.title("Displacement of the cylinder in $y$")
    plt.plot(t_ext,xi_ext, c= 'orange')
    plt.xlabel("Time (s)")
    plt.ylabel("$\\xi(t)$")
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Velocities of the cylinder in $x$")
    plt.plot(t_ext,eta_dot_ext, label= '$\dot{\eta}(t)$')
    plt.plot(t_ext,xi_dot_ext, label= '$\dot{\\xi}(t)$')
    plt.xlabel("Time (s)")
    # plt.ylabel("$\\eta(t)$")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Spectral analysis of the results =================================================================================
    freq, fft_eta_ext = fft_sig_abs(t_ext[-n_final:], eta_ext[-n_final:] - np.mean(eta_ext[-n_final:]))
    freq, fft_xi_ext = fft_sig_abs(t_ext[-n_final:], xi_ext[-n_final:] - np.mean(xi_ext[-n_final:]))

    eta_final = eta_ext[-n_final:]
    xi_final = xi_ext[-n_final:]
    t_final = t_ext[-n_final:]

    sampling_rate = 1/(t_final[1]-t_final[0])

    print("Ur :", Ur)
    print("Amplitude eta :", (np.max(eta_final)-np.min(eta_final))/2 )
    print("Amplitude xi :", (np.max(xi_final)-np.min(xi_final))/2 )


    print("Main frequency eta : ", get_main_freq(freq, fft_eta_ext))

    print("Main frequency xi : ", get_main_freq(freq, fft_xi_ext))


    freqs, psd_eta_final =periodogram(eta_final, sampling_rate,scaling='density')
    freqs, psd_xi_final = periodogram(xi_final, sampling_rate,scaling='density')

    # # Subplots gathering all info on displacements ===================================================================
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

# ======================================================================================================================

# ======================================================================================================================
# Covering all Urs 

# Lambda : linear terms ____________________________________________________________________________________________
    if cover_all_ur:
        all_Ur = np.arange(1.0, 16.0, 0.2)
        # all_Ur = [6.0]

        t_start_one_way = time()        
        eta_amplitudes_max = []
        xi_amplitudes_max = []
        for i_ur in tqdm(range(len(all_Ur))):
            Ur_val = all_Ur[i_ur]
            Lambda = np.zeros((4,4))
            
            # Line 1
            Lambda[0,1] = 1
            # Line 2
            Lambda[1,0] = - (2*np.pi/Ur_val)**2* Mass_number/(1+Mass_number)
            Lambda[1,1] = - (Zeta + 6 / (nCond.Re * Mass_number))* Mass_number/(1+Mass_number)
            
            # Line 3
            Lambda[2,3] = 1
            # Line 4
            Lambda[3,2] = - (2*np.pi/Ur_val)**2* Mass_number/(1+Mass_number)
            Lambda[3,3] = - (Zeta + 6 / (nCond.Re * Mass_number))* Mass_number/(1+Mass_number)

            # Non linear terms -----------------------------------------------------------------------------------------    
            def ODE_cyl_Ur_val(t,a):
                dadt = np.matmul(Lambda,a) + F_NL(t,a)    
                return dadt

            sol = solve_ivp(ODE_cyl_Ur_val,t_span=[param.t_min,param.t_max],y0=intial_conditions_cyl,\
                            t_eval=t_validation, method='RK45')
            t_end_one_way = time()
            print("Solving one way ROM with 1 Ur took: {}s".format(t_end_one_way-t_start_one_way ))

            eta_eval = sol.y[0]
            xi_eval = sol.y[2] 

            sol_disp_ext = solve_ivp(ODE_cyl_Ur_val,t_span=[param.t_min,t_max_ext],y0=intial_conditions_cyl,\
                                     t_eval=t_ext, method='RK45')

            eta_ext = sol_disp_ext.y[0]
            xi_ext = sol_disp_ext.y[2]

            eta_dot_ext = sol_disp_ext.y[1]
            xi_dot_ext = sol_disp_ext.y[3]

            eta_late = eta_ext[-n_final//2:]
            xi_late = xi_ext[-n_final//2:]
            t_late = t_ext[-n_final//2:]

            eta_amplitudes_max.append( get_oscillation_amplitude(t_late, eta_late))
            xi_amplitudes_max.append(get_oscillation_amplitude(t_late, xi_late))   

            # Plot one-way trajectories
            one_way_coupling_path ='C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\one_way_coupling\\'
            plt.figure(figsize=(15,15))
            plt.title("Trajectory of the cylinder at the end of extrapolation - $U_r={:.2f}$".format(Ur_val))
            plt.plot(eta_late,xi_late,c='black')
            plt.xlabel("$\\bar{x}$ [-]")
            plt.ylabel("$\\bar{y}$ [-]")
            plt.xlim(np.mean(eta_late)-0.06,np.mean(eta_late)+0.06)
            plt.ylim(np.mean(xi_late)-1,np.mean(xi_late)+1)
            plt.gca().set_aspect(1/5)
            plt.savefig(one_way_coupling_path + "trajectories\\end_traj_Ur_{:.2f}.svg".format(Ur_val),transparent=True)
            plt.close()

            plt.figure(figsize=(15,15))
            plt.title("Trajectory of the cylinder at the end of extrapolation - $U_r={:.2f}$".format(Ur_val))
            plt.plot(eta_late,xi_late,c='black')
            # plt.xlabel("$\\bar{x}$ [-]")
            # plt.ylabel("$\\bar{y}$ [-]")
            plt.xlim(np.mean(eta_late)-0.06,np.mean(eta_late)+0.06)
            plt.ylim(np.mean(xi_late)-1,np.mean(xi_late)+1)
            plt.gca().set_aspect(1/5)
            tpl.save(one_way_coupling_path + "trajectories\\end_traj_Ur_{:.2f}.tex".format(Ur_val))
            plt.close()
            
        
        columns = {"Ur":all_Ur, "eta max":eta_amplitudes_max,"xi max":xi_amplitudes_max}
        amplitude_results = pd.DataFrame(columns)

        amplitude_results.head()

        if save_coeffs:
            amplitude_results.to_csv("saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_amplitudes.csv".format(N_modes, param.Zeta))

        # Comparison with FSI results (extracted from the results of Boudina et al.) 
        FSI_sim_Ur_x = [1.,2.5,3.,3.5,4.,4.5,5.5,6.5,7.,8.,9.,10.,11.,12.]
        FSI_sim_Ur_y = [1.,2.,4.,4.5,5.,6.,7.,8.,9.,10.,11.,12.,13.]
        true_xi_max = [0.006897,0.124138,0.636207,0.632759,0.615517,0.548276,0.448276,\
                       0.301724,0.093103,0.089655,0.082759,0.075862, 0.075862]   
        true_eta_max =  [0.002619,0.004549,0.026328,0.045075,0.046729,0.04218,0.032256,\
                         0.021779,0.010338,0.003308,0.003308,0.003446,0.003033,0.003308]

        # Plot comparison of maximum amplitudes obtained by the 1W ROM with FSI results --------------------------------
        f, ax = plt.subplots(1,2, figsize=(30,15))
        ax[0].plot(all_Ur, xi_amplitudes_max, '-s',color= 'orange', label = "$\\xi_{max}$ - ROM")
        ax[0].plot(FSI_sim_Ur_y, true_xi_max,color='k', marker='o',fillstyle="none", linestyle='--',label = "$\\xi_{max}$ - FSI simulations")
        ax[0].set_title("Amplitude of oscillation in $y$")
        ax[0].set_xlabel("$U_r$")
        ax[0].set_ylabel("$\\xi_{max}$")
        ax[0].legend(loc='upper right', fontsize=12)

        ax[1].plot(all_Ur, eta_amplitudes_max, '-s',color= 'blue', label = "$\eta_{max}$ - ROM")
        ax[1].plot(FSI_sim_Ur_x, true_eta_max,color='k', marker='o',fillstyle="none",linestyle='--', label = "$\eta_{max}$ - FSI simulations")
        ax[1].set_title("Amplitude of oscillation in $x$")
        ax[1].set_xlabel("$U_r$")
        ax[1].set_ylabel("$\eta_{max}$")
        ax[1].legend(loc='upper right', fontsize=12)
        plt.show()

        # Plot POD coeffs and their spectral analysis ==================================================================
        N_ext = nCond.n_ext
        sol_late = sol_ext.y[:,-n_final:]
        t_late = t_ext[-n_final:]
        sampling_rate = 1/(t_late[1]-t_late[0])

        savefig_path = "C:\\Users\\Student\\Documents\\Thesis\\Homemade_figures\\Fixed_ROM_results\\"

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


