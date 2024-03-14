import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
from text_flow import read_flow
from parameters import Parameters
from auxiliary_functions import *
import pandas as pd
from scipy.interpolate import griddata
from pathlib import Path
import os
import cv2

# Plot parameters =======================
# plt.rcParams['font.size'] = 18

# Set the axes title font size
plt.rc('axes', titlesize=16)
# Set the axes labels font size
plt.rc('axes', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=18)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.figsize'] = (25, 12)
# Importing the NN modes =============================================================================================
param = Parameters()
nCond = Parameters()

N_modes = param.N_modes
N_t = 201
# mode_max = 6
t_validation = tf.linspace(param.t_min, param.t_max, N_t)
t_max_ext = 800.0
t_ext = np.linspace(param.t_min,t_max_ext,5000)


# saved_path = "saved\{}_modes\\zeta1\\Ur_{:.1f}\\".format(param.N_modes,param.Ur)
# figures_path = "figures\{}_modes\\zeta1\\Ur_{:.1f}\\".format(param.N_modes,param.Ur)
# all_results_processed = "saved\\{}_modes\\zeta1\\all_results.csv".format(param.N_modes)

# saved_path = "saved\{}_modes\\zeta_{:.2f}\\Ur_{:.2f}\\".format(param.N_modes,param.Zeta,param.Ur)
# figures_path = "figures\{}_modes\\zeta_{:.2f}\\Ur_{:.2f}\\".format(param.N_modes,param.Zeta,param.Ur)
saved_path = param.saved_path


saved_path_dir = Path(saved_path)


f = open("POD/modes/true_time_coeffs.pl",'rb')
true_time_coeffs = pickle.load(f)

mode_path = "POD/ModePINN/saved/"

mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")
# Initializing the modes list, containing the models for all modes except the mean mode --------------------------------
modes = []
# N_modes counts the POD modes and the mean mode
for i in range(1, N_modes + 1):
    modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))

time_coeffs = pd.read_csv(saved_path +"time_coeffs_ext_{}_modes.csv".format(N_modes), header=None).to_numpy()






# Extraction of the data ===============================================================================================
filename_data = '..\\..\\Data\\fixed_cylinder_atRe100' 

file_data = open(filename_data, 'r')

data = read_flow(filename_data)

Re, Ur = data[0], data[1]
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
close_points = []
close_points_id = []

for i in range(N_points):
    if is_in_domain(points[i], geom=geom):
        close_points.append(points[i])
        close_points_id.append(i)
close_points = np.array(close_points)
close_points_id = np.array(close_points_id)
N_points = len(close_points)
xp = close_points[:,0]
yp = close_points[:,1]

u_data = u_mes[:,close_points_id]
v_data = v_mes[:,close_points_id]
p_data = p_mes[:,close_points_id]



# Reconstruct the flow fields with the ODE time coefficients -----------------------------------------------------------
def reshape_tf(x):
    return tf.cast(tf.reshape(x,(1,len(x))),tf.float32)

def reshape_np(x):
    return np.reshape(x,(1,len(x)))

def reconstruct_all_fields(N_modes, time_coeffs):
    fields = tf.cast(mean_mode(close_points),tf.float32)
    u,v,p = tf.transpose(fields) 
    N_t = np.shape(time_coeffs)[1]

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

# Compute the errors ---------------------------------------------------------------------------------------------------
# mode_ids = np.arange(2,N_modes+1)
# u_error = []
# v_error = []
# p_error = []
# total_error = []

# for n_modes in range(2,N_modes+1):
#     print(n_modes)
#     # f = open("ODE_solver_results/time_coeffs_{}_modes.pl".format(n_modes),'rb')
#     # time_coeffs_ODE = pickle.load(f)
#     # f.close()
#     # time_coeffs_ODE = pd.read_csv("ODE_solver_results/time_coeffs_{}_modes.csv".format(n_modes),usecols=np.arange(1,202)).to_numpy()

#     u_n, v_n, p_n = reconstruct_all_fields(n_modes, time_coeffs)
#     # Error in u
#     u_error.append(tf.reduce_mean(tf.square(u_n - u_data)).numpy())
#     # Error in v
#     v_error.append(tf.reduce_mean(tf.square(v_n - v_data)).numpy())
#     # Error in p
#     p_error.append(tf.reduce_mean(tf.square(p_n - p_data)).numpy())
#     total_error.append(u_error[-1] +v_error[-1] + p_error[-1] )
# Plots ----------------------------------------------------------------------------------------------------------------


# plt.figure()
# plt.plot(mode_ids,u_error,'-+', label = 'Error in u')
# plt.plot(mode_ids,v_error,'-+', label = 'Error in v')
# plt.plot(mode_ids,p_error,'-+', label = 'Error in p')
# plt.plot(mode_ids,total_error,'-+', label = 'Total error')

# plt.grid()
# plt.legend()
# plt.xlabel("# of modes used for the POD")
# plt.ylabel("Errors")
# plt.show()

# Plot the reconstructed fields =======================================================
Nx,Ny = 200,200 
grid_x, grid_y = grid_x, grid_y = np.meshgrid(np.linspace(param.Lxmin, param.Lxmax, Nx), np.linspace(param.Lymin, param.Lymax, Ny))

utest,vtest,ptest = reconstruct_all_fields(param.N_modes,time_coeffs)

u_grid = griddata(close_points,utest[-1],(grid_x,grid_y))

f,ax = plt.subplots(1,1,figsize=(25,15))
plt.imshow(u_grid, extent=(Lxmin, Lxmax, Lymin, Lymax))
cyl = plt.Circle((x_c, y_c), r_c, color='white')
ax.set_aspect(1)
ax.add_artist(cyl)
plt.show()


ODE_frequencies = []

# FFT of the time coeffs of the ODE solver
t_solve = np.linspace(param.t_min,param.t_max,201)
f,ax = plt.subplots(1,1,figsize=(25,15))
plt.title("FFT of the time coefficients - ODE solver")
for i in range(N_modes):
    time_coeffs_i = time_coeffs[i,:]
    xf, yf = fft_sig_abs(t_solve,time_coeffs_i)
    yf_plot =2.0/len(xf) * np.abs(yf[0:len(t_solve)//2])
    freq_i = xf[np.argmax(yf_plot)]
    ODE_frequencies.append(freq_i)
    plt.plot(xf, yf_plot, label="$\mathcal{{F}}(a_{{{mode}}})$".format(mode=i+1))
plt.grid()
plt.xlim(0.0,1.0)
plt.legend()
plt.show()

    
true_frequencies = []
# FFT of the true time coeffs
t_solve = np.linspace(param.t_min,param.t_max,201)
f,ax = plt.subplots(1,1,figsize=(25,15))
plt.title("FFT of the true time coefficients")
for i in range(N_modes):
    time_coeffs_i = true_time_coeffs[i,:]
    xf, yf = fft_sig_abs(t_solve,time_coeffs_i)
    yf_plot =2.0/len(xf) * np.abs(yf[0:len(t_solve)//2])
    # Get the frequencies of each time coeff
    freq_i = xf[np.argmax(yf_plot)]
    true_frequencies.append(freq_i)
    plt.plot(xf, yf_plot, label="$\mathcal{{F}}(a_{{{mode}}})$".format(mode=i+1))
plt.grid()
plt.xlim(0.0,1.0)
plt.legend()
plt.show()

true_frequencies = np.array(true_frequencies)
ODE_frequencies = np.array(ODE_frequencies)

print("True frequencies :",true_frequencies)
print("ODE frequencies :",ODE_frequencies)

MSE_freqs = np.square(true_frequencies - ODE_frequencies)

# Make a video of the flow fields ===================================================

def save_field_grid(field, filename,t_late,field_name ='u'):
    f,ax = plt.subplots(1,1,figsize=(25,15))
    plt.imshow(field, extent=(Lxmin, Lxmax, Lymin, Lymax))
    cyl = plt.Circle((x_c, y_c), r_c, color='white')
    plt.title("$"+field_name+"(x,y,t={:.2f})$".format(t_late))
    ax.set_aspect(1)
    ax.add_artist(cyl)
    plt.colorbar()

    if field_name =='u':
        plt.clim(np.min(u_late),np.max(u_late))
    if field_name =='v':
        plt.clim(np.min(v_late),np.max(v_late))
    if field_name =='p':
        plt.clim(np.min(p_late),np.max(p_late))

    plt.savefig(filename, transparent=True)
    plt.close()
    


n_frames_late = 200
time_coeffs_late = time_coeffs[:,-n_frames_late:] 
u_late,v_late,p_late =reconstruct_all_fields(N_modes=8,time_coeffs=time_coeffs_late)
t_ext_late = t_ext[-n_frames_late:]

frame_images_u =[]

for i_frame in tqdm(range(n_frames_late)):
    u_grid_i_late = griddata(close_points,u_late[i_frame],(grid_x,grid_y))
    v_grid_i_late = griddata(close_points,v_late[i_frame],(grid_x,grid_y))
    p_grid_i_late = griddata(close_points,p_late[i_frame],(grid_x,grid_y))

    # Save u
    filename_u = param.saved_path + "videos\\frames_u\\"
    Path(filename_u).mkdir(exist_ok=True,parents=True)
    save_field_grid(u_grid_i_late, filename_u + "u_{}.png".format(i_frame),t_ext_late[i_frame],field_name='u')
   
    # # Save v
    filename_v = param.saved_path + "videos\\frames_v\\"
    Path(filename_v).mkdir(exist_ok=True,parents=True)
    save_field_grid(v_grid_i_late, filename_v + "v_{}.png".format(i_frame),t_ext_late[i_frame],field_name='v')
    # # Save p
    filename_p = param.saved_path + "videos\\frames_p\\"
    Path(filename_p).mkdir(exist_ok=True,parents=True)
    save_field_grid(p_grid_i_late, filename_p + "p_{}.png".format(i_frame),t_ext_late[i_frame],field_name='p')
   
