import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.chdir("c:\\Users\\Student\\Documents\\Project\\Endgame\\Unified_PODVM_Galerkin")
from parameters import Parameters
import pickle
from scipy.interpolate import griddata
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from tqdm import tqdm
# from numba import jit
from auxiliary_functions import *

param = Parameters()


N_modes = param.N_modes
D = param.D

def inner_product(vec1,vec2):
    return tf.tensordot(vec1, vec2,axes = 1)

def vel_mode_vector(model, x, y):
    """
    Computes a vector that stacks the 2 velocity compenents for the points (x,y) 
    input : NN model, x & y coordinates of the spatial points (2 vectors of shape (N_spatial,))
    output : vector of shape (2 * N_spatial,)  
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec = model(spatial_points)
    u, v = vec[:,0], vec[:,1]
    return tf.cast(tf.concat([u,v],axis=0),tf.float64)

def true_vel_mode_vector(i_mode):
    f = open("POD/ModePINN/mode_data/u_modes.pl", 'rb')
    u_data = pickle.load(f)[:,i_mode]
    f = open("POD/ModePINN/mode_data/v_modes.pl", 'rb')
    v_data = pickle.load(f)[:,i_mode]
    return tf.concat([u_data,v_data],axis=0)


def uv_mode(model, x, y):

    spatial_points = tf.transpose(tf.stack([x,y]))
    vec = model(spatial_points)
    u, v = vec[:,0], vec[:,1]
    return u, v

@tf.function
def Q_velocity(model_1, model_2, x, y):
    """
    Auxiliary function that helps computing the matrices for the linear and non-linear terms in the projected equations.
    input : 2 models to approximate the vectors of the operation, and the spatial points at which we want to compute them 
    output : a 1D tensor of the same shape as the inputs, corresponding to the operation (vec1 . \nabla ) vec2
    
    (vec1 . \nabla ) vec2 = {
        vec1 . \nabla vec2_x ;
        vec1 . \nabla vec2_y
    }
    """
    N_spatial = tf.shape(x)[0]

    spatial_points = tf.transpose(tf.stack([x,y]))
    vec1 = tf.cast(model_1(spatial_points),tf.float64)
    vec2 = tf.cast(model_2(spatial_points),tf.float64)

    vec1_u, vec1_v, vec1_p = vec1[:,0], vec1[:,1], vec1[:,2]
    vec2_u, vec2_v, vec2_p = vec2[:,0], vec2[:,1], vec2[:,2]
    
    vec2_u_dx = tf.gradients(vec2_u, x)[0]
    vec2_u_dy = tf.gradients(vec2_u, y)[0]

    vec2_v_dx = tf.gradients(vec2_v, x)[0]
    vec2_v_dy = tf.gradients(vec2_v, y)[0]
 
    Q_u = tf.math.multiply(vec1_u,vec2_u_dx) + tf.math.multiply(vec1_v,vec2_u_dy)
    Q_v = tf.math.multiply(vec1_u,vec2_v_dx) + tf.math.multiply(vec1_v,vec2_v_dy) 

    return tf.concat([Q_u,Q_v],axis = 0)

# ======================================================================================================================
def all_uv_modes(modes, x, y):
    
    N_spatial = np.shape(x)[0]
    all_modes = np.zeros((N_modes, N_spatial*2))

    for i in range(N_modes):    
        mode_i = vel_mode_vector(modes[i], x, y)
        all_modes[i,:] = all_modes[i,:] + mode_i
    
    return tf.constant(all_modes)

@tf.function
def all_uv_modes_laplacian(modes, x, y):  

    all_modes_laplacian = [[] for i in range(N_modes)]
    for i in range(N_modes):
        mode_i_u, mode_i_v = uv_mode(modes[i], x, y)
        
        mode_i_u_dx = tf.gradients(mode_i_u,x)[0]
        mode_i_u_dy = tf.gradients(mode_i_u,y)[0]

        mode_i_u_dxx = tf.gradients(mode_i_u_dx,x)[0]
        mode_i_u_dyy = tf.gradients(mode_i_u_dy,y)[0]

        mode_i_v_dx = tf.gradients(mode_i_v,x)[0]
        mode_i_v_dy = tf.gradients(mode_i_v,y)[0]

        mode_i_v_dxx = tf.gradients(mode_i_v_dx,x)[0]
        mode_i_v_dyy = tf.gradients(mode_i_v_dy,y)[0]


        mode_i_dxx = tf.concat([mode_i_u_dxx, mode_i_v_dxx], axis=0)
        mode_i_dyy = tf.concat([mode_i_u_dyy, mode_i_v_dyy], axis=0)

        all_modes_laplacian[i].append(mode_i_dxx + mode_i_dyy)

    # Return the tensor version of the previous list
    return tf.squeeze(tf.stack(all_modes_laplacian))

@tf.function
def uv_mode_i_laplacian(modes, x, y,i):  

    all_modes_laplacian = []

    mode_i_u, mode_i_v = uv_mode(modes[i], x, y)
    
    mode_i_u_dx = tf.gradients(mode_i_u,x)[0]
    mode_i_u_dy = tf.gradients(mode_i_u,y)[0]

    mode_i_u_dxx = tf.gradients(mode_i_u_dx,x)[0]
    mode_i_u_dyy = tf.gradients(mode_i_u_dy,y)[0]

    mode_i_v_dx = tf.gradients(mode_i_v,x)[0]
    mode_i_v_dy = tf.gradients(mode_i_v,y)[0]

    mode_i_v_dxx = tf.gradients(mode_i_v_dx,x)[0]
    mode_i_v_dyy = tf.gradients(mode_i_v_dy,y)[0]


    mode_i_dxx = tf.concat([mode_i_u_dxx, mode_i_v_dxx], axis=0)
    mode_i_dyy = tf.concat([mode_i_u_dyy, mode_i_v_dyy], axis=0)

    all_modes_laplacian.append(mode_i_dxx + mode_i_dyy)

    # Return the tensor version of the previous list
    return tf.squeeze(all_modes_laplacian)

    
def all_uv_modes_q_vector_mean(modes, mean_u, x, y):

    all_modes_q_vector = [[] for i in range(N_modes)]
    for i in range(N_modes):                
        mode_i = modes[i]
        q_mode_i = Q_velocity(mean_u,mode_i, x, y) + Q_velocity(mode_i, mean_u, x, y)
        all_modes_q_vector[i].append(q_mode_i)   

    return tf.squeeze(tf.stack(all_modes_q_vector))

# @tf.function
def uv_modes_q_vector_ij(modes_u, x, y, i, j):
    mode_i = modes_u[i]
    mode_j = modes_u[j]
    q_mode_ij = Q_velocity(mode_i, mode_j, x, y)
    return q_mode_ij

def uv_modes_laplacian_inner_prod(modes, x, y, i, j):
    mode_i = modes[i]
    mode_j = vel_mode_vector(modes[j], x, y)
    laplacian_i = uv_mode_i_laplacian(modes,x,y,i)
    return inner_product(mode_j,laplacian_i)

@tf.function
def all_grad_p_modes(modes, x, y):
    all_grad_p_modes = [[] for i in range(N_modes)]

    for i in range(N_modes):                
        spatial_points = tf.transpose(tf.stack([x,y]))
        mode_i = modes[i](spatial_points)
        mode_p_i = mode_i[:,2]
        mode_p_i_dx = tf.gradients(mode_p_i,x)[0]
        mode_p_i_dy = tf.gradients(mode_p_i,y)[0]

        grad_p_mode_i = tf.concat([mode_p_i_dx, mode_p_i_dy], axis=0)
        all_grad_p_modes[i].append(grad_p_mode_i)
    return tf.squeeze(tf.stack(all_grad_p_modes))

@tf.function
def d_mode_dx(mode, x, y):
    spatial_points = tf.transpose(tf.stack([x, y]))

    vec = tf.cast(mode(spatial_points), tf.float64)
    mode_x, mode_y = vec[:,0], vec[:,1]

    mode_x_dx = tf.gradients(mode_x, x)[0]
    mode_y_dx = tf.gradients(mode_y, x)[0]

    return tf.squeeze(tf.concat([mode_x_dx, mode_y_dx],axis=0))

@tf.function
def d_mode_dy(mode, x, y):
    spatial_points = tf.transpose(tf.stack([x, y]))

    vec = tf.cast(mode(spatial_points), tf.float64)
    mode_x, mode_y = vec[:,0], vec[:,1]

    mode_x_dy = tf.gradients(mode_x, y)[0]
    mode_y_dy = tf.gradients(mode_y, y)[0]

    return tf.squeeze(tf.concat([mode_x_dy, mode_y_dy],axis=0)) 


# ======================================================================================================================
def T_matrix(modes, x, y):
    """
    Computes the coefficients for the T matrix in the equation on the time coefficients, which corresponds to the 
    inner product between the mode i and the mode k
    input : the velocity mode vector (shape = (N_modes, 2*N_spatial)) & AM mode vector (same shape)
    output : a matrice of shape (N_modes, N_modes + 2); each line k has the coefficients of the vector L^k which is used 
    in the equation for the k^th time coefficient
    """
    T_matrix = np.zeros((N_modes,N_modes)) 
    velocity_modes = all_uv_modes(modes,x,y)

    # POD coefficients 
    for i in range(N_modes):
        for k in range(N_modes):
            T_matrix[k,i] = T_matrix[k,i] + inner_product(velocity_modes[i,:], velocity_modes[k,:])
        
    return tf.constant(T_matrix,dtype=tf.float64)

def P_matrix(grad_p_modes, velocity_modes):
    P_matrix = np.zeros((N_modes,N_modes))

    for i in range(N_modes):
        for k in range(N_modes):
            P_matrix[k,i] = P_matrix[k,i] + inner_product(grad_p_modes[i,:], velocity_modes[k,:])

    return tf.constant(P_matrix, dtype=tf.float64)

def M_matrix(mean,modes,x,y):
    
    M_matrix = np.zeros((N_modes,N_modes))
    for i in range(N_modes):
        q_vel_i = Q_velocity(mean,modes[i],x,y) +Q_velocity(modes[i],mean,x,y) 
        for k in range(N_modes):
            mode_k = vel_mode_vector(modes[k],x,y)
            M_matrix[k,i] = M_matrix[k,i] + inner_product(q_vel_i,mode_k )
    
    return tf.constant(M_matrix,dtype=tf.float64)

def Q_matrix(modes, x, y):
    Q_matrix = np.zeros((N_modes, N_modes, N_modes))

    for k in range(N_modes):
        mode_u_k = tf.cast(vel_mode_vector(modes[k], x, y),tf.float64)
        for i in range(N_modes):
                for j in range(N_modes):
                        q_ij_vector = uv_modes_q_vector_ij(modes,x, y, i, j)
                        Q_matrix[k, i, j] = Q_matrix[k, i, j] + inner_product(q_ij_vector, mode_u_k)

    return tf.constant(Q_matrix,dtype=tf.float64)

def L_matrix(modes, x, y):

    L_matrix = np.zeros((N_modes, N_modes ))
    
    # POD coefficients
    for i in range(N_modes):
        lap_mode = uv_mode_i_laplacian(modes, x, y, i)
        for k in range(N_modes):
            vel_vect = tf.cast(vel_mode_vector(modes[k],x, y),dtype=tf.float64)
            coeff = inner_product(lap_mode,vel_vect)
            L_matrix[k,i] = L_matrix[k,i] + coeff

    return tf.constant(L_matrix, dtype=tf.float64)


def Gamma_matrix(modes, x, y):

    gamma_matrix = np.zeros((N_modes,N_modes))

    T_mat = T_matrix(modes, x, y)

    gamma_matrix[:N_modes,:] = T_mat

    
    return gamma_matrix

def Lambda_matrix(mean_mode, modes, x, y):
    inv_Re = 1/param.Re
    lambda_matrix = np.zeros((N_modes,N_modes))
    # For P matrix 
    grad_p_modes = all_grad_p_modes(modes, x, y)
    vel_modes = all_uv_modes(modes, x, y)

    L_mat = L_matrix(modes, x, y) 
    M_mat = M_matrix(mean_mode, modes, x, y)
    P_mat = P_matrix(grad_p_modes, vel_modes)

    lambda_matrix[:N_modes,:] = inv_Re * L_mat - M_mat - P_mat
    
    return lambda_matrix


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

if __name__ == "__main__":
    
    f = open("POD/modes/points.pl",'rb')
    close_points = pickle.load(f)
    x_sp, y_sp = tf.transpose(close_points)
    
    nCond = Parameters()
    param = Parameters()

    mode_path = "POD/ModePINN/saved/"

    mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")

    # Initializing the modes list, containing the models for all modes except the mean mode --------------------------------
    modes = []
    # N_modes counts the POD modes and the mean mode
    for i in range(1, N_modes+1):
        modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))
    f = open("POD/modes/points.pl",'rb')
    close_points = pickle.load(f)
    x_sp, y_sp = tf.transpose(close_points)



    Nx,Ny = 500,500 
    grid_x, grid_y = grid_x, grid_y = np.meshgrid(np.linspace(param.Lxmin, param.Lxmax, Nx), np.linspace(param.Lymin, param.Lymax, Ny))

    plt.figure(figsize=(20,20))
    plt.scatter(x_sp, y_sp,c= mean_mode(close_points)[:,0])
    plt.gca().set_aspect('equal')
    plt.show()    

    # ==========================================
    # Testing to generate the matrices 
    #     # print("T")
    # print(T_matrix(modes, x_sp,y_sp))
    # print("M")
    # print(M_matrix(modes, x_sp,y_sp))
    # print("L")
    # print(L_matrix(modes, x_sp,y_sp))
    # print("P")
    # print(P_matrix(modes, x_sp,y_sp))
    # print("Q")
    # print(Q_matrix(modes, x_sp,y_sp))
    # print("Chi")
    # print(Chi_matrix(modes, x_sp,y_sp))
    # print("Beta")
    # print(Beta_matrix(modes, x_sp,y_sp))
    
    # print("Gamma")
    # print(Gamma_matrix(modes, x_sp,y_sp))
    # print("Lambda")
    # print(Lambda_matrix(modes, x_sp,y_sp))
     

    # t_pts = np.linspace(0.0,20.0,100)
    # eta_dot = tf.sin(t_pts * 2 * np.pi /8.0)
    # eta_dot = tf.reshape(eta_dot, (len(t_pts),1)) 