import numpy as np
import tensorflow as tf

import pickle
import scipy.integrate as integrate
from scipy.integrate import cumtrapz
from stokes_modes import *
import os

os.chdir("c:\\Users\\Student\\Documents\\Project\\Endgame\\Unified_PODVM_Galerkin")

from auxiliary_functions import *
from parameters import Parameters

param = Parameters()
nCond = Parameters()

N_modes = param.N_modes
D = param.D
mode_path = "POD/ModePINN/saved/"

mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")

# Initializing the modes list, containing the models for all modes except the mean mode --------------------------------
modes = []
# N_modes counts the POD modes and the mean mode
for i in range(1, N_modes+1):
    modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))

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
# Projection matrices with stokes modes 

# ----------------------------------------------------------------------------------------------------------------------
# Functions to compute the matrices with the stokes modes

def cartesian_to_polar(x, y):
    r = tf.math.sqrt(x**2 + y**2)
    theta = tf.math.atan2(y, x)
    return (r, theta)

def polar_to_cartesian(r,theta):
    x = r* tf.cos(theta)
    y = r* tf.sin(theta)
    return (x,y)

# Stokes modes derivatives ---------------------------------------------------------------------------------------------
# In line flow derivatives 
@tf.function
def u_stokes_eta_x_grad_x(x,y):
    u_eta_x, _ = u_stokes_eta(x,y)
    return tf.gradients(u_eta_x, x)[0]

@tf.function
def u_stokes_eta_x_grad_y(x,y):
    u_eta_x, _ = u_stokes_eta(x,y)
    return tf.gradients(u_eta_x, y)[0] 

@tf.function
def u_stokes_eta_y_grad_x(x,y):
    _, u_eta_y = u_stokes_eta(x,y)
    return tf.gradients(u_eta_y, x)[0]

@tf.function
def u_stokes_eta_y_grad_y(x,y):
    _, u_eta_y = u_stokes_eta(x,y)
    return tf.gradients(u_eta_y, y)[0]

# Cross flow derivatives -----------------------------------------------------------------------------------------------
@tf.function
def u_stokes_xi_x_grad_x(x,y):
    u_xi_x, _ = u_stokes_xi(x,y)
    return tf.gradients(u_xi_x, x)[0]

@tf.function
def u_stokes_xi_x_grad_y(x,y):
    u_xi_x, _ = u_stokes_xi(x,y)
    return tf.gradients(u_xi_x, y)[0] 

@tf.function
def u_stokes_xi_y_grad_x(x,y):
    _, u_xi_y = u_stokes_xi(x,y)
    return tf.gradients(u_xi_y, x)[0]

@tf.function
def u_stokes_xi_y_grad_y(x,y):
    _, u_xi_y = u_stokes_xi(x,y)
    return tf.gradients(u_xi_y, y)[0]

# ======================================================================================================================


# def phi_AM_vx_mode(x, y):
#     phi_AM_x, phi_AM_y = phi_AM_v1(x, y)
#     return tf.concat([phi_AM_x, phi_AM_y], axis=0)


# Old added mass modes functions
# @tf.function
# def phi_AM_v1_x_grad_x(x, y):
#     phi_AM_x, _ = phi_AM_v1(x, y)
#     phi_AM_x_dx = tf.gradients(phi_AM_x, x)[0]
#     return phi_AM_x_dx

# @tf.function
# def phi_AM_v1_x_grad_y(x, y):
#     phi_AM_x, _ = phi_AM_v1(x, y)
#     phi_AM_x_dy = tf.gradients(phi_AM_x, y)[0]
#     return phi_AM_x_dy

# @tf.function
# def phi_AM_v1_y_grad_x(x, y):
#     _, phi_AM_y = phi_AM_v1(x, y)
#     phi_AM_y_dx = tf.gradients(phi_AM_y, x)[0]
#     return phi_AM_y_dx

# @tf.function
# def phi_AM_v1_y_grad_y(x, y):
#     _, phi_AM_y = phi_AM_v1(x, y)
#     phi_AM_y_dy = tf.gradients(phi_AM_y, y)[0]
#     return phi_AM_y_dy


# def phi_AM_v2(x, y):
#     """
#     For (x,y), returns the velocity added mass mode (two components, u and v) for the 2nd AM mode (displacement in y )
#     The mode will then be multiplied by the velocity of the cylinder xi
#     """
#     r, theta = cartesian_to_polar(x, y)

#     phi_ux = 1/(4*r**2) * (tf.sin(2*theta))
#     phi_uy = - 1/(4*r**2) * (tf.cos(2*theta))

#     return phi_ux, phi_uy

# def phi_AM_vy_mode(x, y):
#     phi_AM_x, phi_AM_y = phi_AM_v2(x, y)
#     return tf.concat([phi_AM_x, phi_AM_y], axis=0)

# @tf.function
# def phi_AM_v2_x_grad_x(x, y):
#     phi_AM_x, _ = phi_AM_v2(x, y)
#     phi_AM_x_dx = tf.gradients(phi_AM_x, x)[0]
#     return phi_AM_x_dx

# @tf.function
# def phi_AM_v2_x_grad_y(x, y):
#     phi_AM_x, _ = phi_AM_v2(x, y)
#     phi_AM_x_dy = tf.gradients(phi_AM_x, y)[0]
#     return phi_AM_x_dy

# @tf.function
# def phi_AM_v2_y_grad_x(x, y):
#     _, phi_AM_y = phi_AM_v2(x, y)
#     phi_AM_y_dx = tf.gradients(phi_AM_y, x)[0]
#     return phi_AM_y_dx

# @tf.function
# def phi_AM_v2_y_grad_y(x, y):
#     _, phi_AM_y = phi_AM_v2(x, y)
#     phi_AM_y_dy = tf.gradients(phi_AM_y, y)[0]
#     return phi_AM_y_dy

# ======================================================================================================================




@tf.function
def Q_velocity_mean_stokes_eta(mean, x, y):
    """ 
    Returns Q(mean, phi_AM) + Q(phi_AM, mean)
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mean = tf.cast(mean(spatial_points),tf.float64)

    vec_mean_u, vec_mean_v = vec_mean[:,0], vec_mean[:,1]
    
    vec_mean_u_dx = tf.gradients(vec_mean_u, x)[0]
    vec_mean_u_dy = tf.gradients(vec_mean_u, y)[0]

    vec_mean_v_dx = tf.gradients(vec_mean_v, x)[0]
    vec_mean_v_dy = tf.gradients(vec_mean_v, y)[0]

    phi_stokes_x, phi_stokes_y = u_stokes_eta(x,y)
    phi_stokes_x_dx, phi_stokes_y_dx = u_stokes_eta_x_grad_x(x, y), u_stokes_eta_y_grad_x(x, y)
    phi_stokes_x_dy, phi_stokes_y_dy = u_stokes_eta_x_grad_y(x,y), u_stokes_eta_y_grad_y(x,y) 
    
    Q_u = tf.math.multiply(vec_mean_u, phi_stokes_x_dx) + tf.math.multiply(vec_mean_v, phi_stokes_x_dy) +\
        tf.math.multiply(phi_stokes_x, vec_mean_u_dx) + tf.math.multiply(phi_stokes_y, vec_mean_u_dy)
    Q_v = tf.math.multiply(vec_mean_u, phi_stokes_y_dx) + tf.math.multiply(vec_mean_v, phi_stokes_y_dy)+\
        tf.math.multiply(phi_stokes_x, vec_mean_v_dx) + tf.math.multiply(phi_stokes_y, vec_mean_v_dy) 

    return tf.concat([Q_u,Q_v],axis = 0)

@tf.function
def Q_velocity_mean_stokes_xi(mean, x, y):
    """ 
    Returns Q(mean, phi_AM) + Q(phi_AM, mean)
    """

    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mean = tf.cast(mean(spatial_points),tf.float64)

    vec_mean_u, vec_mean_v = vec_mean[:,0], vec_mean[:,1]
    
    vec_mean_u_dx = tf.gradients(vec_mean_u, x)[0]
    vec_mean_u_dy = tf.gradients(vec_mean_u, y)[0]

    vec_mean_v_dx = tf.gradients(vec_mean_v, x)[0]
    vec_mean_v_dy = tf.gradients(vec_mean_v, y)[0]

    phi_stokes_x, phi_stokes_y = u_stokes_xi(x,y)
    phi_stokes_x_dx, phi_stokes_y_dx = u_stokes_xi_x_grad_x(x, y), u_stokes_xi_y_grad_x(x, y)
    phi_stokes_x_dy, phi_stokes_y_dy = u_stokes_xi_x_grad_y(x,y), u_stokes_xi_y_grad_y(x,y) 
    
    Q_u = tf.math.multiply(vec_mean_u, phi_stokes_x_dx) + tf.math.multiply(vec_mean_v, phi_stokes_x_dy) +\
        tf.math.multiply(phi_stokes_x, vec_mean_u_dx) + tf.math.multiply(phi_stokes_y, vec_mean_u_dy)
    Q_v = tf.math.multiply(vec_mean_u, phi_stokes_y_dx) + tf.math.multiply(vec_mean_v, phi_stokes_y_dy)+\
        tf.math.multiply(phi_stokes_x, vec_mean_v_dx) + tf.math.multiply(phi_stokes_y, vec_mean_v_dy) 
    return tf.concat([Q_u,Q_v],axis = 0)


def phi_i_nabla_phi_stokes_eta(modes, i, x, y):
    """ 
    Returns (phi_i . nabla )phi_AM
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mode = tf.cast(modes[i](spatial_points),tf.float64)

    vec_u, vec_v = vec_mode[:,0], vec_mode[:,1]

    u_stokes_eta_x_dx, u_stokes_eta_y_dx = u_stokes_eta_x_grad_x(x, y), u_stokes_eta_y_grad_x(x, y)
    u_stokes_eta_x_dy, u_stokes_eta_y_dy = u_stokes_eta_x_grad_y(x,y), u_stokes_eta_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(vec_u, u_stokes_eta_x_dx) + tf.math.multiply(vec_v, u_stokes_eta_x_dy)
    Q_v = tf.math.multiply(vec_u, u_stokes_eta_y_dx) + tf.math.multiply(vec_v, u_stokes_eta_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

@tf.function
def phi_stokes_eta_nabla_phi_i(modes, i, x, y):
    """ 
    Returns (phi_i . nabla )phi_AM
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mode = tf.cast(modes[i](spatial_points),tf.float64)

    vec_u, vec_v = vec_mode[:,0], vec_mode[:,1]
    
    vec_u_dx = tf.gradients(vec_u, x)[0]
    vec_u_dy = tf.gradients(vec_u, y)[0]

    vec_v_dx = tf.gradients(vec_v, x)[0]
    vec_v_dy = tf.gradients(vec_v, y)[0]

    phi_stokes_eta_x, phi_stokes_eta_y = u_stokes_eta(x,y)
    
    Q_u = tf.math.multiply(phi_stokes_eta_x, vec_u_dx ) + tf.math.multiply(phi_stokes_eta_y, vec_u_dy)
    Q_v = tf.math.multiply(phi_stokes_eta_x, vec_v_dx) + tf.math.multiply(phi_stokes_eta_y, vec_v_dy)
    
    return tf.concat([Q_u,Q_v],axis = 0)

def phi_i_nabla_phi_stokes_xi(modes, i, x, y):
    """ 
    Returns (phi_i . nabla )phi_AM
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mode = tf.cast(modes[i](spatial_points),tf.float64)

    vec_u, vec_v = vec_mode[:,0], vec_mode[:,1]

    u_stokes_xi_x_dx, u_stokes_xi_y_dx = u_stokes_xi_x_grad_x(x, y), u_stokes_xi_y_grad_x(x, y)
    u_stokes_xi_x_dy, u_stokes_xi_y_dy = u_stokes_xi_x_grad_y(x,y), u_stokes_xi_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(vec_u, u_stokes_xi_x_dx) + tf.math.multiply(vec_v, u_stokes_xi_x_dy)
    Q_v = tf.math.multiply(vec_u, u_stokes_xi_y_dx) + tf.math.multiply(vec_v, u_stokes_xi_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

@tf.function
def phi_stokes_xi_nabla_phi_i(modes, i, x, y):
    """ 
    Returns (phi_i . nabla )phi_AM
    """
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec_mode = tf.cast(modes[i](spatial_points),tf.float64)

    vec_u, vec_v = vec_mode[:,0], vec_mode[:,1]
    
    vec_u_dx = tf.gradients(vec_u, x)[0]
    vec_u_dy = tf.gradients(vec_u, y)[0]

    vec_v_dx = tf.gradients(vec_v, x)[0]
    vec_v_dy = tf.gradients(vec_v, y)[0]

    phi_stokes_xi_x, phi_stokes_xi_y = u_stokes_xi(x,y)
    
    Q_u = tf.math.multiply(phi_stokes_xi_x, vec_u_dx ) + tf.math.multiply(phi_stokes_xi_y, vec_u_dy)
    Q_v = tf.math.multiply(phi_stokes_xi_x, vec_v_dx) + tf.math.multiply(phi_stokes_xi_y, vec_v_dy)
    
    return tf.concat([Q_u,Q_v],axis = 0)


def phi_stokes_eta_nabla_phi_stokes_eta(x, y):
    """ 
    Returns (phi_i . nabla )phi_AM
    """
    phi_stokes_eta_x, phi_stokes_eta_y = u_stokes_eta(x,y)
    u_stokes_eta_x_dx, u_stokes_eta_y_dx = u_stokes_eta_x_grad_x(x, y), u_stokes_eta_y_grad_x(x, y)
    u_stokes_eta_x_dy, u_stokes_eta_y_dy = u_stokes_eta_x_grad_y(x,y), u_stokes_eta_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(phi_stokes_eta_x, u_stokes_eta_x_dx) + tf.math.multiply(phi_stokes_eta_y, u_stokes_eta_x_dy)
    Q_v = tf.math.multiply(phi_stokes_eta_x, u_stokes_eta_y_dx) + tf.math.multiply(phi_stokes_eta_y, u_stokes_eta_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

def phi_stokes_eta_nabla_phi_stokes_xi(x, y):
    """ 
    Returns (phi_AM1 . nabla )phi_AM2
    """
    phi_stokes_eta_x, phi_stokes_eta_y = u_stokes_eta(x,y)
    
    u_stokes_xi_x_dx, u_stokes_xi_y_dx = u_stokes_xi_x_grad_x(x, y), u_stokes_xi_y_grad_x(x, y)
    u_stokes_xi_x_dy, u_stokes_xi_y_dy = u_stokes_xi_x_grad_y(x,y), u_stokes_xi_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(phi_stokes_eta_x, u_stokes_xi_x_dx) + tf.math.multiply(phi_stokes_eta_y, u_stokes_xi_x_dy)
    Q_v = tf.math.multiply(phi_stokes_eta_x, u_stokes_xi_y_dx) + tf.math.multiply(phi_stokes_eta_y, u_stokes_xi_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

def phi_stokes_xi_nabla_phi_stokes_eta(x, y):
    """ 
    Returns (phi_AM2 . nabla )phi_AM1
    """
    phi_stokes_xi_x, phi_stokes_xi_y = u_stokes_xi(x,y)
    
    u_stokes_eta_x_dx, u_stokes_eta_y_dx = u_stokes_eta_x_grad_x(x, y), u_stokes_eta_y_grad_x(x, y)
    u_stokes_eta_x_dy, u_stokes_eta_y_dy = u_stokes_eta_x_grad_y(x,y), u_stokes_eta_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(phi_stokes_xi_x, u_stokes_eta_x_dx) + tf.math.multiply(phi_stokes_xi_y, u_stokes_eta_x_dy)
    Q_v = tf.math.multiply(phi_stokes_xi_x, u_stokes_eta_y_dx) + tf.math.multiply(phi_stokes_xi_y, u_stokes_eta_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

def phi_stokes_xi_nabla_phi_stokes_xi(x, y):
    """ 
    Returns (phi_AM2 . nabla )phi_AM1
    """
    phi_stokes_xi_x, phi_stokes_xi_y = u_stokes_xi(x,y)
    u_stokes_xi_x_dx, u_stokes_xi_y_dx = u_stokes_xi_x_grad_x(x, y), u_stokes_xi_y_grad_x(x, y)
    u_stokes_xi_x_dy, u_stokes_xi_y_dy = u_stokes_xi_x_grad_y(x,y), u_stokes_xi_y_grad_y(x,y) 

    
    Q_u = tf.math.multiply(phi_stokes_xi_x, u_stokes_xi_x_dx) + tf.math.multiply(phi_stokes_xi_y, u_stokes_xi_x_dy)
    Q_v = tf.math.multiply(phi_stokes_xi_x, u_stokes_xi_y_dx) + tf.math.multiply(phi_stokes_xi_y, u_stokes_xi_y_dy)

    return tf.concat([Q_u,Q_v],axis = 0)

# ======================================================================================================================
# TO MODIFY : Stokes pressure modes

@tf.function
def p_stokes_eta_grad_x(x,y):
    p_eta = p_stokes_eta(x, y)
    return tf.gradients(p_eta, x)[0]

@tf.function
def p_stokes_eta_grad_y(x,y):
    p_eta = p_stokes_eta(x, y)
    return tf.gradients(p_eta, y)[0]

@tf.function
def p_stokes_xi_grad_x(x,y):
    p_xi = p_stokes_xi(x, y)
    return tf.gradients(p_xi, x)[0]

@tf.function
def p_stokes_xi_grad_y(x,y):
    p_xi = p_stokes_xi(x, y)
    return tf.gradients(p_xi, y)[0]

# Defining the functions for integration over the cylinder surface -----------------------------------------------------
def n_vector(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def added_mass_value():
    return np.pi / 2


# **********************************************************************************************************************
# This part can be erased
def p_mode_cylinder(r, theta, p_mode):
    xc, yc = polar_to_cartesian(r, theta)
    point = tf.transpose(tf.stack([[xc, yc]]))
    return p_mode(point)[:,2]

def pressure_mode_integration_values_x(theta, p_mode):
    pressure_values = p_mode_cylinder(param.r_c, theta, p_mode).numpy()
    return pressure_values * tf.cos(theta)
    
def pressure_mode_integration_values_y(theta, p_mode):
    pressure_values = p_mode_cylinder(param.r_c, theta, p_mode).numpy()
    return pressure_values * tf.sin(theta)
    
def p_mode_cylinder_integration_x(p_mode):    
    thetas_integration = np.linspace(0.0, 2*np.pi, param.N_cyl_integration)
    f_int_values = pressure_mode_integration_values_x(thetas_integration, p_mode)
    f_pressure_total = cumtrapz(f_int_values, thetas_integration)[-1]
    return f_pressure_total

def p_mode_cylinder_integration_y(p_mode):    
    thetas_integration = np.linspace(0.0, 2*np.pi, param.N_cyl_integration)
    f_int_values = pressure_mode_integration_values_y(thetas_integration, p_mode)
    f_pressure_total = cumtrapz(f_int_values, thetas_integration)[-1]
    return f_pressure_total

# Old integration method, slower than with cumtrapz
def p_mode_cylinder_slow(r, theta, p_mode):
    xc, yc = polar_to_cartesian(r, theta)
    point = tf.stack([[xc, yc]])
    return p_mode(point)[:,2]

def p_mode_cylinder_integration_x_slow(p_mode):
    
    def f_int(theta):
        return p_mode_cylinder_slow(param.r_c, theta, p_mode).numpy()[0] * tf.cos(theta)
        
    cylinder_integral, error = integrate.quad(lambda theta: f_int(theta), 0.0, 2*np.pi)
    return cylinder_integral, error

def p_mode_cylinder_integration_y_slow(p_mode):
    
    def f_int(theta):
        return p_mode_cylinder_slow(param.r_c, theta, p_mode).numpy()[0] * tf.sin(theta)
    cylinder_integral, error = integrate.quad(lambda theta: f_int(theta), 0.0, 2*np.pi)
    return cylinder_integral, error



def gamma_bar_x(mean_mode):
    return p_mode_cylinder_integration_x(p_mode=mean_mode)

def gamma_POD_i_x(modes, i_mode):
    return p_mode_cylinder_integration_x(p_mode=modes[i_mode])

def gamma_bar_y(mean_mode):
    return p_mode_cylinder_integration_y(p_mode=mean_mode)

def gamma_POD_i_y(modes, i_mode):
    return p_mode_cylinder_integration_y(p_mode=modes[i_mode])

def gamma_POD_vector_x(modes):
    gamma_POD = np.zeros((N_modes +4))
    for i in range(N_modes):
        gamma_POD[i] = gamma_POD_i_x(modes, i)
    return gamma_POD

def gamma_POD_vector_y(modes):
    gamma_POD = np.zeros((N_modes +4))
    for i in range(N_modes):
        gamma_POD[i] = gamma_POD_i_y(modes, i)
    return gamma_POD

# Defining functions to compute the terms of the viscosity force integrated on the cylinder surface --------------------
@tf.function
def gradient_tensor_sum_POD_modes(mode, x, y):
    """
    Computes (\nabla u + (\nabla u)^T) for a POD mode
    input: NN model of one of the POD modes (including the mean mode), x & y spatial coordinates
    output: 2D tensor 
    """
    # x, y = polar_to_cartesian(r, theta)
    spatial_points = tf.transpose(tf.stack([x,y]))
    vec = tf.cast(mode(spatial_points), tf.float64)
    mode_x, mode_y = vec[:,0], vec[:,1]
    mode_x_dx = tf.gradients(mode_x, x)[0]
    mode_x_dy = tf.gradients(mode_x, y)[0]
    mode_y_dx = tf.gradients(mode_y, x)[0]
    mode_y_dy = tf.gradients(mode_y, y)[0]

    tensor = tf.stack([[2*mode_x_dx, mode_x_dy + mode_y_dx],
                          [mode_x_dy + mode_y_dx, 2 * mode_y_dy]])
    return tf.squeeze(tensor) 


def mu_visc_i_integration(mode, x, y):
    """
    Computes the integration of local viscosity forces over the points in the contour using the Monte Carlo method
    (same points as in the pressure force integration).  
    input: mode NN model, selected 2D points in the cylinder surface 
    output: Viscosity force vector for the i^th mode
    """

    invRe = 1/param.Re
    tensor = gradient_tensor_sum_POD_modes(mode, x, y)
    r, theta = cartesian_to_polar(x, y)
    n_vectors = n_vector(theta)
    sum = 0
    for i in range(len(x)):
        sum += np.matmul(tensor[:,:,i], n_vectors[:,i])
    return invRe * sum /(2*np.pi)

# This part can be erased
# **********************************************************************************************************************

# ======================================================================================================================
# Matrices with structural damping and 2 DOFs for the cylinder displacement
# ======================================================================================================================
# Every matrix function will have a 2DOF suffix at the end
def T_matrix(modes, x, y):
    """
    Computes the coefficients for the T matrix in the equation on the time coefficients, which corresponds to the 
    inner product between the mode i and the mode k
    input : the velocity mode vector (shape = (N_modes, 2*N_spatial)) & AM mode vector (same shape)
    output : a matrice of shape (N_modes, N_modes + 2); each line k has the coefficients of the vector L^k which is used 
    in the equation for the k^th time coefficient
    """
    T_matrix = np.zeros((N_modes,N_modes+4)) 
    velocity_modes = all_uv_modes(modes,x,y)
    AM_v_mode_x, AM_v_mode_y = u_stokes_eta_mode(x,y), u_stokes_xi_mode(x,y)

    # i in [|0,N_modes + 4|], k in [|0,N_modes|] because the NS eqs are projected on N_modes equations

    # POD coefficients 
    for i in range(N_modes):
        for k in range(N_modes):
            T_matrix[k,i] = T_matrix[k,i] + inner_product(velocity_modes[i,:], velocity_modes[k,:])

    # AM mode coefficients
    for k in range(N_modes):
        T_matrix[k, N_modes+ 2 - 1] = inner_product(AM_v_mode_x, velocity_modes[k,:])
        T_matrix[k, N_modes+ 4 - 1] = inner_product(AM_v_mode_y, velocity_modes[k,:])
        
    return tf.constant(T_matrix,dtype=tf.float64)

def P_matrix(grad_p_modes, velocity_modes):
    P_matrix = np.zeros((N_modes,N_modes + 4))

    for i in range(N_modes):
        for k in range(N_modes):
            P_matrix[k,i] = P_matrix[k,i] + inner_product(grad_p_modes[i,:], velocity_modes[k,:])

    return tf.constant(P_matrix, dtype=tf.float64)

def M_matrix(mean,modes,x,y):
    
    M_matrix = np.zeros((N_modes,N_modes + 4))
    for i in range(N_modes):
        q_vel_i = Q_velocity(mean,modes[i],x,y) +Q_velocity(modes[i],mean,x,y) 
        for k in range(N_modes):
            mode_k = vel_mode_vector(modes[k],x,y)
            M_matrix[k,i] = M_matrix[k,i] + inner_product(q_vel_i,mode_k )
    
    for k in range(N_modes):
        q_vel_stokes_eta = Q_velocity_mean_stokes_eta(mean, x, y)
        q_vel_stokes_xi = Q_velocity_mean_stokes_xi(mean, x, y)
        mode_k = vel_mode_vector(modes[k],x,y)
        M_matrix[k, N_modes + 2 - 1] = M_matrix[k, N_modes + 2 - 1] + inner_product(q_vel_stokes_eta, mode_k )
        M_matrix[k, N_modes + 4 - 1] = M_matrix[k, N_modes + 4 - 1] + inner_product(q_vel_stokes_xi, mode_k ) 

    return tf.constant(M_matrix,dtype=tf.float64)

def Q_matrix(modes, x, y):
    Q_matrix = np.zeros((N_modes, N_modes + 4, N_modes + 4))

    for k in range(N_modes):
        mode_u_k = tf.cast(vel_mode_vector(modes[k], x, y),tf.float64)
        for i in range(N_modes):
                for j in range(N_modes):
                        q_ij_vector = uv_modes_q_vector_ij(modes,x, y, i, j)
                        # print(q_ij_vector)
                        # print(mode_u_k)
                        Q_matrix[k, i, j] = Q_matrix[k, i, j] + inner_product(q_ij_vector, mode_u_k)

        # 1st AM mode --------------------------------------------------------------------------------------------------
        # i = N + 2, j in [1,N]
        for j in range(N_modes):
            q_stokes_j_vector = phi_stokes_eta_nabla_phi_i(modes, j, x, y)
            Q_matrix[k, N_modes+2-1, j] = Q_matrix[k, N_modes+2-1, j] + inner_product(q_stokes_j_vector, mode_u_k)
        # j = N + 2 , i in [1,N]
        for i in range(N_modes):
            q_i_stokes_vector = phi_i_nabla_phi_stokes_eta(modes, i, x, y)
            Q_matrix[k, i, N_modes+2-1] = Q_matrix[k, i, N_modes+2-1] + inner_product(q_i_stokes_vector, mode_u_k)
        
        # 2nd AM mode --------------------------------------------------------------------------------------------------
        # i = N + 4, j in [1,N]
        for j in range(N_modes):
            q_stokes_j_vector = phi_stokes_xi_nabla_phi_i(modes, j, x, y)
            Q_matrix[k, N_modes+4-1, j] = Q_matrix[k, N_modes+4-1, j] + inner_product(q_stokes_j_vector, mode_u_k)
        # j = N + 4 , i in [1,N]
        for i in range(N_modes):
            q_i_stokes_vector = phi_i_nabla_phi_stokes_xi(modes, i, x, y)
            Q_matrix[k, i, N_modes+4-1] = Q_matrix[k, i, N_modes+4-1] + inner_product(q_i_stokes_vector, mode_u_k)
        

        # i = j = N+2
        q_s_eta_s_eta_vector = phi_stokes_eta_nabla_phi_stokes_eta(x, y)
        # print(inner_product(q_AM_AM_vector, mode_u_k))
        Q_matrix[k, N_modes+2-1, N_modes+2-1] = Q_matrix[k, N_modes+2-1, N_modes+2-1] +\
             inner_product(q_s_eta_s_eta_vector, mode_u_k)
        # i = N + 2, j = N + 4
        q_s_eta_s_xi_vector = phi_stokes_eta_nabla_phi_stokes_xi(x,y)
        Q_matrix[k, N_modes+2-1, N_modes+4-1] = Q_matrix[k, N_modes+2-1, N_modes+4-1] +\
        inner_product(q_s_eta_s_xi_vector, mode_u_k)
        # i = N + 4, j = N + 2
        q_s_xi_s_eta_vector = phi_stokes_xi_nabla_phi_stokes_eta(x,y)
        Q_matrix[k, N_modes+4-1, N_modes+2-1] = Q_matrix[k, N_modes+4-1, N_modes+2-1] +\
        inner_product(q_s_xi_s_eta_vector, mode_u_k)
        # i = j = N + 4
        q_s_xi_s_xi_vector = phi_stokes_xi_nabla_phi_stokes_xi(x,y)
        Q_matrix[k, N_modes+4-1, N_modes+4-1] = Q_matrix[k, N_modes+4-1, N_modes+4-1] +\
        inner_product(q_s_xi_s_xi_vector, mode_u_k)

    return tf.constant(Q_matrix,dtype=tf.float64)

def L_matrix(modes, x, y):

    L_matrix = np.zeros((N_modes, N_modes + 4))

    
    # POD coefficients
    for i in range(N_modes):
        lap_mode = uv_mode_i_laplacian(modes, x, y, i)
        for k in range(N_modes):
            vel_vect = tf.cast(vel_mode_vector(modes[k],x, y),dtype=tf.float64)
            coeff = inner_product(lap_mode,vel_vect)
            L_matrix[k,i] = L_matrix[k,i] + coeff

    return tf.constant(L_matrix, dtype=tf.float64)


# Beta and Chi matrices to reevaluate !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Theoritical development of the ROM


def Beta_matrix(modes, x, y):
    beta_matrix = np.zeros((N_modes, N_modes + 4))


    psi_s_eta_dx = p_stokes_eta_grad_x(x, y)
    psi_s_eta_dy = p_stokes_eta_grad_y(x, y)
    
    psi_s_xi_dx = p_stokes_xi_grad_y(x, y)
    psi_s_xi_dy = p_stokes_xi_grad_y(x, y)
    

    grad_psi_s_eta = tf.cast(tf.concat([psi_s_eta_dx, psi_s_eta_dy],axis=0),dtype=tf.float64)
    grad_psi_s_xi = tf.cast(tf.concat([psi_s_xi_dx, psi_s_xi_dy],axis=0),dtype=tf.float64)
    for k in range(N_modes):
        mode_k = tf.cast(vel_mode_vector(modes[k], x, y),tf.float64)
        beta_matrix[k,N_modes+2-1] = beta_matrix[k,N_modes+2-1] + inner_product(grad_psi_s_eta, mode_k)
        beta_matrix[k,N_modes+4-1] = beta_matrix[k,N_modes+4-1] + inner_product(grad_psi_s_xi, mode_k)

    return tf.constant(beta_matrix, tf.float64)

def delta_matrix(mean_mode, modes, x, y):
    delta_mat = np.zeros((N_modes, N_modes + 4))
    
    for k in range(N_modes):

        mode_k = vel_mode_vector(modes[k], x, y)
        mean_mode_dx = d_mode_dx(mean_mode, x, y)
        mean_mode_dy = d_mode_dy(mean_mode, x, y)
        
        delta_mat[k, N_modes+1] += inner_product(mean_mode_dx, mode_k)
        delta_mat[k, N_modes+3] += inner_product(mean_mode_dy, mode_k)

    return tf.constant(delta_mat, tf.float64)

def D_matrix(modes, x, y):
    D_matrix = np.zeros((N_modes, N_modes + 4, N_modes + 4))

    for k in range(N_modes):
        mode_k = vel_mode_vector(modes[k], x, y)
        

        for i in range(N_modes):
            mode_i = modes[i]
            d_mode_i_dx = d_mode_dx(mode_i, x, y)
            d_mode_i_dy = d_mode_dy(mode_i, x, y)
            D_matrix[k, N_modes+1, i] += inner_product(d_mode_i_dx, mode_k)
            D_matrix[k, N_modes+3, i] += inner_product(d_mode_i_dy, mode_k)
        
        d_stokes_eta_x_dx = tf.concat([u_stokes_eta_x_grad_x(x, y), u_stokes_eta_y_grad_x(x, y)],axis=0)
        D_matrix[k, N_modes+1, N_modes+1] += inner_product(d_stokes_eta_x_dx, mode_k)

        d_stokes_xi_dx = tf.concat([u_stokes_xi_x_grad_x(x, y), u_stokes_xi_y_grad_x(x, y)],axis=0)
        D_matrix[k, N_modes+1, N_modes+3] += inner_product(d_stokes_xi_dx, mode_k)

        d_stokes_eta_dy = tf.concat([u_stokes_eta_x_grad_y(x, y), u_stokes_eta_y_grad_y(x, y)],axis=0)
        D_matrix[k, N_modes+3, N_modes+1] += inner_product(d_stokes_eta_dy, mode_k)

        d_stokes_xi_dy = tf.concat([u_stokes_xi_x_grad_y(x, y), u_stokes_xi_y_grad_y(x, y)],axis=0)
        D_matrix[k, N_modes+3, N_modes+3] += inner_product(d_stokes_xi_dy, mode_k)

    return tf.constant(D_matrix,tf.float64)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Gamma_matrix(modes, x, y):

    gamma_matrix = np.zeros((N_modes+4,N_modes+4))

    T_mat = T_matrix(modes, x, y)
    Beta_mat = Beta_matrix(modes, x, y)

    gamma_matrix[:N_modes,:] = T_mat + Beta_mat

    gamma_matrix[N_modes,N_modes] = 1
    gamma_matrix[N_modes+1,N_modes+1] = 1  
    gamma_matrix[N_modes+2,N_modes+2] = 1  
    gamma_matrix[N_modes+3,N_modes+3] = 1  
    
    return gamma_matrix


# Computing gammas with the Monte-Carlo integration method :

def gammas_POD_mean_MC(x, y, p_mean):
    contour_indices = isolate_contour_points(x, y, epsilon=param.epsilon_contour)    
    x_contour, y_contour = x[contour_indices], y[contour_indices]
    p_mean_contour = p_mean[contour_indices]

    N_c = len(x_contour)
    gamma_x, gamma_y = 0.0, 0.0
    for i in range(N_c):
        x_ci = x_contour[i]
        y_ci = y_contour[i]
        p_ci = p_mean_contour[i]
        r,theta = cartesian_to_polar(x_ci,y_ci)
        gamma_x += p_ci * np.cos(theta)
        gamma_y += p_ci * np.sin(theta)
    gamma_x = gamma_x * 2 * np.pi / param.Mass_number / N_c
    gamma_y = gamma_y * 2 * np.pi / param.Mass_number / N_c
    return (gamma_x, gamma_y)

def gammas_POD_mode_MC(x_contour, y_contour, p_mode_contour):
    N_c = len(x_contour)
    gamma_x, gamma_y = 0.0, 0.0
    for i in range(N_c):
        x_ci = x_contour[i]
        y_ci = y_contour[i]
        p_ci = p_mode_contour[i]
        r,theta = cartesian_to_polar(x_ci,y_ci)
        gamma_x += p_ci * np.cos(theta)
        gamma_y += p_ci * np.sin(theta)
    gamma_x = gamma_x * 2 * np.pi / param.Mass_number / N_c
    gamma_y = gamma_y * 2 * np.pi / param.Mass_number / N_c
    return (gamma_x, gamma_y)

def sigmas_xy_MC(x, y, p_POD_modes):
    M = param.Mass_number
    sigma_x_vect = np.zeros(N_modes+4)
    sigma_y_vect = np.zeros(N_modes+4)
    contour_indices = isolate_contour_points(x, y, epsilon=param.epsilon_contour)
    
    x_contour, y_contour = x[contour_indices], y[contour_indices]

    for i in range(N_modes):
        gamma_xi, gamma_yi = gammas_POD_mode_MC(x_contour, y_contour, p_POD_modes[contour_indices, i])
        
        sigma_x_vect[i] = - gamma_xi * 2 / (np.pi * (1+param.Mass_number))
        sigma_y_vect[i] = - gamma_yi * 2 / (np.pi * (1+param.Mass_number))

        if param.include_viscosity:
            mu_visc_xi, mu_visc_yi = mu_visc_i_integration(modes[i], x_contour, y_contour) 
            sigma_x_vect[i] +=  mu_visc_xi * 2 / (np.pi * (1+param.Mass_number))
            sigma_y_vect[i] +=  mu_visc_yi * 2 / (np.pi * (1+param.Mass_number))


    sigma_x_vect[N_modes] = - M/(1+M)* (2*np.pi/param.Ur)**2
    sigma_x_vect[N_modes+1] = - M/(1+M) * param.Zeta

    sigma_y_vect[N_modes+2] = - M/(1+M)* (2*np.pi/param.Ur)**2
    sigma_y_vect[N_modes+3] = - M/(1+M) * param.Zeta
    return (sigma_x_vect, sigma_y_vect)

def Lambda_matrix_MC(mean_mode, modes, x, y):
    inv_Re = 1/param.Re
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
    lambda_matrix[N_modes +1,:], lambda_matrix[N_modes +3,:] = sigmas_xy_MC(x, y,p_POD_modes)    
    
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

    # Testing the new functions for the viscosity forces
    tensor_test = gradient_tensor_sum_POD_modes(modes[0], x_sp, y_sp)

    
    # def phi_AM_u_x(x,y):
    #     N = len(x)
    #     return phi_AM_v(x,y)[0]
    # def phi_AM_u_y(x,y):
    #     N = len(x)
    #     return phi_AM_v(x,y)[1]

    # phi_ux,phi_uy = phi_AM_v2(x_sp,y_sp)
    # phi_uy = phi_AM_u_y(x_sp,y_sp)


    # path_AM = "Misc/AM2/"

    # u_x_AM = phi_ux
    # u_x_grid = griddata(close_points, u_x_AM,(grid_x,grid_y))

    # u_y_AM = phi_uy
    # u_y_grid = griddata(close_points, u_y_AM,(grid_x,grid_y))

    # f,ax = plt.subplots(1,1,figsize=(20,10))
    # plt.imshow(u_x_grid, extent=(param.Lxmin, param.Lxmax, param.Lymin, param.Lymax),cmap="twilight")
    # plt.colorbar()
    # plt.title("Added mass mode - $\phi_x^{AM,2}(x,y)$",fontsize=25)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # cyl = plt.Circle((param.x_c, param.y_c), param.r_c, color='white')
    # ax.set_aspect(1)
    # ax.add_artist(cyl)
    # plt.savefig(path_AM + "u_x_AM.png", transparent=True)
    # plt.close()

    # f,ax = plt.subplots(1,1,figsize=(20,10))
    # plt.imshow(u_y_grid, extent=(param.Lxmin, param.Lxmax, param.Lymin, param.Lymax),cmap="twilight")
    # plt.colorbar()
    # plt.title("Added mass mode - $\phi_y^{AM,2}(x,y)$",fontsize=25)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # cyl = plt.Circle((param.x_c, param.y_c), param.r_c, color='white')
    # ax.set_aspect(1)
    # ax.add_artist(cyl)
    # plt.savefig(path_AM + "u_y_AM.png", transparent=True)
    # plt.close()

    # from time import time
    # Trial on the viscosity force terms 
    

   # F_visc_x_POD = []
    # for i in range(N_modes):
    #     F_visc_x_POD.append(viscosity_integration_cylinder_x(modes[i]))
    # print("Computing F_visc_POD mean")
    # F_visc_x_mean = viscosity_integration_cylinder_x(mean_mode)

    # -------------------------------------------------------------
    # Testing a new method to compute forces 

    # print("Computing F_p_x_POD with old method")
    # print("Forces in x")
    # for i_mode in range(4):
    #     print("=====================")
    #     print("i_mode :",i_mode)
    #     t0 = time() 
    #     fp_old = p_mode_cylinder_integration_x_slow(modes[i_mode])[0]
    #     t_fpod_x = time()
    #     duration_fpod_1 = t_fpod_x - t0
    #     print("Duration fpod old : {:.8f}s".format(duration_fpod_1))

    #     t0 = time() 
    #     print("Computing F_p_x_POD with cumtrapz")
    #     fp_new = p_mode_cylinder_integration_x(modes[i_mode])
    #     t_fpod_x = time()
    #     duration_fpod_1 = t_fpod_x - t0
    #     print("Duration fpod new: {:.8f}s".format(duration_fpod_1))
    #     print("Force value old : {}".format(fp_old))
    #     print("Force value new: {}".format(fp_new))


    # for i_mode in range(4):
    #     print("Forces in y")
    #     print("=====================")
    #     print("i_mode :",i_mode)
    #     t0 = time() 
    #     fp_old = p_mode_cylinder_integration_y_slow(modes[i_mode])[0]
    #     t_fpod_x = time()
    #     duration_fpod_1 = t_fpod_x - t0
    #     print("Duration fpod old : {:.8f}s".format(duration_fpod_1))

    #     t0 = time() 
    #     print("Computing F_p_x_POD with cumtrapz")
    #     fp_new = p_mode_cylinder_integration_y(modes[i_mode])
    #     t_fpod_x = time()
    #     duration_fpod_1 = t_fpod_x - t0
    #     print("Duration fpod new: {:.8f}s".format(duration_fpod_1))
    #     print("Force value old : {}".format(fp_old))
    #     print("Force value new: {}".format(fp_new))
    
    # # lambda_mat = Lambda_matrix_update(mean_mode, modes, x_sp, y_sp)
    # # print(lambda_mat)

    # print(gamma_bar_x(mean_mode))
    # print(gamma_bar_y(mean_mode))


    # t0 = time() 
    # print("Computing F_visc_x_POD with cumtrapz")
    # i_mode = 0
    # fpod_x  = viscosity_integration_x_fast(modes,i_mode)
    # t_fpod_x = time()
    # duration_fpod_1 = t_fpod_x - t0
    # print("Duration fpod : {:.8f}s".format(duration_fpod_1))

    # visc_forces_vect = viscosity_forces_x_vector(modes)
    # print(visc_forces_vect)

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