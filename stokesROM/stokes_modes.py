import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
os.chdir("c:\\Users\\Student\\Documents\\Project\\Endgame\\Unified_PODVM_Galerkin")
from parameters import Parameters

nCond = Parameters()
R = nCond.r_c
Re = nCond.Re

@tf.function
def u_r_stokes(r, theta): 
    return  tf.cos(theta) * (1 + R**3/(2*r**3) - 3*R/(2*r) )

def u_r_stokes_np(r, theta): 
    return  np.cos(theta) * (1 + R**3/(2*r**3) - 3*R/(2*r) )

@tf.function
def u_theta_stokes(r, theta):
    return -  tf.sin(theta)* (1 - R**3 / (4*r**3) - 3*R/(4*r))

def u_theta_stokes_np(r, theta):
    return -  np.sin(theta)* (1 - R**3 / (4*r**3) - 3*R/(4*r))

def cart_to_spher(x,y):
    # return (tf.sqrt(x**2 + y**2), tf.math.atan2(x,y))
    return (tf.sqrt(x**2 + y**2), tf.math.atan2(y,x))

def cart_to_spher_np(x,y):
    # return (np.sqrt(x**2 + y**2), np.arctan2(x,y))
    return (np.sqrt(x**2 + y**2), np.arctan2(y,x))

# def u_z_stokes(x,y):
#     r, theta = cart_to_spher(x,y)
#     ur = u_r_stokes(r, theta)
#     utheta = u_theta_stokes(r, theta)
#     return np.cos(theta) * ur - np.sin(theta) * utheta


# def u_x_stokes(x,y):
#     r, theta = cart_to_spher(x,y)
#     ur = u_r_stokes(r, theta)
#     utheta = u_theta_stokes(r, theta)
#     return np.cos(theta) * utheta + np.sin(theta) * ur 
@tf.function
def u_stokes_eta(x,y):
    """
    Models a Stokes flow in the horizontal direction, by default from left to right.
    input: x,y cartesian coordinate arrays of the spatial points
    output: (u_x, u_y) tuple of the horizontal and vertical components of the fluid velocity 
    """
    r, theta = cart_to_spher(x,y)
    ur = u_r_stokes(r, theta)
    utheta = u_theta_stokes(r, theta)
    ux = tf.cos(theta) * ur - tf.sin(theta) * utheta
    uy = tf.cos(theta) * utheta + tf.sin(theta) * ur
    return (ux, uy ) 

def u_stokes_eta_np(x,y):
    """
    Models a Stokes flow in the horizontal direction, by default from left to right.
    input: x,y cartesian coordinate arrays of the spatial points
    output: (u_x, u_y) tuple of the horizontal and vertical components of the fluid velocity 
    """
    r, theta = cart_to_spher_np(x,y)
    ur = u_r_stokes_np(r, theta)
    utheta = u_theta_stokes_np(r, theta)
    return (np.cos(theta) * ur - np.sin(theta) * utheta, np.cos(theta) * utheta + np.sin(theta) * ur ) 

@tf.function
def u_stokes_xi(x,y):
    """
    Models a Stokes flow in the vertical direction, by default from down to the top.
    input: x,y cartesian coordinate arrays of the spatial points
    output: (u_x, u_y) tuple of the horizontal and vertical components of the fluid velocity 
    """
    r, alpha = cart_to_spher(x,y)
    theta =  alpha - np.pi/2
    ur = u_r_stokes(r, theta)
    utheta = u_theta_stokes(r, theta)

    ux = ur * tf.cos(alpha) - utheta * tf.sin(alpha)
    uy = ur * tf.sin(alpha) + utheta * tf.cos(alpha)
    return (ux, uy)

@tf.function
def u_stokes_eta_mode(x,y):
    """
    Models a Stokes flow in the horizontal direction, by default from left to right, stacked in a single vector of dimension (2*N)
    input: x,y cartesian coordinate arrays of the spatial points
    output: vector of dimension (2*N) containing the velocity compenents for each points, stacked, such that output = (u_x^1,...,u_x^N,u_y^1,...,u_y^N ) 
    """
    ux, uy = u_stokes_eta(x, y)
    return tf.cast(tf.concat([ux,uy],axis=0),tf.float64)

@tf.function
def u_stokes_xi_mode(x,y):
    """
    Models a Stokes flow in the vertical direction, by default from the bottom to the top, stacked in a single vector of dimension (2*N)
    input: x,y cartesian coordinate arrays of the spatial points
    output: vector of dimension (2*N) containing the velocity compenents for each points, stacked, such that output = (u_x^1,...,u_x^N,u_y^1,...,u_y^N )
    """
    ux, uy = u_stokes_xi(x, y)
    return tf.cast(tf.concat([ux,uy],axis=0),tf.float64)

@tf.function
def p_stokes_eta(x,y):
    r, theta = cart_to_spher(x,y)
    return  - 1/ Re  * 3 / 2 * R * tf.cos(theta)/r**2

@tf.function
def p_stokes_xi(x,y):
    r, alpha = cart_to_spher(x,y)
    theta =  alpha - np.pi/2
    return  1/ Re  * 3 / 2 * R * tf.cos(theta)/r**2


def p_stokes_eta_np(x,y):
    r, theta = cart_to_spher_np(x,y)
    # return   1/ Re  * 3 / 2 * R * np.sin(theta)/r**2
    return   - 1/ Re  * 3 / 2 * R * np.cos(theta)/r**2


def p_stokes_xi_np(x,y):
    r, alpha = cart_to_spher_np(x,y)
    theta =  -alpha + np.pi/2
    # return   1/ Re  * 3 / 2 * R * np.sin(theta)/r**2
    return   1/ Re  * 3 / 2 * R * np.cos(theta)/r**2


if __name__ == "__main__":

    # Plot the Stokes mode for a horizontal flow
    Lxmin, Lxmax, Lymin, Lymax = -1,1,-1,1
    # Grid of x, y points
    nx, ny = 500, 500
    x = np.linspace(Lxmin, Lxmax, nx)
    y = np.linspace(Lymin, Lymax, ny)
    X, Y = np.meshgrid(x, y)

    Uz, Ux = np.zeros((ny, nx)), np.zeros((ny, nx)) 

    circle_mask = lambda x,y: x**2+y**2<R**2
    X,Y = np.meshgrid(x,y)

    X,Y = (np.ma.masked_where(circle_mask(X,Y),X), 
            np.ma.masked_where(circle_mask(X,Y),Y))

    u_stokes_eta_values = u_stokes_eta(X,Y)
    u_stokes_xi_values = u_stokes_xi(X,Y)

    plt.figure(figsize=(10,10))
    plt.imshow(u_stokes_eta_values[0], extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.title("Horizontal flow")
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_eta_values[0], u_stokes_eta_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')

    plt.show()


    f,ax = plt.subplots(1,1,figsize=(10,10))
    plt.imshow(u_stokes_eta_values[1], extent=(Lxmin ,Lxmax, Lymin, Lymax), vmin=-0.3, vmax=0.3, cmap='twilight_shifted')
    plt.title("Horizontal flow - $\Phi_y^\eta$")
    plt.colorbar()
    cyl = plt.Circle((nCond.x_c, nCond.y_c), nCond.r_c, color='white')
    ax.set_aspect(1)
    ax.add_artist(cyl)

    plt.gca().set_aspect('equal')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.title("Vertical flow")
    plt.imshow(u_stokes_xi(X,Y)[1], extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_xi_values[0], u_stokes_xi_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.title("Pressure - Horizontal flow")
    plt.imshow(p_stokes_eta(X,Y), extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_eta_values[0], u_stokes_eta_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.title("Pressure - Vertical flow")
    plt.imshow(p_stokes_xi(X,Y), extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_xi_values[0], u_stokes_xi_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')
    plt.show()


    p_stokes_eta_values = p_stokes_eta_np(X,Y)
    p_stokes_xi_values = p_stokes_xi_np(X,Y)


    plt.figure(figsize=(10,10))
    plt.imshow(p_stokes_eta_values, extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.title("Pressure - Horizontal flow")
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_eta_values[0], u_stokes_eta_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')

    plt.show()

    plt.figure(figsize=(10,10))
    plt.imshow(p_stokes_xi_values, extent=(Lxmin ,Lxmax, Lymin, Lymax))
    plt.title("Pressure - Vertical flow")
    plt.colorbar()
    plt.streamplot(x, y, u_stokes_xi_values[0], u_stokes_xi_values[1],
                linewidth = 1,
                density = 2, arrowstyle ='->', 
                arrowsize = 1.5,
                color = 'k')

    plt.show()

    # # Plotting the GIFs ============================================================================================
    # T = 1.0 # Period
    # N_t = 200
    # t = np.linspace(0.0, 5*T, N_t)
    # xi_signal = np.cos(t * 2 * np.pi / T ) 
    # eta_signal = np.cos(t * 2 * np.pi / T * 2) # twice the frequency
    # os.chdir("stokesROM\\")
    # for i_t in tqdm(range(len(t))):
    #     xi_t = xi_signal[i_t]
    #     eta_t = eta_signal[i_t]

    #     f,ax = plt.subplots(1,1,figsize=(20,20))
    #     plt.title("Vertical velocity $v$ for an in-line motion of the cylinder", fontsize=20)
    #     plt.imshow(u_stokes_eta(X,Y)[1]*t[i_t], extent=(Lxmin ,Lxmax, Lymin, Lymax),vmin=-0.3, vmax=0.3)
    #     plt.colorbar()
    #     plt.streamplot(x, y, u_stokes_eta_values[0], u_stokes_eta_values[1],
    #                 linewidth = 1,
    #                 density = 2, arrowstyle ='->', 
    #                 arrowsize = 1.5,
    #                 color = 'k')
        
    #     cyl = plt.Circle((nCond.x_c, nCond.y_c), nCond.r_c, color='white')
    #     ax.set_aspect(1)
    #     ax.add_artist(cyl)
    #     plt.savefig("videos\\frames\\eta_mode_v\\f_{}".format(i_t))
    #     plt.close()

    # def make_video_from_frames(field):
    #     image_folder = "videos\\frames_{}\\".format(field)
    #     video_name = 'stokes_mode_{}.avi'.format(field)

    #     n_frames = N_t

    #     images = [field +"_"+str(i)+".png" for i in range(n_frames)]
    #     frame = cv2.imread(os.path.join(image_folder, images[0]))
    #     height, width, layers = frame.shape

    #     frame_per_s = n_frames//7

    #     video = cv2.VideoWriter(video_name, 0,frame_per_s,(width,height))

    #     for image in images:
    #         video.write(cv2.imread(os.path.join(image_folder, image)))

    #     cv2.destroyAllWindows()
    #     video.release()

