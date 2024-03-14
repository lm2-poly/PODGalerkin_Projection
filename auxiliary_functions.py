import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras 
import io
from scipy.fft import fft, fftfreq
from tqdm import tqdm


from parameters import Parameters
from playsound import playsound
import scipy.integrate as integrate
from scipy.signal import argrelmax, find_peaks


nCond = Parameters()
def sin_act(x):
    return tf.sin(x)

# def snake(x):
#     return x + tf.square(tf.sin(x*nCond.snake_param))/nCond.snake_param

def is_in_domain(point, geom=nCond.geom):
    [Lx_min, Lx_max, Ly_min, Ly_max, xc, yc, rc] = geom
    x, y = point
    return (Lx_min < x < Lx_max) and (Ly_min < y < Ly_max) and ((x ** 2 + y ** 2) > rc**2)


def is_in_wake_rectangle(point, geom=nCond.geom):
    [Lx_min, Lx_max, Ly_min, Ly_max, xc, yc, rc] = geom
    x, y = point

    Lxmin_wake = nCond.Lxmin_wake
    Lxmax_wake = nCond.Lxmax_wake

    Lymin_wake = nCond.Lymin_wake
    Lymax_wake =  nCond.Lymax_wake

    in_wake_rectangle = (Lxmin_wake < x < Lxmax_wake) and (Lymin_wake < y < Lymax_wake)
    return in_wake_rectangle and is_in_domain(point,geom) # verify that the point is in the wake rectangle and not inside the cylinder

def is_around_cylinder(point,geom=nCond.geom):
    [Lx_min, Lx_max, Ly_min, Ly_max, xc, yc, rc] = geom
    x, y = point

    circle_radius = nCond.circle_around_cyl

    in_circle = ((x - xc)**2 + (y - yc)**2 < circle_radius)  

    return is_in_domain(point,geom) and in_circle

def is_in_end_wake(point, geom=nCond.geom):
    [Lx_min, Lx_max, Ly_min, Ly_max, xc, yc, rc] = geom
    x, y = point

    Lxmin_wake = nCond.Lxmin_wake
    Lxmax_wake = nCond.Lxmax_wake

    Lymin_wake = nCond.Lymin_wake
    Lymax_wake =  nCond.Lymax_wake
    
    return (x > Lxmax_wake) and (Lymin_wake < y < Lymax_wake)

def cartesian_to_polar(x, y):
    r = tf.math.sqrt(x**2 + y**2)
    theta = tf.math.atan2(y, x)
    return (r, theta)

def generate_penalization_spatial_points(Npoints, mode='uniform'):
    pen_points = []
    if mode == 'uniform':
        # i = 0
        while len(pen_points) < Npoints:
            p = [np.random.uniform(nCond.Lxmin, nCond.Lxmax), np.random.uniform(nCond.Lymin, nCond.Lymax)]
            if is_in_domain(p):
                pen_points.append(p)


    elif mode == '3 zones':
        counters = [0, 0, 0] # To count the number of points in each zone
        p1, p2, p3 = [3.0/6, 2.0/6, 1.0/6] # probabilities of the dice toss for each zone 
        # i=0
        while len(pen_points) < Npoints:
            p = [np.random.uniform(nCond.Lxmin, nCond.Lxmax), np.random.uniform(nCond.Lymin, nCond.Lymax)]
            dice_toss = np.random.uniform(0.0,1.0)

            in_wake = is_in_wake_rectangle(p,geom=nCond.geom)
            around_cylinder = is_around_cylinder(p,geom=nCond.geom)
            in_domain = is_in_domain(p,geom=nCond.geom)
            end_wake = is_in_end_wake(p,geom=nCond.geom)
            
            # Zone 1 : wake
            if in_wake:
                if dice_toss <= p1:
                    pen_points.append(p)
                    counters[0] = counters[0] + 1
            # Zone 2 : upwind around the cylinder
            if around_cylinder or end_wake and (not in_wake):
                if dice_toss <= p2:
                    pen_points.append(p)
                    counters[1] = counters[1] + 1
            # Zone 3 : outside of the zones
            if (not in_wake) and (not around_cylinder) and in_domain :
                if dice_toss <= p3:
                    pen_points.append(p)
                    counters[2] = counters[2] + 1
        print("Repartition of the points in the zones :")
        print("Zone 1 :",counters[0],"points ("+str(counters[0]/Npoints*100)+"% of the points)")
        print("Zone 2 :",counters[1],"points ("+str(counters[1]/Npoints*100)+"% of the points)")
        print("Zone 3 :",counters[2],"points ("+str(counters[2]/Npoints*100)+"% of the points)")

    print("Number of spatial penalization points :", len(pen_points), "-", mode, "method")
    return tf.constant(pen_points,dtype=tf.float32)

def adapt_learning_rate(patience, lr, stagnation_epochs, prev_loss, current_loss):
    """ 
    Function that reduces the learning rate when the loss stagnates for a certain number of epochs
    patience : number of maximum stagnation epochs before reducing the lr
    lr : current learning rate
    stagnation_epochs : number of epochs for which the loss has stagnated
    """
    if np.abs(prev_loss - current_loss)/prev_loss < nCond.stagnation_tolerance : 
        stagnation_epochs += 1
    else :
        stagnation_epochs = 0

    if stagnation_epochs >= patience :
        if lr*nCond.dim_factor > nCond.lr_min:
            lr = lr*nCond.dim_factor
            print("------------------------------------")
            print("Learning rate set to", lr.numpy())
            stagnation_epochs = 0
    return stagnation_epochs, lr


def plot_to_image(figure):
  """
  Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def recap_parameters(param):
    nCond.recap()

def var_snake_init(a):
    """
    Function to get the optimal variance for the initialization of the weights when using the
    snake activation function. This optimal variance depends on the snake parameter.
    input : a the parameter of the snake function
    output : optimal variance for the initialization of the model
    """
    return 1 + (1/(8*a**2)) * (1 + tf.exp(-8*a**2) - 2 * tf.exp(-4*a**2))

def reshape_for_lstm(x):
    return tf.reshape(x,(len(x),1,1))

def fft_signal(x,y):
    N = len(x)
    yf = fft(y)
    xf = fftfreq(N, x[1]-x[0])[:N//2]
    return xf, yf

def fft_sig_abs(x,y):
    xf, yf = fft_signal(x,y)
    return xf, 2.0/len(xf) * np.abs(yf[0:len(x)//2])


def plot_fft(x,y):
    xf, yf = fft_signal(x,y)
    plt.plot(xf, 2.0/len(xf) * np.abs(yf[0:len(x)//2]))

def beep():
    playsound('Data/beep_end.mp3')

def beep_2():
    playsound('Data/beep_postprocess.mp3')

def heatmap_matrix(Mat,name,chosen_cmap="coolwarm"):
    f, ax = plt.subplots(1,1)
    plt.imshow(Mat,cmap=chosen_cmap)
    for i in range(len(Mat)):
        for j in range(len(Mat[0])):
            text = ax.text(j, i,"{:.3f}".format(Mat[i, j]),
                        ha="center", va="center", color="black",
                        fontsize=18)

    ax.set_xticks(np.arange(len(Mat[0])), labels=np.arange(1,len(Mat[0])+1))
    ax.set_yticks(np.arange(len(Mat)), labels=np.arange(1,len(Mat)+1))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.colorbar()
    ax.set_title("Coefficients of the "+ name +" matrix")
    f.tight_layout()
    plt.show()

def get_oscillation_amplitude(x, y):
    y_osc = y - np.mean(y)
    return np.max(y_osc)

def get_oscillation_A_10(x, y):
    """
    Gets the mean of the 10% of the highest amplitude peaks 
    """
    indices_peaks =  find_peaks(y)[0]
    sorted_peak_amplitudes = y[indices_peaks][np.argsort(-y[indices_peaks])]
    N_peaks = len(indices_peaks)
    N_10 = N_peaks //10
    A_10 = np.mean(sorted_peak_amplitudes[:N_10])
    return A_10


def get_main_freq(xf, yf):
    i_max = np.argmax(yf)
    return xf[i_max]

def get_second_freq(xf, yf):
    i_max = np.argmax(yf)
    i_max_2 = np.argmax(yf[i_max+1:])
    return xf[i_max + i_max_2]

def isolate_contour_points(nodes_x, nodes_y, epsilon):
    contour_indices = []
    r_lim = (nCond.r_c*(1+epsilon))
    for i in range(len(nodes_x)):
        xn,yn = nodes_x[i], nodes_y[i]
        if (nCond.x_c-xn)**2 + (nCond.y_c-yn)**2 < r_lim**2:
            contour_indices.append(i)
    # print("For epsilon = {:.3f} : {} points have been selected.".format(epsilon, len(contour_indices)))
    return np.array(contour_indices)


def get_main_frequencies(signal, sampling_rate, num_frequencies=2, interpolation_range=5):
    N = len(signal)
    # Perform FFT
    spectrum = np.abs(fft(signal))[:N//2]
    
    # Get the frequencies corresponding to the FFT values
    frequencies = fftfreq(N, 1 / sampling_rate)[:N//2]
    
    # Find the indices of the local maxima in the spectrum
    peak_indices = argrelmax(np.abs(spectrum))[0]
    
    # Sort the peak indices based on the magnitudes of the spectrum
    sorted_indices = np.argsort(np.abs(spectrum[peak_indices]))[::-1]
    
    main_frequencies = []
    for i in range(num_frequencies):
        # Retrieve the current peak index
        peak_index = peak_indices[sorted_indices[i]]
        
        # Calculate the interpolation range around the peak index
        start_index = max(0, peak_index - interpolation_range)
        end_index = min(len(spectrum), peak_index + interpolation_range + 1)
        
        # Interpolate the peak location using quadratic peak interpolation
        interpolated_index = np.argmax(np.abs(spectrum[start_index:end_index])) + start_index
        
        # Get the interpolated frequency
        interpolated_frequency = frequencies[interpolated_index]
        
        # Add the interpolated frequency to the main frequencies list
        main_frequencies.append(interpolated_frequency)
    
    return main_frequencies

def gammas_POD_mean(x_contour,y_contour,p_mean_POD):
    N_c = len(x_contour)
    gamma_x, gamma_y = 0.0, 0.0
    for i in range(N_c):
        x_ci = x_contour[i]
        y_ci = y_contour[i]
        p_ci = p_mean_POD[i]
        r,theta = cartesian_to_polar(x_ci,y_ci)
        # BIG CHANGE
        gamma_x += p_ci * np.cos(theta)* nCond.r_c
        gamma_y += p_ci * np.sin(theta)* nCond.r_c
        # gamma_x += p_ci * np.cos(theta)
        # gamma_y += p_ci * np.sin(theta)
    gamma_x = -gamma_x * 2 * 2 / nCond.Mass_number / N_c
    gamma_y = -gamma_y * 2 * 2 / nCond.Mass_number / N_c

    return (gamma_x, gamma_y)

def gammas_POD_mode_i(i_mode, x_contour,y_contour,p_modes_POD):
    N_c = len(x_contour)
    gamma_x, gamma_y = 0.0, 0.0
    for i in range(N_c):
        x_ci = x_contour[i]
        y_ci = y_contour[i]
        p_ci = p_modes_POD[i, i_mode]
        r,theta = cartesian_to_polar(x_ci,y_ci) 
        # BIG CHANGE
        gamma_x += p_ci * np.cos(theta)* nCond.r_c
        gamma_y += p_ci * np.sin(theta)* nCond.r_c
        # gamma_x += p_ci * np.cos(theta)
        # gamma_y += p_ci * np.sin(theta)
    gamma_x = -gamma_x * 2 * 2 / nCond.Mass_number / N_c
    gamma_y = -gamma_y * 2 * 2 / nCond.Mass_number / N_c
    return (gamma_x, gamma_y)

def gammas_POD_mode_i_PINN(mode, x_contour,y_contour):
    N_c = len(x_contour)
    spatial_points = tf.transpose(tf.stack([x_contour,y_contour]))
    vec = tf.cast(mode(spatial_points), tf.float64)
    vec_p = vec[:,2]

    r, theta = cartesian_to_polar(x_contour, y_contour)

    n_vec = n_vector(theta)
    vec_p_n_vec = vec_p * n_vec
    gamma_x, gamma_y = -tf.reduce_sum(vec_p_n_vec, axis=1) *2* 2 / nCond.Mass_number / N_c
    return (gamma_x, gamma_y)

def n_vector(theta):
    return np.array([np.cos(theta), np.sin(theta)])*nCond.r_c

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

    invRe = 1/nCond.Re
    tensor = gradient_tensor_sum_POD_modes(mode, x, y)
    r, theta = cartesian_to_polar(x, y)
    n_vectors = n_vector(theta)
    sum = 0
    for i in range(len(x)):
        sum += np.matmul(tensor[:,:,i], n_vectors[:,i])
    return invRe * sum /(2*np.pi)/len(x)


if __name__ == "__main__":

    # Convergence test of the integration of mu_i ======================================================================
    # First import & define the necessary modes/variables:
    N_modes = nCond.N_modes
    mode_path = "POD/ModePINN/saved/"

    mean_mode = tf.keras.models.load_model(mode_path + "mode_0/mode_0.h5")
    modes = []
    # N_modes counts the POD modes and the mean mode
    for i in range(1, N_modes+1):
        modes.append(tf.keras.models.load_model(mode_path +"/mode_{}/mode_{}.h5".format(i,i)))
    
        # N_theta_integration = 3000
    # values_N_integration = [1000,2000,3000,5000,10000,20000, 30000,40000, 50000,70000, 100000,200000]
    # values_N_integration = np.concatenate([[1*10**i,2*10**i,3*10**i, 5*10**i] for i in range(3,6)])
    values_N_integration = np.concatenate([[j*10**i for j in range(1,10)] for i in range(3,6)])

    def convergence_mu_integration(values_N_integration):
        mu_x_N_values = np.zeros((len(values_N_integration),N_modes + 1))
        mu_y_N_values = np.zeros((len(values_N_integration),N_modes + 1))

        for i_integration in range(len(values_N_integration)):
            N_integration = values_N_integration[i_integration]    
            theta_values = np.sort(np.random.rand(N_integration)*2*np.pi)
            x_int, y_int = nCond.r_c*np.cos(theta_values), nCond.r_c*np.sin(theta_values)

            print("Starting integration with {} points:".format(N_integration))
            # # Show the integration points
            # plt.figure(figsize=(10,10))
            # plt.rc('font',size=18)
            # plt.scatter(x_int, y_int, s=0.6)
            # plt.title("{} integration points".format(N_integration))
            # plt.show()

            mu_x_N_values[i_integration, 0],mu_y_N_values[i_integration, 0] =\
                mu_visc_i_integration(mean_mode, x_int, y_int)
            

            for i_mode in range(N_modes):
                mu_x_N_values[i_integration, i_mode+1],mu_y_N_values[i_integration, i_mode+1] =\
                    mu_visc_i_integration(modes[i_mode], x_int, y_int)
        return mu_x_N_values, mu_y_N_values
    
    mu_x_N_values, mu_y_N_values = convergence_mu_integration(values_N_integration)

    # Plot the evolution of the value with the nummber of integration points: 
    plt.figure(figsize=(20,20))
    plt.rc('font',size = 18)
    plt.title("Values of $\mu^i_x$ integrated for different numbers of integration points")
    for i_mode in range(N_modes +1):
        plt.semilogx(values_N_integration, mu_x_N_values[:,i_mode],"-o", label ="$\mu_x^{}$".format(i_mode))
    plt.xlabel("Number of integration points")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    plt.figure(figsize=(20,20))
    plt.rc('font',size = 18)
    plt.title("Values of $\mu^i_y$ integrated for different numbers of integration points")
    for i_mode in range(N_modes +1):
        plt.semilogx(values_N_integration, mu_y_N_values[:,i_mode],"-o", label ="$\mu_y^{}$".format(i_mode))
    plt.xlabel("Number of integration points")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

# Let's do the same for gammas 
    def convergence_gamma_integration(values_N_integration):
        gamma_x_N_values = np.zeros((len(values_N_integration),N_modes + 1))
        gamma_y_N_values = np.zeros((len(values_N_integration),N_modes + 1))

        for i_integration in range(len(values_N_integration)):
            N_integration = values_N_integration[i_integration]    
            theta_values = np.sort(np.random.rand(N_integration)*2*np.pi)
            x_int, y_int = nCond.r_c*np.cos(theta_values), nCond.r_c*np.sin(theta_values)

            print("Starting integration with {} points:".format(N_integration))
            # # Show the integration points
            # plt.figure(figsize=(10,10))
            # plt.rc('font',size=18)
            # plt.scatter(x_int, y_int, s=0.6)
            # plt.title("{} integration points".format(N_integration))
            # plt.show()

            gamma_x_N_values[i_integration, 0], gamma_y_N_values[i_integration, 0] =\
                gammas_POD_mode_i_PINN(mean_mode, x_int, y_int)
            

            for i_mode in range(N_modes):
                gamma_x_N_values[i_integration, i_mode+1],gamma_y_N_values[i_integration, i_mode+1] =\
                                    gammas_POD_mode_i_PINN(modes[i_mode], x_int, y_int)
        return gamma_x_N_values, gamma_y_N_values
    
    gamma_x_N_values, gamma_y_N_values = convergence_gamma_integration(values_N_integration)
    # Plot the evolution of the value with the nummber of integration points: 
    plt.figure(figsize=(20,20))
    plt.rc('font',size = 18)
    plt.title("Values of $\gamma^i_x$ integrated for different numbers of integration points")
    for i_mode in range(N_modes +1):
        plt.semilogx(values_N_integration, gamma_x_N_values[:,i_mode],"-o", label ="$\gamma_x^{}$".format(i_mode))
    plt.xlabel("Number of integration points")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    plt.figure(figsize=(20,20))
    plt.rc('font',size = 18)
    plt.title("Values of $\gamma^i_y$ integrated for different numbers of integration points")
    for i_mode in range(N_modes +1):
        plt.semilogx(values_N_integration, gamma_y_N_values[:,i_mode],"-o", label ="$\gamma_y^{}$".format(i_mode))
    plt.xlabel("Number of integration points")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # Verify that the forces can be reconstructed ===============================================
    import pandas as pd
    time_coeffs_fixed = pd.read_csv("saved\\8_modes\\fixed\\data\\time_coeffs_ext_8_modes.csv")


    # # TO MODIFY
    def F_pressure(time_coeffs):
        N_modes, N_t = np.shape(time_coeffs)
        gamma_bar_x, gamma_bar_y = gamma_x_N_values[-1,0],gamma_y_N_values[-1,0]  
        Fx_tot = gamma_bar_x * np.ones(N_t)
        Fy_tot = gamma_bar_y * np.ones(N_t)
        for i_mode in range(N_modes):
            gamma_i_x = gamma_x_N_values[-1,i_mode+1]
            gamma_i_y = gamma_y_N_values[-1,i_mode+1]
            Fx_tot += time_coeffs[i_mode] * gamma_i_x
            Fy_tot += time_coeffs[i_mode] * gamma_i_y
        return (Fx_tot, Fy_tot)


    def F_viscosity(time_coeffs):
        N_modes, N_t = np.shape(time_coeffs)
        mu_bar_x, mu_bar_y = mu_x_N_values[-1,0],mu_y_N_values[-1,0]  
        Fx_tot = mu_bar_x * np.ones(N_t)
        Fy_tot = mu_bar_y * np.ones(N_t)
        for i_mode in range(N_modes):
            mu_i_x = mu_x_N_values[-1,i_mode+1]
            mu_i_y = mu_y_N_values[-1,i_mode+1]
            Fx_tot += time_coeffs[i_mode] * mu_i_x
            Fy_tot += time_coeffs[i_mode] * mu_i_y
        return (Fx_tot, Fy_tot)


