import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Re_eff_values = [10 * i for i in range(1,11)]

cmap_eta = plt.cm.Blues
cmap_xi = plt.cm.Oranges

# Comparison with FSI results
FSI_sim_Ur_x = np.array([1.,2.5,3.,3.5,4.,4.5,5.5,6.5,7.,8.,9.,10.,11.,12.])

FSI_sim_Ur_y = np.array([1.,2.,4.,4.5,5.,6.,7.,8.,9.,10.,11.,12.,13.])

true_xi_max = np.array([0.006897,0.124138,0.636207,0.632759,0.615517,0.548276,0.448276,0.301724,0.093103,0.089655,0.082759,0.075862, 0.075862])
true_eta_max =  np.array([0.002619,0.004549,0.026328,0.045075,0.046729,0.04218,0.032256,0.021779,0.010338,0.003308,0.003308,0.003446,0.003033,0.003308])

# def F_total(time_coeffs):
plt.figure(figsize=(16,23))
plt.title("Variation of the in-line oscillation amplitude with the value of $Re_{eff}$", pad = 50)
# Plot the initial value for reference
plt.plot(FSI_sim_Ur_x, true_eta_max, 'k--', marker='o', markerfacecolor='none', markeredgecolor='k',  label = "FSI sim. results" )


path = "amplitude_data_Re_eff_{:.2f}"

for i_Re in range(10):
    Re_eff = Re_eff_values[i_Re]
    data_amp = pd.read_csv(path.format(Re_eff))
    if i_Re<10:
        plt.scatter(data_amp['ur'],data_amp['eta_max'],  marker = 's', label = "$Re_{eff}"+ " = {}$".format("{:.2f}".format(Re_eff_values[i_Re])), color = cmap_eta((i_Re*5+60)))    
    else:
        plt.plot(data_amp['ur'],data_amp['eta_max'],'-k',  marker = 's', label = "$Re_{eff}"+ " = {}$".format("{:.2f}".format(Re_eff_values[i_Re]))) 

plt.xlim(0.0,17)
plt.xlabel("$U_r$")
plt.ylabel("$\eta_{max}$")
plt.legend(fontsize=18, loc='upper right')
plt.show()