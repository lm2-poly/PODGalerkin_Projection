import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from parameters import Parameters

plt.rc('font',size=18)

FSI_sim_Ur_x = [1.,2.5,3.,3.5,4.,4.5,5.5,6.5,7.,8.,9.,10.,11.,12.]

FSI_sim_Ur_y = [1.,2.,4.,4.5,5.,6.,7.,8.,9.,10.,11.,12.,13.]

true_xi_max = [0.006897,0.124138,0.636207,0.632759,0.615517,0.548276,0.448276,0.301724,0.093103,0.089655,0.082759,0.075862, 0.075862]
true_eta_max =  [0.002619,0.004549,0.026328,0.045075,0.046729,0.04218,0.032256,0.021779,0.010338,0.003308,0.003308,0.003446,0.003033,0.003308]


param = Parameters()
path_results ="saved\\{}_modes\\zeta_{:.2f}\\data\\allUr_amplitudes_test_damping_coeff.csv".format(param.N_modes, param.Zeta)

res_dataframe = pd.read_csv(path_results)
res_dataframe = res_dataframe.sort_values('Ur')

Ur = res_dataframe['Ur']
eta_max = res_dataframe['eta max']
xi_max = res_dataframe['xi max']


f, ax = plt.subplots(1,2)
ax[0].plot(Ur, xi_max, '-s',color= 'orange', label = "$\\xi_{max}$ - ROM")
ax[0].plot(FSI_sim_Ur_y, true_xi_max,color='k', marker='o',fillstyle="none", linestyle='--',label = "$\\xi_{max}$ - FSI simulations")
ax[0].set_title("Amplitude of oscillation in the cross-flow direction",y=1.1, pad=-14,fontsize=17)
ax[0].set_xlabel("$U_r$")
ax[0].set_ylabel("$\\xi_{max}$")
ax[0].legend(loc='upper right', fontsize=12)

ax[1].plot(Ur, eta_max, '-s',color= 'blue', label = "$\eta_{max}$ - ROM")
ax[1].plot(FSI_sim_Ur_x, true_eta_max,color='k', marker='o',fillstyle="none",linestyle='--', label = "$\eta_{max}$ - FSI simulations")
ax[1].set_title("Amplitude of oscillation in the flow direction",y=1.1, pad=-14,fontsize=17)
ax[1].set_xlabel("$U_r$")
ax[1].set_ylabel("$\eta_{max}$")
ax[1].legend(loc='upper right', fontsize=12)
plt.tight_layout()



plt.show()