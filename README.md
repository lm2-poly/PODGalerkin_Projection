# POD Galerkin Projection


This code was developed during Omar Tazi Labzour's Master thesis on combining PINNs and Reduced Order Model (ROM) methods to model Vortex-Induced Vibrations (VIV). The situation modeled is the case of a 2D laminar flow (Re = 100) on a vibrating cylinder. It is strongly recommended to refer to this thesis before using this code. The data used for reference and to train the PINNs comes from simulations M. Boudina (see Vortex-induced vibrations: a soft coral feeding strategy?, Boudina et al (2021))  and is available here: https://zenodo.org/record/5039610.

The main physical and numerical parameters are set in parameters.py, which defines a parameter class that is used in all the other scripts.

The POD folder holds the scripts to perform POD on the Data file from the CFD simulation.

The PINN models are already trained and can be loaded using the Keras library.

Auxiliary functions and postprocessing functions are defined in the corresponding files.

The matrices used in the ROMs are computed in the projected_matrices scripts and saved to be used in the ODE solver scripts.

The ODE_solver files correspond to the scripts where the projected equations are solved to predict the POD time coefficients describing the system. The suffix 1W and 2W respectively correspond to One-Way and Two-Way coupling models. The all_Ur suffix corresponds to solving the projected eauqtions for a range of reduced velocity (Ur) values, in order to plot the evolution of the maximum amlpitude of vibration predicted against Ur and validate that the model is able to capture the lock-in phenomenon.     
