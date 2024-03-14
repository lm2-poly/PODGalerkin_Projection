"""This defines an object PhysicalConditions() that contains all the physical properties of the problem"""

class Parameters:
    def __init__(self):
        # Number of modes : 
        self.N_modes = 8
        
        # Physical parameters ------------------------------------------------------------------------------------------
        # Volumic mass
        self.rho = 1000.0 
        # Time domain
        self.t_min = 400.0
        self.t_max = 420.0
        self.t_max_ext = 800.0 # Maximum timestep in the extended time domain
        self.n_ext = 5000
        self.n_final = self.n_ext //4 # Number of points to examine at the end of the time domain 
        # Non dimensional numbers --------------------------------------------------------------------------------------
        # Reynolds number
        # self.Re = 100.0
        self.Re = 100.0
        # Non dimensional damping parameter 
        self.Zeta = 0.0
        # self.Zeta = 0.0
        # Mass number with added mass
        # self.Mass_number = 1.0
        self.Mass_number = 1.0
        # Reduced velocity
        self.Ur = 6.
        # Drag coefficient for a cylinder at Re = 100
        # self.C_D = 0.9732
        # Parameter to control if the program prints the precise advancement of the process
        self.verbose = False
        
        # Geometry studied ---------------------------------------------------------------------------------------------
        self.Lxmin =  -5.  # -40.
        self.Lxmax = 15.  # 120.
        self.Lx = self.Lxmax - self.Lxmin
        self.Lymin = -5.  # -60.
        self.Lymax =  5.  # 60.
        self.Ly = self.Lymax - self.Lymin
        self.x_c = 0.
        self.y_c = 0.
        self.r_c = 0.5
        self.D = 2*self.r_c
        self.geom = [self.Lxmin, self.Lxmax, self.Lymin, self.Lymax,self.x_c, self.y_c, self.r_c]
        self.epsilon_contour = 0.05 #0.05 # Defines the contour points around the cylinder
        self.N_cyl_integration = int(1e4) # Number of points used for integration on the cylinder surface using cumtrapz method
        self.norm_coeff = 1e-3
        self.include_viscosity = False
        self.include_added_damping_force = True

        
        self.saved_path = "saved\\{}_modes\\zeta_{:.2f}\\data\\Ur_{:.2f}\\".format(self.N_modes,self.Zeta,self.Ur)
        self.figures_path = "saved\\{}_modes\\zeta_{:.2f}\\figures\\Ur_{:.2f}\\".format(self.N_modes,self.Zeta,self.Ur)

    
    def get_geom(self):
        return self.geom
    
    def recap(self):
        print("===============================================")
        print("Parameters")
        print("===============================================")
        print("Number of modes :", self.N_modes)

        print("\nNon dimensional numbers :")
        print("Re :", self.Re)
        print("Ur :", self.Ur)
        print("Zeta :", self.Zeta)
        print("Mass number:", self.Mass_number)

        print("\nGeometry parameters")
        print("Lxmin :", self.Lxmin)
        print("Lxmax :", self.Lxmax)
        print("Lymin :", self.Lymin)
        print("Lymax :", self.Lymax)
        print("\nCylinder parameter")
        print("x_c :", self.x_c)
        print("y_c :", self.y_c)
        print("r_c :", self.r_c)
        print("rho :", self.rho)

        print("\nTime domain :")
        print("[t_min , t_max] =","[{} , {}]".format(self.t_min, self.t_max))