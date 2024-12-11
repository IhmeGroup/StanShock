import os
from tqdm import tqdm
import numpy as np
from scipy import optimize, integrate, interpolate, special, stats
import cantera as ct

from  StanScram.fpv_table import FPVTable

datadir = "./data"

class JICModel():
    '''
    Class: JICModel
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the Jet-in-Crossflow model
    '''
    def __init__(self, gas, fuel,
                 x, x_inj, w, h, n_inj, d_inj,
                 rho_inj, u_inj, T_inj,
                 rho, u, T,
                 alpha,
                 time_delay=0.0,
                 fpv_table=None,
                 load_adjustment_factor=False,
                 load_Z_avg_var_profiles=False):
        '''
        Method: __init__
        --------------------------------------------------------------------------
        This method initializes the Jet-in-Crossflow model with the following
        parameters:
        gas: Cantera.Solution
            The Cantera gas object
        fuel: str
            The fuel species
        x: float
            The 1D mesh
        x_inj: float
            The x-coordinate of the injection point
        w: float
            The width of the domain
        h: float
            The height of the domain
        n_inj: float
            The number of injected jets
        d_inj: float
            The diameter of the injected jet
        rho_inj: float
            The density of the injected jet
        u_inj: float
            The velocity of the injected jet
        T_inj: float
            The temperature of the injected jet
        rho: float
            The density of the crossflow
        u: float
            The velocity of the crossflow
        T: float
            The temperature of the crossflow
        alpha: float
            The relaxation parameter (used here only for storage)
        time_delay: float
            The time delay for the start of the injection
        fpv_table: FPVTable
            The FPV table object, used for the chemical source terms
        load_adjustment_factor: bool
            Whether to load the adjustment factor from file
        load_Z_avg_var_profiles: bool
            Whether to load the Z average and variance profiles from file
        '''
        self.gas = gas
        self.fuel = fuel

        self.x = x
        self.x_inj = x_inj
        self.w = w
        self.h = h
        self.n_inj = n_inj
        self.d_inj = d_inj

        self.rho_inj = rho_inj
        self.u_inj = u_inj
        self.T_inj = T_inj
        self.rho = rho
        self.u = u
        self.T = T

        self.alpha = alpha
        self.time_delay = time_delay
        self.fpv_table = fpv_table

        # Geometry parameters
        self.A = self.w * self.h
        self.x_max = self.x[-1]

        # Other properties
        self.gas.TDX = self.T_inj, self.rho_inj, "{0}:1".format(self.fuel)
        self.p_inj = self.gas.P
        self.W_inj = self.gas.mean_molecular_weight
        self.gamma_inj = self.gas.cp / self.gas.cv
        self.c_inj = gas.sound_speed
        self.M_inj = self.u_inj / self.c_inj
        self.E_inj = self.gas.int_energy_mass + 0.5 * self.u_inj**2

        self.gas.TDX = self.T, self.rho, "O2:0.21,N2:0.79"
        self.p = self.gas.P
        self.W = self.gas.mean_molecular_weight
        self.gamma = self.gas.cp / self.gas.cv
        self.c = gas.sound_speed
        self.M = self.u / self.c

        # Position of the first injected fluid particle
        self.x_fluid_tip = self.x_inj

        self.__prep_zbilger()

        # Properties behind the bow shock (assuming normal shock)
        self.rho_2 = self.rho * (self.gamma + 1) * self.M**2 / ((self.gamma - 1) * self.M**2 + 2)
        self.u_2 = self.u * self.rho / self.rho_2

        # Integral of Yf across centerline normal plane
        A_inj = np.pi * (self.d_inj / 2.0)**2
        self.Yf_cl_int = self.rho_inj * self.u_inj * A_inj / (self.rho_2 * self.u_2)
        self.d_eff = np.sqrt(self.Yf_cl_int / (2 * np.pi))

        # Compute the non-dimensional parameters
        # self.r_u = np.sqrt(self.rho_inj / self.rho *
        #                    (self.u_inj / self.u)**2)
        self.r_u = self.u_inj / self.u
        self.r_W = self.W_inj / self.W
        self.J = (self.gamma_inj * self.p_inj * self.M_inj) / (self.gamma * self.p * self.M)
        
        # Create the array of injectors
        self.z_inj = np.linspace(-self.w/2, self.w/2, n_inj+2)[1:-1]

        # Precompute the adjustment factor for the boundary clipping
        if load_adjustment_factor:
            data = np.loadtxt(os.path.join(datadir, "adjustment_factor.csv"), delimiter=',')
            self.x_cl_arr = data[:, 0]
            self.y_cl_arr = data[:, 1]
            self.adjustment_factor_arr = data[:, 2]

            print("Building interpolators...")
            self.adjustment_factor_interp_x = interpolate.CubicSpline(self.x_cl_arr, self.adjustment_factor_arr)
            self.adjustment_factor_interp_y = interpolate.CubicSpline(self.y_cl_arr, self.adjustment_factor_arr)
        else:
            self.calc_adjustment_factor(write=True)
        
        # Precompute the axial mean and variance profiles of Z
        if load_Z_avg_var_profiles:
            data = np.loadtxt(os.path.join(datadir, "Z_avg_var_profiles.csv"), delimiter=',')
            self.Z_avg_profile = data[:, 0]
            self.Z_var_profile = data[:, 1]
        else:
            self.calc_Z_avg_var_profiles(write=True)
    
    def __prep_zbilger(self):
        #             2(Y_C - Yo_C)/W_C + (Y_H - Yo_H)/2W_H - (Y_O - Yo_O)/W_O
        # ZBilger =  -----------------------------------------------------------
        #            2(Yf_C - Yo_C)/W_C + (Yf_H - Yo_H)/2W_H - (Yf_O - Yo_O)/W_O
        has_C = "C" in self.gas.element_names

        i_C = self.gas.element_index("C") if has_C else 0
        i_H = self.gas.element_index("H")
        i_O = self.gas.element_index("O")

        W_C = self.gas.atomic_weight(i_C) if has_C else 1.0
        W_H = self.gas.atomic_weight(i_H)
        W_O = self.gas.atomic_weight(i_O)

        self.gas.TDX = self.T, self.rho, "O2:0.21,N2:0.79"
        Yo_C = self.gas.elemental_mass_fraction("C") if has_C else 0.0
        Yo_H = self.gas.elemental_mass_fraction("H")
        Yo_O = self.gas.elemental_mass_fraction("O")

        self.gas.TDX = self.T_inj, self.rho_inj, "{0}:1".format(self.fuel)
        Yf_C = self.gas.elemental_mass_fraction("C") if has_C else 0.0
        Yf_H = self.gas.elemental_mass_fraction("H")
        Yf_O = self.gas.elemental_mass_fraction("O")

        s = 1.0 / (  2.0 * (Yf_C - Yo_C) / W_C
                   + 0.5 * (Yf_H - Yo_H) / W_H
                   - 1.0 * (Yf_O - Yo_O) / W_O)
        
        self.Z_weights = np.zeros(self.gas.n_species)
        for k in range(self.gas.n_species):
            self.Z_weights[k] = (  2.0 * (self.gas.n_atoms(k, i_C) if has_C else 0.0)
                                 + 0.5 *  self.gas.n_atoms(k, i_H)
                                 - 1.0 *  self.gas.n_atoms(k, i_O)) / \
                                self.gas.molecular_weights[k]
        self.Z_offset = -(  2.0 * Yo_C / W_C
                          + 0.5 * Yo_H / W_H
                          - 1.0 * Yo_O / W_O)
        
        self.Z_weights *= s
        self.Z_offset *= s

        self.Z_weight_f = self.Z_weights[self.gas.species_index(self.fuel)]
        self.Z_weight_O2 = self.Z_weights[self.gas.species_index("O2")]
        self.Z_weight_N2 = self.Z_weights[self.gas.species_index("N2")]
    
    def y_cl(self, x_cl):
        # SUBSONIC VERSION
        # return self.d_inj * 1.6 * (x_cl / self.d_inj)**(1.0/3.0) * self.r_u**(2.0/3.0) # Torrez 2011 (Same as Margason 1968)
        # return 1.6 * x_cl**(1.0/3.0) * (self.d_inj * self.r_u)**(2.0/3.0) # Margason 1968
        return self.r_u * self.d_inj * 1.6 * (x_cl / (self.r_u * self.d_inj))**(1.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2
        # return self.d_inj * 0.527 * self.r_u**1.178 * (x_cl / self.d_inj)**0.314 # Karagozian 1986

        # SUPERSONIC VERSION
        # return self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**0.344 # Gruber 1997 Phys. Fluids
        # return self.d_inj * 2.173 / self.J**0.276 * (x_cl / self.d_inj)**0.281 # Rothstein and Wantuck 1992
    
    def x_cl_from_y_cl(self, y_cl):
        return (y_cl / (self.r_u * self.d_inj * 1.6))**(3.0) * self.r_u * self.d_inj # Hasselbrink and Mungal 2001 Pt. 2
    
    def dy_cl_dx(self, x_cl):
        return (self.r_u * self.d_inj)**(2.0/3.0) * 1.6 * (1.0/3.0) * x_cl**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2
    
    def __nearest_on_cl_single(self, x, y, dz=0.0):
        # Offset coordinate
        x_local = x - self.x_inj

        # Define the centerline
        n2_func_x_cl = lambda x_cl: (x_local - x_cl                     )**2 + (y - self.y_cl(x_cl))**2 + dz**2
        n2_func_y_cl = lambda y_cl: (x_local - self.x_cl_from_y_cl(y_cl))**2 + (y - y_cl           )**2 + dz**2

        # Compute the x_cl which minimizes n2
        x_cl = optimize.fminbound(n2_func_x_cl, 0.0, self.x_max, disp=False)
        y_cl = self.y_cl(x_cl)
        n2 = n2_func_x_cl(x_cl)

        if self.dy_cl_dx(x_cl) > 1:
            # Compute the y_cl which minimizes n2
            y_cl = optimize.fminbound(n2_func_y_cl, 0.0, self.h, disp=False)
            x_cl = self.x_cl_from_y_cl(y_cl)
            n2 = n2_func_y_cl(y_cl)

        return x_cl, y_cl, n2
    
    def __match_ndarray_shapes(self, *args):
        is_ndarray = [isinstance(arg, np.ndarray) for arg in args]
        if not any(is_ndarray):
            return args
        shape = np.shape(args[is_ndarray.index(True)])
        args_out = []
        for i in range(0, len(args)):
            if is_ndarray[i]:
                if np.shape(args[i]) != shape:
                    raise ValueError("Shapes do not match")
                args_out.append(args[i])
            else:
                args_out.append(np.full(shape, args[i]))
        return args_out
    
    def nearest_on_cl(self, x, y, dz=0.0):
        x_match, y_match, dz_match = self.__match_ndarray_shapes(x, y, dz)

        if isinstance(x_match, np.ndarray):
            x_flat = x_match.flatten()
            y_flat = y_match.flatten()
            dz_flat = dz_match.flatten()
            x_cl = np.zeros_like(x_flat)
            y_cl = np.zeros_like(x_flat)
            n2 = np.zeros_like(x_flat)
            for i in range(len(x_flat)):
                x_cl[i], y_cl[i], n2[i] = self.__nearest_on_cl_single(x_flat[i], y_flat[i], dz_flat[i])
            x_cl = x_cl.reshape(x_match.shape)
            y_cl = y_cl.reshape(x_match.shape)
            n2 = n2.reshape(x_match.shape)
            return x_cl, y_cl, n2
        else:
            return self.__nearest_on_cl_single(x, y, dz)
    
    def Yf_cl(self, x_cl):
        # Centerline mole fraction
        y_cl = self.y_cl(x_cl)
        xi = 0.4 * (1 / self.r_u) * (self.rho_inj / self.rho)**(0.5) * (y_cl / (self.r_u * self.d_inj))**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 1
        X_f = np.minimum(xi, 1.0)
        Y_f = X_f * self.W_inj / (X_f * self.W_inj + (1 - X_f) * self.W)
        return Y_f
    
    def calc_adjustment_factor(self, write=False):
        # Create array of points, L-shaped surrounding trajectory
        y_cl_max = self.y_cl(self.x_max - self.x_inj)
        x_arr = np.concatenate([
            np.full(20, self.x_inj),
            np.linspace(self.x_inj, self.x_max, 50)[1:]])
        y_arr = np.concatenate([
            np.linspace(0, y_cl_max, 20),
            np.full(50, y_cl_max)[1:]])
        self.x_cl_arr = np.zeros_like(x_arr)
        self.y_cl_arr = np.zeros_like(y_arr)
        print("Computing trajectory...")
        self.x_cl_arr, self.y_cl_arr, _ = self.nearest_on_cl(x_arr, y_arr)

        # Iterate over the centerline
        print("Computing adjustment factor...")
        self.adjustment_factor_arr = np.zeros_like(self.x_cl_arr)
        idx = (self.x_cl_arr > 0)
        self.adjustment_factor_arr[idx] = self.calc_adjustment_factor_xy(self.x_cl_arr[idx], self.y_cl_arr[idx])
        
        print("Building interpolators...")
        self.adjustment_factor_interp_x = interpolate.CubicSpline(self.x_cl_arr, self.adjustment_factor_arr)
        self.adjustment_factor_interp_y = interpolate.CubicSpline(self.y_cl_arr, self.adjustment_factor_arr)
        
        if write:
            print("Writing adjustment factor to file...")
            np.savetxt(os.path.join(datadir, "adjustment_factor.csv"),
                       np.stack((self.x_cl_arr, self.y_cl_arr, self.adjustment_factor_arr), axis=1),
                       delimiter=',')
    
    def calc_adjustment_factor_xy(self, x_cl, y_cl):
        # Compute the normal to the centerline
        dy_cl_dx = self.dy_cl_dx(x_cl)
        ds = np.stack([np.full_like(x_cl, 1.0), dy_cl_dx], axis=0)
        ds /= np.linalg.norm(ds, axis=0)
        dn = np.array([-ds[1], ds[0]])

        # Intersection of the normal with the top and bottom boundaries
        x_top = x_cl + dn[0] * (self.h - y_cl)
        x_bot = x_cl - dn[0] * (         y_cl)
        xi_lo = -np.sqrt((x_bot - x_cl)**2 + (     0 - y_cl)**2)
        xi_hi =  np.sqrt((x_top - x_cl)**2 + (self.h - y_cl)**2)
        
        
        Yf_int_nobound = self.n_inj * self.Yf_cl_int

        Yf_cl = self.Yf_cl(x_cl)
        sigma2 = self.Yf_cl_int / (2 * np.pi * Yf_cl)
        s2s = np.sqrt(2 * sigma2)

        Yf_int_bound = 0.0
        for z_inj in self.z_inj:
            Yf_int_bound += Yf_cl * (np.pi / 2) * sigma2 * \
                            (special.erf(xi_hi              / s2s) - special.erf(xi_lo               / s2s)) * \
                            (special.erf((self.w/2 - z_inj) / s2s) - special.erf((-self.w/2 - z_inj) / s2s))

        # max_xi_val = 0.01
        # xi_lo = max(xi_lo, -max_xi_val)
        # xi_hi = min(xi_hi, max_xi_val)
        # def integrand(z, xi):
        #     x = x_cl + xi * dn[0] + self.x_inj
        #     y = y_cl + xi * dn[1]
        #     # print("x = {0}, y = {1}, z = {2}".format(x, y, z))
        #     return self.Yf_3D(x, y, z)
        # Yf_int_bound = integrate.dblquad(
        #     integrand,
        #     xi_lo, xi_hi,
        #     lambda xi: -self.w/2,
        #     lambda xi: self.w/2)[0]

        fac = Yf_int_nobound / Yf_int_bound
        return fac
    
    def get_adjustment_factor(self, x, y):
        # if x <  self.x_inj - (self.d_eff / 2):
        #     return 1.0

        x_cl, y_cl, _ = self.nearest_on_cl(x, y)
        dy_cl_dx = self.dy_cl_dx(x_cl)
        idx_g1 = dy_cl_dx > 1
        fac = np.zeros_like(x_cl)
        fac[ idx_g1] = self.adjustment_factor_interp_x(x_cl[ idx_g1])
        fac[~idx_g1] = self.adjustment_factor_interp_y(y_cl[~idx_g1])
        return fac
    
    def Yf_3D(self, x, y, z):
        '''
        Method: Yf_3D
        --------------------------------------------------------------------------
        This method computes the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        Yf = 0.0
        for z_inj in self.z_inj:
            Yf += self.Yf_3D_single_inj(x, y, z, z_inj)
        return Yf
    
    def grad_Yf_3D(self, x, y, z):
        '''
        Method: grad_Yf_3D
        --------------------------------------------------------------------------
        This method computes the gradient of the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        grad_Yf = np.zeros([3] + list(y.shape))
        for z_inj in self.z_inj:
            grad_Yf += self.grad_Yf_3D_single_inj(x, y, z, z_inj)
        return grad_Yf
    
    def grad_Z_3D(self, x, y, z):
        '''
        Method: grad_Z_3D
        --------------------------------------------------------------------------
        This method computes the gradient of the Bilger mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        grad_Yf = self.grad_Yf_3D(x, y, z)
        grad_Ya = -grad_Yf
        grad_YO2 = 0.23291 * grad_Ya
        grad_YN2 = grad_Ya - grad_YO2
        grad_Z = (self.Z_weight_f  * grad_Yf +
                  self.Z_weight_O2 * grad_YO2 +
                  self.Z_weight_N2 * grad_YN2)
        return grad_Z
    
    def __Yf_to_Z(self, Yf):
        Ya = 1 - Yf
        YO2 = 0.23291 * Ya
        YN2 = Ya - YO2
        Z = (self.Z_weight_f  * Yf +
             self.Z_weight_O2 * YO2 +
             self.Z_weight_N2 * YN2) + self.Z_offset
        return np.minimum(np.maximum(Z, 0.0), 1.0)
    
    def Z_3D(self, x, y, z):
        '''
        Method: Z_3D
        --------------------------------------------------------------------------
        This method computes the Bilger mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        Yf = self.Yf_3D(x, y, z)
        return self.__Yf_to_Z(Yf)
    
    def Yf_3D_adjusted(self, x, y, z):
        '''
        Method: Yf_3D_adjusted
        --------------------------------------------------------------------------
        This method computes the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        Yf_adjusted = self.Yf_3D(x, y, z) * self.get_adjustment_factor(x, y)
        # return Yf_adjusted
        return np.minimum(Yf_adjusted, 1.0) # TODO: This cap introduces error in the integral. Distribute somehow?
    
    def grad_Yf_3D_adjusted(self, x, y, z):
        '''
        Method: grad_Yf_3D_adjusted
        --------------------------------------------------------------------------
        This method computes the gradient of the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        grad_Yf = self.grad_Yf_3D(x, y, z) * self.get_adjustment_factor(x, y)
        return grad_Yf
    
    def grad_Z_3D_adjusted(self, x, y, z):
        '''
        Method: grad_Z_3D_adjusted
        --------------------------------------------------------------------------
        This method computes the gradient of the Bilger mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        grad_Z = self.grad_Z_3D(x, y, z) * self.get_adjustment_factor(x, y)
        return grad_Z
    
    def Z_3D_adjusted(self, x, y, z):
        '''
        Method: Z_3D_adjusted
        --------------------------------------------------------------------------
        This method computes the Bilger mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        '''
        Yf = self.Yf_3D_adjusted(x, y, z)
        return self.__Yf_to_Z(Yf)
    
    def Yf_3D_single_inj(self, x, y, z, z_inj):
        '''
        Method: Yf_3D_single_inj
        --------------------------------------------------------------------------
        This method computes the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        z_inj: float
            The injector z-coordinate
        '''

        # Compute the nearest point on the centerline and the distance squared
        x_cl, y_cl, n2 = self.nearest_on_cl(x, y, z - z_inj)

        # If we haven't passed the edge of the injector, assume it's still a perfect cylinder
        # if x_cl < self.d_inj / 2:
        #     if n2 < (self.d_inj / 2)**2:
        #         return 1.0
        #     else:
        #         return 0.0

        # Compute the centerline fuel mass fraction
        Yf_cl = self.Yf_cl(x_cl)

        # Spreading scaling law based on Hasselbrink and Mungal 2001 Pt. 2 (u_rms)
        # b = self.d_inj * 0.76 * self.r_u**(2.0/3.0) * (x_cl / self.d_inj)**(1.0/3.0)
        # b = max(b, self.d_inj / 2.0)
        # Yf = Yf_cl * np.exp(-n2 / (2*b**2))

        # Spreading based on scalar conservation
        # (Assume gaussian, rho_inj * u_inj * Yf_inj * A_inj = rho * u * int(Yf * dA))
        # where int(Yf * dA) = 2 * pi * sigma^2 * Yf_cl
        sigma2 = self.Yf_cl_int / (Yf_cl * 2 * np.pi)
        Yf = Yf_cl * np.exp(-n2 / (2 * sigma2))

        return Yf
    
    def grad_Yf_3D_single_inj(self, x, y, z, z_inj):
        '''
        Method: grad_Yf_3D_single_inj
        --------------------------------------------------------------------------
        This method computes the gradient of the fuel mass fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        z_inj: float
            The injector z-coordinate
        '''

        # Compute the nearest point on the centerline and the distance squared
        x_cl, y_cl, n2 = self.nearest_on_cl(x, y, z - z_inj)

        # Compute the centerline fuel mass fraction
        Yf_cl = self.Yf_cl(x_cl)

        # Spreading scaling law based on Hasselbrink and Mungal 2001 Pt. 2 (u_rms)
        # b = self.d_inj * 0.76 * self.r_u**(2.0/3.0) * (x_cl / self.d_inj)**(1.0/3.0)
        # b = max(b, self.d_inj / 2.0)
        # Yf = Yf_cl * np.exp(-n2 / (2*b**2))

        # Spreading based on scalar conservation
        # (Assume gaussian, rho_inj * u_inj * Yf_inj * A_inj = rho * u * int(Yf * dA))
        # where int(Yf * dA) = 2 * pi * sigma^2 * Yf_cl
        sigma2 = self.Yf_cl_int / (Yf_cl * 2 * np.pi)
        grad_Yf = -Yf_cl * np.array([x - x_cl, y - y_cl, z - z_inj]) / (2 * sigma2) * np.exp(-n2 / (2 * sigma2))

        return grad_Yf

    def Z_avg(self, x):
        func = lambda z, y: self.Z_3D(x, y, z)
        return integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
    
    def Z_avg_adjusted(self, x):
        func = lambda z, y: self.Z_3D_adjusted(x, y, z)
        return integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
    
    def Z_avg_var(self, x):
        func = lambda z, y: self.Z_3D(x, y, z)
        Z_avg = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        func = lambda z, y: (self.Z_3D(x, y, z) - Z_avg)**2
        Z_var = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        return Z_avg, Z_var
    
    def Z_avg_var_adjusted(self, x):
        func = lambda z, y: self.Z_3D_adjusted(x, y, z)
        Z_avg = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        func = lambda z, y: (self.Z_3D_adjusted(x, y, z) - Z_avg)**2
        Z_var = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        return Z_avg, Z_var
    
    def calc_Z_avg_var_profiles(self, write=False):
        self.Z_avg_profile = np.zeros_like(self.x)
        self.Z_var_profile = np.zeros_like(self.x)
        print("Computing Z average and variance profiles...")
        for i in tqdm(range(len(self.x))):
            self.Z_avg_profile[i], self.Z_var_profile[i] = self.Z_avg_var_adjusted(self.x[i])
        
        if write:
            print("Writing Z average and variance profiles to file...")
            np.savetxt(os.path.join(datadir, "Z_avg_var_profiles.csv"),
                       np.stack((self.Z_avg_profile, self.Z_var_profile), axis=1),
                       delimiter=',')
    
    def estimate_p_Z(self, x, Z):
        '''
        Method: estimate_p_Z
        --------------------------------------------------------------------------
        This method estimates the PDF of the mixture fraction at a given point
        using a Beta distribution.
        x: float
            The query x-coordinate
        Z: float
            The query mixture fraction
        '''
        Z_avg, Z_var = self.Z_avg_var_adjusted(x)
        if Z_avg == 0.0:
            return 0.0
        a = ((Z_avg * (1 - Z_avg) / Z_var) - 1) * Z_avg
        b = a * (1 - Z_avg) / Z_avg
        return stats.beta.pdf(Z, a, b)
    
    def update_fluid_tip_position(self, dt, t, u):
        '''
        Method: update_fluid_tip_position
        --------------------------------------------------------------------------
        This method updates the position of the fluid tip based on the velocity
        of the fluid.
        t: float
            The current time
        dt: float
            The time step
        x: float
            The current x-coordinate of the fluid tip
        u: float
            The current velocity of the fluid
        '''
        if t < self.time_delay:
            return
        self.x_fluid_tip += dt * np.interp(self.x_fluid_tip, self.x, u)
    
    def get_injector_sources(self, rho, rhoU, E, rhoY, gamma, t):
        '''
        Method: get_injector_sources
        --------------------------------------------------------------------------
        This method computes a fuel injector source term to target the desired
        mixture fraction profile.
        '''
        rhs = np.zeros((rho.shape[0], 3+self.gas.n_species))
        if t < self.time_delay:
            return rhs

        Y = rhoY / np.tile(rho, (self.gas.n_species, 1)).T
        Z = np.sum(self.Z_weights * Y, axis=1) + self.Z_offset
        Z_target = self.Z_avg_profile
        mdot = rho * self.alpha * (Z_target - Z)

        # Compute the source term
        i_fuel = self.gas.species_index(self.fuel)
        rhs[:, 0         ] = mdot
        rhs[:, 1         ] = 0.0
        rhs[:, 2         ] = mdot * self.E_inj
        rhs[:, 3 + i_fuel] = mdot

        # Only apply source terms in regions that the fluid has reached
        rhs[self.x > self.x_fluid_tip, :] = 0.0

        return rhs
    
    def get_chemical_sources(self, C, t):
        '''
        Method: get_chemical_sources
        --------------------------------------------------------------------------
        This method computes the chemical source terms using the FPV table.
        C: float
            The progress variable
        '''
        omega_Y = np.zeros((len(C), self.gas.n_species))
        if t < self.time_delay:
            return omega_Y

        Z_probe = np.linspace(0.0, 1.0, 1000)
        # TODO ^ cluster these points around min and max Z

        for i in range(len(C)):
            Z_avg = self.Z_avg_profile[i]
            Z_var = self.Z_var_profile[i]

            if Z_avg == 0.0:
                continue

            a = ((Z_avg * (1 - Z_avg) / Z_var) - 1) * Z_avg
            b = a * (1 - Z_avg) / Z_avg
            Z_pdf = lambda Z: stats.beta.pdf(Z, a, b)
            p_Z = Z_pdf(Z_probe)
            L = self.fpv_table.L_from_C(Z_avg, C[i])
            
            omega_Y_k_probe = np.zeros_like(Z_probe)

            for k in range(self.gas.n_species):
                # Area integral
                # def integrand(z, y):
                #     src_name = "SRC_{0}".format(self.gas.species_name(k))
                #     Z = self.Z_3D_adjusted(self.x[i], y, z)
                #     L = self.fpv_table.L_from_C(Z, C[i])
                #     return self.fpv_table.lookup(src_name, Z, 0.0, L)
                # omega_Y[i,k] = integrate.dblquad(integrand,
                #                                  0, self.h,
                #                                  lambda y: -self.w/2, lambda y: self.w/2,
                #                                  epsabs=1e-2)[0] / (self.w * self.h)
                
                # Presume Beta PDF for Z
                # def integrand(Z):
                #     src_name = "SRC_{0}".format(self.gas.species_name(k))
                #     L = self.fpv_table.L_from_C(Z, C[i])
                #     p_Z = self.estimate_p_Z(self.x[i], Z)
                #     return self.fpv_table.lookup(src_name, Z, 0.0, L) * p_Z
                # omega_Y[i,k] = integrate.quad(integrand, 0.0, 1.0, epsabs=1e2)[0]

                # Manual approach with trapezoidal rule
                src_name = "SRC_{0}".format(self.gas.species_name(k))
                omega_Y_k_probe = self.fpv_table.lookup(src_name, Z_probe, 0.0, L)
                # PDF may have singularities at the boundaries, but the source term should be
                # zero there anyway
                p_Z[0] = 0.0
                p_Z[-1] = 0.0
                omega_Y_k_probe[0] = 0.0
                omega_Y_k_probe[-1] = 0.0
                omega_Y[i,k] = np.trapz(omega_Y_k_probe * p_Z, Z_probe)

        # Only apply source terms in regions that the fluid has reached
        omega_Y[self.x > self.x_fluid_tip, :] = 0.0

        return omega_Y
