import os
from tqdm import tqdm
import pickle
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
                 x, x_inj, x_noz, w, h, n_inj, d_inj,
                 t_inj, rho_inj, u_inj, T_inj,
                 rho, u, T,
                 alpha,
                 fpv_table=None,
                 load_Z_avg_var_profiles=False,
                 load_chemical_sources=False):
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
        x_noz: float
            The x-coordinate of the nozzle start
        w: float
            The width of the domain
        h: float
            The height of the domain
        n_inj: float
            The number of injected jets
        d_inj: float
            The diameter of the injected jet
        t_inj: np.ndarray
            Time array for the injection profile
        rho_inj: np.ndarray
            The density of the injected jet, as a function of time
        u_inj: np.ndarray
            The velocity of the injected jet
        T_inj: np.ndarray
            The temperature of the injected jet
        rho: float
            The density of the crossflow
        u: float
            The velocity of the crossflow
        T: float
            The temperature of the crossflow
        alpha: float
            The relaxation parameter (used here only for storage)
        fpv_table: FPVTable
            The FPV table object, used for the chemical source terms
        load_Z_avg_var_profiles: bool
            Whether to load the Z average and variance profiles
        load_chemical_sources: bool
            Whether to load the chemical source terms
        '''
        self.gas = gas
        self.fuel = fuel

        self.x = x
        self.x_inj = x_inj
        self.x_noz = x_noz
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
        self.fpv_table = fpv_table

        # Geometry parameters
        self.A = self.w * self.h
        self.x_max = self.x[-1]
        self.A_inj = np.pi * (self.d_inj / 2.0)**2

        # Free stream properties
        self.gas.TDX = self.T, self.rho, "O2:0.21,N2:0.79"
        self.p = self.gas.P
        self.W = self.gas.mean_molecular_weight
        self.gamma = self.gas.cp / self.gas.cv
        self.c = gas.sound_speed
        self.M = self.u / self.c

        self.__prep_zbilger()

        # Properties of the injected fluid
        self.t_inj = t_inj
        self.p_inj = np.zeros_like(self.t_inj)
        for i in range(len(self.t_inj)):
            if np.isnan(self.rho_inj[i]):
                continue
            self.gas.TDX = self.T_inj, self.rho_inj[i], "{0}:1".format(self.fuel)
            self.p_inj[i] = self.gas.P
        self.E_inj = self.gas.int_energy_mass + 0.5 * self.u_inj**2
        self.W_inj = self.gas.mean_molecular_weight
        self.gamma_inj = self.gas.cp / self.gas.cv
        self.c_inj = gas.sound_speed
        self.M_inj = 1.0
        self.mdot_inj = self.rho_inj * self.u_inj * self.A_inj

        self.mdot_inj[np.isnan(self.mdot_inj)] = 0.0
        self.mdot_inj_unique, self.mdot_inj_unique_idx = np.unique(self.mdot_inj, return_index=True)
        self.mdot_inj_unique_idx = self.mdot_inj_unique_idx[np.argsort(self.mdot_inj_unique)]
        self.mdot_inj_unique = self.mdot_inj[self.mdot_inj_unique_idx]
        self.rho_inj_unique = self.rho_inj[self.mdot_inj_unique_idx]
        self.p_inj_unique = self.p_inj[self.mdot_inj_unique_idx]

        # Position of the first injected fluid particle
        self.fluid_tips = np.array([[self.x_inj, self.mdot_inj[0]]])

        # Properties behind the bow shock (assuming normal shock)
        # self.rho_2 = self.rho * (self.gamma + 1) * self.M**2 / ((self.gamma - 1) * self.M**2 + 2)
        # self.u_2 = self.u * self.rho / self.rho_2

        # Integral of Yf across centerline normal plane
        A_inj = np.pi * (self.d_inj / 2.0)**2
        self.Yf_cl_int = self.rho_inj_unique * self.u_inj * A_inj / (self.rho * self.u)
        self.d_eff = np.sqrt(self.Yf_cl_int / (2 * np.pi))

        # Compute the non-dimensional parameters
        self.r_u = self.u_inj / self.u
        self.r_W = self.W_inj / self.W
        self.J = (self.gamma_inj * self.p_inj * self.M_inj) / (self.gamma * self.p * self.M)
        # ^!!!! ASSUMES CHOKED FLOW
        
        # Create the array of injectors
        self.z_inj = np.linspace(-self.w/2, self.w/2, n_inj+2)[1:-1]

        # Precompute the adjustment factor for the boundary clipping
        self.calc_adjustment_factor()
        
        # Precompute the axial mean and variance profiles of Z
        if load_Z_avg_var_profiles:
            self.Z_avg_profile = np.load(os.path.join(datadir, "Z_avg_profile.npy"))
            self.Z_var_profile = np.load(os.path.join(datadir, "Z_var_profile.npy"))
        else:
            self.calc_Z_avg_var_profiles(write=True)
        
        # Precompute the mapping from mdot to Z mean and variance profiles
        self.Z_avg_profile_interp = interpolate.RegularGridInterpolator((self.mdot_inj_unique, self.x),
                                                                        self.Z_avg_profile)
        self.Z_var_profile_interp = interpolate.RegularGridInterpolator((self.mdot_inj_unique, self.x),
                                                                        self.Z_var_profile)
        
        # Precompute and tabulate chemical source terms
        if load_chemical_sources:
            L_probe = np.load(os.path.join(datadir, "L_probe.npy"))
            omega_C = np.load(os.path.join(datadir, "omega_C.npy"))
            self.omega_C_interpolators = []
            for i in range(len(self.x)):
                omega_C_i = interpolate.RegularGridInterpolator((self.mdot_inj_unique, L_probe),
                                                                omega_C[i])
                self.omega_C_interpolators.append(omega_C_i)
        else:
            self.calc_chemical_sources(write=True)
   
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

        self.gas.X = "O2:0.21,N2:0.79"
        Yo_C = self.gas.elemental_mass_fraction("C") if has_C else 0.0
        Yo_H = self.gas.elemental_mass_fraction("H")
        Yo_O = self.gas.elemental_mass_fraction("O")

        self.gas.X = "{0}:1".format(self.fuel)
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
        # SUBSONIC VERSION
        return (y_cl / (self.r_u * self.d_inj * 1.6))**(3.0) * self.r_u * self.d_inj # Hasselbrink and Mungal 2001 Pt. 2

        # SUPERSONIC VERSION
        # return (y_cl / (self.d_inj * self.J * 1.20))**(1.0 / 0.344) * self.d_inj * self.J - self.d_inj/2 # Gruber 1997 Phys. Fluids
    
    def dy_cl_dx(self, x_cl):
        # SUBSONIC VERSION
        return (self.r_u * self.d_inj)**(2.0/3.0) * 1.6 * (1.0/3.0) * x_cl**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2

        # SUPERSONIC VERSION
        # return self.d_inj * self.J * 1.20 * 0.344 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**(0.344 - 1.0) # Gruber 1997 Phys. Fluids
    
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
        if isinstance(x_cl, np.ndarray):
            rho_inj_unique = self.rho_inj_unique[:, np.newaxis]
        else:
            rho_inj_unique = self.rho_inj_unique

        # Centerline mole fraction
        y_cl = self.y_cl(x_cl)
        xi = 0.4 * (1 / self.r_u) * (rho_inj_unique / self.rho)**(0.5) * (y_cl / (self.r_u * self.d_inj))**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 1
        X_f = np.minimum(xi, 1.0)
        Y_f = X_f * self.W_inj / (X_f * self.W_inj + (1 - X_f) * self.W)
        return Y_f
    
    def calc_adjustment_factor(self):
        # Create array of points, L-shaped surrounding trajectory
        y_cl_max = self.y_cl(self.x_max - self.x_inj)
        x_arr = np.concatenate([
            np.full(20, self.x_inj),
            np.linspace(self.x_inj, self.x_max, 50)[1:]])
        y_arr = np.concatenate([
            np.linspace(0, y_cl_max, 20),
            np.full(50, y_cl_max)[1:]])
        x_cl_arr = np.zeros_like(x_arr)
        y_cl_arr = np.zeros_like(y_arr)
        print("Computing trajectory...")
        x_cl_arr, y_cl_arr, _ = self.nearest_on_cl(x_arr, y_arr)

        # Iterate over the centerline
        print("Computing adjustment factor...")
        adjustment_factor_arr = np.zeros([len(self.mdot_inj_unique), len(x_cl_arr)])
        idx = (x_cl_arr > 0)
        adjustment_factor_arr[:, idx] = self.calc_adjustment_factor_xy(x_cl_arr[idx], y_cl_arr[idx])
        adjustment_factor_arr[np.isnan(adjustment_factor_arr)] = 1.0
        
        print("Building interpolators...")
        self.adjustment_factor_interp_x = interpolate.CubicSpline(x_cl_arr, adjustment_factor_arr, axis=1)
        self.adjustment_factor_interp_y = interpolate.CubicSpline(y_cl_arr, adjustment_factor_arr, axis=1)
    
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
        
        Yf_int_nobound = self.n_inj * self.Yf_cl_int[:, np.newaxis]

        Yf_cl = self.Yf_cl(x_cl)
        sigma2 = self.Yf_cl_int[:, np.newaxis] / (2 * np.pi * Yf_cl)
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
        if np.isscalar(x):
            x_arr = np.array([x])
            y_arr = np.array([y])

        x_cl, y_cl, _ = self.nearest_on_cl(x_arr, y_arr)
        dy_cl_dx = self.dy_cl_dx(x_cl)
        idx_g1 = dy_cl_dx > 1
        fac = np.zeros([len(self.mdot_inj_unique), len(x_cl)])
        fac[:,  idx_g1] = self.adjustment_factor_interp_x(x_cl[ idx_g1])
        fac[:, ~idx_g1] = self.adjustment_factor_interp_y(y_cl[~idx_g1])

        if np.isscalar(x):
            return fac.T[0]
        else:
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

    def Z_avg_var(self, x):
        Z_avg = np.zeros_like(self.mdot_inj_unique)
        Z_var = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            if np.isnan(self.rho_inj_unique[i_m]):
                Z_avg[i_m] = 0.0
                Z_var[i_m] = 0.0
                continue

            func = lambda z, y: self.Z_3D(x, y, z)[i_m]
            Z_avg[i_m] = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
            func = lambda z, y: (self.Z_3D(x, y, z)[i_m] - Z_avg[i_m])**2
            Z_var[i_m] = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        return Z_avg, Z_var
    
    def Z_avg_var_adjusted(self, x):
        Z_avg = np.zeros_like(self.mdot_inj_unique)
        Z_var = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            if np.isnan(self.rho_inj_unique[i_m]):
                Z_avg[i_m] = 0.0
                Z_var[i_m] = 0.0
                continue

            func = lambda z, y: self.Z_3D_adjusted(x, y, z)[i_m]
            Z_avg[i_m] = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
            func = lambda z, y: (self.Z_3D_adjusted(x, y, z)[i_m] - Z_avg[i_m])**2
            Z_var[i_m] = integrate.dblquad(func, 0, self.h, lambda y: -self.w/2, lambda y: self.w/2)[0] / (self.w * self.h)
        return Z_avg, Z_var
    
    def calc_Z_avg_var_profiles(self, write=False):
        self.Z_avg_profile = np.zeros([len(self.mdot_inj_unique), len(self.x)])
        self.Z_var_profile = np.zeros([len(self.mdot_inj_unique), len(self.x)])
        print("Computing Z average and variance profiles...")
        for i in tqdm(range(len(self.x))):
            if self.x[i] > self.x_noz:
                # Freeze the profiles in the nozzle
                self.Z_avg_profile[:, i] = self.Z_avg_profile[:, i-1]
                self.Z_var_profile[:, i] = self.Z_var_profile[:, i-1]
            else:
                self.Z_avg_profile[:, i], self.Z_var_profile[:, i] = self.Z_avg_var_adjusted(self.x[i])
        
        if write:
            np.save(os.path.join(datadir, "Z_avg_profile.npy"), self.Z_avg_profile)
            np.save(os.path.join(datadir, "Z_var_profile.npy"), self.Z_var_profile)
    
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
    
    def update_fluid_tip_positions(self, dt, t, u):
        '''
        Method: update_fluid_tip_positions
        --------------------------------------------------------------------------
        This method updates the position of the fluid tips based on the velocity
        of the fluid.
        t: float
            The current time
        dt: float
            The time step
        x: float
            The current x-coordinate of the fluid tips
        u: float
            The current velocity of the fluid
        '''
        # Update the fluid tip positions
        self.fluid_tips[:, 0] += dt * np.interp(self.fluid_tips[:, 0], self.x, u)
        
        # Emit a new fluid tip
        mdot = np.interp(t, self.t_inj, self.mdot_inj)
        next_tip = np.array([[self.x_inj, mdot]])
        self.fluid_tips = np.concatenate([self.fluid_tips, next_tip], axis=0)

        # Drop fluid tips that have passed the end of the domain
        self.fluid_tips = self.fluid_tips[self.fluid_tips[:, 0] < self.x_max]

    def get_injector_sources(self, rho, rhoU, E, rhoZ, rhoC, gamma, t):
        '''
        Method: get_injector_sources
        --------------------------------------------------------------------------
        This method computes a fuel injector source term to target the desired
        mixture fraction profile.
        '''
        rhs = np.zeros((rho.shape[0], 5))

        Z = rhoZ / rho
        last_fluid_tip = len(self.fluid_tips) - np.searchsorted(self.fluid_tips[::-1, 0], self.x, side='right') - 1
        mdot_profile = np.zeros_like(self.x)
        mdot_profile[last_fluid_tip > 0] = self.fluid_tips[last_fluid_tip[last_fluid_tip > 0], 1]
        Z_target = self.Z_avg_profile_interp((mdot_profile, self.x))
        mdot = rho * self.alpha * (Z_target - Z)

        # Treat small (pre-injector) values
        mdot[Z_target < 1e-6] = 0.0

        # Compute the source term
        rhs[:, 0] = mdot
        rhs[:, 1] = 0.0
        rhs[:, 2] = mdot * self.E_inj
        rhs[:, 3] = mdot
        rhs[:, 4] = 0.0

        # DEBUG: Add some progress variable to trigger ignition
        # i_ig = np.logical_and(self.x > self.x_inj, self.x < self.x_inj + 0.01)
        # if (t < 1e-4):
        #     rhs[i_ig, 4] = 1e1

        return rhs
    
    def calc_chemical_sources(self, write=False):
        '''
        Method: calc_chemical_sources
        --------------------------------------------------------------------------
        This method precomputes the chemical source terms as a function of x and L.
        '''
        print("Precomputing chemical sources...")
        self.omega_C_interpolators = []

        Z_probe = np.linspace(0.0, 1.0, 1000)
        # TODO ^ cluster these points around min and max Z
        L_probe = np.linspace(0.0, 1.0, 200)
        Z_mesh_ZL, L_mesh_ZL = np.meshgrid(Z_probe, L_probe, indexing='ij')

        if write:
            np.save(os.path.join(datadir, "L_probe.npy"), L_probe)
            omega_C = np.zeros((len(self.x),
                                len(self.mdot_inj_unique),
                                len(L_probe)))

        for i in tqdm(range(len(self.x))):
            Z_avg = self.Z_avg_profile[:, i]
            Z_var = self.Z_var_profile[:, i]
            omega_C_i = []

            if self.x[i] > self.x_noz:
                # Freeze the chemistry in the nozzle
                omega_C_i = lambda mdot, L : 0.0
                if write:
                    omega_C[i, :, :] = 0.0
            elif np.all(Z_avg == 0.0):
                # No fuel, don't bother calculating
                omega_C_i = lambda mdot, L : 0.0
                if write:
                    omega_C[i, :, :] = 0.0
            else:
                a = ((Z_avg * (1 - Z_avg) / Z_var) - 1) * Z_avg
                b = a * (1 - Z_avg) / Z_avg
                p_Z = stats.beta.pdf(np.tile(Z_probe, (len(self.mdot_inj_unique), 1)),
                                     np.tile(a, (len(Z_probe), 1)).T,
                                     np.tile(b, (len(Z_probe), 1)).T)
                omega_C_probe = self.fpv_table.lookup("SRC_PROG", Z_mesh_ZL, 0.0, L_mesh_ZL) # [1/s]
                # PDF may have singularities at the boundaries, but the source term should be
                # zero there anyway
                p_Z[:,  0] = 0.0
                p_Z[:, -1] = 0.0
                omega_C_probe[ 0, :] = 0.0
                omega_C_probe[-1, :] = 0.0
                omega_C_probe_mdots = np.zeros((len(self.mdot_inj_unique), len(L_probe)))
                for i_m in range(len(self.mdot_inj_unique)):
                    omega_C_probe_mdots[i_m, :] = np.trapz(omega_C_probe * p_Z[i_m, :].reshape((-1, 1)),
                                                           Z_probe, axis=0)
                omega_C_probe_mdots[np.isnan(omega_C_probe_mdots)] = 0.0
                omega_C_i = interpolate.RegularGridInterpolator((self.mdot_inj_unique, L_probe),
                                                                omega_C_probe_mdots)
                if write:
                    omega_C[i, :, :] = omega_C_probe_mdots

            self.omega_C_interpolators.append(omega_C_i)
        
        if write:
            np.save(os.path.join(datadir, "omega_C.npy"), omega_C)
   
    def get_chemical_sources(self, Z, C):
        '''
        Method: get_chemical_sources
        --------------------------------------------------------------------------
        This method computes the chemical source terms [1/s] using the FPV table.
        Z: float
            The array of mixture fraction values at different grid points
        C: float
            The array of progress variable values at different grid points
        t: float
            The current time
        '''
        omega_C = np.zeros(len(self.x))
        
        for i_x in range(len(self.x)):
            i_last_tip = np.argmax(self.fluid_tips[:, 0] >= self.x[i_x]) # Note that fluid_tips is descending
            L = self.fpv_table.L_from_C(Z[i_x], C[i_x])
            omega_C[i_x] = self.omega_C_interpolators[i_x]((self.fluid_tips[i_last_tip, 1], L))
        
        return omega_C
    