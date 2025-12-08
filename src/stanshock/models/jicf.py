from __future__ import annotations

import functools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import integrate, interpolate, optimize, special, stats
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import Box

XSMALL_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": SMALL_SIZE,
        "axes.titlesize": SMALL_SIZE,
        "axes.labelsize": MEDIUM_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": XSMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
    }
)

datadir = Path("./data")


class JICModel(RightHandSide):
    """
    This is a class defined to encapsulate the Jet-in-Crossflow model
    """

    def __init__(
        self,
        fuel: str,
        x_inj: float,
        x_noz: float,
        n_inj: float,
        d_inj: float,
        t_inj: Array,
        rho_inj: Array,
        u_inj: Array,
        T_inj: Array,
        rho: float,
        u: float,
        T: float,
        alpha: float,
        load_Z_3D: bool = False,
        load_Z_avg_var_profiles: bool = False,
        load_chemical_sources: bool = False,
        load_MIB_profile: bool = False,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        """
        This method initializes the Jet-in-Crossflow model with the following
        parameters:
        fuel: str
            The fuel species
        x_inj: float
            The x-coordinate of the injection point
        x_noz: float
            The x-coordinate of the nozzle start
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
        load_Z_3D: bool
            Whether to load the 3D Z table
        load_Z_avg_var_profiles: bool
            Whether to load the Z average and variance profiles
        load_chemical_sources: bool
            Whether to load the chemical source terms
        load_MIB_profile: bool
            Whether to load the MIB profile
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        """
        super().__init__(**precompute_steps)
        assert isinstance(self.geometry, Box)
        assert isinstance(self.physics, FPVTable)
        self.fpv_table: FPVTable = self.physics

        gas = self.physics.gas
        self.fuel = fuel

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.idx_input]
        self.cell_volumes: Array | None = None
        if self.geometry.dlnA_dt is None:
            self.cell_volumes = self.geometry.volume()[:, None]

        self.x_inj = x_inj
        self.x_noz = x_noz
        self.w = float(self.geometry.w(0.0, np.array(self.x_inj)))
        self.h = float(self.geometry.h(0.0, np.array(self.x_inj)))
        self.n_inj = n_inj
        self.d_inj = d_inj

        self.rho_inj = rho_inj
        self.u_inj = u_inj
        self.T_inj = T_inj
        self.rho = rho
        self.u = u
        self.T = T

        self.alpha = alpha

        # Geometry parameters
        self.L = self.xc[-1] - self.xc[0]
        self.A = self.w * self.h
        self.A_inj = np.pi * (self.d_inj / 2.0) ** 2

        # Free stream properties
        gas.TDX = self.T, self.rho, "O2:0.21,N2:0.79"
        self.p = gas.P
        self.W = gas.mean_molecular_weight
        self.gamma = gas.cp / gas.cv
        self.c = gas.sound_speed
        self.M = self.u / self.c

        # Properties of the injected fluid
        self.t_inj = t_inj
        self.p_inj = np.zeros_like(self.t_inj)
        for i in range(len(self.t_inj)):
            if np.isnan(self.rho_inj[i]):
                continue
            gas.TDX = self.T_inj, self.rho_inj[i], f"{self.fuel}:1"
            self.p_inj[i] = gas.P
        self.E_inj = gas.int_energy_mass + 0.5 * self.u_inj**2
        self.W_inj = gas.mean_molecular_weight
        self.gamma_inj = gas.cp / gas.cv
        self.c_inj = gas.sound_speed
        self.M_inj = 1.0
        self.mdot_inj = self.n_inj * self.rho_inj * self.u_inj * self.A_inj
        self.mdot_inj[np.isnan(self.mdot_inj)] = 0.0
        self.mdot_inj_unique, self.mdot_inj_unique_idx = np.unique(
            self.mdot_inj, return_index=True
        )
        self.mdot_inj_unique_idx = self.mdot_inj_unique_idx[
            np.argsort(self.mdot_inj_unique)
        ]
        self.mdot_inj_unique = self.mdot_inj[self.mdot_inj_unique_idx]
        self.rho_inj_unique = self.rho_inj[self.mdot_inj_unique_idx]
        self.p_inj_unique = self.p_inj[self.mdot_inj_unique_idx]

        # Position of the first injected fluid particle
        self.fluid_tips = np.array([[self.x_inj, self.mdot_inj[0]]])

        # Properties behind the bow shock (assuming normal shock)
        # self.rho_2 = self.rho * (self.gamma + 1) * self.M**2 / ((self.gamma - 1) * self.M**2 + 2)
        # self.u_2 = self.u * self.rho / self.rho_2

        # Integral of Z across centerline normal plane
        A_inj = np.pi * (self.d_inj / 2.0) ** 2
        self.Z_cl_int = self.rho_inj_unique * self.u_inj * A_inj / (self.rho * self.u)
        self.d_eff = np.sqrt(self.Z_cl_int / (2 * np.pi))

        # Stoichiometry
        mdot_a = self.rho * self.u * self.A
        mdot_f = self.mdot_inj
        mdot_f_unique = self.mdot_inj_unique
        self.phi_gl_unique = np.zeros_like(self.mdot_inj_unique)
        self.Z_gl_unique = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            gas.TDY = (
                self.T,
                self.rho,
                f"O2:{0.233 * mdot_a},N2:{0.767 * mdot_a},{self.fuel}:{mdot_f_unique[i_m]}",
            )
            self.phi_gl_unique[i_m] = gas.equivalence_ratio(
                self.fuel, "O2:0.21,N2:0.79"
            )
            self.Z_gl_unique[i_m] = gas.mixture_fraction(self.fuel, "O2:0.21,N2:0.79")
        self.mdot_f_interp = interpolate.interp1d(
            self.t_inj, mdot_f, bounds_error=False, fill_value=0.0
        )

        # Compute the non-dimensional parameters
        self.J = (self.rho_inj * self.u_inj**2) / (
            self.rho * self.u**2
        )  # Momentum flux ratio
        self.J_unique = self.J[self.mdot_inj_unique_idx]
        self.r_u = np.sqrt(self.J)  # Blowing ratio = sqrt(J)
        self.r_u_unique = np.sqrt(self.J_unique)
        self.r_W = self.W_inj / self.W  # Molecular weight ratio

        # Create the array of injectors
        self.z_inj = np.linspace(-self.w / 2, self.w / 2, n_inj + 2)[1:-1]

        # Precompute the adjustment factor for the boundary clipping
        self.calc_adjustment_factor()

        # Precompute a 3D array of the mixture fraction and generate an interpolator
        if load_Z_3D:
            self.x_3D_data = np.load(datadir / "Z_3D_x.npy")
            self.y_3D_data = np.load(datadir / "Z_3D_y.npy")
            self.z_3D_data = np.load(datadir / "Z_3D_z.npy")
            self.Z_3D_data = np.load(datadir / "Z_3D.npy")
            self.Z_3D_interp = []
            for i_m in range(len(self.mdot_inj_unique)):
                interp = interpolate.RegularGridInterpolator(
                    (self.x_3D_data, self.y_3D_data, self.z_3D_data),
                    self.Z_3D_data[i_m],
                    method="cubic",
                )
                self.Z_3D_interp.append(interp)
        else:
            self.calc_Z_3D_interp(write=True)

        # Precompute the axial mean and variance profiles of Z
        if load_Z_avg_var_profiles:
            self.Z_avg_profile = np.load(datadir / "Z_avg_profile.npy")
            self.Z_var_profile = np.load(datadir / "Z_var_profile.npy")
        else:
            self.calc_Z_avg_var_profiles(write=True)

        # Precompute the mapping from mdot to Z mean and variance profiles
        self.Z_avg_profile_interp = interpolate.RegularGridInterpolator(
            (self.mdot_inj_unique, self.xc), self.Z_avg_profile
        )
        self.Z_var_profile_interp = interpolate.RegularGridInterpolator(
            (self.mdot_inj_unique, self.xc), self.Z_var_profile
        )

        # Precompute and tabulate chemical source terms
        if load_chemical_sources:
            self.Zbar_vec = np.load(datadir / "Zbar_vec.npy")
            self.Lbar_vec = np.load(datadir / "Lbar_vec.npy")
            self.logsigma2_vec = np.load(datadir / "logsigma2_vec.npy")
            self.omega_C_int = np.load(datadir / "omega_C_int.npy")
            self.omega_C_int_interp = interpolate.RegularGridInterpolator(
                (self.Zbar_vec, self.Lbar_vec, self.logsigma2_vec), self.omega_C_int
            )
        else:
            self.calc_chemical_sources(write=True)

        # Calculate the progress variable and chemical energy profiles for the
        # mixed is burned (MIB) model
        if load_MIB_profile:
            self.C_profile = np.load(datadir / "C_profile_MIB.npy")
            self.E_CHEM_profile = np.load(datadir / "E_CHEM_profile_MIB.npy")
        else:
            self.calc_MIB_profile(write=True)

    def y_cl(self, x_cl):
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return self.d_inj * 1.6 * (x_cl / self.d_inj)**(1.0/3.0) * self.r_u**(2.0/3.0) # Torrez 2011 (Same as Margason 1968)
        # return 1.6 * x_cl**(1.0/3.0) * (self.d_inj * self.r_u)**(2.0/3.0) # Margason 1968
        # return self.r_u * self.d_inj * 1.6 * (x_cl / (self.r_u * self.d_inj))**(1.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2
        # return self.d_inj * 0.527 * self.r_u**1.178 * (x_cl / self.d_inj)**0.314 # Karagozian 1986

        # SONIC VERSION
        return (
            self.d_inj
            * self.J_unique
            * 1.23
            * (x_cl / (self.d_inj * self.J_unique)) ** 0.344
        )  # Gruber 1995 JPP
        # return self.d_inj * self.J_unique * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J_unique))**0.344 # Gruber 1997 Phys. Fluids
        # return self.d_inj * 2.173 / self.J_unique**0.276 * (x_cl / self.d_inj)**0.281 # Rothstein and Wantuck 1992

    def x_cl_from_y_cl(self, y_cl):
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (y_cl / (self.r_u * self.d_inj * 1.6))**(3.0) * self.r_u * self.d_inj # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        return (
            (y_cl / (self.d_inj * self.J_unique * 1.23)) ** (1.0 / 0.344)
            * self.d_inj
            * self.J_unique
        )  # Gruber 1995 JPP
        # return (y_cl / (self.d_inj * self.J_unique * 1.20))**(1.0 / 0.344) * self.d_inj * self.J_unique - self.d_inj/2 # Gruber 1997 Phys. Fluids

    def dy_cl_dx(self, x_cl):
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (self.r_u * self.d_inj)**(2.0/3.0) * 1.6 * (1.0/3.0) * x_cl**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        return (
            0.344
            * self.d_inj
            * self.J_unique
            * 1.23
            * (x_cl / (self.d_inj * self.J_unique)) ** (0.344 - 1.0)
            * (1.0 / (self.d_inj * self.J_unique))
        )  # Gruber 1995 JPP
        # return (0.344 *
        #         self.d_inj * self.J_unique * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J_unique))**(0.344 - 1.0) *
        #         (1.0 / (self.d_inj * self.J_unique))) # Gruber 1997 Phys. Fluids

    def __nearest_on_cl_single(self, x, y, dz, i_m):
        # Offset coordinate
        x_local = x - self.x_inj

        # Define the centerline
        # Note: dz makes no difference in the minimization, but it's more convenient to include it here
        # so that the n2 is correct
        def n2_func_x_cl(x_cl):
            return (x_local - x_cl) ** 2 + (y - self.y_cl(x_cl)[i_m]) ** 2 + dz**2

        def n2_func_y_cl(y_cl):
            return (
                (x_local - self.x_cl_from_y_cl(y_cl)[i_m]) ** 2
                + (y - y_cl) ** 2
                + dz**2
            )

        # Compute the x_cl which minimizes n2
        x_cl = optimize.fminbound(n2_func_x_cl, self.xc[0], self.xc[-1], disp=False)
        y_cl = self.y_cl(x_cl)[i_m]
        n2 = n2_func_x_cl(x_cl)

        if self.dy_cl_dx(x_cl)[i_m] > 1:
            # Compute the y_cl which minimizes n2
            y_cl = optimize.fminbound(n2_func_y_cl, 0.0, self.h, disp=False)
            x_cl = self.x_cl_from_y_cl(y_cl)[i_m]
            n2 = n2_func_y_cl(y_cl)

        return x_cl, y_cl, n2

    def __match_ndarray_shapes(self, *args):
        is_ndarray = [isinstance(arg, np.ndarray) for arg in args]
        if not any(is_ndarray):
            return args
        shape = np.shape(args[is_ndarray.index(True)])
        args_out = []
        for i in range(len(args)):
            if is_ndarray[i]:
                if np.shape(args[i]) != shape:
                    msg = "Shapes do not match"
                    raise ValueError(msg)
                args_out.append(args[i])
            else:
                args_out.append(np.full(shape, args[i]))
        return args_out

    def nearest_on_cl(self, x, y, dz, i_m):
        x_match, y_match, dz_match = self.__match_ndarray_shapes(x, y, dz)

        if isinstance(x_match, np.ndarray):
            x_flat = x_match.flatten()
            y_flat = y_match.flatten()
            dz_flat = dz_match.flatten()
            x_cl = np.zeros_like(x_flat)
            y_cl = np.zeros_like(x_flat)
            n2 = np.zeros_like(x_flat)
            for i in range(len(x_flat)):
                x_cl[i], y_cl[i], n2[i] = self.__nearest_on_cl_single(
                    x_flat[i], y_flat[i], dz_flat[i], i_m
                )
            x_cl = x_cl.reshape(x_match.shape)
            y_cl = y_cl.reshape(x_match.shape)
            n2 = n2.reshape(x_match.shape)
            return x_cl, y_cl, n2
        return self.__nearest_on_cl_single(x, y, dz)

    def Z_cl(self, x_cl):
        # if isinstance(x_cl, np.ndarray):
        #     rho_inj_unique = self.rho_inj_unique[:, np.newaxis]
        #     r_u_unique = self.r_u_unique[:, np.newaxis]
        # else:
        #     rho_inj_unique = self.rho_inj_unique
        #     r_u_unique = self.r_u_unique

        # Following Torrez 2011
        # xi = 0.85 * ((self.rho_inj_unique / self.rho) * (self.u / self.u_inj) * (self.d_inj / x_cl)**2)**(1.0/3.0)
        # Z = xi * self.r_W / (1 + (self.r_W - 1) * xi)

        # Taking Z directly
        Z = (
            0.85
            * (1 / self.r_u_unique)
            * (self.rho_inj_unique / self.rho) ** (0.5)
            * (x_cl / (self.r_u_unique * self.d_inj)) ** (-2.0 / 3.0)
        )  # Hasselbrink and Mungal 2001 Pt. 1
        return np.clip(Z, self.Z_gl_unique, 1.0)

    def calc_adjustment_factor(self):
        print("Computing adjustment factor...")
        self.adjustment_factor_interp = []
        for i_m in tqdm(range(len(self.mdot_inj_unique))):
            if np.isnan(self.rho_inj_unique[i_m]):
                self.adjustment_factor_interp.append(lambda x: np.ones_like(x))
                continue

            # Create grid along the centerline
            y_cl_max = self.y_cl(self.xc[-1] - self.x_inj)[i_m]
            y_cl_arr = np.linspace(0, y_cl_max, 1000)
            x_cl_arr = self.x_cl_from_y_cl(y_cl_arr[:, np.newaxis])[:, i_m]

            # Iterate over the centerline
            adjustment_factor_arr = self.calc_adjustment_factor_xy(
                x_cl_arr, y_cl_arr, i_m
            )
            adjustment_factor_arr[np.isnan(adjustment_factor_arr)] = 1.0

            # Interpolate over y because the most rapid variation is near the injection point
            self.adjustment_factor_interp.append(
                interpolate.CubicSpline(y_cl_arr, adjustment_factor_arr, axis=1)
            )

    def calc_adjustment_factor_xy(self, x_cl, y_cl, i_m):
        # Compute the normal to the centerline
        dy_cl_dx = self.dy_cl_dx(x_cl[:, np.newaxis])[:, i_m]
        ds = np.stack([np.full_like(x_cl, 1.0), dy_cl_dx], axis=0)
        ds /= np.linalg.norm(ds, axis=0)
        dn = np.array([-ds[1], ds[0]])

        # Intersection of the normal with the top and bottom boundaries
        x_top = x_cl + dn[0] * (self.h - y_cl)
        x_bot = x_cl - dn[0] * (y_cl)
        xi_lo = -np.sqrt((x_bot - x_cl) ** 2 + (0 - y_cl) ** 2)
        xi_hi = np.sqrt((x_top - x_cl) ** 2 + (self.h - y_cl) ** 2)

        Z_int_nobound = self.n_inj * self.Z_cl_int[i_m]

        Z_cl = self.Z_cl(x_cl[:, np.newaxis])[:, i_m]
        sigma2 = self.Z_cl_int[i_m] / (2 * np.pi * Z_cl)
        s2s = np.sqrt(2 * sigma2)

        Z_int_bound = 0.0
        for z_inj in self.z_inj:
            Z_int_bound += (
                Z_cl
                * (np.pi / 2)
                * sigma2
                * (special.erf(xi_hi / s2s) - special.erf(xi_lo / s2s))
                * (
                    special.erf((self.w / 2 - z_inj) / s2s)
                    - special.erf((-self.w / 2 - z_inj) / s2s)
                )
            )

        return Z_int_nobound / Z_int_bound

    def get_adjustment_factor(self, x, y):
        if np.isscalar(x):
            x_arr = np.array([x])
            y_arr = np.array([y])
        else:
            x_arr = x
            y_arr = y

        fac = np.zeros([len(x_arr), len(self.mdot_inj_unique)])
        for i_m in range(len(self.mdot_inj_unique)):
            _, y_cl, _ = self.nearest_on_cl(x_arr, y_arr, np.zeros_like(x_arr), i_m)
            fac[:, i_m] = self.adjustment_factor_interp[i_m](y_cl)

        if np.isscalar(x):
            return fac[0]
        return fac

    def Z_3D(self, x, y, z):
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        """
        Z = 0.0
        for z_inj in self.z_inj:
            Z += self.Z_3D_single_inj(x, y, z, z_inj)
        return Z

    def grad_Z_3D(self, x, y, z):
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        """
        grad_Z = np.zeros((3, *y.shape))
        for z_inj in self.z_inj:
            grad_Z += self.grad_Z_3D_single_inj(x, y, z, z_inj)
        return grad_Z

    def Z_3D_adjusted(self, x, y, z):
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        """
        Z_adjusted = self.Z_3D(x, y, z) * self.get_adjustment_factor(x, y)
        # return Z_adjusted
        return np.minimum(
            Z_adjusted, 1.0
        )  # TODO: This cap introduces error in the integral. Distribute somehow?

    def grad_Z_3D_adjusted(self, x, y, z):
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        """
        return self.grad_Z_3D(x, y, z) * self.get_adjustment_factor(x, y)

    def Z_3D_single_inj(self, x, y, z, z_inj):
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        z_inj: float
            The injector z-coordinate
        """
        if np.isscalar(x):
            x_arr = np.array([x])
            y_arr = np.array([y])
            z_arr = np.array([z])
        else:
            x_arr = x
            y_arr = y
            z_arr = z

        # Compute the nearest point on the centerline and the distance squared
        x_cl = np.zeros([len(x_arr), len(self.mdot_inj_unique)])
        n2 = np.zeros_like(x_cl)
        for i_m in range(len(self.mdot_inj_unique)):
            x_cl[:, i_m], _, n2[:, i_m] = self.nearest_on_cl(
                x_arr, y_arr, z_arr - z_inj, i_m
            )

        # If we haven't passed the edge of the injector, assume it's still a perfect cylinder
        # if x_cl < self.d_inj / 2:
        #     if n2 < (self.d_inj / 2)**2:
        #         return 1.0
        #     else:
        #         return 0.0

        # Compute the centerline fuel mass fraction
        Z_cl = self.Z_cl(x_cl)

        # Spreading based on scalar conservation
        # (Assume gaussian, rho_inj * u_inj * Z_inj * A_inj = rho * u * int(Z * dA))
        # where int(Z * dA) = 2 * pi * sigma^2 * Z_cl
        sigma2 = self.Z_cl_int / (Z_cl * 2 * np.pi)
        Z = Z_cl * np.exp(-n2 / (2 * sigma2))

        if np.isscalar(x):
            return Z[0]
        return Z

    def grad_Z_3D_single_inj(self, x, y, z, z_inj):
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: float
            The query x-coordinate
        y: float
            The query y-coordinate
        z: float
            The query z-coordinate
        z_inj: float
            The injector z-coordinate
        """
        # Compute the nearest point on the centerline and the distance squared
        x_cl, y_cl, n2 = self.nearest_on_cl(x, y, z - z_inj)

        # Compute the centerline fuel mass fraction
        Z_cl = self.Z_cl(x_cl)

        # Spreading based on scalar conservation
        # (Assume gaussian, rho_inj * u_inj * Z_inj * A_inj = rho * u * int(Z * dA))
        # where int(Z * dA) = 2 * pi * sigma^2 * Z_cl
        sigma2 = self.Z_cl_int / (Z_cl * 2 * np.pi)
        return (
            -Z_cl
            * np.array([x - x_cl, y - y_cl, z - z_inj])
            / (2 * sigma2)
            * np.exp(-n2 / (2 * sigma2))
        )

    def __stretched_grid(self, x_start, x_end, dx, growth_rate, target_x):
        x_grid = [x_start, x_end]
        for direction in [-1, 1]:
            x, spacing = target_x, dx
            while x_start <= x <= x_end:
                x_grid.append(x)
                x += direction * spacing
                spacing *= growth_rate
        return np.sort(np.unique(x_grid))

    def calc_Z_3D_interp(self, write=False):
        print("Computing Z 3D array...")
        dx = 5.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        self.y_3D_data = np.linspace(0, self.h, Ny)
        self.z_3D_data = np.linspace(-self.w / 2, self.w / 2, Nz)
        self.x_3D_data = self.__stretched_grid(
            self.xc[0], self.xc[-1], dx, 1.1, self.x_inj
        )
        Nx = len(self.x_3D_data)
        self.Z_3D_data = np.zeros([len(self.mdot_inj_unique), Nx, Ny, Nz])
        for i in tqdm(range(Nx)):
            for j in range(Ny):
                for k in range(Nz):
                    self.Z_3D_data[:, i, j, k] = self.Z_3D_adjusted(
                        self.x_3D_data[i], self.y_3D_data[j], self.z_3D_data[k]
                    )
        self.Z_3D_data[np.isnan(self.rho_inj_unique)] = 0.0

        if write:
            np.save(datadir / "Z_3D_x.npy", self.x_3D_data)
            np.save(datadir / "Z_3D_y.npy", self.y_3D_data)
            np.save(datadir / "Z_3D_z.npy", self.z_3D_data)
            np.save(datadir / "Z_3D.npy", self.Z_3D_data)

        self.Z_3D_interp = []
        for i_m in range(len(self.mdot_inj_unique)):
            interp = interpolate.RegularGridInterpolator(
                (self.x_3D_data, self.y_3D_data, self.z_3D_data),
                self.Z_3D_data[i_m],
                method="cubic",
            )
            self.Z_3D_interp.append(interp)

    def eval_Z_3D_interp(self, x, y, z):
        Z_arr = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            Z_arr[i_m] = self.Z_3D_interp[i_m]((x, y, z))
        return Z_arr

    def Z_avg_var(self, x):
        Z_avg = np.zeros_like(self.mdot_inj_unique)
        Z_var = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            if np.isnan(self.rho_inj_unique[i_m]):
                Z_avg[i_m] = 0.0
                Z_var[i_m] = 0.0
                continue

            def func(z, y, i_m=i_m):
                return self.Z_3D(x, y, z)[i_m]

            Z_avg[i_m] = (
                2.0
                * integrate.dblquad(
                    func, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + 0 * y
                )[0]
                / (self.w * self.h)
            )

            def func(z, y, i_m=i_m):
                return (self.Z_3D(x, y, z)[i_m] - Z_avg[i_m]) ** 2

            Z_var[i_m] = (
                2.0
                * integrate.dblquad(
                    func, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + 0 * y
                )[0]
                / (self.w * self.h)
            )
        return Z_avg, Z_var

    def Z_avg_var_adjusted(self, x):
        Z_avg = np.zeros_like(self.mdot_inj_unique)
        Z_var = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            if np.isnan(self.rho_inj_unique[i_m]):
                Z_avg[i_m] = 0.0
                Z_var[i_m] = 0.0
                continue

            # func = lambda z, y: self.Z_3D_adjusted(x, y, z)[i_m]
            def func(z, y, i_m=i_m):
                return self.Z_3D_interp[i_m]((x, y, z))

            Z_avg[i_m] = (
                2.0
                * integrate.dblquad(
                    func, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + 0 * y
                )[0]
                / (self.w * self.h)
            )

            # func = lambda z, y: (self.Z_3D_adjusted(x, y, z)[i_m] - Z_avg[i_m])**2
            def func(z, y, i_m=i_m):
                return (self.Z_3D_interp[i_m]((x, y, z)) - Z_avg[i_m]) ** 2

            Z_var[i_m] = (
                2.0
                * integrate.dblquad(
                    func, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + 0 * y
                )[0]
                / (self.w * self.h)
            )
        return Z_avg, Z_var

    def calc_Z_avg_var_profiles(self, write=False):
        print("Computing Z average and variance profiles...")
        self.Z_avg_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])
        self.Z_var_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])
        for i in tqdm(range(len(self.xc))):
            if self.xc[i] > self.x_noz:
                # Freeze the profiles in the nozzle
                self.Z_avg_profile[:, i] = self.Z_avg_profile[:, i - 1]
                self.Z_var_profile[:, i] = self.Z_var_profile[:, i - 1]
            else:
                self.Z_avg_profile[:, i], self.Z_var_profile[:, i] = (
                    self.Z_avg_var_adjusted(self.xc[i])
                )

        if write:
            np.save(datadir / "Z_avg_profile.npy", self.Z_avg_profile)
            np.save(datadir / "Z_var_profile.npy", self.Z_var_profile)

    def C_E_CHEM_avg_MIB(self, x):
        C_avg = np.zeros_like(self.mdot_inj_unique)
        E_CHEM_avg = np.zeros_like(self.mdot_inj_unique)
        for i_m in range(len(self.mdot_inj_unique)):
            if np.isnan(self.rho_inj_unique[i_m]):
                C_avg[i_m] = self.fpv_table.lookup_direct("PROG", 0.0, 0.0, 0.0)
                E_CHEM_avg[i_m] = self.fpv_table.lookup_direct("E_CHEM", 0.0, 0.0, 0.0)
                continue

            def integrand(z, y, i_m=i_m):
                # Z = self.Z_3D_adjusted(x, y, z)[i_m]
                Z = self.Z_3D_interp[i_m]((x, y, z))
                return self.fpv_table.lookup_direct("PROG", Z, 0.0, 1.0)

            C_avg[i_m] = (
                2.0
                * integrate.dblquad(
                    integrand, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + y * 0
                )[0]
                / (self.w * self.h)
            )

            def integrand(z, y, i_m=i_m):
                # Z = self.Z_3D_adjusted(x, y, z)[i_m]
                Z = self.Z_3D_interp[i_m]((x, y, z))
                return self.fpv_table.lookup_direct("E_CHEM", Z, 0.0, 1.0)

            E_CHEM_avg[i_m] = (
                2.0
                * integrate.dblquad(
                    integrand, 0, self.h, lambda y: 0 * y, lambda y: self.w / 2 + y * 0
                )[0]
                / (self.w * self.h)
            )
        return C_avg, E_CHEM_avg

    def calc_MIB_profile(self, write=False):
        print("Computing MIB profile...")
        self.C_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])
        self.E_CHEM_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])

        # Debugging way
        C = self.fpv_table.lookup_direct("PROG", self.Z_3D_data, 0.0, 1.0)
        E_CHEM = self.fpv_table.lookup_direct("E_CHEM", self.Z_3D_data, 0.0, 1.0)

        C_profile = np.mean(C, axis=(2, 3))
        E_CHEM_profile = np.mean(E_CHEM, axis=(2, 3))
        self.C_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])
        self.E_CHEM_profile = np.zeros([len(self.mdot_inj_unique), len(self.xc)])
        for i_m in range(len(self.mdot_inj_unique)):
            self.C_profile[i_m] = np.interp(self.xc, self.x_3D_data, C_profile[i_m])
            self.E_CHEM_profile[i_m] = np.interp(
                self.xc, self.x_3D_data, E_CHEM_profile[i_m]
            )

        # # Real way
        # for i in tqdm(range(len(self.x))):
        #     if self.x[i] < self.x_inj:
        #         # Assume no fuel in the domain
        #         self.C_profile[:, i] = self.fpv_table.lookup_direct('PROG', 0.0, 0.0, 0.0)
        #         self.E_CHEM_profile[:, i] = self.fpv_table.lookup_direct('E_CHEM', 0.0, 0.0, 0.0)
        #     elif self.x[i] > self.x_noz:
        #         # Freeze the profiles in the nozzle
        #         self.C_profile[:, i] = self.C_profile[:, i-1]
        #         self.E_CHEM_profile[:, i] = self.E_CHEM_profile[:, i-1]
        #     elif self.x[i] < self.x_inj + 0.001:
        #         # DEBUG: Assume nearly no mixing, so no burning
        #         Z_avg = self.Z_avg_profile[:, i]
        #         self.C_profile[:, i] = self.fpv_table.lookup_direct('PROG', Z_avg, 0.0, 0.0)
        #         self.E_CHEM_profile[:, i] = self.fpv_table.lookup_direct('E_CHEM', Z_avg, 0.0, 0.0)
        #     else:
        #         self.C_profile[:,i], self.E_CHEM_profile[:, i] = self.C_E_CHEM_avg_MIB(self.x[i])

        if write:
            np.save(datadir / "C_profile_MIB.npy", self.C_profile)
            np.save(datadir / "E_CHEM_profile_MIB.npy", self.E_CHEM_profile)

    def estimate_p_Z(self, x, Z):
        """
        This method estimates the PDF of the mixture fraction at a given point
        using a Beta distribution.
        x: float
            The query x-coordinate
        Z: float
            The query mixture fraction
        """
        Z_avg, Z_var = self.Z_avg_var_adjusted(x)
        if Z_avg == 0.0:
            return 0.0
        a = ((Z_avg * (1 - Z_avg) / Z_var) - 1) * Z_avg
        b = a * (1 - Z_avg) / Z_avg
        return stats.beta.pdf(Z, a, b)

    def update_fluid_tip_positions(self, dt, t, u):
        """
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
        """
        # Update the fluid tip positions
        self.fluid_tips[:, 0] += dt * np.interp(self.fluid_tips[:, 0], self.xc, u)

        # Emit a new fluid tip
        mdot = np.interp(t, self.t_inj, self.mdot_inj)
        next_tip = np.array([[self.x_inj, mdot]])
        self.fluid_tips = np.concatenate([self.fluid_tips, next_tip], axis=0)

        # Drop fluid tips that have passed the end of the domain
        self.fluid_tips = self.fluid_tips[self.fluid_tips[:, 0] < self.xc[-1]]

    def source(
        self,
        time: float,
        state_array_local: Array | None,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Compute a fuel injector source term to target the desired mixture fraction profile."""
        _ = gamma_star, e0_star
        assert self.geometry is not None
        assert state_array_local is not None
        state_array_local = np.reshape(state_array_local, self.shape_input)
        rhs = np.zeros_like(state_array_local)

        # rho = state_array_local[:, 2]
        # rhoZ = state_array_local[:, 3]

        # Z = rhoZ / rho
        # mdot_inj = np.interp(
        #     self.xc,
        #     np.flip(self.fluid_tips, axis=0)[:, 0],
        #     np.flip(self.fluid_tips, axis=0)[:, 1],
        # )
        # Z_target = self.Z_avg_profile_interp((mdot_inj, self.xc))
        # mdot = rho * self.alpha * (Z_target - Z)

        # # Treat small (pre-injector) values
        # mdot[Z_target < 1e-6] = 0.0

        mdot = np.interp(time, self.t_inj, self.mdot_inj)

        L_src = 3e-2
        xf = self.geometry.xf
        idx = (
            np.where(np.logical_and(xf >= self.x_inj, xf < self.x_inj + L_src))[0]
            + self.geometry.n_ghost_layers
        )
        dx = self.geometry.dx
        if isinstance(dx, np.ndarray):
            dx = dx[idx]
        mdot *= dx / L_src

        # Compute the source term
        rhs[idx, 0] = mdot * self.u_inj  # momentum
        rhs[idx, 1] = mdot * self.E_inj  # total energy
        rhs[idx, 2] = mdot  # density
        rhs[idx, 3] = mdot  # mixture fraction
        rhs[idx, 4] = 0.0  # progress variable

        # Divide by the cell volumes
        if self.cell_volumes is None:
            assert self.geometry is not None
            vol = self.geometry.volume(time)[:, None]
            return np.ravel(rhs / vol)

        return np.ravel(rhs / self.cell_volumes)

    @staticmethod
    def _compute_omega_C_int(
        i_Zbar,
        i_Lbar,
        i_S,
        Zbar_vec,
        Lbar_vec,
        logsigma2_vec,
        uv_vec,
        W_vec,
        omega_C_interp,
    ):
        Zbar = Zbar_vec[i_Zbar]
        Lbar = Lbar_vec[i_Lbar]
        logsigma2 = logsigma2_vec[i_S]
        sigma2 = 10**logsigma2

        eps = 1.0e-6
        if (Zbar < eps) or (Zbar > 1 - eps) or (Lbar < eps) or (Lbar > 1 - eps):
            result = omega_C_interp((Zbar, Lbar))
            return (i_Zbar, i_Lbar, i_S, result)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Compute the shape parameters
            alpha_Z = ((Zbar * (1 - Zbar) / sigma2) - 1) * Zbar
            beta_Z = alpha_Z * (1 - Zbar) / Zbar
            alpha_L = ((Lbar * (1 - Lbar) / sigma2) - 1) * Lbar
            beta_L = alpha_L * (1 - Lbar) / Lbar

            # Compute the quadrature points and weights
            Z_int_vec = stats.beta.ppf(uv_vec, alpha_Z, beta_Z)
            L_int_vec = stats.beta.ppf(uv_vec, alpha_L, beta_L)

            # Evaluate the integrand at the quadrature points
            W_Z_mesh, W_L_mesh = np.meshgrid(W_vec, W_vec, indexing="ij")
            W_mesh = W_Z_mesh * W_L_mesh
            Z_int_mesh, L_int_mesh = np.meshgrid(Z_int_vec, L_int_vec, indexing="ij")
            integrand = omega_C_interp((Z_int_mesh, L_int_mesh))

            # Perform the integration
            result = np.sum(integrand * W_mesh)

        return (i_Zbar, i_Lbar, i_S, result)

    def calc_chemical_sources(self, write=False):
        """
        This method precomputes the chemical source terms as a function of x, mdot_f, and L.
        """
        print("Precomputing chemical sources...")

        # Build omega_C interpolator
        Z_sample = np.linspace(0.0, 1.0, 100)
        L_sample = np.linspace(0.0, 1.0, 100)
        Z_sample_mesh, L_sample_mesh = np.meshgrid(Z_sample, L_sample, indexing="ij")
        omega_C = self.fpv_table.lookup_direct(
            "SRC_PROG", Z_sample_mesh, 0.0, L_sample_mesh
        )
        omega_C_interp = interpolate.RegularGridInterpolator(
            (Z_sample, L_sample), omega_C, bounds_error=False, fill_value=0.0
        )

        # Grid in Zbar, Lbar, logsigma2 dimensions (to be tabulated over)
        n_tab = (100, 100, 100)
        self.Zbar_vec = np.linspace(0.0, 1.0, n_tab[0])
        self.Lbar_vec = np.linspace(0.0, 1.0, n_tab[1])
        self.logsigma2_vec = np.linspace(-4.0, -1.5, n_tab[2])

        uv_vec, W_vec = special.roots_legendre(200)
        W_vec = W_vec / 2
        uv_vec = uv_vec / 2 + 0.5

        self.omega_C_int = np.zeros(n_tab)
        compute_func = functools.partial(
            self._compute_omega_C_int,
            Zbar_vec=self.Zbar_vec,
            Lbar_vec=self.Lbar_vec,
            logsigma2_vec=self.logsigma2_vec,
            uv_vec=uv_vec,
            W_vec=W_vec,
            omega_C_interp=omega_C_interp,
        )
        tasks = [
            (i_Zbar, i_Lbar, i_S)
            for i_Zbar in range(n_tab[0])
            for i_Lbar in range(n_tab[1])
            for i_S in range(n_tab[2])
        ]

        # Parallel version
        with tqdm_joblib(tqdm(desc="Assembling table", total=len(tasks))):
            results = Parallel(n_jobs=-1)(
                delayed(compute_func)(i_Zbar, i_Lbar, i_S)
                for i_Zbar, i_Lbar, i_S in tasks
            )

        for i_Zbar, i_Lbar, i_S, value in results:
            self.omega_C_int[i_Zbar, i_Lbar, i_S] = value
        self.omega_C_int[np.isnan(self.omega_C_int)] = 0.0

        if write:
            np.save(datadir / "Zbar_vec.npy", self.Zbar_vec)
            np.save(datadir / "Lbar_vec.npy", self.Lbar_vec)
            np.save(datadir / "logsigma2_vec.npy", self.logsigma2_vec)
            np.save(datadir / "omega_C_int.npy", self.omega_C_int)

        # Build 3D table interpolator
        self.omega_C_int_interp = interpolate.RegularGridInterpolator(
            (self.Zbar_vec, self.Lbar_vec, self.logsigma2_vec), self.omega_C_int
        )

    def get_MIB_profiles(self):
        """
        This method returns the MIB profiles for the progress variable and chemical
        energy.
        """
        C = np.zeros(len(self.xc))
        E_CHEM = np.zeros(len(self.xc))
        mdot_inj = np.interp(
            self.xc,
            np.flip(self.fluid_tips, axis=0)[:, 0],
            np.flip(self.fluid_tips, axis=0)[:, 1],
        )

        for i_x in range(len(self.xc)):
            if self.xc[i_x] < self.x_inj:
                continue
            # Interpolate into fluid tip positions to get the mass flow rate
            C[i_x] = np.interp(
                mdot_inj[i_x], self.mdot_inj_unique, self.C_profile[:, i_x]
            )
            E_CHEM[i_x] = np.interp(
                mdot_inj[i_x], self.mdot_inj_unique, self.E_CHEM_profile[:, i_x]
            )

        return C, E_CHEM


class JICFChemistrySource(RightHandSide):
    def __init__(
        self, jicf: JICModel, **precompute_steps: Unpack[PrecomputeSteps]
    ) -> None:
        super().__init__(**precompute_steps)
        self.jicf = jicf

        # Reactions only directly affect progress variable
        self.idx_source = np.array([4])
        self.shape_output = (self.shape_output[0], 1)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Compute the chemical source terms [1/s] using the FPV table."""
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert isinstance(self.physics, FPVTable)
        assert state is not None
        assert state.density is not None
        assert state.mixture_fraction is not None
        assert state.normalized_progress_variable is not None
        factor = self.physics.get_source_progress_variable_compressibility_factor(state)

        # Get the mixture fraction variance profile
        mdot_inj = np.interp(
            self.jicf.xc,
            np.flip(self.jicf.fluid_tips, axis=0)[:, 0],
            np.flip(self.jicf.fluid_tips, axis=0)[:, 1],
        )
        Zvar = self.jicf.Z_var_profile_interp((mdot_inj, self.jicf.xc))
        Zvar = np.maximum(Zvar, 10 ** self.jicf.logsigma2_vec.min())

        return (
            factor
            * state.density
            * self.jicf.omega_C_int_interp(
                (
                    state.mixture_fraction,
                    state.normalized_progress_variable,
                    np.log10(Zvar),
                )
            )
        )
