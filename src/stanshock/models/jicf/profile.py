from __future__ import annotations

import numpy as np
from scipy import integrate, interpolate, optimize, special
from tqdm import tqdm

from stanshock.physics.flamelet import FPVTable
from stanshock.system.backend import Array


class AnalyticJICF:
    """Direct evaluation of analytical models of a jet-in-crossflow."""

    def __init__(
        self,
        x: Array,
        w: float,
        h: float,
        n_inj: int,
        d_inj: float,
        rho_inj: Array,
        u_inj: Array,
        rho: float,
        u: float,
        physics: FPVTable,
        theta_inj: float = 0.0,
    ) -> None:
        """
        This method initializes the Jet-in-Crossflow model with the following
        parameters:
        x: Array
            The x-coordinates relative to the injector
        w: float
            The width of the combustor
        h: float
            The height of the combustor
        n_inj: int
            The number of injected jets
        d_inj: float
            The diameter of the injected jet
        rho_inj: np.ndarray
            The density of the injected jet, as a function of time
        u_inj: np.ndarray
            The velocity of the injected jet
        rho: float
            The density of the crossflow
        u: float
            The velocity of the crossflow
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        theta_inj: float
            The angle of the jet relative to the x axis (rads)
        """
        self.physics = physics

        assert self.physics.fuel_def is not None
        self.fuel_def = self.physics.fuel_def
        assert self.physics.ox_def is not None
        self.ox_def = self.physics.ox_def

        gas = self.physics.gas

        self.n_inj = n_inj
        self.d_inj = d_inj
        self.theta_inj = theta_inj
        self.rho_inj = rho_inj
        self.u_inj = u_inj

        # Geometry parameters
        self.x = x
        self.w = w
        self.h = h
        self.L = self.x[-1] - self.x[0]
        self.A = self.w * self.h
        self.A_inj = np.pi * (self.d_inj / 2.0) ** 2

        # Free stream properties
        self.rho = rho
        self.u = u
        gas.X = self.ox_def
        self.Y_ox = gas.Y

        # Properties of the injected fluid
        self.mdot_inj = self.rho_inj * self.u_inj * self.A_inj
        gas.X = self.fuel_def
        self.Y_fuel = gas.Y

        # Integral of Z across centerline normal plane
        A_inj = np.pi * (self.d_inj / 2.0) ** 2
        self.Z_cl_int = self.rho_inj * self.u_inj * A_inj / (self.rho * self.u)
        self.d_eff = np.sqrt(self.Z_cl_int / (2 * np.pi))

        # Stoichiometry
        mdot_a = self.rho * self.u * self.A
        self.phi_gl = np.zeros_like(self.mdot_inj)
        self.Z_gl = np.zeros_like(self.mdot_inj)
        for i_m in range(len(self.mdot_inj)):
            Y_ox = mdot_a / (mdot_a + self.mdot_inj[i_m])
            gas.Y = Y_ox * self.Y_ox + (1.0 - Y_ox) * self.Y_fuel
            self.phi_gl[i_m] = gas.equivalence_ratio(self.fuel_def, self.ox_def)
            self.Z_gl[i_m] = gas.mixture_fraction(self.fuel_def, self.ox_def)

        # Compute the non-dimensional parameters
        self.J = (self.rho_inj * self.u_inj**2) / (
            self.rho * self.u**2
        )  # Momentum flux ratio
        self.r_u = np.sqrt(self.J)  # Blowing ratio = sqrt(J)

        # Create the array of injectors
        self.z_inj = np.linspace(-self.w / 2, self.w / 2, n_inj + 2)[1:-1]

        # Precompute the adjustment factor for the boundary clipping
        self.calc_adjustment_factor()

    def y_cl(self, x_cl: Array | float) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return self.d_inj * 1.6 * (x_cl / self.d_inj)**(1.0/3.0) * self.r_u**(2.0/3.0) # Torrez 2011 (Same as Margason 1968)
        # return 1.6 * x_cl**(1.0/3.0) * (self.d_inj * self.r_u)**(2.0/3.0) # Margason 1968
        # return self.r_u * self.d_inj * 1.6 * (x_cl / (self.r_u * self.d_inj))**(1.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2
        # return self.d_inj * 0.527 * self.r_u**1.178 * (x_cl / self.d_inj)**0.314 # Karagozian 1986

        # SONIC VERSION
        denom = np.divide(
            1.0, self.d_inj * self.J, out=np.zeros_like(self.J), where=self.J > 0
        )
        term = x_cl * denom
        term = np.power(term, 0.344, out=np.zeros_like(term), where=term > 0)
        return self.d_inj * self.J * 1.23 * term  # Gruber 1995 JPP
        # return self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**0.344 # Gruber 1997 Phys. Fluids
        # return self.d_inj * 2.173 / self.J**0.276 * (x_cl / self.d_inj)**0.281 # Rothstein and Wantuck 1992

    def x_cl_from_y_cl(self, y_cl: Array | float) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (y_cl / (self.r_u * self.d_inj * 1.6))**(3.0) * self.r_u * self.d_inj # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        denom = np.divide(
            1.0, self.d_inj * self.J * 1.23, out=np.zeros_like(self.J), where=self.J > 0
        )
        return (y_cl * denom) ** (1.0 / 0.344) * self.d_inj * self.J  # Gruber 1995 JPP
        # return (y_cl / (self.d_inj * self.J * 1.20))**(1.0 / 0.344) * self.d_inj * self.J - self.d_inj/2 # Gruber 1997 Phys. Fluids

    def dy_cl_dx(self, x_cl: Array | float) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (self.r_u * self.d_inj)**(2.0/3.0) * 1.6 * (1.0/3.0) * x_cl**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        denom = np.divide(
            1.0, self.d_inj * self.J, out=np.zeros_like(self.J), where=self.J > 0
        )
        term = x_cl * denom
        term = np.power(term, 0.344 - 1.0, out=np.zeros_like(term), where=term > 0)
        return 0.344 * self.d_inj * self.J * 1.23 * term * denom  # Gruber 1995 JPP
        # return (0.344 *
        #         self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**(0.344 - 1.0) *
        #         (1.0 / (self.d_inj * self.J))) # Gruber 1997 Phys. Fluids

    def __nearest_on_cl_single(self, x, y, dz, i_m):
        # Define the centerline
        # Note: dz makes no difference in the minimization, but it's more convenient to include it here
        # so that the n2 is correct
        def n2_func_x_cl(x_cl):
            return (x - x_cl) ** 2 + (y - self.y_cl(x_cl)[i_m]) ** 2 + dz**2

        def n2_func_y_cl(y_cl):
            return (x - self.x_cl_from_y_cl(y_cl)[i_m]) ** 2 + (y - y_cl) ** 2 + dz**2

        # Compute the x_cl which minimizes n2
        x_cl = optimize.fminbound(n2_func_x_cl, self.x[0], self.x[-1], disp=False)
        y_cl = self.y_cl(x_cl)[i_m]
        n2 = n2_func_x_cl(x_cl)

        if self.dy_cl_dx(x_cl)[i_m] > 1:
            # Compute the y_cl which minimizes n2
            y_cl = optimize.fminbound(n2_func_y_cl, 0.0, self.h, disp=False)
            x_cl = self.x_cl_from_y_cl(y_cl)[i_m]
            n2 = n2_func_y_cl(y_cl)

        return x_cl, y_cl, n2

    def nearest_on_cl(self, x, y, dz, i_m):
        x_match, y_match, dz_match = np.broadcast_arrays(x, y, dz)

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

    def Z_cl(self, x_cl: Array | float) -> Array:
        denom = np.divide(
            1.0, self.r_u, out=np.zeros_like(self.r_u), where=self.r_u > 0
        )
        term = x_cl * denom / self.d_inj
        term = np.power(term, -2.0 / 3.0, out=np.zeros_like(term), where=term > 0)
        Z = (
            0.85 * denom * np.sqrt(self.rho_inj / self.rho) * term
        )  # Hasselbrink and Mungal 2001 Pt. 1
        return np.clip(Z, self.Z_gl, 1.0)

    def calc_adjustment_factor(self):
        print("Computing adjustment factor...")
        self.adjustment_factor_interp = []
        y_cl_max = self.y_cl(self.x[-1])

        # Create grid along the centerline
        y_cl_arr = np.linspace(0, y_cl_max, 1000, axis=0)
        x_cl_arr = self.x_cl_from_y_cl(y_cl_arr)

        for i_m in tqdm(range(len(self.mdot_inj))):
            if self.u_inj[i_m] == 0.0:
                self.adjustment_factor_interp.append(np.ones_like)
                continue

            # Iterate over the centerline
            adjustment_factor_arr = self.calc_adjustment_factor_xy(
                x_cl_arr[:, i_m], y_cl_arr[:, i_m], i_m
            )
            adjustment_factor_arr[np.isnan(adjustment_factor_arr)] = 1.0

            # Interpolate over y because the most rapid variation is near the injection point
            self.adjustment_factor_interp.append(
                interpolate.CubicSpline(y_cl_arr[:, i_m], adjustment_factor_arr, axis=1)
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

        fac = np.zeros([len(x_arr), len(self.mdot_inj)])
        for i_m in range(len(self.mdot_inj)):
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
            Z_temp = self.Z_3D_single_inj(x, y, z, z_inj)
            if isinstance(Z_temp, np.ndarray):
                Z_temp = Z_temp[0]
            Z += Z_temp
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
        x_cl = np.zeros([len(x_arr), len(self.mdot_inj)])
        n2 = np.zeros_like(x_cl)
        for i_m in range(len(self.mdot_inj)):
            x_cl[:, i_m], _, n2[:, i_m] = self.nearest_on_cl(
                x_arr, y_arr, z_arr - z_inj, i_m
            )

        Z_cl = self.Z_cl(x_cl)

        sigma2 = np.divide(
            self.Z_cl_int, Z_cl * 2 * np.pi, out=np.zeros_like(Z_cl), where=Z_cl > 0
        )
        return Z_cl * np.exp(
            np.divide(-n2, 2 * sigma2, out=np.zeros_like(sigma2), where=sigma2 > 0)
        )

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

    def Z_avg_var(self, x):
        Z_avg = np.zeros_like(self.mdot_inj)
        Z_var = np.zeros_like(self.mdot_inj)
        for i_m in range(len(self.mdot_inj)):
            if np.isnan(self.rho_inj[i_m]):
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
