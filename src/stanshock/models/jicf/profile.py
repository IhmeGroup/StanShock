from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import interpolate, special
from scipy.optimize.elementwise import find_minimum
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
        self.nJ = len(self.mdot_inj)
        gas.X = self.fuel_def
        self.Y_fuel = gas.Y

        # Index into the mass flux array
        self.i_m: slice | int = slice(None)

        # Integral of Z across centerline normal plane
        A_inj = np.pi * (self.d_inj / 2.0) ** 2
        self.Z_cl_int = self.rho_inj * self.u_inj * A_inj / (self.rho * self.u)
        self.d_eff = np.sqrt(self.Z_cl_int / (2 * np.pi))

        # Stoichiometry
        mdot_a = self.rho * self.u * self.A
        self.phi_gl = np.zeros_like(self.mdot_inj)
        self.Z_gl = np.zeros_like(self.mdot_inj)
        for i_m in range(self.nJ):
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

    def y_cl(self, x_cl: Array) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return self.d_inj * 1.6 * (x_cl / self.d_inj)**(1.0/3.0) * self.r_u**(2.0/3.0) # Torrez 2011 (Same as Margason 1968)
        # return 1.6 * x_cl**(1.0/3.0) * (self.d_inj * self.r_u)**(2.0/3.0) # Margason 1968
        # return self.r_u * self.d_inj * 1.6 * (x_cl / (self.r_u * self.d_inj))**(1.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2
        # return self.d_inj * 0.527 * self.r_u**1.178 * (x_cl / self.d_inj)**0.314 # Karagozian 1986

        # SONIC VERSION
        J = self.J[self.i_m]
        denom = np.divide(1.0, self.d_inj * J, out=np.zeros_like(J), where=J > 0)
        term = x_cl * denom
        term = np.power(term, 0.344, out=np.zeros_like(term), where=term > 0)
        return self.d_inj * J * 1.23 * term  # Gruber 1995 JPP
        # return self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**0.344 # Gruber 1997 Phys. Fluids
        # return self.d_inj * 2.173 / self.J**0.276 * (x_cl / self.d_inj)**0.281 # Rothstein and Wantuck 1992

    def x_cl_from_y_cl(self, y_cl: Array) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (y_cl / (self.r_u * self.d_inj * 1.6))**(3.0) * self.r_u * self.d_inj # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        J = self.J[self.i_m]
        denom = np.divide(1.0, self.d_inj * J * 1.23, out=np.zeros_like(J), where=J > 0)
        return (y_cl * denom) ** (1.0 / 0.344) * self.d_inj * J  # Gruber 1995 JPP
        # return (y_cl / (self.d_inj * self.J * 1.20))**(1.0 / 0.344) * self.d_inj * self.J - self.d_inj/2 # Gruber 1997 Phys. Fluids

    def dy_cl_dx(self, x_cl: Array) -> Array:
        # SUBSONIC VERSION - CHECK THESE FOR CORRECTNESS
        # return (self.r_u * self.d_inj)**(2.0/3.0) * 1.6 * (1.0/3.0) * x_cl**(-2.0/3.0) # Hasselbrink and Mungal 2001 Pt. 2

        # SONIC VERSION
        J = self.J[self.i_m]
        denom = np.divide(1.0, self.d_inj * J, out=np.zeros_like(J), where=J > 0)
        term = x_cl * denom
        term = np.power(term, 0.344 - 1.0, out=np.zeros_like(term), where=term > 0)
        return 0.344 * self.d_inj * J * 1.23 * term * denom  # Gruber 1995 JPP
        # return (0.344 *
        #         self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**(0.344 - 1.0) *
        #         (1.0 / (self.d_inj * self.J))) # Gruber 1997 Phys. Fluids

    def nearest_on_cl(
        self, x: Array, y: Array, dz: Array
    ) -> tuple[Array, Array, Array]:
        # Define the centerline
        # Note: dz makes no difference in the minimization, but it's more convenient to include it here
        # so that the n2 is correct
        def n2_func_x_cl(x_cl: Array, x: Array, y: Array, dz: Array) -> Array:
            return (x - x_cl) ** 2 + (y - self.y_cl(x_cl)) ** 2 + dz**2

        def n2_func_y_cl(y_cl: Array, x: Array, y: Array, dz: Array) -> Array:
            return (x - self.x_cl_from_y_cl(y_cl)) ** 2 + (y - y_cl) ** 2 + dz**2

        x_bracket: tuple[float, float, float] = (
            self.x[0],
            0.5 * (self.x[0] + self.x[-1]),
            self.x[-1],
        )
        y_bracket: tuple[float, float, float] = (0.0, 0.5 * self.h, self.h)

        x, y, dz = np.broadcast_arrays(x, y, dz)
        x_cl = np.zeros((*x.shape, self.nJ))
        y_cl = np.zeros((*x.shape, self.nJ))
        n2 = np.zeros((*x.shape, self.nJ))
        for i_m in range(self.nJ):
            self.i_m = i_m

            # Compute the x_cl which minimizes n2
            res = find_minimum(n2_func_x_cl, x_bracket, args=(x, y, dz))
            x_cl[..., i_m] = res.x
            n2[..., i_m] = res.f_x
            y_cl[..., i_m] = self.y_cl(x_cl[..., i_m])

            # For points where dy_cl/dx > 1
            dy_cl_dx = self.dy_cl_dx(x_cl[..., i_m])
            idx = dy_cl_dx > 1

            # Compute the y_cl which minimizes n2
            res = find_minimum(n2_func_y_cl, y_bracket, args=(x[idx], y[idx], dz[idx]))
            y_cl[idx, i_m] = res.x
            n2[idx, i_m] = res.f_x
            x_cl[idx, i_m] = self.x_cl_from_y_cl(y_cl[idx, i_m])

        return x_cl, y_cl, n2

    def Z_cl(self, x_cl: Array) -> Array:
        r_u = self.r_u[self.i_m]
        denom = np.divide(1.0, r_u, out=np.zeros_like(r_u), where=r_u > 0)
        term = x_cl * denom / self.d_inj
        term = np.power(term, -2.0 / 3.0, out=np.zeros_like(term), where=term > 0)
        Z = (
            0.85 * denom * np.sqrt(self.rho_inj[self.i_m] / self.rho) * term
        )  # Hasselbrink and Mungal 2001 Pt. 1
        return np.clip(Z, self.Z_gl[self.i_m], 1.0)

    def calc_adjustment_factor(self) -> None:
        print("Computing adjustment factor...")
        self.adjustment_factor_interp: list[Callable[[Array], Array]] = []
        y_cl_max = self.y_cl(self.x[-1])

        # Create grid along the centerline
        y_cl_arr = np.linspace(0, y_cl_max, 1000, axis=0)

        for i_m in tqdm(range(self.nJ)):
            if self.u_inj[i_m] == 0.0:
                self.adjustment_factor_interp.append(np.ones_like)
                continue

            # Iterate over the centerline
            self.i_m = i_m
            x_cl_arr = self.x_cl_from_y_cl(y_cl_arr[:, i_m])
            adjustment_factor_arr = self.calc_adjustment_factor_xy(
                x_cl_arr, y_cl_arr[:, i_m]
            )
            adjustment_factor_arr[np.isnan(adjustment_factor_arr)] = 1.0

            # Interpolate over y because the most rapid variation is near the injection point
            self.adjustment_factor_interp.append(
                interpolate.CubicSpline(y_cl_arr[:, i_m], adjustment_factor_arr)
            )

    def calc_adjustment_factor_xy(self, x_cl: Array, y_cl: Array) -> Array:
        # Compute the normal to the centerline
        dy_cl_dx = self.dy_cl_dx(x_cl)
        ds = np.stack([np.full_like(x_cl, 1.0), dy_cl_dx], axis=0)
        ds /= np.linalg.norm(ds, axis=0)
        dn = np.array([-ds[1], ds[0]])

        # Intersection of the normal with the top and bottom boundaries
        x_top = x_cl + dn[0] * (self.h - y_cl)
        x_bot = x_cl - dn[0] * (y_cl)
        xi_lo = -np.sqrt((x_bot - x_cl) ** 2 + (0 - y_cl) ** 2)
        xi_hi = np.sqrt((x_top - x_cl) ** 2 + (self.h - y_cl) ** 2)

        Z_int_nobound = self.n_inj * self.Z_cl_int[self.i_m]

        Z_cl = self.Z_cl(x_cl)
        sigma2 = self.Z_cl_int[self.i_m] / (2 * np.pi * Z_cl)
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

        return np.asarray(Z_int_nobound / Z_int_bound)

    def get_adjustment_factor(self, x: Array, y: Array) -> Array:
        shape = np.broadcast_shapes(x.shape, y.shape)
        fac = np.zeros((*shape, self.nJ))
        _, y_cl, _ = self.nearest_on_cl(x, y, np.zeros_like(x))
        for i_m in range(self.nJ):
            fac[..., i_m] = self.adjustment_factor_interp[i_m](y_cl[..., i_m])

        return fac

    def Z_3D(self, x: Array, y: Array, z: Array) -> Array:
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
            The query z-coordinate
        """
        Z: Array = np.zeros((*x.shape, self.nJ))
        for z_inj in self.z_inj:
            Z = Z + self.Z_3D_single_inj(x, y, z, float(z_inj))
        return Z

    def grad_Z_3D(self, x: Array, y: Array, z: Array) -> Array:
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed).
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
            The query z-coordinate
        """
        grad_Z = np.zeros((3, *y.shape, self.nJ))
        for z_inj in self.z_inj:
            grad_Z += self.grad_Z_3D_single_inj(x, y, z, float(z_inj))
        return grad_Z

    def Z_3D_adjusted(self, x: Array, y: Array, z: Array) -> Array:
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
            The query z-coordinate
        """
        Z_adjusted = self.Z_3D(x, y, z) * self.get_adjustment_factor(x, y)
        # return Z_adjusted
        return np.minimum(
            Z_adjusted, 1.0
        )  # TODO: This cap introduces error in the integral. Distribute somehow?

    def grad_Z_3D_adjusted(self, x: Array, y: Array, z: Array) -> Array:
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for all injectors (summed) with the boundary clipping adjustment.
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
            The query z-coordinate
        """
        return self.grad_Z_3D(x, y, z) * self.get_adjustment_factor(x, y)

    def Z_3D_single_inj(self, x: Array, y: Array, z: Array, z_inj: float) -> Array:
        """
        This method computes the mixture fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
            The query z-coordinate
        z_inj: float
            The injector z-coordinate
        """
        # Compute the nearest point on the centerline and the distance squared
        x_cl, _, n2 = self.nearest_on_cl(x, y, z - z_inj)

        # Compute the centerline fuel mass fraction
        Z_cl = self.Z_cl(x_cl)

        sigma2 = np.divide(
            self.Z_cl_int, Z_cl * 2 * np.pi, out=np.zeros_like(Z_cl), where=Z_cl > 0
        )
        return Z_cl * np.exp(
            np.divide(-n2, 2 * sigma2, out=np.zeros_like(sigma2), where=sigma2 > 0)
        )

    def grad_Z_3D_single_inj(self, x: Array, y: Array, z: Array, z_inj: float) -> Array:
        """
        This method computes the gradient of the mixture fraction for the Jet-in-Crossflow
        model in 3D for a single injector at (x_inj, 0, z_inj).
        x: Array
            The query x-coordinate
        y: Array
            The query y-coordinate
        z: Array
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
        dxyz = np.stack((x - x_cl, y - y_cl, z - z_inj), axis=0)
        return -Z_cl * dxyz / (2 * sigma2) * np.exp(-n2 / (2 * sigma2))
