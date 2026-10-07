from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy import interpolate, special
from scipy.optimize.elementwise import find_root

# from scipy.integrate import cubature
# from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from collections.abc import Callable

    from stanshock.system.backend import Array


class AnalyticJICF:
    """Direct evaluation of analytical models of a jet-in-crossflow.

    Computed values of mixture fraction, Z, are normalized by sqrt(rho_fuel / rho_ox).
    """

    def __init__(
        self,
        x: Array,
        w: float,
        h: float,
        n_inj: int,
        d_inj: float,
        J: Array,
        theta_inj: float = 0.0,
        Cd: float = 1.0,
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
        J: np.ndarray
            The momentum flux ratios over which to discretize the profiles
        theta_inj: float
            The angle of the jet relative to the x axis (rads)
        Cd: float
            Discharge coefficient for the injector
        """
        self.n_inj = n_inj
        self.d_inj = d_inj
        self.theta_inj = theta_inj

        # Geometry parameters
        self.x = x
        self.w = w
        self.h = h
        self.L = self.x[-1] - self.x[0]
        self.A = self.w * self.h
        self.A_inj = np.pi * (0.5 * self.d_inj) ** 2
        Ae = Cd * self.A_inj

        # Create the array of injectors
        self.z_inj = np.linspace(-self.w / 2, self.w / 2, n_inj + 2)[1:-1]

        # Non-dimensional parameters
        self.J = J  # Momentum flux ratio
        self.r_u = np.sqrt(self.J)  # Blowing ratio = sqrt(rho_inj / rho) * u_inj / u
        self.nJ = len(self.J)
        # Minimum mixture fraction (i.e. fully-mixed) - currently not clipping by this
        self.Z_gl = np.zeros_like(self.J)

        # Index into the J array
        self.i_m: slice | int = slice(None)

        # Integral of Z across centerline normal plane
        self.Z_cl_int = self.r_u * Ae

        # Extra data for debugging
        self._aux_adjust_data: Array | None = None

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
        term = np.power(term, 0.344 - 1.0, out=np.full_like(term, 1e20), where=term > 0)
        return 0.344 * 1.23 * term  # Gruber 1995 JPP
        # return (0.344 *
        #         self.d_inj * self.J * 1.20 * ((x_cl + self.d_inj/2) / (self.d_inj * self.J))**(0.344 - 1.0) *
        #         (1.0 / (self.d_inj * self.J))) # Gruber 1997 Phys. Fluids

    def dx_cl_dy(self, y_cl: Array) -> Array:
        # SONIC VERSION
        J = self.J[self.i_m]
        denom = np.divide(1.0, self.d_inj * J * 1.23, out=np.zeros_like(J), where=J > 0)
        term = y_cl * denom
        c = 1.0 / 0.344
        term = np.power(term, c - 1.0, out=np.zeros_like(term), where=term > 0)
        return c / 1.23 * term

    def nearest_on_cl(
        self, x: Array, y: Array, dz: Array
    ) -> tuple[Array, Array, Array]:
        # Define distance to the centerline as function of x or y
        def n2_func_x_cl(x_cl: Array, x: Array, y: Array) -> Array:
            return (x - x_cl) ** 2 + (y - self.y_cl(x_cl)) ** 2

        def n2_func_y_cl(y_cl: Array, x: Array, y: Array) -> Array:
            return (x - self.x_cl_from_y_cl(y_cl)) ** 2 + (y - y_cl) ** 2

        def distance_derivative_x(x_cl: Array, x: Array, y: Array) -> Array:
            """Derivative of the squared Euclidean distance w.r.t. x_cl"""
            dydx = self.dy_cl_dx(x_cl)
            return x_cl - x + (self.y_cl(x_cl) - y) * dydx

        def distance_derivative_y(y_cl: Array, x: Array, y: Array) -> Array:
            """Derivative of the squared Euclidean distance w.r.t. y_cl"""
            dxdy = self.dx_cl_dy(y_cl)
            return y_cl - y + (self.x_cl_from_y_cl(y_cl) - x) * dxdy

        x_bracket: tuple[float, float] = (0.0, self.x[-1])

        x, y = np.broadcast_arrays(x, y)
        x_cl = np.zeros((*x.shape, self.nJ))
        y_cl = np.zeros((*x.shape, self.nJ))
        n2 = np.zeros((*x.shape, self.nJ))
        for i_m in range(self.nJ):
            if self.J[i_m] == 0.0:
                continue

            self.i_m = i_m

            # Compute the x_cl which minimizes n2
            res = find_root(distance_derivative_x, x_bracket, args=(x, y))
            x_cl[..., i_m] = res.x
            n2[..., i_m] = n2_func_x_cl(x_cl[..., i_m], x, y)
            y_cl[..., i_m] = self.y_cl(x_cl[..., i_m])

            # Retry points where root solve failed
            # idx = res.status < 0
            idx = np.logical_or(res.status < 0, x == 0.0)

            # Compute the y_cl which minimizes n2
            y_bracket: tuple[float, float] = (0.0, self.h)
            res = find_root(distance_derivative_y, y_bracket, args=(x[idx], y[idx]))
            y_cl[idx, i_m] = res.x
            x_cl[idx, i_m] = self.x_cl_from_y_cl(y_cl[idx, i_m])
            n2[idx, i_m] = n2_func_y_cl(y_cl[idx, i_m], x[idx], y[idx])

        x_cl[np.isnan(x_cl)] = 0.0
        y_cl[np.isnan(y_cl)] = 0.0
        n2[np.isnan(n2)] = 0.0

        # Add z-offset to the Euclidean distance
        n2 = n2 + dz[..., None] ** 2
        n2, x_cl, y_cl = np.broadcast_arrays(n2, x_cl, y_cl)

        return x_cl, y_cl, n2

    def Z_cl(self, y_cl: Array) -> Array:
        """Centerline concentration as a function of y.

        Equations from Hasselbrink and Mungal 2001 Pt. 1, normalized by the square root
        of the density ratio.
        """
        # Normalized coordinates
        y = y_cl / self.d_inj
        Zmax = self.Z_cl_int[self.i_m] / self.A_inj
        r_u = self.r_u[self.i_m]
        denom = np.divide(1.0, r_u, out=np.zeros_like(r_u), where=r_u > 0)

        # Linear slope computed from intersection with jet-like or wake-like curves
        a = np.where(
            r_u > 20.0,
            0.1 * Zmax - 0.05,  # Intersects with jet-like at y/d = 10
            # Intersects wake-like at y/d = r*3/4:
            denom / 0.75 * (Zmax - (1.6 / 0.73 * 0.75**2) * denom**2),
        )

        # Potential core region (y/d < 10), linear from (y=0, Z=Zmax) to intersection point
        Z = Zmax - a * y

        # Free jet region (10 < y/d < 3r/4), only if r > 20, eq. 3.39:
        Z = np.divide(5.0, y, out=Z, where=np.logical_and(y > 10.0, r_u > 20.0))

        # Wake region (y/rd > 3/4), eq. 3.40:
        y = y * denom
        Z = (1.6 / 0.73) * denom * np.power(y, -2.0, out=Z, where=y > 0.75)

        return np.clip(Z, self.Z_gl[self.i_m], Zmax)

    @cached_property
    def adjustment_factors(self) -> dict[float, list[Callable[[Array], Array]]]:
        print("Computing adjustment factors...")
        adjustment_factors: dict[float, list[Callable[[Array], Array]]] = {}
        self.i_m = slice(None)
        y_cl_max = self.y_cl(self.x[-1])

        # Create grid along the centerline
        n = 1000
        y_cl_arr = np.linspace(0, y_cl_max, n, axis=0)
        x_cl_arr = self.x_cl_from_y_cl(y_cl_arr)

        # Allocate array for auxiliary info
        self._aux_adjust_data = np.zeros((4, *y_cl_arr.shape))
        self._aux_adjust_data[0, ...] = x_cl_arr
        self._aux_adjust_data[1, ...] = y_cl_arr

        for z_inj in self.z_inj:
            interp_inj: list[Callable[[Array], Array]] = []
            for i_m in range(self.nJ):
                if self.J[i_m] == 0.0:
                    interp_inj.append(np.ones_like)
                    continue

                # Iterate over the centerline
                self.i_m = i_m
                adjustment_factor_arr = self.calc_adjustment_factor_xy(
                    x_cl_arr[:, i_m], y_cl_arr[:, i_m], float(z_inj)
                )
                adjustment_factor_arr[np.isnan(adjustment_factor_arr)] = 1.0

                # Interpolate over y because the most rapid variation is near the injection point
                interp_inj.append(
                    interpolate.CubicSpline(y_cl_arr[:, i_m], adjustment_factor_arr)
                )

            adjustment_factors[float(z_inj)] = interp_inj
        return adjustment_factors

    def calc_adjustment_factor_xy(
        self, x_cl: Array, y_cl: Array, z_inj: float
    ) -> Array:
        # Compute the normal to the centerline
        dy_cl_dx = self.dy_cl_dx(x_cl)
        ds = np.stack([np.ones_like(x_cl), dy_cl_dx], axis=0)
        ds /= np.linalg.norm(ds, axis=0)
        dn = np.array([-ds[1], ds[0]])

        # Intersection of the normal with the top and bottom boundaries
        x_top = x_cl + dn[0] * (self.h - y_cl)
        x_bot = x_cl - dn[0] * (y_cl)
        xi_lo = -np.sqrt((x_bot - x_cl) ** 2 + (0.0 - y_cl) ** 2)
        xi_hi = np.sqrt((x_top - x_cl) ** 2 + (self.h - y_cl) ** 2)
        xi_hi *= np.sign(self.h - y_cl)
        yi_lo = -0.5 * self.w - z_inj
        yi_hi = 0.5 * self.w - z_inj
        area = (xi_hi - xi_lo) * (yi_hi - yi_lo)

        # Predicted jet spreading
        Z_cl = self.Z_cl(y_cl)
        sigma2 = np.divide(
            self.Z_cl_int[self.i_m],
            Z_cl * 2 * np.pi,
            out=np.zeros_like(Z_cl),
            where=Z_cl > 0,
        )

        # Analytical denominator of the truncated normal distribution
        s2s = np.sqrt(2.0 * sigma2)
        phix_hi = special.erf(xi_hi / s2s)
        phix_lo = special.erf(xi_lo / s2s)
        phiy_hi = special.erf(yi_hi / s2s)
        phiy_lo = special.erf(yi_lo / s2s)
        x00 = phix_lo * phiy_lo
        x01 = phix_lo * phiy_hi
        x10 = phix_hi * phiy_lo
        x11 = phix_hi * phiy_hi
        V = 0.25 * (x11 - x01 - x10 + x00)

        # # Check against numerically integrated volume under the distribution
        # def gauss(
        #     xy: Array,
        #     sigma2: Array = sigma2,
        #     xi_lo: Array = xi_lo,
        #     xi_hi: Array = xi_hi,
        #     yi_lo: float = yi_lo,
        #     yi_hi: float = yi_hi,
        # ) -> Array:
        #     # Rescale dimensions to the physical domain
        #     x = xy[:, 0, None] * (xi_hi - xi_lo) + xi_lo
        #     y = xy[:, 1, None] * (yi_hi - yi_lo) + yi_lo
        #     n2 = x**2 + y**2

        #     # Return the 2D Gaussian
        #     return np.exp(
        #         np.divide(-n2, 2 * sigma2, out=np.zeros_like(n2), where=sigma2 > 0)
        #     )

        # xy_min = [0.0, 0.0]
        # xy_max = [1.0, 1.0]
        # result = cubature(gauss, a=xy_min, b=xy_max)
        # V2 = result.estimate * area

        # # Check against scipy.stats calculation at single point
        # idx = 100
        # cov = np.diag([sigma2[idx], sigma2[idx]])
        # dist = multivariate_normal([0.0, 0.0], cov)
        # x00 = dist.cdf([xi_lo[idx], yi_lo])
        # x01 = dist.cdf([xi_lo[idx], yi_hi])
        # x10 = dist.cdf([xi_hi[idx], yi_lo])
        # x11 = dist.cdf([xi_hi[idx], yi_hi])
        # V3 = x11 - x01 - x10 + x00

        # print(f"{V[idx] = }")
        # print(f"{V2[idx] = }")
        # print(f"{V3 = }")

        assert self._aux_adjust_data is not None
        self._aux_adjust_data[2, :, self.i_m] = area
        self._aux_adjust_data[3, :, self.i_m] = V

        return np.divide(1.0, V, out=np.ones_like(x_cl), where=V > 0.0)

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
        _, y_cl, n2 = self.nearest_on_cl(x, y, z - z_inj)

        # Compute the centerline fuel mass fraction
        Z_cl = self.Z_cl(y_cl)

        sigma2 = np.divide(
            self.Z_cl_int, Z_cl * 2 * np.pi, out=np.zeros_like(Z_cl), where=Z_cl > 0
        )
        Z = Z_cl * np.exp(
            np.divide(-n2, 2 * sigma2, out=np.zeros_like(sigma2), where=sigma2 > 0)
        )

        # Apply adjustment factor (truncated normal distribution)
        interp_list = self.adjustment_factors[z_inj]
        for i_m in range(self.nJ):
            Z[..., i_m] *= interp_list[i_m](y_cl[..., i_m])

        return Z

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
        Z_cl = self.Z_cl(y_cl)

        # Spreading based on scalar conservation
        # (Assume gaussian, rho_inj * u_inj * Z_inj * A_inj = rho * u * int(Z * dA))
        # where int(Z * dA) = 2 * pi * sigma^2 * Z_cl
        sigma2 = np.divide(
            self.Z_cl_int, Z_cl * 2 * np.pi, out=np.zeros_like(Z_cl), where=Z_cl > 0
        )
        idx = sigma2 > 0

        dxyz = np.stack((x - x_cl, y - y_cl, z - z_inj), axis=0)
        Z = Z_cl * np.exp(
            np.divide(-n2, 2 * sigma2, out=np.zeros_like(sigma2), where=idx)
        )
        grad_Z = np.divide(-dxyz * Z, 2 * sigma2, out=np.zeros_like(sigma2), where=idx)

        # Apply adjustment factor (truncated normal distribution)
        interp_list = self.adjustment_factors[z_inj]
        for i_m in range(self.nJ):
            grad_Z[..., i_m] *= interp_list[i_m](y_cl[..., i_m])

        return grad_Z
