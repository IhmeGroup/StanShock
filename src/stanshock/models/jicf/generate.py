from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy import stats
from scipy.integrate import cubature
from scipy.interpolate import RegularGridInterpolator

from stanshock.models.jicf.profile import AnalyticJICF
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState
from stanshock.system.geometry import Box
from stanshock.utils.h5 import RectilinearVtkhdf, h5_getarray, h5_has, h5_write

if TYPE_CHECKING:
    from stanshock.system.backend import Array


class JICModel:
    """
    This is a class defined to encapsulate the Jet-in-Crossflow model
    """

    def __init__(
        self,
        x_inj: float,
        x_noz: float,
        n_inj: int,
        d_inj: float,
        J: Array,
        manifold_state: FluidState,
        geometry: Box,
        physics: FPVTable,
        theta_inj: float = 0.0,
        alpha: float = 1e6,
        Cd: float = 1.0,
        datadir: Path | str = "./data",
        model_file: str = "jicf_model.vtkhdf",
        x_profile: Array | None = None,
    ) -> None:
        """
        This method initializes the Jet-in-Crossflow model with the following
        parameters:

        The model is tabulated over a range of momentum-flux ratios ``J``
        (the throttle-agnostic table dimension) rather than a specific
        schedule of mass flow rates. The runtime throttle schedule lives in the
        :class:`~stanshock.models.jicf.source.FuelInjector`, which maps the
        live throttle/inflow state onto this table via ``J``.

        x_inj: float
            The x-coordinate of the injection point
        x_noz: float
            The x-coordinate of the nozzle start
        n_inj: float
            The number of injected jets
        d_inj: float
            The diameter of the injected jet
        J: np.ndarray
            The grid of momentum-flux ratios to tabulate over. Sets the table
            resolution and range; each entry corresponds to one injected-fluid
            state ``(rho_inj[i], u_inj[i], T_inj[i])``.
        manifold_state: FluidState
            Thermodynamic state in the propellant manifold.
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        theta_inj: float
            The angle of the jet relative to the x axis (rads)
        alpha: float
            The relaxation parameter (used here only for storage)
        Cd: float
            Discharge coefficient for the injector
        datadir: str
            Where to access or store tables written for this injector
        model_file: str
            Name of the HDF5 file caching the generated model tables, resolved
            relative to ``datadir`` (i.e. ``<datadir>/<model_file>``). Tables
            already present in the file are loaded; any that are missing are
            computed and written to it.
        x_profile: np.ndarray | None
            Axial mesh on which to generate the stored Z mean/variance profiles.
            When ``None`` (default), the stretched grid ``self.x_3D_data`` used
            for the 3D field is reused. When supplied, it is used verbatim.
            Ignored when the profiles are loaded from an existing model file
            (the stored mesh is used instead).
        """
        self.geometry = geometry
        self.physics = physics

        # Set the manifold properties
        self.manifold_state = manifold_state

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.geometry.idx_cells]
        self.x_inj = x_inj
        self.x_noz = x_noz
        self.w = float(self.geometry.w(0.0, np.array(self.x_inj)))
        self.h = float(self.geometry.h(0.0, np.array(self.x_inj)))
        self.n_inj = n_inj
        self.d_inj = d_inj
        self.theta_inj = theta_inj

        self.alpha = alpha
        self.datadir = Path(datadir)
        self.datadir.mkdir(exist_ok=True)
        self.model_file = self.datadir / model_file

        # Optional user-supplied mesh for the stored Z profiles; when None the
        # stretched 3D grid (self.x_3D_data) is used (see calc_Z_avg_var_profiles).
        self._x_profile_input = x_profile

        # Geometry parameters
        self.A = self.w * self.h
        self.A_inj = np.pi * (0.5 * self.d_inj) ** 2
        self.Ae = Cd * self.n_inj * self.A_inj

        # Momentum-flux-ratio grid (the table dimension). Sort ascending and
        # carry the per-grid injected-fluid state along so the tables and the
        # RegularGridInterpolators built below share a monotone axis.
        order = np.argsort(J)
        self.J = J[order]
        self.nJ = len(self.J)

        # Set up the analytic JICF model
        self.analytic = AnalyticJICF(
            x=self.xc - x_inj,
            w=self.w,
            h=self.h,
            n_inj=n_inj,
            d_inj=d_inj,
            J=J,
            theta_inj=theta_inj,
            Cd=Cd,
        )

        # Precompute a 3D array of the mixture fraction and generate an interpolator.
        # Only reuse a cached table if it was generated on the same J grid;
        # otherwise it belongs to a different throttle range and is recomputed.
        if h5_has(self.model_file, "VTKHDF/PointData/Z") and self._cached_J_matches(
            "VTKHDF/Steps/Values"
        ):
            vtk = RectilinearVtkhdf.from_vtkhdf(self.model_file)
            self.x_3D_data = vtk.x
            self.y_3D_data = vtk.y
            self.z_3D_data = vtk.z
            self.Z_3D_data = vtk.vals["Z"]
        else:
            self.calc_Z_3D_interp(write=True)

        # Precompute the axial mean and variance profiles of Z, along with the
        # mesh they were generated on.
        if h5_has(self.model_file, "Z_profiles/Z_avg") and self._cached_J_matches(
            "Z_profiles/J"
        ):
            with h5py.File(str(self.model_file), "r") as f:
                group = f["Z_profiles"]
                assert isinstance(group, h5py.Group)
                self.x_profile = h5_getarray(group, "x")
                self.Z_avg_profile = h5_getarray(group, "Z_avg")
                self.Z_var_profile = h5_getarray(group, "Z_var")
        else:
            self.calc_Z_avg_var_profiles(write=True)

        # Map J -> Z mean/variance on the *generation* mesh (self.x_profile).
        # Building the interpolators on the stored mesh rather than the current
        # simulation mesh decouples the profiles from the simulation grid, so the
        # model does not need to be regenerated when the mesh changes. Out-of-range
        # query points (e.g. ghost cells) are linearly extrapolated.
        self.Z_avg_profile_interp = RegularGridInterpolator(
            (self.x_profile, self.J),
            self.Z_avg_profile,
            bounds_error=False,
            fill_value=None,
        )
        self.Z_var_profile_interp = RegularGridInterpolator(
            (self.x_profile, self.J),
            self.Z_var_profile,
            bounds_error=False,
            fill_value=None,
        )

    @property
    def manifold_state(self) -> FluidState:
        """Thermodynamic properties in the propellant manifold."""
        return self._manifold_state

    @manifold_state.setter
    def manifold_state(self, state: FluidState) -> None:
        # Update stored properties
        self._manifold_state = state
        self.gamma0 = float(self.physics.get_gamma(state)[0])
        self.p0 = float(self.physics.get_pressure(state)[0])
        self.rho0 = float(self.physics.get_density(state)[0])
        self.T0 = float(self.physics.get_temperature(state)[0])
        self.R0 = float(self.physics.get_specific_gas_constant(state)[0])

    def _cached_J_matches(self, dataset: str) -> bool:
        """Return True if the cached group was generated on the current J grid.

        Guards against silently reusing a table tabulated over a different
        throttle range (momentum-flux-ratio grid).
        """
        if not self.model_file.exists():
            return False
        with h5py.File(str(self.model_file), "r") as f:
            if dataset not in f:
                return False
            J_cached = h5_getarray(f, dataset)
        return J_cached.shape == self.J.shape and np.allclose(J_cached, self.J)

    def _stretched_grid(
        self,
        x_start: float,
        x_end: float,
        dx: float,
        growth_rate: float,
        target_x: float,
    ) -> Array:
        x_grid = [x_start, x_end]
        for direction in [-1, 1]:
            x, spacing = target_x, dx
            while x_start <= x <= x_end:
                x_grid.append(x)
                x += direction * spacing
                spacing *= growth_rate
        return np.sort(np.unique(x_grid))

    def calc_Z_3D_interp(self, write: bool = False, debug: bool = False) -> None:
        print("Computing Z 3D array...")
        dx = 1.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        self.y_3D_data = np.linspace(0, self.h, Ny)
        self.z_3D_data = np.linspace(-self.w / 2, self.w / 2, Nz)
        self.x_3D_data = self._stretched_grid(
            self.xc[0], self.xc[-1], dx, 1.1, self.x_inj
        )
        self.analytic.i_m = slice(None)
        self.Z_3D_data = self.analytic.Z_3D(  # Z_3D_adjusted(
            self.x_3D_data[:, None, None] - self.x_inj,
            self.y_3D_data[None, :, None],
            self.z_3D_data[None, None, :],
        )
        self.Z_3D_data[..., self.J == 0.0] = 0.0
        self.Z_3D_data[np.isnan(self.Z_3D_data)] = 0.0

        results: dict[str, Array] = {"Z": self.Z_3D_data}

        if debug:
            z_inj = self.analytic.z_inj[0]
            x_cl, y_cl, n2 = self.analytic.nearest_on_cl(
                self.x_3D_data[:, None, None] - self.x_inj,
                self.y_3D_data[None, :, None],
                self.z_3D_data[None, None, :] - z_inj,
            )
            Z_cl = self.analytic.Z_cl(y_cl)
            sigma2 = np.divide(
                self.analytic.Z_cl_int,
                Z_cl * 2 * np.pi,
                out=np.zeros_like(Z_cl),
                where=Z_cl > 0,
            )
            results["x_cl"] = x_cl
            results["y_cl"] = y_cl
            results["n"] = np.sqrt(n2)
            results["sigma2"] = sigma2

        if write:
            vtk = RectilinearVtkhdf(
                self.x_3D_data,
                self.y_3D_data,
                self.z_3D_data,
                results,
                self.model_file,
                self.J,
            )
            vtk.save()

    @cached_property
    def Z_3D_interp(self) -> list[RegularGridInterpolator[np.float64]]:
        """Interpolators for the 3D mixture fraction fields.

        These are computed once when first accessed, and cached for reuse.
        """
        Z_3D_interp: list[RegularGridInterpolator[np.float64]] = []
        for i_m in range(self.nJ):
            interp = RegularGridInterpolator(
                (self.x_3D_data, self.y_3D_data, self.z_3D_data),
                self.Z_3D_data[..., i_m],
                method="cubic",
            )
            Z_3D_interp.append(interp)
        return Z_3D_interp

    def _mu_Z(self, yz: Array, x: Array) -> Array:
        """Vectorized evaluation of mixture fraction over set of y + z points.

        Returns flattened Z array for all [nyz * nx * nm] points.
        """
        y: Array
        z: Array
        y, z = yz[:, 0, None], yz[:, 1, None]
        return np.reshape(self.analytic.Z_3D(x, y, z), (y.shape[0], -1))

    def _sigma_Z(self, yz: Array, x: Array, Z_avg: Array) -> Array:
        """Vectorized evaluation of mixture fraction variance over set of y + z points.

        Returns flattened Z variance array for all [nyz * nx * nm] points.
        """
        Z = self._mu_Z(yz, x)
        return (Z - np.reshape(Z_avg, (1, -1))) ** 2

    def Z_avg_var(self, x: Array) -> tuple[Array, Array]:
        """Compute the mean and variance of the mixture fraction profiles at given axial locations.

        Returns axial mean and variance profiles for all mass flow rates.
        """
        # res = cubature(
        #     self._mu_Z,
        #     a=[0.0, 0.0],
        #     b=[self.h, 0.5 * self.w],
        #     args=(x,),
        #     # workers=-1,
        # )
        # Z_avg: Array = np.asarray(
        #     2.0
        #     * np.reshape(res.estimate, (*x.shape, *self.mdot_inj_unique.shape))
        #     / self.A,
        #     dtype=float,
        # )
        #
        # res = cubature(
        #     self._sigma_Z,
        #     a=[0.0, 0.0],
        #     b=[self.h, 0.5 * self.w],
        #     args=(x, Z_avg),
        #     # workers=-1,
        # )
        # Z_var: Array = np.asarray(
        #     2.0
        #     * np.reshape(res.estimate, (*x.shape, *self.mdot_inj_unique.shape))
        #     / self.A,
        #     dtype=float,
        # )

        dx = 1e-4  # 5.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        y = np.linspace(0.0, self.h, Ny)[None, :, None]
        z = np.linspace(0.0, 0.5 * self.w, Nz)[None, None, :]
        self.analytic.i_m = slice(None)
        # Z = self.analytic.Z_3D_adjusted(x[:, None, None] - self.x_inj, y, z)
        Z = self.analytic.Z_3D(x[:, None, None] - self.x_inj, y, z)
        Z_avg = np.mean(Z, axis=(1, 2))
        Z_var = np.mean((Z - Z_avg[:, None, None, :]) ** 2, axis=(1, 2))

        return Z_avg, Z_var

    def _mu_Z_adjusted(self, yz: Array, x: Array, i_m: int) -> Array:
        """Vectorized evaluation of mixture fraction over set of y + z points.

        Returns flattened Z array for all [nyz * nx * nm] points.
        """
        y: Array
        z: Array
        y, z = yz[:, 0, None], yz[:, 1, None]
        return np.reshape(self.Z_3D_interp[i_m]((x, y, z)), (y.shape[0], -1))

    def _sigma_Z_adjusted(self, yz: Array, x: Array, Z_avg: Array, i_m: int) -> Array:
        """Vectorized evaluation of mixture fraction variance over set of y + z points.

        Returns flattened Z variance array for all [nyz * nx * nm] points.
        """
        Z = self._mu_Z_adjusted(yz, x, i_m)
        return (Z - np.reshape(Z_avg, (1, -1))) ** 2

    def Z_avg_var_adjusted(self, x: Array) -> tuple[Array, Array]:
        """Compute the mean and variance of the mixture fraction profiles at given axial locations.

        Returns axial mean and variance profiles for all mass flow rates.
        """
        Z_avg = np.zeros((*x.shape, self.nJ))
        Z_var = np.zeros((*x.shape, self.nJ))
        for i_m in range(self.nJ):
            res = cubature(
                self._mu_Z_adjusted,
                a=[0.0, 0.0],
                b=[self.h, 0.5 * self.w],
                args=(x, i_m),
                # workers=-1,
            )
            Z_avg[..., i_m] = 2.0 * res.estimate / self.A

            res = cubature(
                self._sigma_Z_adjusted,
                a=[0.0, 0.0],
                b=[self.h, 0.5 * self.w],
                args=(x, Z_avg[..., i_m], i_m),
                # workers=-1,
            )
            Z_var[..., i_m] = 2.0 * res.estimate / self.A

        return Z_avg, Z_var

    def calc_Z_avg_var_profiles(self, write: bool = False) -> None:
        print("Computing Z average and variance profiles...")
        # Record the mesh the profiles are generated on so the interpolators can
        # be rebuilt independently of the simulation mesh. Use the caller-supplied
        # mesh if given, otherwise the stretched 3D grid (which spans the same
        # [xc[0], xc[-1]] interval as the simulation mesh).
        # self.x_profile = (
        #     self.x_3D_data if self._x_profile_input is None else self._x_profile_input
        # )
        # self.Z_avg_profile = np.zeros((self.x_profile.shape[0], self.nJ))
        # self.Z_var_profile = np.zeros((self.x_profile.shape[0], self.nJ))

        # # Compute profiles between injector and nozzle
        # idx = np.logical_and(self.x_profile > self.x_inj, self.x_profile < self.x_noz)
        # self.Z_avg_profile[idx, :], self.Z_var_profile[idx, :] = (
        #     # self.Z_avg_var_adjusted(self.x_profile[idx])
        #     self.Z_avg_var(self.x_profile[idx])
        # )

        # Freeze profiles downstream of the nozzle
        # ifreeze = len(idx) - 1 - np.argmax(idx[::-1])
        # idx = self.x_profile >= self.x_noz
        # self.Z_avg_profile[idx, :] = self.Z_avg_profile[ifreeze, :]
        # self.Z_var_profile[idx, :] = self.Z_var_profile[ifreeze, :]

        self.x_profile = self.x_3D_data
        Z = self.Z_3D_data
        self.Z_avg_profile = np.mean(Z, axis=(1, 2))
        self.Z_var_profile = np.mean(
            (Z - self.Z_avg_profile[:, None, None, :]) ** 2, axis=(1, 2)
        )

        if write:
            h5_write(
                self.model_file,
                "Z_profiles",
                {
                    "J": self.J,
                    "x": self.x_profile,
                    "Z_avg": self.Z_avg_profile,
                    "Z_var": self.Z_var_profile,
                },
            )

    def estimate_p_Z(self, x: Array, Z: Array) -> Array:
        """
        This method estimates the PDF of the mixture fraction at a given point
        using a Beta distribution.
        x: Array
            The query x-coordinate
        Z: Array
            The query mixture fraction
        """
        Z_avg, Z_var = self.Z_avg_var_adjusted(x)
        if np.all(Z_avg == 0.0):
            return Z_avg
        a = ((Z_avg * (1 - Z_avg) / Z_var) - 1) * Z_avg
        b = a * (1 - Z_avg) / Z_avg
        return np.asarray(stats.beta.pdf(Z, a, b))
