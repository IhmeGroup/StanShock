from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy import special, stats
from scipy.integrate import cubature
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

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
        fuel_state: FluidState,
        geometry: Box,
        physics: FPVTable,
        datadir: Path | str = "./data",
        theta_inj: float = 0.0,
        alpha: float = 1e6,
        Cd: float = 0.7,
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
        theta_inj: float
            The angle of the jet relative to the x axis (rads)
        J: np.ndarray
            The grid of momentum-flux ratios to tabulate over. Sets the table
            resolution and range; each entry corresponds to one injected-fluid
            state ``(rho_inj[i], u_inj[i], T_inj[i])``.
        fuel_state: FluidState
            Thermodynamic state of the fuel manifold.
        alpha: float
            The relaxation parameter (used here only for storage)
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
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        """
        self.geometry = geometry
        self.physics = physics

        # Set the fuel properties
        self.fuel_state = fuel_state

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
        self.A_inj = np.pi * (self.d_inj / 2.0) ** 2
        self.Ae = Cd * self.A_inj

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
            self._build_Z_3D_interp()
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

        # Precompute and tabulate chemical source terms
        if h5_has(self.model_file, "chemical_sources/omega_C_int"):
            with h5py.File(str(self.model_file), "r") as f:
                group = f["chemical_sources"]
                assert isinstance(group, h5py.Group)
                self.Zbar_vec = h5_getarray(group, "Zbar")
                self.Lbar_vec = h5_getarray(group, "Lbar")
                self.logsigma2_vec = h5_getarray(group, "logsigma2")
                self.omega_C_int = h5_getarray(group, "omega_C_int")
            self.omega_C_int_interp = RegularGridInterpolator(
                (self.Zbar_vec, self.Lbar_vec, self.logsigma2_vec), self.omega_C_int
            )
        else:
            self.calc_chemical_sources(write=True)

    @property
    def fuel_state(self) -> FluidState:
        return self._fuel_state

    @fuel_state.setter
    def fuel_state(self, state: FluidState) -> None:
        # Update stored properties of the fuel
        self._fuel_state = state
        self.gamma_inj = float(self.physics.get_gamma(state))
        self.p_inj = float(self.physics.get_pressure(state))
        self.rho_inj = float(self.physics.get_pressure(state))
        self.T_inj = float(self.physics.get_temperature(state))
        self.R_inj = float(self.physics.get_specific_gas_constant(state))
        self.e0_inj = float(self.physics.get_internal_energy(state))

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
        dx = 5.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        self.y_3D_data = np.linspace(0, self.h, Ny)
        self.z_3D_data = np.linspace(-self.w / 2, self.w / 2, Nz)
        self.x_3D_data = self._stretched_grid(
            self.xc[0], self.xc[-1], dx, 1.1, self.x_inj
        )
        self.analytic.i_m = slice(None)
        self.Z_3D_data = self.analytic.Z_3D_adjusted(
            self.x_3D_data[:, None, None] - self.x_inj,
            self.y_3D_data[None, :, None],
            self.z_3D_data[None, None, :],
        )
        self.Z_3D_data[..., np.isnan(self.rho_inj)] = 0.0
        self.Z_3D_data[..., self.u_inj == 0.0] = 0.0
        self.Z_3D_data[np.isnan(self.Z_3D_data)] = 0.0

        results: dict[str, Array] = {"Z": self.Z_3D_data}

        if debug:
            z_inj = self.analytic.z_inj[0]
            x_cl, y_cl, n2 = self.analytic.nearest_on_cl(
                self.x_3D_data[:, None, None] - self.x_inj,
                self.y_3D_data[None, :, None],
                self.z_3D_data[None, None, :] - z_inj,
            )
            results["x_cl"] = x_cl
            results["y_cl"] = y_cl
            results["n"] = np.sqrt(n2)

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

        self._build_Z_3D_interp()

    def _build_Z_3D_interp(self) -> None:
        self.Z_3D_interp: list[RegularGridInterpolator[np.float64]] = []
        for i_m in range(self.nJ):
            interp = RegularGridInterpolator(
                (self.x_3D_data, self.y_3D_data, self.z_3D_data),
                self.Z_3D_data[..., i_m],
                method="cubic",
            )
            self.Z_3D_interp.append(interp)

    def _mu_Z(self, yz: Array, x: Array) -> Array:
        """Vectorized evaluation of mixture fraction over set of y + z points.

        Returns flattened Z array for all [nyz * nx * nm] points.
        """
        y: Array
        z: Array
        y, z = yz[:, 0, None], yz[:, 1, None]
        return np.reshape(self.analytic.Z_3D_adjusted(x, y, z), (y.shape[0], -1))

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

        dx = 5.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        y = np.linspace(0.0, self.h, Ny)[None, :, None]
        z = np.linspace(0.0, 0.5 * self.w, Nz)[None, None, :]
        self.analytic.i_m = slice(None)
        Z = self.analytic.Z_3D_adjusted(x[:, None, None] - self.x_inj, y, z)
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
        self.x_profile = (
            self.x_3D_data if self._x_profile_input is None else self._x_profile_input
        )
        self.Z_avg_profile = np.zeros((self.x_profile.shape[0], self.nJ))
        self.Z_var_profile = np.zeros((self.x_profile.shape[0], self.nJ))

        # Compute profiles between injector and nozzle
        idx = np.logical_and(self.x_profile > self.x_inj, self.x_profile < self.x_noz)
        self.Z_avg_profile[idx, :], self.Z_var_profile[idx, :] = (
            # self.Z_avg_var_adjusted(self.x_profile[idx])
            self.Z_avg_var(self.x_profile[idx])
        )

        # Freeze profiles downstream of the nozzle
        ifreeze = len(idx) - 1 - np.argmax(idx[::-1])
        idx = self.x_profile >= self.x_noz
        self.Z_avg_profile[idx, :] = self.Z_avg_profile[ifreeze, :]
        self.Z_var_profile[idx, :] = self.Z_var_profile[ifreeze, :]

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

    @staticmethod
    def _compute_omega_C_int(
        i_Zbar: int,
        i_Lbar: int,
        i_S: int,
        Zbar_vec: Array,
        Lbar_vec: Array,
        logsigma2_vec: Array,
        uv_vec: Array,
        W_vec: Array,
        omega_C_interp: RegularGridInterpolator[np.float64],
    ) -> tuple[int, int, int, float]:
        Zbar = Zbar_vec[i_Zbar]
        Lbar = Lbar_vec[i_Lbar]
        logsigma2 = logsigma2_vec[i_S]
        sigma2 = 10**logsigma2

        eps = 1.0e-6
        if (Zbar < eps) or (Zbar > 1 - eps) or (Lbar < eps) or (Lbar > 1 - eps):
            return (i_Zbar, i_Lbar, i_S, float(omega_C_interp((Zbar, Lbar))))

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
            result = float(np.sum(integrand * W_mesh))

        return (i_Zbar, i_Lbar, i_S, result)

    def calc_chemical_sources(self, write: bool = False) -> None:
        """
        This method precomputes the chemical source terms as a function of x, mdot_f, and L.
        """
        print("Precomputing chemical sources...")

        # Build omega_C interpolator
        Z_sample = np.linspace(0.0, 1.0, 100)
        L_sample = np.linspace(0.0, 1.0, 100)
        Z_sample_mesh, L_sample_mesh = np.meshgrid(Z_sample, L_sample, indexing="ij")
        omega_C = self.physics.lookup_direct(
            "SRC_PROG", Z_sample_mesh, 0.0, L_sample_mesh
        )
        omega_C_interp = RegularGridInterpolator(
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
        gen = Parallel(n_jobs=-1, return_as="generator")(
            delayed(compute_func)(i_Zbar, i_Lbar, i_S) for i_Zbar, i_Lbar, i_S in tasks
        )
        results = list(tqdm(gen, total=len(tasks)))

        for i_Zbar, i_Lbar, i_S, value in results:
            self.omega_C_int[i_Zbar, i_Lbar, i_S] = value
        self.omega_C_int[np.isnan(self.omega_C_int)] = 0.0

        if write:
            h5_write(
                self.model_file,
                "chemical_sources",
                {
                    "Zbar": self.Zbar_vec,
                    "Lbar": self.Lbar_vec,
                    "logsigma2": self.logsigma2_vec,
                    "omega_C_int": self.omega_C_int,
                },
            )

        # Build 3D table interpolator
        self.omega_C_int_interp = RegularGridInterpolator(
            (self.Zbar_vec, self.Lbar_vec, self.logsigma2_vec), self.omega_C_int
        )
