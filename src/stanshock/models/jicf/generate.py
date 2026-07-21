from __future__ import annotations

import functools
import warnings
from pathlib import Path

import cantera as ct
import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy import special, stats
from scipy.integrate import cubature
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from tqdm import tqdm

from stanshock.models.jicf.profile import AnalyticJICF
from stanshock.physics.flamelet import FPVTable
from stanshock.system.backend import Array
from stanshock.system.geometry import Box


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
        t_inj: Array,
        phi_inj: Array,
        rho_inj: Array,
        u_inj: Array,
        T_inj: Array,
        rho: float,
        u: float,
        T: float,
        geometry: Box,
        physics: FPVTable,
        datadir: Path | str = "./data",
        theta_inj: float = 0.0,
        alpha: float = 1e6,
        model_file: str = "jicf_model.h5",
        x_profile: Array | None = None,
    ) -> None:
        """
        This method initializes the Jet-in-Crossflow model with the following
        parameters:
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
        t_inj: np.ndarray
            Time array for the injection profile
        phi_inj: np.ndarray
            Scheduled equivalence ratio of injected jet
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

        assert self.physics.fuel_def is not None
        self.fuel_def = self.physics.fuel_def
        assert self.physics.ox_def is not None
        self.ox_def = self.physics.ox_def

        gas = self.physics.gas

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.geometry.idx_cells]
        self.x_inj = x_inj
        self.x_noz = x_noz
        self.w = float(self.geometry.w(0.0, np.array(self.x_inj)))
        self.h = float(self.geometry.h(0.0, np.array(self.x_inj)))
        self.n_inj = n_inj
        self.d_inj = d_inj
        self.theta_inj = theta_inj

        self.rho_inj = rho_inj
        self.u_inj = u_inj
        self.T_inj = T_inj

        self.rho = rho

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

        # Free stream properties
        self.u = u
        self.T = T
        gas.TDX = self.T, self.rho, self.ox_def
        self.p = gas.P
        self.W = gas.mean_molecular_weight
        self.gamma = gas.cp / gas.cv
        self.c = gas.sound_speed
        self.M = self.u / self.c
        self.Y_ox = gas.Y

        # Properties of the injected fluid
        self.t_inj = t_inj
        self.phi_inj = phi_inj
        sol = ct.SolutionArray(gas, shape=self.t_inj.shape)
        sol.TDX = self.T_inj, self.rho_inj, self.fuel_def  # type: ignore[assignment]
        self.p_inj = sol.P
        self.E_inj = sol.int_energy_mass + 0.5 * self.u_inj**2
        self.W_inj = sol.mean_molecular_weight
        self.gamma_inj = sol.cp / sol.cv
        self.c_inj = sol.sound_speed
        self.Y_fuel = sol.Y[0]
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
        self.u_inj_unique = self.u_inj[self.mdot_inj_unique_idx]
        self.p_inj_unique = self.p_inj[self.mdot_inj_unique_idx]

        # Mass flow rate and equivalence ratio schedules
        self.mdot_f_interp = make_interp_spline(self.t_inj, self.mdot_inj, k=1)
        self.phi_f_interp = make_interp_spline(self.t_inj, self.phi_inj, k=1)

        # Set up the analytic JICF model
        self.analytic = AnalyticJICF(
            x=self.xc - x_inj,
            w=self.w,
            h=self.h,
            n_inj=n_inj,
            d_inj=d_inj,
            rho_inj=self.rho_inj_unique,
            u_inj=self.u_inj_unique,
            rho=rho,
            u=u,
            physics=self.physics,
            theta_inj=theta_inj,
        )

        # Precompute a 3D array of the mixture fraction and generate an interpolator
        if self._h5_has("Z_3D/Z"):
            with h5py.File(str(self.model_file), "r") as f:
                group = f["Z_3D"]
                assert isinstance(group, h5py.Group)
                self.x_3D_data = self._h5_getarray(group, "x")
                self.y_3D_data = self._h5_getarray(group, "y")
                self.z_3D_data = self._h5_getarray(group, "z")
                self.Z_3D_data = self._h5_getarray(group, "Z")
            self._build_Z_3D_interp()
        else:
            self.calc_Z_3D_interp(write=True)

        # Precompute the axial mean and variance profiles of Z, along with the
        # mesh they were generated on.
        if self._h5_has("Z_profiles/Z_avg"):
            with h5py.File(str(self.model_file), "r") as f:
                group = f["Z_profiles"]
                assert isinstance(group, h5py.Group)
                self.x_profile = self._h5_getarray(group, "x")
                self.Z_avg_profile = self._h5_getarray(group, "Z_avg")
                self.Z_var_profile = self._h5_getarray(group, "Z_var")
        else:
            self.calc_Z_avg_var_profiles(write=True)

        # Map mdot -> Z mean/variance on the *generation* mesh (self.x_profile).
        # Building the interpolators on the stored mesh rather than the current
        # simulation mesh decouples the profiles from the simulation grid, so the
        # model does not need to be regenerated when the mesh changes. Out-of-range
        # query points (e.g. ghost cells) are linearly extrapolated.
        self.Z_avg_profile_interp = RegularGridInterpolator(
            (self.x_profile, self.mdot_inj_unique),
            self.Z_avg_profile,
            bounds_error=False,
            fill_value=None,
        )
        self.Z_var_profile_interp = RegularGridInterpolator(
            (self.x_profile, self.mdot_inj_unique),
            self.Z_var_profile,
            bounds_error=False,
            fill_value=None,
        )

        # Precompute and tabulate chemical source terms
        if self._h5_has("chemical_sources/omega_C_int"):
            with h5py.File(str(self.model_file), "r") as f:
                group = f["chemical_sources"]
                assert isinstance(group, h5py.Group)
                self.Zbar_vec = self._h5_getarray(group, "Zbar")
                self.Lbar_vec = self._h5_getarray(group, "Lbar")
                self.logsigma2_vec = self._h5_getarray(group, "logsigma2")
                self.omega_C_int = self._h5_getarray(group, "omega_C_int")
            self.omega_C_int_interp = RegularGridInterpolator(
                (self.Zbar_vec, self.Lbar_vec, self.logsigma2_vec), self.omega_C_int
            )
        else:
            self.calc_chemical_sources(write=True)

    def _h5_has(self, key: str) -> bool:
        """Return True if the model file exists and contains the given dataset."""
        if not self.model_file.exists():
            return False
        with h5py.File(str(self.model_file), "r") as f:
            return key in f

    def _h5_getarray(self, h: h5py.File | h5py.Group, key: str) -> Array:
        """Load array data from h5 file in a way that respects type checking."""
        data_handle = h[key]
        assert isinstance(data_handle, h5py.Dataset)
        return np.asarray(data_handle[...], dtype=float)

    def _h5_write(self, group: str, data: dict[str, Array]) -> None:
        """Write (overwriting if present) a group of named arrays to the model file."""
        with h5py.File(str(self.model_file), "a") as f:
            grp = f.require_group(group)
            for name, array in data.items():
                if name in grp:
                    del grp[name]
                grp[name] = array

    def __stretched_grid(
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

    def calc_Z_3D_interp(self, write: bool = False) -> None:
        print("Computing Z 3D array...")
        dx = 5.0e-4
        Ny = int(np.ceil(self.h / dx))
        Nz = int(np.ceil(self.w / dx))
        self.y_3D_data = np.linspace(0, self.h, Ny)
        self.z_3D_data = np.linspace(-self.w / 2, self.w / 2, Nz)
        self.x_3D_data = self.__stretched_grid(
            self.xc[0], self.xc[-1], dx, 1.1, self.x_inj
        )
        self.analytic.i_m = slice(None)
        self.Z_3D_data = self.analytic.Z_3D_adjusted(
            self.x_3D_data[:, None, None] - self.x_inj,
            self.y_3D_data[None, :, None],
            self.z_3D_data[None, None, :],
        )
        self.Z_3D_data[..., np.isnan(self.rho_inj_unique)] = 0.0
        self.Z_3D_data[..., self.u_inj_unique == 0.0] = 0.0
        self.Z_3D_data[np.isnan(self.Z_3D_data)] = 0.0

        if write:
            self._h5_write(
                "Z_3D",
                {
                    "x": self.x_3D_data,
                    "y": self.y_3D_data,
                    "z": self.z_3D_data,
                    "Z": self.Z_3D_data,
                },
            )

        self._build_Z_3D_interp()

    def _build_Z_3D_interp(self) -> None:
        self.Z_3D_interp: list[RegularGridInterpolator[np.float64]] = []
        for i_m in range(len(self.mdot_inj_unique)):
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
        res = cubature(
            self._mu_Z,
            a=[0.0, 0.0],
            b=[self.h, 0.5 * self.w],
            args=(x,),
            # workers=-1,
        )
        Z_avg: Array = np.asarray(
            2.0
            * np.reshape(res.estimate, (*x.shape, *self.mdot_inj_unique.shape))
            / self.A,
            dtype=float,
        )

        res = cubature(
            self._sigma_Z,
            a=[0.0, 0.0],
            b=[self.h, 0.5 * self.w],
            args=(x, Z_avg),
            # workers=-1,
        )
        Z_var: Array = np.asarray(
            2.0
            * np.reshape(res.estimate, (*x.shape, *self.mdot_inj_unique.shape))
            / self.A,
            dtype=float,
        )

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
        Z_avg = np.zeros((*x.shape, len(self.mdot_inj_unique)))
        Z_var = np.zeros((*x.shape, len(self.mdot_inj_unique)))
        for i_m in range(len(self.mdot_inj_unique)):
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

        # Compute profiles between injector and nozzle
        idx = np.logical_and(self.x_profile > self.x_inj, self.x_profile < self.x_noz)
        self.Z_avg_profile[:, idx], self.Z_var_profile[:, idx] = (
            # self.Z_avg_var_adjusted(self.xc[idx])
            self.Z_avg_var(self.x_profile[idx])
        )

        # Freeze profiles downstream of the nozzle
        ifreeze = len(idx) - 1 - np.argmax(idx[::-1])
        idx = self.x_profile >= self.x_noz
        self.Z_avg_profile[:, idx] = self.Z_avg_profile[:, ifreeze]
        self.Z_var_profile[:, idx] = self.Z_var_profile[:, ifreeze]

        if write:
            self._h5_write(
                "Z_profiles",
                {
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
            self._h5_write(
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
