from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide


class SkinFriction:
    """Compressibility-aware skin-friction model with tabulated Cf(Re, M, T/Tw).

    This is unchanged from the original implementation. It returns an *effective*
    skin-friction coefficient (Cf) meant to be used with:

        tau_w = Cf * (1/2) * rho * U^2

    Notes
    -----
    - The underlying fit assumes a recovery factor of ~0.88 inside the table build.
      That is fine for wall shear modeling; for wall heat transfer we separately
      handle recovery/adiabatic wall temperature in BoundaryLayer.
    """

    def __init__(
        self,
        ReMin: float = 100.0,
        ReMax: float = 1e7,
        MachMin: float = 0.01,
        MachMax: float = 4.0,
        T_ratMin: float = 0.01,
        T_ratMax: float = 8.0,
    ):
        self.ReMax = ReMax
        self.ReMin = ReMin
        self.MachMin = MachMin
        self.MachMax = MachMax
        self.T_ratMin = T_ratMin
        self.T_ratMax = T_ratMax

        self.N_Re = 30
        self.N_Mach = 30
        self.N_T_rat = 11

        self.gamma = 1.4

        self.ReTable = np.logspace(np.log10(self.ReMin), np.log10(self.ReMax), self.N_Re)
        self.MachTable = np.linspace(self.MachMin, self.MachMax, int(self.N_Mach))
        self.T_ratTable = np.linspace(self.T_ratMin, self.T_ratMax, self.N_T_rat)

        self.M_grid, self.T_grid = np.meshgrid(
            self.MachTable, self.T_ratTable, indexing="ij"
        )
        self.F, self.G = self._precompute_FG(self.M_grid, self.T_grid)

        self.cfTable = np.zeros((self.N_Re, self.N_Mach, self.N_T_rat))
        self._build_table()

        self.interp = RegularGridInterpolator(
            (self.ReTable, self.MachTable, self.T_ratTable),
            self.cfTable,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, Re: Array, M: Array, T_rat: Array) -> Array:
        pts = np.array([Re, M, T_rat]).T
        return self.interp(pts)

    def _precompute_FG(self, M: Array, T_rat: Array) -> tuple[Array, Array]:
        gamma = self.gamma
        recovery = 0.88
        C_axi = -0.6005

        k = 0.5 * (gamma - 1.0) * M**2
        A = (T_rat - 1.0) + k * T_rat
        B = np.sqrt(k * T_rat * recovery)

        D = np.sqrt(A**2 + 4.0 * B**2)
        C = np.arcsin((2.0 * B**2 - A) / D) + np.arcsin(A / D)

        E = T_rat * (1.505 / (1.0 + (0.505 / T_rat)))

        F = C / np.sqrt(k * recovery)
        G = 1.77 * np.log(E) + C_axi
        return F, G

    def _f(self, x: Array, Re: float) -> Array:
        return 1.77 * np.log(x * Re) - (self.F / x) + self.G

    def _dfdx(self, x: Array, Re: float) -> Array:
        return (1.77 / x) + (self.F / x**2)

    def _solve_cf_slice(self, Re: float) -> Array:
        x0 = np.sqrt(3e-3) * np.ones_like(self.F)
        sqrt_cf = newton(
            func=self._f,
            fprime=self._dfdx,
            x0=x0,
            args=(Re,),
            tol=1e-10,
            maxiter=100,
        )
        return sqrt_cf**2

    def _build_table(self) -> None:
        for i, Re in enumerate(self.ReTable):
            self.cfTable[i, :, :] = self._solve_cf_slice(float(Re))


class BoundaryLayer(RightHandSide):
    """Wall shear + wall heat transfer source terms for quasi-1D internal flow.

    This version is designed to be *defensible for scramjet-like, high-speed flows*.

    Key changes vs. the experimental "parallel plates" version:
    - Removes ad-hoc aspect-ratio multipliers and geometry-specific methods (w/h).
    - Uses perimeter/area (P/A) directly, which is the correct conversion from wall
      fluxes (tau_w, q'') to volumetric sink terms in a 1D control-volume model.
    - Uses a recovery/adiabatic-wall driving temperature by default:
          q'' = St * rho * |U| * cp * (T_aw - T_w)
      rather than q'' ~ h (T - T_w), which often underpredicts inert heat flux in
      compressible, high-Mach flows.

    Parameters
    ----------
    wall_temperature:
        Wall temperature Tw [K]. If None, only momentum loss is applied.

    heated_perimeter_fraction:
        Fraction of the wetted perimeter that should exchange heat with the wall.
        - 1.0 => all walls heated/cooled
        - 0.0 => adiabatic walls (no heat loss)
        If you only want top/bottom walls in a wide duct, a reasonable first model
        is: heated_perimeter_fraction = P_heated / P_wetted.

    use_adiabatic_wall_temperature:
        If True (default), uses T_aw as the driving temperature.
        If False, uses static T as the driving temperature (less recommended).

    recovery_model:
        "turbulent_Pr13" (default): r = Pr^(1/3)
        "constant": r = recovery_factor_constant

    Notes on correlations
    ---------------------
    - Turbulent heat transfer is computed via a Reynolds-analogy style relation
      between Cf and Stanton number, matching your original Kays-like form.
    - Laminar is handled via a constant fully-developed Nu (3.657) with Dh.
      This laminar branch is rarely dominant in scramjet combustors but keeps the
      model well-posed at low Re.
    """

    REQUIRED_PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = ("geometry", "physics")

    def __init__(
        self,
        wall_temperature: Array | float | None = None,
        skin_friction_coefficient: SkinFriction | None = None,
        *,
        heated_perimeter_fraction: float = 1.0,
        use_adiabatic_wall_temperature: bool = True,
        recovery_model: str = "turbulent_Pr13",
        recovery_factor_constant: float = 0.88,
        Re_critical_laminar: float = 1000.0,
        Nu_laminar: float = 3.657,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)

        self.wall_temperature = wall_temperature

        self.heated_perimeter_fraction = float(heated_perimeter_fraction)
        if not (0.0 <= self.heated_perimeter_fraction <= 1.0):
            raise ValueError("heated_perimeter_fraction must be in [0, 1].")

        self.use_adiabatic_wall_temperature = bool(use_adiabatic_wall_temperature)
        self.recovery_model = str(recovery_model)
        self.recovery_factor_constant = float(recovery_factor_constant)

        self.Re_critical_laminar = float(Re_critical_laminar)
        self.Nu_laminar = float(Nu_laminar)

        if skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()
        else:
            self.skin_friction_coefficient = skin_friction_coefficient

        # Provides momentum and energy source terms
        self.idx_source: Index = np.array([0, 1])
        self.shape_output = (self.shape_output[0], 2)

    def _recovery_factor(self, Pr: Array) -> Array:
        if self.recovery_model == "turbulent_Pr13":
            # Common turbulent recovery approximation
            return Pr ** (1.0 / 3.0)
        if self.recovery_model == "constant":
            return np.full_like(Pr, self.recovery_factor_constant)
        raise ValueError(
            "Unknown recovery_model. Use 'turbulent_Pr13' or 'constant'."
        )

    def _adiabatic_wall_temperature(self, T: Array, M: Array, Pr: Array) -> Array:
        # T_aw = T * (1 + r * (gamma-1)/2 * M^2)
        assert self.physics is not None
        gamma = getattr(self.physics, "gamma", 1.4)
        r = self._recovery_factor(Pr)
        return T * (1.0 + r * 0.5 * (gamma - 1.0) * M**2)

    def get_stanton_number(self, Re: Array, Pr: Array, cf: Array, Dh: Array) -> Array:
        """Return Stanton number St.

        - Laminar: St = Nu / (Re Pr) using Dh-based Reynolds number.
        - Turbulent: Kays-like relation between Cf and St.
        """
        St = np.zeros_like(Re)

        idx_lam = Re < self.Re_critical_laminar
        if np.any(idx_lam):
            # Fully-developed laminar, constant wall temperature (canonical)
            St[idx_lam] = self.Nu_laminar / (Re[idx_lam] * Pr[idx_lam])

        idx_turb = ~idx_lam
        if np.any(idx_turb):
            cf_t = cf[idx_turb]
            Pr_t = Pr[idx_turb]

            # Kays / Reynolds-analogy style relation
            # St = (Cf/2) / (1 + 13*(Pr^(2/3)-1)*sqrt(Cf/2))
            St[idx_turb] = (cf_t / 2.0) / (
                1.0 + 13.0 * (Pr_t ** (2.0 / 3.0) - 1.0) * np.sqrt(cf_t / 2.0)
            )

        return St

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert self.physics is not None
        assert self.geometry is not None
        assert state is not None
        assert state.density is not None
        assert state.velocity is not None

        rhs = np.zeros((*state.shape, 2))

        x = self.geometry.xc[self.idx_input]

        # Geometry
        A = self.geometry.area(time, x)
        P = self.geometry.perimeter(time, x)
        Dh = self.geometry.hydraulic_diameter(time, x)

        # Convert wall fluxes (per area) to volumetric sinks (per volume): (P/A)*flux
        P_over_A = P / A
        P_h_over_A = self.heated_perimeter_fraction * P_over_A

        # Thermo/transport
        T = state.temperature = self.physics.get_temperature(state)
        mu = self.physics.get_mu(state)
        a = self.physics.get_sound_speed(state)

        U = state.velocity
        rho = state.density

        Re = np.abs(rho * U * Dh / mu)
        Mach = np.abs(U / a)

        # Cf table uses T/Tw; if Tw is not set, pass 1.
        if self.wall_temperature is None:
            T_Tw = np.ones_like(T)
        else:
            # Avoid divide-by-zero if user passes Tw=0 accidentally
            Tw = np.maximum(np.asarray(self.wall_temperature), 1e-12)
            T_Tw = T / Tw

        cf = self.skin_friction_coefficient(Re, Mach, T_Tw)

        # Wall shear stress (sign with flow direction)
        tau_w = cf * (0.5 * rho * U**2.0) * np.sign(U)
        rhs[:, 0] = -P_over_A * tau_w

        # Wall heat transfer (optional)
        if self.wall_temperature is not None and self.heated_perimeter_fraction > 0.0:
            cp = self.physics.get_cp(state)
            k = self.physics.get_thermal_conductivity(state)
            Pr = cp * mu / k

            St = self.get_stanton_number(Re=Re, Pr=Pr, cf=cf, Dh=Dh)

            if self.use_adiabatic_wall_temperature:
                T_drive = self._adiabatic_wall_temperature(T=T, M=Mach, Pr=Pr)
            else:
                T_drive = T

            q_w = St * rho * np.abs(U) * cp * (T_drive - self.wall_temperature)
            rhs[:, 1] = -P_h_over_A * q_w

        return np.ravel(rhs)
