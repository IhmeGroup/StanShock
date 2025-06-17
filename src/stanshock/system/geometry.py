from __future__ import annotations

import numpy as np
from scipy import integrate

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide


class Geometry(RightHandSide):
    def __init__(
        self,
        x: Array,
        h=None,
        w=None,
        d_inner=None,
        d_outer=None,
        dlnA_dt=None,
        dlnA_dx=None,
    ) -> None:
        self.x = x
        self.n = len(self.x)
        self.dx = self.x[1] - self.x[0]

        self.h = h
        self.w = w
        self.d_inner = d_inner
        self.d_outer = d_outer
        self.dlnA_dt = dlnA_dt
        self.dlnA_dx = dlnA_dx

        if self.h is not None and self.w is not None:
            self.hydraulic_diameter = 2 * self.h * self.w / (self.h + self.w)
            self.characteristic_length = self.hydraulic_diameter.copy()
        elif self.d_outer is not None:
            self.hydraulic_diameter = self.d_outer(self.x)
            self.characteristic_length = self.hydraulic_diameter.copy()

            if self.d_inner is not None:
                self.hydraulic_diameter -= self.d_inner(self.x)
                self.characteristic_length = 0.5 * self.hydraulic_diameter

                noInsert = self.d_inner(self.x) == 0.0
                self.characteristic_length[noInsert] = self.hydraulic_diameter[noInsert]

        self.integrator = integrate.ode(self.source_fast).set_integrator("lsoda")

        # Define global indices
        self.idx_locations = np.s_[:]
        self.idx_source_terms = np.s_[:]

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
        dt: float,
    ):
        # Compute the density from the state array
        rY0 = state_array[:, 2:]
        r0 = rY0[..., : physics.n_scalars_rho_sum].sum(axis=-1)
        Y0 = rY0 / r0[..., None]

        state0_compact = np.zeros((state_array.shape[0], 3))
        state0_compact[:, 0] = r0
        state0_compact[:, 1:] = state_array[:, :2]  # ru and rE

        # Divide domain between explicit and implicit source terms
        idx_explicit = np.arange(self.x.shape[0])
        idx_implicit = []
        if self.dlnA_dt is not None:
            dlnA_dt = self.dlnA_dt(self.x, time)
            idx_implicit = np.where(dlnA_dt != 0.0)
            idx_explicit = np.where(dlnA_dt == 0.0)

        # Integrate fast terms implicitly
        rhs = np.zeros(state_array[self.idx_locations, self.idx_source_terms].shape)
        for i in idx_implicit:
            # Initialize
            y0 = state0_compact[i, :].copy()
            args = self.x[i], gamma_star[i]
            self.integrator.set_initial_value(y0, time)
            self.integrator.set_f_params(args)

            # Solve
            self.integrator.integrate(time + dt)

            # Store RHS source term
            rhs_compact = (self.integrator.y - state0_compact[i, :]) / dt
            rhs[i, 0:2] += rhs_compact[1:]  # ru and rE
            rhs[i, 2:] += rhs_compact[0] * Y0[i, :]  # rY sources

        # Add slow source terms
        state = physics.conservative_to_primitive(state_array, gamma_star)
        rhs_compact = self.source_slow(time, state0_compact, state, idx_explicit)
        rhs[idx_explicit, 0:2] += rhs_compact[idx_explicit, 1:]  # ru and rE
        rhs[idx_explicit, 2:] += (
            rhs_compact[idx_explicit, 0:1] * Y0[idx_explicit, :]
        )  # rY sources

        return rhs

    def source_slow(
        self, time: float, state0_compact: Array, state: FluidState, idx: Array
    ) -> Array:
        """Area change contributions to RHS."""
        rhs_compact = np.zeros((idx.shape[0], 3))

        if self.dlnA_dt is not None:
            dlnA_dt = self.dlnA_dt(self.x, time)[idx]
            rhs_compact -= state0_compact[idx, :] * dlnA_dt

        if self.dlnA_dx is not None:
            dlnA_dx = self.dlnA_dx(self.x, time)[idx]
            rhs_compact[:, 0] -= state0_compact[idx, 1] * dlnA_dx
            rhs_compact[:, 1] -= (
                state0_compact[idx, 1] ** 2.0 / state0_compact[idx, 0]
            ) * dlnA_dx
            rhs_compact[:, 2] -= (
                state.velocity[idx] * (state0_compact[idx, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs_compact

    def source_fast(self, time: float, y: Array, args: tuple[float, float]):
        """Fast source terms for quasi-1D geometry."""
        # Unpack the input and initialize
        x, gamma = args
        r, ru, rE = y
        p = (gamma - 1.0) * (rE - 0.5 * ru**2.0 / r)
        rhs_compact = np.zeros(3)

        # create quasi-1D right hand side
        if self.dlnA_dt is not None:
            dlnA_dt = self.dlnA_dt([x], time)[0]
            rhs_compact[0] -= r * dlnA_dt
            rhs_compact[1] -= ru * dlnA_dt
            rhs_compact[2] -= rE * dlnA_dt

        if self.dlnA_dx is not None:
            dlnA_dx = self.dlnA_dx([x], time)[0]
            rhs_compact[0] -= ru * dlnA_dx
            rhs_compact[1] -= (ru**2.0 / r) * dlnA_dx
            rhs_compact[2] -= (ru / r * (rE + p)) * dlnA_dx

        return rhs_compact
