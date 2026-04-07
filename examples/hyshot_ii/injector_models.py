from __future__ import annotations

import cantera as ct
import numpy as np
from scipy import optimize

from stanshock.physics.fluid_base import FluidPhysics


def f(M: float, gamma: float) -> float:
    expon = (gamma + 1) / (2 * (gamma - 1))
    tmp = ((gamma + 1) / 2) ** expon
    return tmp * M / (1 + (gamma - 1) / 2 * M**2) ** expon


def calc_M(P0: float, Pa: float, gamma: float) -> float:
    P0_choked = Pa * ((gamma + 1) / 2) ** (gamma / (gamma - 1))
    if P0_choked > P0:
        # Exit pressure is equal to the ambient pressure
        M = float(np.sqrt(2 / (gamma - 1) * ((P0 / Pa) ** ((gamma - 1) / gamma) - 1)))
    else:
        # Exit pressure is no longer equal to the ambient pressure
        # We know based on geometry that the Mach number at the orifice is 1
        M = 1.0
    return M


def calc_mdot(
    P0: float, T0: float, A: float, Pa: float, gamma: float, R: float
) -> float:
    M = calc_M(P0, Pa, gamma)
    gamma_term = gamma / ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
    return gamma_term * P0 * A / np.sqrt(gamma * R * T0) * f(M, gamma)


def P0_from_mdot(
    mdot: float, T0: float, A: float, P_in: float, gamma: float, R: float
) -> tuple[float, float]:
    def eqn(P0: float) -> float:
        return calc_mdot(P0, T0, A, P_in, gamma, R) - mdot

    result = optimize.root_scalar(eqn, x0=P_in)
    P0 = result.root
    M = calc_M(P0, P_in, gamma)
    return P0, M


def fuel_props_from_phi(
    physics: FluidPhysics,
    phi_gl: float,
    mdot_ox: float,
    T0_f: float,
    P_in: float,
    A_f_tot: float,
) -> tuple[float, float, float]:
    gas = physics.gas
    X_f = physics.fuel_def
    assert X_f is not None
    X_ox = physics.ox_def
    assert X_ox is not None

    # Get fuel properties at combustor conditions
    gas.TPX = T0_f, P_in, X_f
    if phi_gl == 0.0:
        return gas.density_mass, 0.0, 300.0

    gamma_f = gas.cp / gas.cv
    R_f = ct.gas_constant / gas.mean_molecular_weight

    # Compute the fuel mass flow from the target fuel/air ratio
    gas.set_equivalence_ratio(phi_gl, X_f, X_ox)
    # X_mix = gas.X
    Y_ox = sum(gas.Y[gas.species_index(sp)] for sp in X_ox)
    Y_f = sum(gas.Y[gas.species_index(sp)] for sp in X_f)
    OF = Y_ox / Y_f
    mdot_f = mdot_ox / OF
    # Z_gl = gas.mixture_fraction(X_f, X_ox)

    # Compute the fuel plenum (stagnation) pressure to achieve mdot_f
    _, M_f = P0_from_mdot(mdot_f, T0_f, A_f_tot, P_in, gamma_f, R_f)
    T_f = T0_f / (1 + (gamma_f - 1) / 2 * M_f**2)
    a_f = float(np.sqrt(gamma_f * R_f * T_f))
    U_f = M_f * a_f
    rho_f = mdot_f / (U_f * A_f_tot)
    # gas.TDX = T_f, rho_f, X_f
    # P_f = gas.P
    # H_f = gas.enthalpy_mass

    # Compute the estimated temperature of the mixture
    # (Pressure will change but this doesn't affect the temperature)
    # gas.HPX = (mdot_a * H_in + mdot_f * H_f) / (mdot_a + mdot_f), P_in, X_mix
    # T_mix = gas.T
    # delta_T = T_mix - T_in

    return rho_f, U_f, T_f
