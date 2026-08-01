from __future__ import annotations

import time
import traceback
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.models.wall_models import (
    CompressibleHeatFlux,
    CompressibleInertSkinFriction,
)
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import get_variable_info_map
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array
from stanshock.system.geometry import Box

root_dir = Path(__file__).resolve().parents[2]
plt.style.use(root_dir / "data" / "stylelib" / "publication.mplstyle")

"""
Emami scramjet unstart example.


"""


# Run controls
closing = True
pseudoshock = True
desired_img_count = 200
t_sim = 0.025
N_x = 300
dt_estimate = 7.0e-7

t_stab = t_sim / 100.0
t_close = 0.5 * t_sim
tFinal = t_sim + t_stab

AR_i = 4.15
AR_f = 1.0 if closing else AR_i
plot_interval = max(
    1, int(10 ** np.round(np.log10(tFinal / (dt_estimate * desired_img_count))))
)


# Data
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)


# Chemistry
mech = Path(__file__).resolve().parent / "../../data/mechanisms/N2O2HeAr.yaml"
X_amb = "O2:0.21,N2:0.79"


"""
Geometry inputs from:
"Experimental Investigation of Inlet-Combustor Isolators for a Dual-Mode Scramjet..."

The station labels are preserved from the full-domain file:
    x1: isolator end
    x2: first diffuser end
    x3: second diffuser end
    x4: combustor end / nozzle start
    x5: nozzle end
"""
H_ramp = 0.04826
theta_diff = 20.0
H_th = 0.01016
L_H_th = 12.7
L_iso = L_H_th * H_th
L_diff = H_ramp / np.tan(np.radians(theta_diff))
H_diff = H_th + H_ramp
theta_diff2 = 14.11

H_cc = 0.06985
L_diff2 = (H_cc - H_diff) / np.tan(np.radians(theta_diff2))
L_cc = 0.2636

L_flap = 0.150
theta_noz = np.degrees(np.atan2((H_cc - H_diff), L_flap))
H_noz = L_flap * np.tan(np.radians(theta_noz))

x1 = L_iso
x2 = x1 + L_diff
x3 = x2 + L_diff2
x4 = x3 + L_cc
x5 = x4 + L_flap
L = x5

W = 0.0508

x_u = np.array([0.0, x1, x2, x3, x4, x5])
y_u = np.array([H_th, H_th, H_diff, H_cc, H_cc, H_cc - H_noz])

x_d_incomp = np.array([0.0, x1, x2, x3, x4])
y_d_incomp = np.zeros_like(x_d_incomp)

regions = {
    "isolator": (0.0, x1),
    "diffuser": (x1, x3),
    "combustor": (x3, x4),
    "nozzle": (x4, x5),
}


def _restore_scalar(value: Array, scalar_input: bool) -> Array | float:
    return float(value[0]) if scalar_input else value


def flap_coords(t: float) -> tuple[Array, Array]:
    """Return lower-wall nozzle coordinates for the current area ratio."""
    H_exit = H_th * AR(t)
    H_flap = H_cc - H_noz - H_exit
    return np.array([x4, x5]), np.array([0.0, H_flap])


def AR(t: float | Array) -> Array | float:
    """Nozzle-to-throat area ratio from Deng et al."""
    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(t).astype(float)
    AR_vals = np.full_like(t_arr, AR_i, dtype=float)

    closing_mask = (t_arr >= t_stab) & (t_arr <= t_stab + t_close)
    closed_mask = t_arr > t_stab + t_close

    slope = (AR_f - AR_i) / t_close
    AR_vals[closing_mask] = slope * (t_arr[closing_mask] - t_stab) + AR_i
    AR_vals[closed_mask] = AR_f

    return _restore_scalar(AR_vals, scalar_input)


def dAR_dt(t: float | Array) -> Array | float:
    """Time derivative of the nozzle-to-throat area ratio."""
    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(t).astype(float)
    dAR = np.zeros_like(t_arr, dtype=float)
    closing_mask = (t_arr >= t_stab) & (t_arr <= t_stab + t_close)
    dAR[closing_mask] = (AR_f - AR_i) / t_close
    return _restore_scalar(dAR, scalar_input)


def H(t: float, x: Array | float) -> Array | float:
    """Full-domain channel height with the reduced-domain interpolation form."""
    scalar_input = np.isscalar(x)
    x_arr = np.atleast_1d(x).astype(float)
    x_fp, y_fp = flap_coords(t)

    x_d = np.append(x_d_incomp, x_fp[1])
    y_d = np.append(y_d_incomp, y_fp[1])

    y_d_arr = np.interp(x_arr, x_d, y_d)
    y_u_arr = np.interp(x_arr, x_u, y_u)
    heights = y_u_arr - y_d_arr

    return _restore_scalar(heights, scalar_input)


def D_H(t: float, x: Array | float) -> Array | float:
    """Hydraulic diameter for the rectangular full-domain flowpath."""
    h = H(t, x)
    return (2.0 * W * h) / (W + h)


def dHdx(t: float, x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros_like(x)
    return np.gradient(H(t, x), x)


def dHdt(t: float, x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    dH_dt = np.zeros_like(x)
    mask_noz = (x > x4) & (x <= x5)
    x_noz = x[mask_noz] - x4
    dH_dt[mask_noz] = (x_noz / L_flap) * H_th * dAR_dt(t)
    return dH_dt


def A(t: float, x: Array | float) -> Array | float:
    return H(t, x) * W


def dAdx(t: float, x: Array) -> Array:
    return W * dHdx(t, x)


def dAdt(t: float, x: Array) -> Array:
    return W * dHdt(t, x)


def dlnAdx(t: float, x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros_like(x)
    return np.gradient(np.log(A(t, x)), x)


def dlnAdt(t: float, x: Array) -> Array:
    return dAdt(t, x) / A(t, x)


# Flow properties
gas1 = ct.Solution(mech)
M1 = 2.05
p1 = 85515.38
T1 = 182.74
gas1.TPX = T1, p1, X_amb
u1 = M1 * gas1.sound_speed

p2 = 8278.763


# Solver setup
x = np.linspace(0.0, L, N_x)
geometry = Box(
    xf=x,
    h=H,
    w=W,
    dlnA_dx=dlnAdx,
    dlnA_dt=dlnAdt,
    regions=regions,
)

physics_model = CanteraInterface(gas1)
variable_info_map = get_variable_info_map(physics_model)

BC_inlet = SpecifiedFace(
    location="left",
    reference_state=(gas1.density, u1, gas1.P, None),
)
BC_outlet = SpecifiedFace(location="right", reference_state=(None, None, p2, None))
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}

init = InitializeConstant(geometry, physics_model, gas1, u1)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())

ss = Combustor(
    geometry=geometry,
    wall_temperature=330.0,
    wall_models=wall_models,
    initialization=init,
    boundary_conditions=BCs,
    physics=physics_model,
    cfl=1.0,
    include_pseudoshock=pseudoshock,
    include_diffusion=False,
    output_every=plot_interval,
    plot_state_interval=plot_interval,
    plot_state_variables=["mach", "p", "T"],
    plot_state_variable_info_map=variable_info_map,
    use_double_flux=False,
)

ss.probes.append(Probe(geometry, physics_model, 0.01))
ss.probes.append(Probe(geometry, physics_model, x1))
ss.probes.append(Probe(geometry, physics_model, 0.5 * (x3 + x4)))

ss.csv_writers = [
    CSVWriter(
        combustor=ss,
        filename=figdir / "csv" / "state.csv",
        interval=ss.plot_state_interval,
        variables=["x", "mach", "p", "T"],
        variable_info_map=variable_info_map,
    )
]


try:
    t0 = time.perf_counter()
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    if hasattr(ss, "pseudoshock"):
        ind_s = np.array(ss.pseudoshock.sf_array).flatten()
        if len(ind_s) != 0:
            t_ps = np.array(ss.pseudoshock.t_ps).flatten()
            x_s = geometry.xc[geometry.idx_cells][ind_s]
            sigma = np.array(ss.pseudoshock.sigma_ss).flatten()

            fig, ax = plt.subplots()
            ax.plot(x_s, t_ps * 1000.0, c="r")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("t [ms]")
            ax.set_xlim([x[0], x[-1]])
            fig.tight_layout()
            fig.savefig(figdir / "shock_location.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(t_ps * 1000.0, sigma, c="r")
            ax.set_ylabel(r"$\sigma  [(P_2 / P_1)_{SS} / (P_2 / P_1)_{NS}]$")
            ax.set_ylim([0, 1.01])
            ax.set_xlabel(r"$t$ $[\mathrm{ms}]$")
            fig.tight_layout()
            fig.savefig(figdir / "pseudoshock_sigma.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)
