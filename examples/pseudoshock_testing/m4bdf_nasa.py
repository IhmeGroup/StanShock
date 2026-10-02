from __future__ import annotations

import time
import traceback
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from inlet_moc.moc_solution import MOCSolution
from inlet_moc.planar_inlet import PlanarInlet
from inlet_moc.processing import process_solution
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
from stanshock.system.geometry import AsymmetricBox

#############################################
#   NASA Technical Paper 3502 (Emami, 1995) #
#   Data and geometry from figure 19e.      #
#############################################
### Data ###
data_dir = Path(__file__).resolve().parents[2] / "data"
plt.style.use(data_dir / "stylelib" / "publication.mplstyle")
mech = data_dir / "mechanisms" / "N2O2HeAr.yaml"

### Simulation Output ###
case_dir = Path()
figdir = case_dir / "figures"
animdir = figdir / "anim"
csvdir = figdir / "csv"
figdir.mkdir(exist_ok=True)
animdir.mkdir(exist_ok=True)
csvdir.mkdir(exist_ok=True)


### Geometry ###
N_x = 150
L_H_th = 8.7
L_rp = 2.4816e-01
H_th = 1.0160e-02
H_rp = 4.8260e-02
H_tot = 6.9850e-02
L_cf = 5.0165e-01
L_bd = 1.2903e-01
Lc_Hth = 6.25  # cowl/throat ratio

W = 5.08e-02
AR_i = 4.15
AR_f = 0.99
t_f = 0.05

L_iso = L_H_th * H_th
H_rt = H_rp + H_th

L_c_aft = 1.24 * L_bd
L_cf_real = L_cf - L_c_aft
L_bf_real = L_cf - L_bd
dH_c = H_tot - H_rt
dH_b = -H_rp


xc_0 = 0.0
xc_1 = xc_0 + L_iso
xc_2 = xc_1 + L_bd
xc_3 = xc_2 + 0.24 * L_bd
xc_4 = xc_3 + 0.63 * L_cf_real
xc_5 = xc_3 + 0.82 * L_cf_real
xc_6 = xc_3 + L_cf_real

yc_0 = H_th
yc_1 = yc_0
yc_2 = yc_1
yc_3 = yc_2 + dH_c
yc_4 = yc_3
yc_5 = yc_4 - 1.1 * dH_c
yc_6 = yc_5

L_tot = xc_6


xyc = np.column_stack(
    (
        [xc_0, xc_1, xc_2, xc_3, xc_4, xc_5, xc_6],
        [yc_0, yc_1, yc_2, yc_3, yc_4, yc_5, yc_6],
    )
)

xb_0 = 0.0
xb_1 = xb_0 + L_iso
xb_2 = xb_1 + L_bd
xb_3 = xb_2 + 0.6 * L_bf_real
xb_4 = xc_6
shoulder_smooth_dx = 0.5 * H_th

yb_0 = 0.0
yb_1 = yb_0
yb_2 = yb_1 + dH_b
yb_3 = yb_2
m_b = (yb_2 - yb_1) / (xb_2 - xb_1)
xb_1_l = xb_1 - shoulder_smooth_dx
xb_1_r = xb_1 + shoulder_smooth_dx
yb_1_l = yb_1
yb_1_r = yb_1 + m_b * (xb_1_r - xb_1)


def shoulder_y(x_q: np.ndarray) -> np.ndarray:
    s = (x_q - xb_1_l) / (xb_1_r - xb_1_l)
    h = xb_1_r - xb_1_l
    return (
        (2.0 * s**3 - 3.0 * s**2 + 1.0) * yb_1_l
        + (s**3 - 2.0 * s**2 + s) * h * 0.0
        + (-2.0 * s**3 + 3.0 * s**2) * yb_1_r
        + (s**3 - s**2) * h * m_b
    )


def yb_4(AR: float) -> float:
    return yc_6 - AR * H_th


def _restore_scalar(value: np.ndarray, scalar_input: bool) -> np.ndarray | float:
    return float(value[0]) if scalar_input else value


def AR(t: float | np.ndarray) -> np.ndarray | float:
    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(t).astype(float)
    ramp = np.clip(t_arr / t_f, 0.0, 1.0)
    return _restore_scalar(AR_i + ramp * (AR_f - AR_i), scalar_input)


def dAR_dt(t: float | np.ndarray) -> np.ndarray | float:
    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(t).astype(float)
    dAR = np.zeros_like(t_arr)
    dAR[(t_arr >= 0.0) & (t_arr <= t_f)] = (AR_f - AR_i) / t_f
    return _restore_scalar(dAR, scalar_input)


xyb = np.column_stack(
    (
        [xb_0, xb_1_l, xb_1, xb_1_r, xb_2, xb_3, xb_4],
        [
            yb_0,
            yb_1_l,
            shoulder_y(np.asarray([xb_1]))[0],
            yb_1_r,
            yb_2,
            yb_3,
            yb_4(AR_i),
        ],
    )
)

x_local = np.linspace(0.0, L_tot, N_x)
x = x_local + L_rp
yc = np.interp(x_local, xyc[:, 0], xyc[:, 1])


def lower_wall(t: float, x_q: np.ndarray | float) -> np.ndarray | float:
    scalar_input = np.isscalar(x_q)
    x_q = np.atleast_1d(x_q).astype(float) - L_rp
    y = np.full_like(x_q, yb_0)
    mask = (x_q > xb_1_l) & (x_q < xb_1_r)
    y[mask] = shoulder_y(x_q[mask])
    mask = (x_q >= xb_1_r) & (x_q <= xb_2)
    y[mask] = yb_1 + m_b * (x_q[mask] - xb_1)
    y[x_q > xb_2] = yb_3
    y_tail = yb_4(float(AR(t)))
    mask = (x_q > xb_3) & (x_q <= xb_4)
    y[mask] = yb_3 + (x_q[mask] - xb_3) / (xb_4 - xb_3) * (y_tail - yb_3)
    y[x_q > xb_4] = y_tail
    return _restore_scalar(y, scalar_input)


def H(t: float, x_q: np.ndarray | float) -> np.ndarray | float:
    return np.interp(np.asarray(x_q) - L_rp, x_local, yc) - lower_wall(t, x_q)


def dHdx(t: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros_like(x)
    return np.gradient(H(t, x), x)


def dHdt(t: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_local = x - L_rp
    dH_dt = np.zeros_like(x)
    mask_flap = (x_local > xb_3) & (x_local <= xb_4)
    dH_dt[mask_flap] = (x_local[mask_flap] - xb_3) / (xb_4 - xb_3) * H_th * dAR_dt(t)
    return dH_dt


def dlnAdx(t: float, x: np.ndarray) -> np.ndarray:
    return dHdx(t, x) / H(t, x)


def dlnAdt(t: float, x: np.ndarray) -> np.ndarray:
    return dHdt(t, x) / H(t, x)


yb = lower_wall(0.0, x)

geometry = AsymmetricBox(
    xf=x,
    upper_wall=(x, yc),
    lower_wall=lower_wall,
    w=W,
    dlnA_dx=dlnAdx,
    dlnA_dt=dlnAdt,
)


theta_c = np.radians(2.2)
L_c = Lc_Hth * H_th  # m
dy_c = L_c * np.sin(theta_c)
dx_c = L_c * np.cos(theta_c)

x_cle = L_rp - dx_c  # cowl leading edge
y_cle = (H_th + H_rp) - dy_c

x_cent = [0.0, L_rp, L_rp + L_iso]
y_cent = [0.0, H_rp, H_rp]

x_cowl = [x_cle, L_rp, L_rp + L_iso]
y_cowl = [y_cle, H_rp + H_th, H_rp + H_th]

centerbody = np.column_stack((x_cent, y_cent))
cowl = np.column_stack((x_cowl, y_cowl))

inlet = PlanarInlet(centerbody, cowl)

M_amb, p_amb, T_amb = 4.03, 8729.0, 70.69

N_idl = 100

moc = MOCSolution(
    inlet=inlet,
    Mach=M_amb,
    theta=0.0,
    T_amb=T_amb,
    p_amb=p_amb,
    N_idl=N_idl,
    x_stop=L_rp,
    verbose=False,
    plot_during_solve=False,
    figdir=animdir,
)

moc.solve_inlet()
inflow_moc = process_solution(moc, mode="direct")
### Boundary Conditions ###
gas = ct.Solution(mech)
physics_model = CanteraInterface(gas)
X_amb = "O2:0.21,N2:0.79"

# inlet
_, rho1, u1, p1, *_ = inflow_moc.final_state
gas1 = ct.Solution(mech)
gas1.DPX = rho1, p1, X_amb


# outlet
BC_inlet = SpecifiedFace(
    location="left",
    reference_state=(gas1.density, u1, gas1.P, None),
)
BC_outlet = SpecifiedFace(
    location="right",
    reference_state=(None, None, gas1.P, None),
)
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}


### Initialization ###
init = InitializeConstant(geometry, physics_model, gas1, u1)


### Sim params ###
pseudoshock = True
tFinal = 0.5
output_interval = 100
plot_interval = output_interval
csv_interval = output_interval

variable_info_map = get_variable_info_map(physics_model)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())
plot_variables = ["mach", "p", "T"]

ss = Combustor(
    geometry=geometry,
    wall_temperature=None,
    wall_models=wall_models,
    include_pseudoshock=pseudoshock,
    include_diffusion=False,
    initialization=init,
    boundary_conditions=BCs,
    physics=physics_model,
    cfl=1.0,
    output_every=output_interval,
    plot_state_interval=plot_interval,
    plot_state_variables=plot_variables,
    plot_state_variable_info_map=variable_info_map,
    use_double_flux=False,
)

ss.csv_writers = [
    CSVWriter(
        domain=ss,
        filename=csvdir / "state.csv",
        interval=csv_interval,
        variables=["x", "rho", "u", "mach", "p", "T"],
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
    ### Pseudoshock plots ###
    if hasattr(ss, "pseudoshock"):
        x_sf = np.array(ss.pseudoshock.x_sf).flatten()
        t_ps = np.array(ss.pseudoshock.t_ps).flatten()
        n_history = min(x_sf.size, t_ps.size)
        if n_history != 0:
            x_sf = x_sf[:n_history]
            t_ps = t_ps[:n_history]

            fig, ax = plt.subplots()
            ax.plot(x_sf, t_ps * 1000.0, c="r")
            ax.set_xlabel(r"$x~[\mathrm{m}]$")
            ax.set_ylabel(r"$t~[\mathrm{ms}]$")
            ax.set_xlim([x[0], x[-1]])
            fig.tight_layout()
            fig.savefig(
                animdir / "m4bdf_shock_location.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=animdir)
