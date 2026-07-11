from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _scalar_or_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


class ReferenceState:
    def __init__(self, T0: float, gamma: float, R: float):
        # assigns invariant properties
        self.gamma = float(gamma)
        self.R = float(R)
        self.T0 = float(T0)
        self.a0 = math.sqrt(self.gamma * self.R * self.T0)
        self._beta = 0.5 * (self.gamma - 1.0)
        self._a02 = self.a0**2
        self.a_fxn = self.sound_speed_from_vel

    @classmethod
    def from_static(cls, M, T, gamma, R):
        T0 = T * (1.0 + 0.5 * (gamma - 1.0) * M**2)
        return cls(T0, gamma, R)

    def sound_speed_from_vel(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        V2 = u * u + v * v
        a2 = self._a02 - self._beta * V2
        a = np.sqrt(np.where(a2 >= 0.0, a2, np.nan))
        return _scalar_or_array(a)

    def vel_to_mach(self, u: float, v: float):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        V_mag = np.hypot(u, v)
        a = self.a_fxn(u, v)
        theta = np.arctan2(v, u)
        return _scalar_or_array(V_mag / a), _scalar_or_array(theta)

    def mach_to_vel(self, M: float, theta: float):
        M = float(M)
        theta = float(theta)
        a = self.a0 / math.sqrt(1.0 + self._beta * M * M)
        V_mag = M * a
        return V_mag * math.cos(theta), V_mag * math.sin(theta)

    def static_temp_from_mach(self, M: float):
        M = np.asarray(M, dtype=float)
        return _scalar_or_array(self.T0 / total_temperature(M, self.gamma))

    def static_temp_from_vel(self, u: float, v: float):
        M, _ = self.vel_to_mach(u, v)
        return self.static_temp_from_mach(M)


@dataclass(frozen=True)
class RegionState:
    ref: ReferenceState
    p0: float

    @classmethod
    def from_static(cls, M, p, T, gamma, R):
        ref = ReferenceState.from_static(M, T, gamma, R)
        p0 = p * total_pressure(M, gamma)
        return cls(ref, p0)

    def get_static_props(self, u: float, v: float):
        M, theta = self.ref.vel_to_mach(u, v)
        T = self.ref.static_temp_from_mach(M)
        p = self.p0 / total_pressure(M, self.ref.gamma)
        return M, theta, T, p
    

    def get_rho_p_a(self, u: np.ndarray, v:np.ndarray):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        M, _ = self.ref.vel_to_mach(u, v)
        rho0 = self.p0 / (self.ref.R * self.ref.T0)
        rho = rho0 / total_density_ratio(M, self.ref.gamma)
        p = self.p0 / total_pressure(M, self.ref.gamma)
        a = self.ref.sound_speed_from_vel(u, v)
        return rho, p, a
    



    def get_angles(self, u: float, v: float):
        M, theta = self.ref.vel_to_mach(u, v)
        if not math.isfinite(M) or (M <= 1.0):
            from inlet_moc.char_shock_solvers import SubsonicFlowError

            msg = (
                "Subsonic flow encountered while computing characteristic angles: "
                f"M={M:.6g}."
            )
            raise SubsonicFlowError(msg)
        alpha = math.asin(1.0 / M)
        return theta, alpha
    
    def get_lamd_plus(self, pt: np.ndarray):
        u, v = pt[2:]
        theta, alpha = self.get_angles(u, v)
        return (theta + alpha)
    def get_lamd_minus(self, pt: np.ndarray):
        u, v = pt[2:]
        theta, alpha = self.get_angles(u, v)
        return (theta - alpha)


@dataclass(frozen=True)
class FlowCell:
    verts: np.ndarray
    u: float
    v: float
    region: RegionState

    def xyuv_points(self) -> np.ndarray:
        xy = np.asarray(self.verts, dtype=float)
        uv = np.empty((xy.shape[0], 2), dtype=float)
        uv[:, 0] = float(self.u)
        uv[:, 1] = float(self.v)
        return np.column_stack((xy, uv))

def total_temperature_ratio(M, gamma: float = 1.4):
    return (1.0 + 0.5 * (gamma - 1.0) * M * M)

def total_temperature(M, gamma: float = 1.4):
    return total_temperature_ratio(M, gamma)

def total_pressure_ratio(M: float, gamma: float = 1.4):
    T0_T = total_temperature_ratio(M, gamma)
    return T0_T ** (gamma / (gamma - 1.0))

def total_pressure(M, gamma: float = 1.4):
    return total_pressure_ratio(M, gamma)

def total_density_ratio(M, gamma: float = 1.4):
    T0_T = total_temperature_ratio(M, gamma)
    return T0_T ** (1 / (gamma - 1.0))
