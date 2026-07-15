from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def static_state_from_mach(
    M: float,
    theta: float,
    T: float,
    p: float,
    gamma: float,
    R: float,
) -> tuple[float, float, float, float]:
    a = math.sqrt(gamma * R * T)
    V = M * a
    rho = p / (R * T)
    return V, theta, p, rho


def sound_speed_from_state(pt: np.ndarray, gamma: float):
    _, _, _, _, p, rho = pt
    return math.sqrt(gamma * p / rho)


def mach_from_state(pt: np.ndarray, gamma: float):
    _, _, V, _, p, rho = pt
    return V / math.sqrt(gamma * p / rho)


@dataclass(frozen=True)
class FlowCell:
    verts: np.ndarray
    V: float
    theta: float
    p: float
    rho: float

    def state_points(self) -> np.ndarray:
        xy = np.asarray(self.verts, dtype=float)
        state = np.empty((xy.shape[0], 4), dtype=float)
        state[:, 0] = float(self.V)
        state[:, 1] = float(self.theta)
        state[:, 2] = float(self.p)
        state[:, 3] = float(self.rho)
        return np.column_stack((xy, state))

    def xyuv_points(self) -> np.ndarray:
        xy_state = self.state_points()
        V = xy_state[:, 2]
        theta = xy_state[:, 3]
        u = V * np.cos(theta)
        v = V * np.sin(theta)
        return np.column_stack((xy_state[:, :2], u, v))


def total_temperature_ratio(M, gamma: float = 1.4):
    return 1.0 + 0.5 * (gamma - 1.0) * M * M


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
