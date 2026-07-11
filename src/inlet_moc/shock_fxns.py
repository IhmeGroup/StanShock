from __future__ import annotations

import math


def obl_shock_angle(
    M_in: float, gamma: float, d: float, a: int = 1
):  # Returns: shock angle (rads) | assumes weak shock.
    # Inputs: gamma, Mach number, flow-turning angle d (rads)
    return _obl_shock_angle_scalar(M_in, gamma, d, a=a)


def _obl_shock_angle_scalar(M_in: float, gamma: float, d: float, a: int = 1) -> float:
    d = abs(float(d))
    M_in = float(M_in)
    gamma = float(gamma)
    if not math.isfinite(M_in) or (M_in <= 1.0):
        return math.nan

    gp = gamma + 1.0
    gm = gamma - 1.0
    M2 = M_in * M_in
    tan_d = math.tan(d)
    tand2 = tan_d * tan_d
    discr = (M2 - 1.0) ** 2 - 3.0 * (1.0 + (0.5 * gm) * M2) * (
        1.0 + (0.5 * gp) * M2
    ) * tand2
    if discr < 0.0:
        return math.nan

    lamb = math.sqrt(discr)
    if not math.isfinite(lamb) or math.isclose(lamb, 0.0):
        return math.nan

    chi = (1.0 / (lamb**3)) * (
        (M2 - 1.0) ** 3
        - 9.0
        * (1.0 + (0.5 * gm) * M2)
        * (1.0 + (0.5 * gm) * M2 + (0.25 * gp) * M2 * M2)
        * tand2
    )
    chi = min(max(chi, -1.0), 1.0)
    numerator = (
        M2
        - 1.0
        + 2.0 * lamb * math.cos((4.0 * math.pi * int(a) + math.acos(chi)) / 3.0)
    )
    denominator = 3.0 * (1.0 + (0.5 * gm) * M2) * tan_d
    if not math.isfinite(denominator) or math.isclose(denominator, 0.0):
        return math.nan
    return math.atan(numerator / denominator)


def obl_shock_mach_out(M_in: float, gamma: float, beta: float):
    M_out, _, _, _ = _obl_shock_downstream_values(M_in, gamma, beta)
    return M_out


def obl_shock_property_ratios(M_in, gamma, beta):
    _, p2p1, T2T1, p02p01 = _obl_shock_downstream_values(M_in, gamma, beta)
    return p2p1, T2T1, p02p01


def _obl_shock_downstream_values(
    M_in: float,
    gamma: float,
    beta: float,
) -> tuple[float, float, float, float]:
    M_in = float(M_in)
    gamma = float(gamma)
    beta = float(beta)

    gp = gamma + 1.0
    gm = gamma - 1.0
    sin_beta = math.sin(beta)
    sin_beta_2 = sin_beta * sin_beta
    M2 = M_in * M_in
    f = M2 * sin_beta_2

    M_out_num = _sqrt_or_nan(
        1.0 + gm * f + (((0.5 * gp) ** 2 - gamma * sin_beta_2) * M2 * M2 * sin_beta_2)
    )
    M_out_den = _sqrt_or_nan(gamma * f - 0.5 * gm) * _sqrt_or_nan(0.5 * gm * f + 1.0)
    if not math.isfinite(M_out_den) or math.isclose(M_out_den, 0.0):
        return math.nan, math.nan, math.nan, math.nan
    M_out = M_out_num / M_out_den

    p2p1 = (2.0 * gamma * f / gp) - (gm / gp)
    T2T1_den = f * gp * gp / (2.0 * gm)
    if math.isclose(T2T1_den, 0.0):
        return M_out, math.nan, math.nan, math.nan
    T2T1 = (1.0 + 0.5 * gm * f) * ((2.0 * gamma * f / gm) - 1.0) / T2T1_den
    p02p01 = T2T1 ** (-gamma / gm) * p2p1
    return M_out, p2p1, T2T1, p02p01


def _sqrt_or_nan(value: float) -> float:
    if value < 0.0:
        return math.nan
    return math.sqrt(value)


def obl_shock_post_state(
    M_in: float,
    gamma: float,
    d: float,
    a: int = 1,
) -> tuple[float, float, float]:
    """
    Return ``(beta, M_out, p02p01)`` for the weak attached oblique shock.

    ``d`` is the flow-turning angle magnitude in radians. Invalid or detached
    states return ``nan`` values, matching the legacy scalar helper behavior.
    """
    beta = _obl_shock_angle_scalar(M_in, gamma, d, a=a)
    if not math.isfinite(beta):
        return math.nan, math.nan, math.nan

    try:
        M_out, _, _, p02p01 = _obl_shock_downstream_values(M_in, gamma, beta)
    except ValueError:
        return beta, math.nan, math.nan
    return beta, M_out, p02p01


def normal_shock_mach_out(M_in, gamma):
    M_in = float(M_in)
    gamma = float(gamma)
    return math.sqrt(
        (M_in * M_in * (gamma - 1.0) + 2.0)
        / (2.0 * gamma * M_in * M_in - (gamma - 1.0))
    )


def normal_shock_property_ratios(M_in, gamma):
    h = math.pi / 2.0
    return obl_shock_property_ratios(M_in, gamma, h)
