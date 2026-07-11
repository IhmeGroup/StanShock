from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from inlet_moc import utils_moc


class PiecewiseLinearCurve:
    def __init__(self, xy: np.ndarray, normal_sign: float):
        """
        xy : (N,2) array, strictly increasing in x
        normal_sign : +1 or -1 to control normal direction
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            msg = "xy must be (N,2)"
            raise ValueError(msg)
        order = np.argsort(xy[:, 0])
        xy = xy[order]

        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.pts = xy

        if np.any(np.diff(self.x) <= 0.0):
            msg = "x must be strictly increasing"
            raise ValueError(msg)

        self.normal_sign = np.sign(normal_sign)

        # Precompute slopes on each interval
        dx = np.diff(self.x)
        self.slopes = np.diff(self.y) / dx  # length N-1
        self.dx = dx  # store for interpolation
        self.x_min = self.x[0]
        self.x_max = self.x[-1]
        self.y_min = min(self.y)
        self.y_max = max(self.y)

    def interval_index(self, x):
        """
        Returns interval index i such that
        x in [x_i, x_{i+1}]
        """
        if x < self.x_min or x > self.x_max:
            return None

        i = np.searchsorted(self.x, x) - 1
        if i == len(self.slopes):
            i -= 1
        return max(i, 0)

    def get_y(self, x):
        i = self.interval_index(x)
        if i is None:
            return None

        return self.y[i] + self.slopes[i] * (x - self.x[i])

    def get_dydx(self, x):
        i = self.interval_index(x)
        if i is None:
            return None
        return self.slopes[i]

    def get_angle(self, x):
        i = self.interval_index(x)
        if i is None:
            return None
        return np.atan(self.slopes[i])

    def normal_of(self, x):
        """
        Returns outward unit normal.
        For tangent (1, m), normal is (-m, 1).
        """
        m = self.get_dydx(x)
        if m is None:
            return None

        n = np.array([-m, 1.0])
        n *= self.normal_sign
        n /= np.linalg.norm(n)
        return n

    def slope_change_points(self, include_first=False):
        internal = self.x[1:-1]

        if include_first:
            return np.concatenate(([self.x[0]], internal))
        return internal


class PlanarInlet:
    """
    Two bounding curves:
    - centerbody
    - cowl
    """

    def __init__(self, xy_cent, xy_cowl):
        if max(xy_cent[:, 1]) > max(xy_cowl[:, 1]):
            n_cent, n_cowl = +1.0, -1.0
        else:
            n_cowl, n_cent = -1.0, +1.0
        self.centerbody = PiecewiseLinearCurve(xy_cent, normal_sign=n_cent)
        self.cowl = PiecewiseLinearCurve(xy_cowl, normal_sign=n_cowl)

    def plot_inlet(self, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))
        elif fig is None:
            fig = ax.figure

        ax.fill_between(
            self.centerbody.x,
            0,
            self.centerbody.y,
            facecolor="white",
            edgecolor="0.35",
            hatch="///",
            linewidth=0.0,
            zorder=8,
        )
        ax.plot(self.centerbody.x, self.centerbody.y, color="k", lw=0.5, zorder=50)
        ax.plot(self.cowl.x, self.cowl.y, color="k", lw=0.5, zorder=50)
        ax.set_aspect("equal")
        return fig, ax

    def get_infl0(self):
        x_cl_min = self.cowl.x_min
        x_cb_min = self.centerbody.x_min
        wall_le = self.centerbody if x_cb_min < x_cl_min else self.cowl
        return wall_le, wall_le.pts[0], wall_le.normal_sign

    def build_idl_pts(self, char_init_slope: float, N_idl: int):
        """
        Returns equally spaced (x,y) coords along the initial data line (leading char)
        """
        xy_cl0 = self.cowl.pts[0]
        xy_cb0 = self.centerbody.pts[0]

        m_cl0 = self.cowl.slopes[0]
        m_cb0 = self.centerbody.slopes[0]

        if abs(m_cl0) < 1e-10:
            x_cl_max = xy_cl0[0]
            y_hit = self.centerbody.get_y(x_cl_max)
            if y_hit is not None:
                xy_hit = np.array([x_cl_max, y_hit])
            else:
                return None
        else:
            cowl_norm_eqn = utils_moc.get_norm(xy_cl0, m_cl0)
            cent_tang_eqn = utils_moc.get_tang(xy_cb0, m_cb0)
            xy_hit = utils_moc.get_intersection(cowl_norm_eqn, cent_tang_eqn)

        t = np.linspace(0, 1, N_idl)
        if np.isinf(char_init_slope):
            x_hit = xy_cl0[0]
            y_hit = self.centerbody.get_y(x_hit)
            xy_hit = np.array([x_hit, y_hit])
            return xy_cl0 + t[:, None] * (xy_hit - xy_cl0)
        cowl_tang_eqn = utils_moc.get_tang(xy_cl0, char_init_slope)
        cent_tang_eqn = utils_moc.get_tang(xy_cb0, m_cb0)
        xy_hit = utils_moc.get_intersection(cowl_tang_eqn, cent_tang_eqn)

        idl_pts = xy_cl0 + t[:, None] * (xy_hit - xy_cl0)
        order = np.argsort(idl_pts[:, 0])
        return idl_pts[order]


def identify_wall(xy_p: np.ndarray, inlet: PlanarInlet, tol=1e-8):
    """
    Determine whether (xp, yp) lies on cowl or centerbody.

    Returns
    -------
    wall : PiecewiseLinearCurve
    name : str
    """
    xp, yp = xy_p
    yc = inlet.cowl.get_y(xp)
    yb = inlet.centerbody.get_y(xp)

    dc = np.inf if yc is None else abs(yp - yc)
    db = np.inf if yb is None else abs(yp - yb)

    if dc < tol:
        return inlet.cowl
    if db < tol:
        return inlet.centerbody
    msg = f"Point ({xp:.6g},{yp:.6g}) not on either wall."
    raise RuntimeError(msg)


def get_opposite_wall(xy_p: np.ndarray, inlet: PlanarInlet, tol=1e-8):
    xp, yp = xy_p

    yc = inlet.cowl.get_y(xp)
    yb = inlet.centerbody.get_y(xp)

    dc = np.inf if yc is None else abs(yp - yc)
    db = np.inf if yb is None else abs(yp - yb)

    if dc < tol:
        return inlet.centerbody
    if db < tol:
        return inlet.cowl
    msg = f"Point ({xp:.6g},{yp:.6g}) not on either wall."
    raise RuntimeError(msg)
