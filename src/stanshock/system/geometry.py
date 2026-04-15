from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from stanshock.system.backend import Array, Index, TypeAlias, at, is_array, to_array, xp

SpatioTemporalFunction: TypeAlias = Callable[[float, Array], Array | float]
SpatioTemporalLike: TypeAlias = (
    SpatioTemporalFunction | tuple[Array, Array] | Array | float
)


# Classes which turn scalars and arrays into spatiotemporal functions
class ConstantValue:
    def __init__(self, constant: float) -> None:
        self.constant: float = constant

    def __call__(self, _time: float, _x: Array) -> float:
        return self.constant


class LinearInterpolator:
    def __init__(self, xp: Array, fp: Array) -> None:
        self.xp: Array = xp
        self.fp: Array = fp

    def __call__(self, _time: float, x: Array) -> Array:
        return xp.interp(x=x, xp=self.xp, fp=self.fp)


class Geometry:
    def __init__(
        self,
        xf: Array,
        area: SpatioTemporalLike = 1.0,
        perimeter: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
        n_ghost_layers: int = 0,
    ) -> None:
        self.xf: Array = xf  # Face-center locations
        self.n_faces: int = len(self.xf)
        self.xc: Array = 0.5 * (xf[1:] + xf[:-1])  # Cell-center locations
        self.n_cells: int = self.n_faces - 1
        self.n_cells_interior = self.n_cells + 0
        self.dx: Array | float = xp.diff(self.xf)
        # If mesh spacing is constant, simplify this to a float
        if xp.max(xp.abs(xp.diff(self.dx))) < 1e-8:
            self.dx = self.dx[0]

        # Denote distinct regions by their x-range, mapping them to the mesh index
        if regions is None:
            regions = {"domain": (self.xf[0], self.xf[-1])}
        self.regions: dict[str, tuple[float, float]] = regions
        self._region_indices: dict[str, Index] = {}

        # Turn constant-value areas+perimeters into functions of t and x
        self.area: SpatioTemporalFunction = self.to_spatiotemporal(value=area)
        self.perimeter: SpatioTemporalFunction = self.to_spatiotemporal(value=perimeter)

        # Set up area derivatives
        self.dlnA_dx: SpatioTemporalFunction | None
        self.dlnA_dt: SpatioTemporalFunction | None = dlnA_dt

        if dlnA_dx is None:
            if isinstance(area, float | int):
                self.dlnA_dx = None
            else:
                x_tmp: Array
                y_tmp: Array | float
                if is_array(area):
                    x_tmp, y_tmp = self.xf, area
                elif isinstance(area, tuple):
                    x_tmp, y_tmp = area
                else:
                    x_tmp = self.xf
                    y_tmp = area(0.0, self.xf)

                if isinstance(y_tmp, float | int):
                    self.dlnA_dx = None
                else:
                    x_midpoint: Array = 0.5 * (x_tmp[1:] + x_tmp[:-1])
                    area_midpoint: Array = 0.5 * (y_tmp[1:] + y_tmp[:-1])
                    dlnA_dx_fd: Array = xp.diff(y_tmp) / (
                        xp.diff(x_tmp) * area_midpoint
                    )
                    self.dlnA_dx = LinearInterpolator(xp=x_midpoint, fp=dlnA_dx_fd)
        else:
            self.dlnA_dx = dlnA_dx

        # Add ghost layers
        self.n_ghost_layers = 0
        self.idx_cells: Index = xp.s_[:]
        self.setup_ghost_layers(n_ghost_layers)

    def to_spatiotemporal(
        self, value: SpatioTemporalLike | None
    ) -> SpatioTemporalFunction:
        if value is None:
            value = 0.0

        func: SpatioTemporalFunction
        if isinstance(value, float | int):
            func = ConstantValue(constant=value)
        elif isinstance(value, tuple):
            func = LinearInterpolator(xp=value[0], fp=value[1])
        elif is_array(value):
            func = LinearInterpolator(xp=self.xf, fp=value)
        else:
            func = value

        return func

    def setup_ghost_layers(self, n_ghost_layers: int) -> None:
        n_added = n_ghost_layers - self.n_ghost_layers
        if n_added <= 0:
            return

        if isinstance(self.dx, float):
            xc_left = self.xc[0] - xp.arange(n_added, 0, -1) * self.dx
            xc_right = self.xc[-1] + xp.arange(1, n_added + 1) * self.dx
        else:
            self.dx = xp.pad(self.dx, pad_width=n_added, mode="edge")
            xc_left = self.xc[0] - xp.cumulative_sum(self.dx[n_added - 1 :: -1])[::-1]
            xc_right = self.xc[-1] + xp.cumulative_sum(self.dx[-n_added:])

        self.xc = xp.concatenate([xc_left, self.xc, xc_right])
        self.n_ghost_layers = n_ghost_layers
        self.n_cells = len(self.xc)
        self.idx_cells = xp.s_[self.n_ghost_layers : -self.n_ghost_layers]

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.xc[self.idx_cells]

        return to_array(4.0 * self.area(time, x) / self.perimeter(time, x))

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.xc[self.idx_cells]

        return self.hydraulic_diameter(time, x)

    def get_region_index(self, region_name: str) -> Index:
        if region_name not in self.regions:
            return xp.s_[:0]

        if region_name not in self._region_indices:
            x_region_start, x_region_end = self.regions[region_name]
            start_index: xp.intp = xp.argmin(xp.abs(self.xc - x_region_start))
            end_index: xp.intp = xp.argmin(xp.abs(self.xc - x_region_end))

            self._region_indices[region_name] = xp.s_[start_index:end_index]

        return self._region_indices[region_name]

    def surface_area(self, time: float = 0.0, x: Array | None = None) -> Array:
        """Use Simpson's rule to return surface area between given points."""
        msg = "Surface area is not fully determined by perimeter alone."
        raise NotImplementedError(msg)

    def volume(self, time: float = 0.0, x: Array | None = None) -> Array:
        """Use Simpson's rule to return volume between given points."""
        if x is None:
            x = self.xf
            dx = self.dx
            if not isinstance(dx, float | int):
                dx = dx[self.idx_cells]
        else:
            dx = xp.diff(x)

        x_half = 0.5 * (x[1:] + x[:-1])

        area = self.area(time, x)
        area_half = self.area(time, x_half)
        if isinstance(area, float | int):
            area = xp.full_like(x, area)
            area_half = xp.full_like(x_half, area_half)

        return to_array(area[1:] + 4.0 * area_half + area[:-1] * dx / 6.0)


class Cylinder(Geometry):
    def __init__(
        self,
        xf: Array,
        d_outer: SpatioTemporalLike,
        d_inner: SpatioTemporalLike | None = None,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
        n_ghost_layers: int = 0,
    ) -> None:
        self.xf: Array = xf
        if d_inner is None:
            d_inner = 0.0

        # Set up functional form of inner and outer diameters
        self.d_outer: SpatioTemporalFunction = self.to_spatiotemporal(value=d_outer)
        self.d_inner: SpatioTemporalFunction = self.to_spatiotemporal(value=d_inner)

        # Set up functional forms of area and perimeter
        area: SpatioTemporalLike = self._area
        perimeter: SpatioTemporalLike = self._perimeter

        if (is_array(d_outer) or isinstance(d_outer, float | int)) and (
            is_array(d_inner) or isinstance(d_inner, float | int)
        ):
            area = 0.25 * xp.pi * (d_outer**2 - d_inner**2)
            perimeter = xp.pi * (d_outer + d_inner)

        super().__init__(xf, area, perimeter, dlnA_dt, dlnA_dx, regions, n_ghost_layers)

    def _area(self, t: float, x: Array) -> Array | float:
        return 0.25 * xp.pi * (self.d_outer(t, x) ** 2 - self.d_inner(t, x) ** 2)

    def _perimeter(self, t: float, x: Array) -> Array | float:
        return xp.pi * (self.d_outer(t, x) + self.d_inner(t, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.xc[self.idx_cells]

        return self.d_outer(time, x) - self.d_inner(time, x)

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.xc[self.idx_cells]

        d_outer: Array | float = self.d_outer(time, x)
        d_inner: Array | float = self.d_inner(time, x)
        characteristic_length: Array | float = d_outer - d_inner

        if isinstance(d_inner, float | int) and d_inner > 0:
            characteristic_length *= 0.5
        else:
            assert not isinstance(characteristic_length, float | int)
            characteristic_length = at(characteristic_length)[d_inner > 0].multiply(0.5)

        return characteristic_length

    def surface_area(self, time: float = 0.0, x: Array | None = None) -> Array:
        """Use Simpson's rule to return surface area between given points."""
        if x is None:
            x = self.xf
            dx = self.dx
            if not isinstance(dx, float | int):
                dx = dx[self.idx_cells]
        else:
            dx = xp.diff(x)

        x_half = 0.5 * (x[1:] + x[:-1])

        ro = 0.5 * self.d_outer(time, x)
        ro_half = 0.5 * self.d_outer(time, x_half)
        if isinstance(ro, float | int):
            ro = xp.full_like(x, ro)
            ro_half = xp.full_like(x_half, ro_half)

        drdx = xp.diff(ro) / dx
        arc = (ro[1:] + 4.0 * ro_half + ro[:-1]) * xp.sqrt(1.0 + drdx**2)

        return to_array(xp.pi * arc * dx / 3.0)


class Box(Geometry):
    def __init__(
        self,
        xf: Array,
        h: SpatioTemporalLike = 1.0,
        w: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
        n_ghost_layers: int = 0,
    ) -> None:
        self.xf: Array = xf
        # Set up functional forms of height and width
        self.h: SpatioTemporalFunction = self.to_spatiotemporal(value=h)
        self.w: SpatioTemporalFunction = self.to_spatiotemporal(value=w)

        # Set up functional forms of area and perimeter
        area: SpatioTemporalLike = self._area
        perimeter: SpatioTemporalLike = self._perimeter

        if (is_array(h) or isinstance(h, float | int)) and (
            is_array(w) or isinstance(w, float | int)
        ):
            area = h * w
            perimeter = 2 * (h + w)

        super().__init__(xf, area, perimeter, dlnA_dt, dlnA_dx, regions, n_ghost_layers)

    def _area(self, time: float, x: Array) -> Array | float:
        return self.h(time, x) * self.w(time, x)

    def _perimeter(self, time: float, x: Array) -> Array | float:
        return 2.0 * (self.h(time, x) + self.w(time, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.xc[self.idx_cells]

        h: Array | float = self.h(time, x)
        w: Array | float = self.w(time, x)
        return to_array(2 * h * w / (h + w))

    def surface_area(self, time: float = 0.0, x: Array | None = None) -> Array:
        """Use Simpson's rule to return surface area between given points."""
        if x is None:
            x = self.xf
            dx = self.dx
            if not isinstance(dx, float | int):
                dx = dx[self.idx_cells]
        else:
            dx = xp.diff(x)

        x_half = 0.5 * (x[1:] + x[:-1])

        # Treat as symmetric across y = 0
        y = 0.5 * self.h(time, x)
        y_half = 0.5 * self.h(time, x_half)
        if isinstance(y, float | int):
            y = xp.full_like(x, y)
            y_half = xp.full_like(x_half, y_half)
        dydx = xp.diff(y) / dx

        w = 0.5 * self.w(time, x)
        w_half = 0.5 * self.w(time, x_half)
        if isinstance(w, float | int):
            w = xp.full_like(x, w)
            w_half = xp.full_like(x_half, w_half)
        dwdx = xp.diff(w) / dx

        # Side wall area
        area_sides = (
            2.0 * (y[1:] + 4.0 * y_half + y[:-1]) * xp.sqrt(1.0 + dwdx**2) * dx / 3.0
        )

        # Upper and lower wall area
        area_horizontal = (
            2.0 * (w[1:] + 4.0 * w_half + w[:-1]) * xp.sqrt(1.0 + dydx**2) * dx / 3.0
        )

        return to_array(area_sides + area_horizontal)


class AsymmetricBox(Box):
    def __init__(
        self,
        xf: Array,
        upper_wall: SpatioTemporalLike,
        lower_wall: SpatioTemporalLike | None = None,
        w: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
        n_ghost_layers: int = 0,
    ) -> None:
        self.xf: Array = xf
        # Set up functional forms of height and width
        self.upper_wall: SpatioTemporalFunction = self.to_spatiotemporal(
            value=upper_wall
        )
        self.lower_wall: SpatioTemporalFunction = self.to_spatiotemporal(
            value=lower_wall
        )

        super().__init__(xf, self._h, w, dlnA_dt, dlnA_dx, regions, n_ghost_layers)

    def _h(self, time: float, x: Array) -> Array | float:
        return self.upper_wall(time, x) - self.lower_wall(time, x)

    def surface_area(
        self,
        time: float = 0.0,
        x: Array | None = None,
        surface: Literal["top", "bottom", "sides", "all"] = "all",
    ) -> Array:
        """Use Simpson's rule to return surface area between given points."""
        if x is None:
            x = self.xf
            dx = self.dx
            if not isinstance(dx, float | int):
                dx = dx[self.idx_cells]
        else:
            dx = xp.diff(x)

        x_half = 0.5 * (x[1:] + x[:-1])

        yu = self.upper_wall(time, x)
        yu_half = self.upper_wall(time, x_half)
        if isinstance(yu, float | int):
            yu = xp.full_like(x, yu)
            yu_half = xp.full_like(x_half, yu_half)
        dyudx = xp.diff(yu) / dx

        yl = self.lower_wall(time, x)
        yl_half = self.lower_wall(time, x_half)
        if isinstance(yl, float | int):
            yl = xp.full_like(x, yl)
            yl_half = xp.full_like(x_half, yl_half)
        dyldx = xp.diff(yl) / dx

        w = 0.5 * self.w(time, x)
        w_half = 0.5 * self.w(time, x_half)
        if isinstance(w, float | int):
            w = xp.full_like(x, w)
            w_half = xp.full_like(x_half, w_half)
        dwdx = xp.diff(w) / dx

        area = xp.zeros_like(x_half)

        if surface in ["sides", "all"]:
            # Side wall area
            h = yu - yl
            area += (
                (h[1:] + 4.0 * (yu_half - yl_half) + h[:-1])
                * xp.sqrt(1.0 + dwdx**2)
                * dx
                / 3.0
            )

        if surface in ["top", "all"]:
            # Upper wall area
            area += (w[1:] + 4.0 * w_half + w[:-1]) * xp.sqrt(1.0 + dyudx**2) * dx / 3.0

        if surface in ["bottom", "all"]:
            # Lower wall area
            area += (w[1:] + 4.0 * w_half + w[:-1]) * xp.sqrt(1.0 + dyldx**2) * dx / 3.0

        return area


def initialize_geometry(
    xf: Array,
    area: SpatioTemporalLike = 1.0,  # Cross-sectional area of flow
    perimeter: SpatioTemporalLike = 1.0,  # Perimeter of the flow cross section
    d_outer: SpatioTemporalLike | None = None,  # Outer diameter of the cylinder/annulus
    d_inner: SpatioTemporalLike | None = None,  # Inner diameter of the cylinder/annulus
    h: SpatioTemporalLike | None = None,  # height of the channel
    w: SpatioTemporalLike | None = None,  # width of the channel
    upper_wall: SpatioTemporalLike
    | None = None,  # Input: 2 x N list: 1st row is x_locs, 2nd row either "wall" or "open". Used for simulating freestream or internal flow (ceiling)
    lower_wall: SpatioTemporalLike
    | None = None,  # Same format and meaning as upper_wall, but for floor
    dlnA_dt: SpatioTemporalFunction
    | None = None,  # derivative of the natural log of the area of the shock tube with respect to time (needed for quasi-1D)
    dlnA_dx: SpatioTemporalFunction
    | None = None,  # derivative of the natural log of the area of the shock tube with respect to x (needed for quasi-1D)
    regions: dict[str, tuple[float, float]] | None = None,
    n_ghost_layers: int = 0,
    **_kwargs: float,
) -> Geometry:
    geometry: Geometry

    if w is None:
        w = 1.0

    if upper_wall is not None:
        geometry = AsymmetricBox(
            xf=xf,
            upper_wall=upper_wall,
            lower_wall=lower_wall,
            w=w,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
            n_ghost_layers=n_ghost_layers,
        )
    elif h is not None:
        geometry = Box(
            xf=xf,
            h=h,
            w=w,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
            n_ghost_layers=n_ghost_layers,
        )
    elif d_outer is not None:
        geometry = Cylinder(
            xf=xf,
            d_outer=d_outer,
            d_inner=d_inner,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
            n_ghost_layers=n_ghost_layers,
        )
    else:
        geometry = Geometry(
            xf=xf,
            area=area,
            perimeter=perimeter,
            dlnA_dx=dlnA_dx,
            dlnA_dt=dlnA_dt,
            regions=regions,
            n_ghost_layers=n_ghost_layers,
        )

    return geometry
