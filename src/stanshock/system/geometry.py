from __future__ import annotations

from collections.abc import Callable

import numpy as np

from stanshock.system.backend import Array, Index, TypeAlias

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
        return np.interp(x=x, xp=self.xp, fp=self.fp)


class Geometry:
    def __init__(
        self,
        x: Array,
        area: SpatioTemporalLike = 1.0,
        perimeter: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.x: Array = x
        self.n: int = len(self.x)
        self.dx: float = self.x[1] - self.x[0]

        # Denote distinct regions by their x-range, mapping them to the mesh index
        if regions is None:
            regions = {"domain": (x[0], x[-1])}
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
                if isinstance(area, np.ndarray):
                    x_tmp, y_tmp = self.x, area
                elif isinstance(area, tuple):
                    x_tmp, y_tmp = area
                else:
                    msg = "Cannot (yet) automatically determine dlnA_dx from callable area."
                    raise NotImplementedError(msg)
                x_midpoint: Array = 0.5 * (x_tmp[1:] + x_tmp[:-1])
                area_midpoint: Array = 0.5 * (y_tmp[1:] + y_tmp[:-1])
                dlnA_dx_fd: Array = np.diff(y_tmp) / (np.diff(x_tmp) * area_midpoint)
                self.dlnA_dx = LinearInterpolator(xp=x_midpoint, fp=dlnA_dx_fd)
        else:
            self.dlnA_dx = dlnA_dx

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
        elif isinstance(value, np.ndarray):
            func = LinearInterpolator(xp=self.x, fp=value)
        else:
            func = value

        return func

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return 4.0 * self.area(time, x) / self.perimeter(time, x)

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return self.hydraulic_diameter(time, x)

    def get_region_index(self, region_name: str) -> Index:
        if region_name not in self.regions:
            return np.s_[:0]

        if region_name not in self._region_indices:
            x_region_start, x_region_end = self.regions[region_name]
            start_index: np.intp = np.argmin(np.abs(self.x - x_region_start))
            end_index: np.intp = np.argmin(np.abs(self.x - x_region_end))

            self._region_indices[region_name] = np.s_[start_index:end_index]

        return self._region_indices[region_name]


class Cylinder(Geometry):
    def __init__(
        self,
        x: Array,
        d_outer: SpatioTemporalLike,
        d_inner: SpatioTemporalLike | None = None,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.x: Array = x
        # Set up functional form of inner and outer diameters
        self.d_outer: SpatioTemporalFunction = self.to_spatiotemporal(value=d_outer)
        self.d_inner: SpatioTemporalFunction = self.to_spatiotemporal(value=d_inner)

        # Set up functional forms of area and perimeter
        area: SpatioTemporalLike = self._area
        perimeter: SpatioTemporalLike = self._perimeter

        if isinstance(d_outer, np.ndarray | float | int) and isinstance(
            d_inner, np.ndarray | float | int
        ):
            area = 0.25 * np.pi * (d_outer**2 - d_inner**2)
            perimeter = 0.5 * np.pi * (d_outer + d_inner)

        super().__init__(x, area, perimeter, dlnA_dt, dlnA_dx, regions)

    def _area(self, t: float, x: Array) -> Array | float:
        return 0.25 * np.pi * (self.d_outer(t, x) ** 2 - self.d_inner(t, x) ** 2)

    def _perimeter(self, t: float, x: Array) -> Array | float:
        return 0.5 * np.pi * (self.d_outer(t, x) + self.d_inner(t, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return self.d_outer(time, x) - self.d_inner(time, x)

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        d_outer: Array | float = self.d_outer(time, x)
        d_inner: Array | float = self.d_inner(time, x)
        characteristic_length: Array | float = d_outer - d_inner

        if isinstance(d_inner, float | int) and d_inner > 0:
            characteristic_length *= 0.5
        else:
            assert isinstance(characteristic_length, np.ndarray)
            characteristic_length[d_inner > 0] *= 0.5

        return characteristic_length


class Box(Geometry):
    def __init__(
        self,
        x: Array,
        h: SpatioTemporalLike = 1.0,
        w: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.x: Array = x
        # Set up functional forms of height and width
        self.h: SpatioTemporalFunction = self.to_spatiotemporal(value=h)
        self.w: SpatioTemporalFunction = self.to_spatiotemporal(value=w)

        # Set up functional forms of area and perimeter
        area: SpatioTemporalLike = self._area
        perimeter: SpatioTemporalLike = self._perimeter

        if isinstance(h, np.ndarray | float | int) and isinstance(
            w, np.ndarray | float | int
        ):
            area = h * w
            perimeter = 2 * (h + w)

        super().__init__(x, area, perimeter, dlnA_dt, dlnA_dx, regions)

    def _area(self, time: float, x: Array) -> Array | float:
        return self.h(time, x) * self.w(time, x)

    def _perimeter(self, time: float, x: Array) -> Array | float:
        return 2.0 * (self.h(time, x) + self.w(time, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        h: Array | float = self.h(time, x)
        w: Array | float = self.w(time, x)
        return 2 * h * w / (h + w)


class AsymmetricBox(Box):
    def __init__(
        self,
        x: Array,
        upper_wall: SpatioTemporalLike,
        lower_wall: SpatioTemporalLike | None = None,
        w: SpatioTemporalLike = 1.0,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.x: Array = x
        # Set up functional forms of height and width
        self.upper_wall: SpatioTemporalFunction = self.to_spatiotemporal(
            value=upper_wall
        )
        self.lower_wall: SpatioTemporalFunction = self.to_spatiotemporal(
            value=lower_wall
        )

        super().__init__(x, self._h, w, dlnA_dt, dlnA_dx, regions)

    def _h(self, time: float, x: Array) -> Array | float:
        return self.upper_wall(time, x) - self.lower_wall(time, x)


def initialize_geometry(
    x: Array,
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
    **_kwargs: float,
) -> Geometry:
    geometry: Geometry

    if w is None:
        w = 1.0

    if upper_wall is not None:
        geometry = AsymmetricBox(
            x=x,
            upper_wall=upper_wall,
            lower_wall=lower_wall,
            w=w,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
        )
    elif h is not None:
        geometry = Box(
            x=x,
            h=h,
            w=w,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
        )
    elif d_outer is not None:
        geometry = Cylinder(
            x=x,
            d_outer=d_outer,
            d_inner=d_inner,
            dlnA_dt=dlnA_dt,
            dlnA_dx=dlnA_dx,
            regions=regions,
        )
    else:
        geometry = Geometry(
            x=x,
            area=area,
            perimeter=perimeter,
            dlnA_dx=dlnA_dx,
            dlnA_dt=dlnA_dt,
            regions=regions,
        )

    return geometry
