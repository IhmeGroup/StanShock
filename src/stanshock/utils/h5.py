from __future__ import annotations

from pathlib import Path
from typing import Self

from h5py import Dataset, File, Group, string_dtype

from stanshock.system.backend import Array, IntArray, np


def h5_has(filename: Path, key: str) -> bool:
    """Return True if the model file exists and contains the given dataset."""
    if not filename.exists():
        return False
    with File(str(filename), "r") as f:
        return key in f


def h5_getarray(h: File | Group, key: str) -> Array:
    """Load array data from h5 file in a way that respects type checking."""
    data_handle = h[key]
    assert isinstance(data_handle, Dataset)
    return np.asarray(data_handle[...], dtype=float)


def h5_write(filename: Path, group: str, data: dict[str, Array]) -> None:
    """Write (overwriting if present) a group of named arrays to the model file."""
    with File(str(filename), "a") as f:
        grp = f.require_group(group)
        for name, array in data.items():
            if name in grp:
                del grp[name]
            grp[name] = array


class RectilinearVtkhdf:
    def __init__(
        self,
        x: Array,
        y: Array,
        z: Array,
        vals: dict[str, Array],
        filename: Path,
        t: Array | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.vals = vals
        self.filename = filename

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        self.nt = len(self.t) if self.t is not None else 0

        ts = (self.nt,) if self.t is not None else ()
        self.shape = (self.nx, self.ny, self.nz, *ts)
        # Verify data shape
        for val in self.vals.values():
            assert self.shape == val.shape

    @classmethod
    def from_vtkhdf(cls, filename: Path) -> Self:
        with File(str(filename), "r") as f:
            root = f.require_group("VTKHDF")
            x = h5_getarray(root, "XCoordinates")
            y = h5_getarray(root, "YCoordinates")
            z = h5_getarray(root, "ZCoordinates")

            t: Array | None = None
            move_time = False
            if "Steps" in root:
                steps = root.require_group("Steps")
                t = h5_getarray(steps, "Values")
                move_time = True

            data = root.require_group("PointData")
            vals: dict[str, Array] = {}
            for key in data:
                val = h5_getarray(data, key)
                if move_time:
                    val = np.moveaxis(val, 0, -1)
                vals[key] = np.swapaxes(val, 0, 2)

        return cls(x, y, z, vals, filename, t)

    def save(self) -> None:
        with File(str(self.filename), "w") as f:
            root = f.require_group("VTKHDF")
            root.attrs["Version"] = (2, 8)
            ascii_type = "RectilinearGrid".encode("ascii")
            root.attrs.create(  # type: ignore[attr-defined]
                "Type", ascii_type, dtype=string_dtype("ascii", len(ascii_type))
            )
            dimensions = (self.nx, self.ny, self.nz)
            root.attrs.create("Dimensions", dimensions, dtype="i4")  # type: ignore[attr-defined]

            move_time = False
            if self.t is not None and self.nt > 0:
                steps = root.require_group("Steps")
                steps.attrs["NSteps"] = self.nt
                steps.create_dataset("Values", data=self.t)

                offsets: IntArray = np.arange(self.nt + 1, dtype=np.int64)
                point_offsets = steps.require_group("PointDataOffsets")
                for key in self.vals:
                    point_offsets.create_dataset(key, data=offsets, maxshape=(None,))

                offsets = np.zeros((self.nt,), dtype=np.int64)
                for dim in ["X", "Y", "Z"]:
                    steps.create_dataset(
                        f"{dim}CoordinatesOffsets", data=offsets, maxshape=(None,)
                    )
                move_time = True

            root.create_dataset("XCoordinates", data=self.x, maxshape=(None,))
            root.create_dataset("YCoordinates", data=self.y, maxshape=(None,))
            root.create_dataset("ZCoordinates", data=self.z, maxshape=(None,))

            data = root.require_group("PointData")
            for key, val in self.vals.items():
                # write_val = np.asarray(val, dtype=np.float32)
                write_val = np.swapaxes(val, 0, 2)
                if move_time:
                    # Move time axis to first dimension
                    write_val = np.copy(np.moveaxis(write_val, -1, 0))
                data.create_dataset(
                    key,
                    data=write_val,
                    maxshape=(None, *dimensions[::-1]),
                    chunks=(1, *dimensions[::-1]),
                )
