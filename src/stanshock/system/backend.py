from __future__ import annotations

from typing import TypeAlias

import numpy as np

__all__: list[str] = ["Array", "Index", "TypeAlias", "np"]

# Define the Array type for use in type hints to make future changes easier
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Index: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.int64]] | slice
