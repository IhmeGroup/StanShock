from __future__ import annotations

import sys

import numpy as np

if sys.version_info >= (3, 13):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

__all__: list[str] = ["Array", "Index", "TypeAlias", "np"]

# Define the Array type for use in type hints to make future changes easier
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Index: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.int64]] | slice
