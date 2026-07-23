from __future__ import annotations

import sys
from typing import TypeAlias

import numpy as np

if sys.version_info >= (3, 13):
    from typing import NotRequired, Unpack
else:
    from typing_extensions import NotRequired, Unpack

__all__: list[str] = [
    "Array",
    "Composition",
    "Index",
    "NotRequired",
    "TypeAlias",
    "Unpack",
    "np",
]

# Define the Array type for use in type hints to make future changes easier
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
Index: TypeAlias = IntArray | slice

Composition: TypeAlias = dict[str, float]
