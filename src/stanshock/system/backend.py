from __future__ import annotations

from typing import Union

import numpy as np
from typing_extensions import TypeAlias

__all__ = ["Array", "Index", "np"]

# Define the Array type for use in type hints to make future changes easier
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
# Note: Must use Union instead of |, but only in Python <3.10 and only in TypeAlias
Index: TypeAlias = Union[np.ndarray[tuple[int, ...], np.dtype[np.int64]], slice]
