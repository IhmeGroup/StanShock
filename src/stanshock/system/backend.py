from __future__ import annotations

__all__: list[str] = [
    "Array",
    "ArrayType",
    "Composition",
    "Index",
    "NotRequired",
    "TypeAlias",
    "Unpack",
    "apply_where",
    "at",
    "jit",
    "np",
    "xp",
]

import os
import sys
import warnings
from typing import TYPE_CHECKING, TypeAlias, TypeVar, overload

import array_api_compat
import array_api_extra as xpx
import numpy as np

from stanshock import config

if sys.version_info >= (3, 13):
    from typing import NotRequired, Self, Unpack
else:
    from typing_extensions import NotRequired, Self, Unpack


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import EllipsisType, MethodType, ModuleType

    from _numtype import Array as ArrayType
    from array_api_extra._lib._at import Undef, _AtOp, _undef

    xp = np
    backend_choice = config.backend

    _T = TypeVar("_T")

    _ReturnInput: TypeAlias = Callable[[_T], _T]
    jit: _ReturnInput[MethodType]
    jit_method: _ReturnInput[MethodType]

    # Custom workaround for typing the xpx.at and apply_where methods
    _IndexItem = int | slice | EllipsisType | ArrayType
    _SetIndex = _IndexItem | tuple[_IndexItem, ...]
    _GetIndex = _IndexItem | None | tuple[_IndexItem | None, ...]

    class at:
        def __init__(
            self, x: ArrayType, idx: _SetIndex | Undef = _undef, /
        ) -> None: ...
        def __getitem__(self, idx: _SetIndex, /) -> Self: ...
        def _op(
            self,
            at_op: _AtOp,
            in_place_op: Callable[[ArrayType, ArrayType | complex], ArrayType] | None,
            out_of_place_op: Callable[[ArrayType, ArrayType], ArrayType] | None,
            y: ArrayType | complex,
            /,
            copy: bool | None,
            xp: ModuleType | None,
        ) -> ArrayType: ...
        def set(
            self,
            y: ArrayType | complex,
            /,
            copy: bool | None = None,
            xp: ModuleType | None = None,
        ) -> ArrayType: ...

        # Each operation has the same signature:
        add = subtract = multiply = divide = power = min = max = set

    @overload
    def apply_where(
        cond: ArrayType,
        args: ArrayType | tuple[ArrayType, ...],
        f1: Callable[..., ArrayType],
        f2: Callable[..., ArrayType],
        /,
        *,
        kwargs: dict[str, ArrayType] | None = None,
        xp: ModuleType | None = None,
    ) -> ArrayType: ...

    @overload
    def apply_where(
        cond: ArrayType,
        args: ArrayType | tuple[ArrayType, ...],
        f1: Callable[..., ArrayType],
        /,
        *,
        fill_value: ArrayType | complex,
        kwargs: dict[str, ArrayType] | None = None,
        xp: ModuleType | None = None,
    ) -> ArrayType: ...

    def apply_where(
        cond: ArrayType,
        args: ArrayType | tuple[ArrayType, ...],
        f1: Callable[..., ArrayType],
        f2: Callable[..., ArrayType] | None = None,
        /,
        *,
        fill_value: ArrayType | complex | None = None,
        kwargs: dict[str, ArrayType] | None = None,
        xp: ModuleType | None = None,
    ) -> ArrayType: ...

else:
    backend_choice = os.getenv("STANSHOCK_BACKEND", config.backend)
    if backend_choice != config.backend:
        print(
            f"NOTE: Backend overridden by environment variable STANSHOCK_BACKEND={backend_choice}."
        )

    device_choice: str = os.getenv("STANSHOCK_DEVICE", config.device)
    if device_choice != config.device:
        print(
            f"NOTE: Device overridden by environment variable STANSHOCK_DEVICE={device_choice}."
        )

    try:
        if backend_choice == "torch":
            import torch

            torch.set_default_device(device_choice)
            if config.precision == "float64":
                torch.set_default_dtype(torch.float64)
            from torch import ones
        elif backend_choice == "jax":
            import jax

            jax.config.update("jax_platforms", device_choice)
            if config.precision == "float64":
                jax.config.update("jax_enable_x64", True)
            from jax.numpy import ones
        elif backend_choice == "numpy":
            ones = np.ones
    except ImportError:
        warnings.warn(
            f"Failed to import requested backend {backend_choice}, falling back to NumPy.",
            UserWarning,
            stacklevel=1,
        )
        backend_choice = "numpy"
        ones = np.ones

    xp = array_api_compat.array_namespace(ones((1,)))

    # Backend-specific options
    if array_api_compat.is_torch_namespace(xp):
        # print("Using PyTorch Backend")
        ArrayType = torch.Tensor
        jit = xp.compile
    elif array_api_compat.is_jax_namespace(xp):
        # print("Using Jax Backend")
        ArrayType = jax.Array
        jit = jax.jit
    elif array_api_compat.is_numpy_namespace(xp):
        # print("Using NumPy Backend")
        ArrayType = np.array

        # Set jit compilers to no-ops:
        def jit(fun: _T) -> _T:
            return fun
    else:
        msg = "Array backend could not be determined."
        raise ValueError(msg)

    at = xpx.at
    apply_where = xpx.apply_where


# Define the Array type for use in type hints to make future changes easier
Float = xp.float64
Int = xp.int64
Array: TypeAlias = ArrayType[Float, tuple[int, ...]]
Index: TypeAlias = ArrayType[Int, tuple[int, ...]] | slice

Composition: TypeAlias = dict[str, float]
