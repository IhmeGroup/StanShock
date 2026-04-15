from __future__ import annotations

__all__ = ["config"]


from typing import Literal


class GlobalConfiguration:
    backend: Literal["numpy", "jax", "torch"] = "numpy"
    device: Literal["cpu", "cuda"] = "cpu"
    precision: str = "float64"


config = GlobalConfiguration()
