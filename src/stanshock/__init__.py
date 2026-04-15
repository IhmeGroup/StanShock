"""
Copyright (c) 2025 Stanford University. All rights reserved.

StanShock: Quasi-1D gas dynamics solver designed to model shock tube
experiments and scramjet engines.
"""

from __future__ import annotations

__all__ = ["__version__", "config"]

from ._config import config
from ._version import version as __version__
