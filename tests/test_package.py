from __future__ import annotations

import importlib.metadata

import stanscram as m


def test_version():
    assert importlib.metadata.version("stanscram") == m.__version__
