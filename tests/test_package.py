from __future__ import annotations

import importlib.metadata

import stanshock as m


def test_version():
    assert importlib.metadata.version("stanshock") == m.__version__
