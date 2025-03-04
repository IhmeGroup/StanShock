# StanShock

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/IhmeGroup/StanShock/workflows/CI/badge.svg
[actions-link]:             https://github.com/IhmeGroup/StanShock/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/StanShock
[conda-link]:               https://github.com/conda-forge/StanShock-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/IhmeGroup/StanShock/discussions
[pypi-link]:                https://pypi.org/project/StanShock/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/StanShock
[pypi-version]:             https://img.shields.io/pypi/v/StanShock
[rtd-badge]:                https://readthedocs.org/projects/StanShock/badge/?version=latest
[rtd-link]:                 https://StanShock.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

StanShock is a quasi-1D gas dynamics solver designed to model shock tube
experiments and, more recently, scramjet engines. It is currently under heavy
development to bring in new capabilities and modernize the infrastructure, and
should thus be considered unstable until further notice.

## Installation

StanShock has been tested using python 3.9 and 3.13. It is recommended to
install the requirements into a virtual environment such as that provided by
[conda](https://github.com/conda-forge/miniforge). With conda one can create and
activate a new virtual environment named _stanshock_ for Python 3.13 using:

```bash
conda create --name stanshock python=3.13
conda activate stanshock
```

Alternatively, you can create and activate a local virtual environment with:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

Finally, install StanShock and its dependencies from source with:

```bash
pip install .
```

## Documentation

To manually build the documentation locally using Sphinx, first install the
optional dependencies with:

```bash
pip install .[docs]
```

To build the documentation statically and view in Firefox, execute:

```bash
sphinx-build --keep-going -n -T -b=html docs docs/_build/html
firefox docs/_build/html/index.html
```

Or to serve the documentation, pip install sphinx-autobuild and execute:

```bash
sphinx-autobuild --open-browser -n -T -b=html docs docs/_build/html
```

Which will automatically rebuild the documentation when changes to documented
files are detected.

## Citation

To cite StanShock, please refer the following article:

```bibtex
@article{stanshock2020,
    Author = {Grogan, K. and Ihme, M.},
    Title = {StanShock: a gas-dynamic model for shock tube simulations with non-ideal effects and chemical kinetics},
    Journal = {Shock Waves},
    Year = {2020},
    Volume = {30},
    Number = {4},
    Pages = {425--438},
    Doi = {10.1007/s00193-019-00935-x},
}
```

<!-- SPHINX-END -->

## Development

To contribute to StanShock, see [CONTRIBUTING.md](.github/CONTRIBUTING.md).
