# StanShock

StanShock is a quasi-1D gas dynamics solver designed model shock tube experiments. 

The provided version stanShock has the following capabilities:

	Variable cross-sectional area
	Boundary layer modeling
	Multicomponent gas interfaces
	
	Reaction Chemistry
	Species and thermal diffusion
	Geometric Optimization
	

StanShock is writen in object-oriented pure-python, which allows the client to flexibly script and run stanShock cases. 

## Installation

StanShock has been tested using python 3.9. 
It is recommended to install the requirements into a virtual environment such as that provided by [conda](https://docs.conda.io/en/latest/).
With conda one can create a new virtual environment named _stanshock_ for python 3.9 using

`conda create --name stanshock python=3.9`

The requirements are listed in the _requirements.txt_ file. One may install these requirements into the current python environment using

`python -m pip install -r requirements.txt`

## Usage
Included are six examples in the _examples_ folder:
```
laminarFlame.py (laminar flame test case of stoichiometric H2/Air)
optimization.py (driver insert optimization)
validation (four validation test cases)
│─── case1.py (baseline)
│─── case2.py (step change in driver/driven area)
│─── case3.py (driver insert case)
│─── case4.py (disparate driver/driven mixtures)
```

These may be run from their containing directories. 
Ensure that your _PYTHONPATH_ environment variable is set appropriately.
For example, from the project directory

    export PYTHONPATH=.
    python examples/validation/case1.py

Note that the matplotlib plots in these examples use LaTeX to render the fonts. 
See [here](https://matplotlib.org/stable/tutorials/text/usetex.html) for more information on the requirements.
Alternatively, one may remove the LaTeX rendering of fonts in these examples (e.g., commenting out `plt.rc('text',usetex=True)`).
## Structure
Files include:

	stanShock.py (entirety of the StanShock solver code)
	*.{xml,cti} (cantera files containing the thermodiffusive properties)
	{laminarFlame,optimization,case{1..4}}.py (python driver scripts)
	case{1..4}.csv (experimental shock tube data for the validation cases)

## Test
To run the test suite, first ensure that the test dependencies are installed: 

    python -m pip install -r requirements-test.txt

The tests may be run from the project directory using

    python -m unittest discover

For a breakdown of the test coverage, run from the project directory

    coverage run -m unittest discover
    coverage html

This will create an HTML report, which can be explored by a browser.

## Citation
To cite StanShock, please refer the following article:

```
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
## Contact
Please report any issues to the GitHub [site](https://github.com/IhmeGroup/StanShock). 
If you are interested in contributing or collaborating, please contact kevin.p.grogan@gmail.com.
