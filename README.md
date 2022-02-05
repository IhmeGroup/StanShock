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
Included are six examples:
```
laminarFlame (laminar flame test case of stoichiometric H2/Air)
optimization (driver insert optimization)
validationCases (four validation test cases)
│─── case1 (baseline)
│─── case2 (step change in driver/driven area)
│─── case3 (driver insert case)
│─── case4 (disparate driver/driven mixtures)
```

These may be run from their containing directories. 
Ensure that your _PYTHONPATH_ environment variable is set appropriately.
For example, from the _validationCases/case1_ directory

    export PYTHONPATH=../../
    python case1.py

## Structure
Files include:

	stanShock.py (entirety of the StanShock solver code)
	*.{xml,cti} (cantera files containing the thermodiffusive properties)
	{laminarFlame,optimization,case{1..4}}.py (python driver scripts)
	case{1..4}.csv (experimental shock tube data for the validation cases)

## Contact
For timely responses, please report any issues or bugs to kevin.p.grogan@gmail.com. 
