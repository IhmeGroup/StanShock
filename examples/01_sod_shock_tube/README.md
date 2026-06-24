# 01_sod_shock_tube

This example solves the 1D Sod shock tube problem with several numerical-scheme
configurations and compares them against the analytical solution.

## Primary feature demonstrated

This case showcases StanShock's user-selectable numerical scheme options:

- WENO5 vs. first-order face extrapolation
- HLLC vs. Lax-Friedrichs inviscid fluxes
- double-flux enabled vs. disabled
- CSV export of final-state solution data

## Validation target

The solution is validated against the analytical Sod shock tube solution.

## Cases compared

- `weno5_hllc`
- `weno5_hllc_df`
- `o1_hllc`
- `o1_lf`

## Outputs

Running `sod_shock_tube.py` creates an `output/` folder containing:

- `summary.csv`: runtime and L1 error metrics for each configuration
- `figures/sod_comparison.png`: combined density/pressure/velocity comparison
- `figures/sod_rho.png`, `figures/sod_p.png`, `figures/sod_u.png`: individual
  comparison figures
- `csv/<case>_00000.csv`: final-state CSV export for each configuration

## Run

From the repository root:

```bash
python examples/01_sod_shock_tube/sod_shock_tube.py
```

## Expected behavior

All configurations should reproduce the basic analytical wave structure. The
higher-order WENO5 + HLLC cases should be less diffusive than the first-order
options.
