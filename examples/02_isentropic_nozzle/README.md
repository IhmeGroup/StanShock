# 02_isentropic_nozzle

## Purpose

This example showcases StanShock's **geometry** and **area-change** capabilities
using a smooth converging-diverging nozzle. The flow is initialized with
quasi-1D **isentropic nozzle theory**, then advanced a short time to verify that
the solver preserves the quasi-steady balance between fluxes and area-change
source terms.

## Primary feature demonstrated

- Smooth nozzle geometry definition using the modern `Geometry` / `Box`
  interface
- Isentropic flow initialization with `InitializeIsentropic`
- Short-time preservation of quasi-1D isentropic nozzle flow under numerical
  integration
- Optional export of the nozzle state using `CSVWriter`

## Validation target

This case is validated against the standard quasi-1D isentropic relations for a
converging-diverging nozzle:

- area-Mach relation
- isentropic pressure and density ratios
- corresponding velocity profile

The example reports both:

- **initialization error** relative to theory
- **final error** after a short integration interval
- **short-time drift** between the initialized and advanced solutions

## What the example produces

Running `isentropic_nozzle.py` generates an `output/` folder containing:

- `figures/isentropic_nozzle_profiles.png`
- `figures/isentropic_nozzle_mach.png`
- `summary.csv`
- `csv/isentropic_nozzle_state_00000.csv`

## Notes

- The flow is inert and intentionally remains a **geometry/validation** case
  rather than a chemistry case.
- By default, the example initializes the nozzle state, advances it a small
  number of numerical time steps, and compares both the initialized and advanced
  profiles against theory.
- The nozzle is defined as a smooth, symmetric converging-diverging profile with
  a single throat.
- `advance_time` can be set directly in `main()` if you want more or less
  short-time evolution; otherwise the script uses `n_preservation_steps` times
  the CFL-limited time step.
