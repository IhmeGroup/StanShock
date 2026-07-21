# hyshot_ii

## Purpose

This example models the **HyShot II** hydrogen-fueled scramjet combustor and
validates StanShock's 1D reduced-order solution against 3D RANS (CTR) results
and experimental wall measurements. It is the primary demonstration of the
**jet-in-crossflow (JICF) fuel-injection model** coupled to the flamelet /
progress-variable (FPV) chemistry approach.

## Primary feature demonstrated

- JICF fuel injection via `JICModel` with FPV chemistry (`FPVTable`)
- Compressible wall models (skin friction + heat flux) for a reacting flow
- Stream-averaged inflow boundary condition from RANS data
- Quantitative validation of wall pressure and heat flux against experiment and
  3D RANS, with L1 / L-infinity error metrics

## Validation target

The case is compared against the reference data in `reference_data/`, which
contains experimental measurements (with uncertainty bounds) and 3D RANS (CTR)
profiles for both the **body** and **cowl** walls. Three comparisons are
produced, reproducing the project reference figures:

1. **Fuel-on (reacting)** wall pressure and heat flux
2. **Fuel-off (inert)** wall pressure and heat flux
3. **Equivalence-ratio sweep** — fuel-on body pressure at phi = 0.3 vs 0.5

The reference CSVs store _nondimensional_ quantities; `validation.py`
redimensionalizes them using the freestream reference dynamic pressure (`17.7e6`
Pa) and reference heat flux (`12.37e9` W/m^2).

## Layout

- `case_setup.py` — `Hyshot2Interface`: geometry, conditions, boundary
  conditions, injector, and `Combustor` assembly
- `injector_models.py` — choked-flow fuel property calculations
- `hyshot_ii_jic.py` — FPV (flamelet/progress-variable) driver case
- `hyshot_ii_frc.py` — finite-rate-chemistry variant _(work in progress)_
- `validation.py` — reference-data parsing, comparison plots, error metrics
- `reference_data/` — experimental + CTR RANS wall pressure/heat-flux CSVs

## Outputs

Running the case produces an `output/` folder containing:

- `figures/hyshot_ii_fuel_on.png` — reacting pressure + heat flux
- `figures/hyshot_ii_fuel_off.png` — inert pressure + heat flux
- `figures/hyshot_ii_phi_sweep.png` — phi = 0.3 vs 0.5 body pressure
- `summary.csv` — L1 / L-infinity error metrics vs. experiment

## Run

From the example directory (relative paths to the FPV table are resolved from
here):

```bash
cd examples/hyshot_ii
python hyshot_ii_jic.py
```

The validation plots can also be regenerated from existing StanShock result CSVs
without re-running the simulation:

```bash
python validation.py --reacting-csv <reacting.csv> \
    --inert-csv <inert.csv> --phi-high-csv <reacting_phi0.5.csv>
```

## Notes

- **FPV table:** the flamelet table (`h2_table/flamelet_results/*.h5`) and its
  FlameMaster solutions are large and are **not** tracked in git. Generate them
  with FPVgen (see `h2_table/input.toml`) or point the case at an existing
  table.
- The **finite-rate-chemistry** variant (`hyshot_ii_frc.py`) is not yet
  functional and is retained as work in progress.
- The JICF model pre-computes mixture-fraction profiles on first run and caches
  them as `data/*.npy`; delete this folder to force regeneration.
