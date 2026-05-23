# 04_autoignition_vitiated_flow

This example is the **chemistry-model comparison case** in the StanShock flagship
example suite. It demonstrates that the same hydrogen autoignition problem can be run
with both:

- **finite-rate chemistry (FRC)** via `CanteraInterface`, and
- **flamelet/progress-variable chemistry (FPV)** via `FPVTable`.

The scientific reference is the 1D mixture-fraction-space analysis in:

> Caban, L.; Tyliszczak, A. *A Comparative Study of the Hydrogen Auto-Ignition Process
> in Oxygen–Nitrogen and Oxygen–Water Vapor Oxidizer: Numerical Investigations in
> Mixture Fraction Space and 3D Forced Homogeneous Isotropic Turbulent Flow Field*.
> Energies 2024, 17, 4525.

## Final shipped case

The shipped comparison reproduces the **H$_2$–O$_2$–N$_2$** case with:

- oxidizer temperature: **T$_O$ = 1400 K**
- oxidizer composition at Z = 0: **Y$_{N_2}$ = 0.767**, **Y$_{O_2}$ = 0.233**
- fuel temperature: **T$_F$ = 300 K**
- scalar dissipation parameter in the reference study: **χ$_0$ = 0**
- reference temperature-based scalar values from Appendix B:
  - **ξ$_{MR}$ = 7.1 × 10$^{-3}$**
  - **t$_{ign}$ = 0.034 ms**

## What this StanShock case demonstrates

This is intentionally a **chemistry-focused** example. It avoids injector, wall-model,
and geometry complexity so it can answer one clear question:

> **Can StanShock reproduce the same ignition-development trend with both FRC and FPV?**

To stay honest to the paper’s reduced-coordinate formulation, the StanShock setup is a
**1D reacting surrogate**:

- the computational coordinate `x` is used as a monotonic surrogate for the mixture fraction `Z`,
- each cell is initialized as an inert mixture between a cold hydrogen stream and a hot oxidizer stream,
- the solver is advanced in time,
- and ignition delay is extracted cell-by-cell using the same **temperature-rise criterion**
  used in the paper family discussion.

## Validation target

The primary validation is the **ignition delay curve** in mixture-fraction space,
digitized from the paper for the shipped case:

- `reference_data/ignition_delay_reference.csv`

The script also compares against the paper’s appendix scalar values for:

- most reactive mixture fraction, ξ$_{MR}$
- ignition delay at ξ$_{MR}$

## Running the example

### FRC only

This requires only the mechanism already stored in the repo:

```bash
python autoignition_vitiated_flow.py
```

If the FPV table is missing, the script will automatically skip the FPV branch.

### FRC + FPV

Generate the flamelet table first using the TOML files in `table_input/`, then place
the resulting table at:

```text
table_input/fpv_table.h5
```

Then rerun:

```bash
python autoignition_vitiated_flow.py
```

The script will run both chemistry models and generate:

- **Reference vs FRC vs FPV ignition-delay comparison**
- scalar error metrics
- final-state CSV output
- ignition-delay curve CSV output

## Outputs

The example writes to local case folders:

- `figures/`
- `output/`

Key outputs are:

- `figures/oxygen_nitrogen_1400K_ignition_delay.png`
- `figures/oxygen_nitrogen_1400K_final_profiles.png`
- `output/oxygen_nitrogen_1400K_summary.csv`
- `output/frc_state_00000.csv`
- `output/fpv_state_00000.csv` (if FPV is run)
- `output/frc_ignition_delay_curve.csv`
- `output/fpv_ignition_delay_curve.csv` (if FPV is run)

## CSVWriter

This example also demonstrates the new `CSVWriter` feature by exporting the final
state of each chemistry branch to CSV.

## Reference-data note

The digitized ignition-delay curve in `reference_data/ignition_delay_reference.csv`
comes from the paper figure and therefore may contain small extraction noise. The
script automatically takes the absolute value of the imported ignition-delay column in
case the digitizer exported the axis with an inverted sign.
