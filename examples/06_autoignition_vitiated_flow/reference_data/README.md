# reference_data

This folder contains the lightweight in-repo reference dataset used by the shipped
04 autoignition example.

## Shipped case

- Mixture: **H2–O2–N2**
- Oxidizer composition: **Y_N2 = 0.767**, **Y_O2 = 0.233**
- Oxidizer temperature: **T_O = 1400 K**
- Fuel temperature: **T_F = 300 K**
- Reference study scalar dissipation setting: **chi_0 = 0**

## Files

- `ignition_delay_reference.csv`
  - digitized from the ignition-delay-versus-mixture-fraction curve for the shipped
    case in Caban & Tyliszczak (2024)

## Appendix scalar values used by the example

From the paper appendices for the shipped case:

- temperature-based most reactive mixture fraction:
  - **xi_MR = 7.1e-3**
- temperature-based ignition delay:
  - **t_ign = 0.034 ms**

The example script compares both FRC and FPV against these scalar values in addition
to the digitized ignition-delay curve.
