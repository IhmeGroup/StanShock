# 04_jicf_heat_release_profile

This example pivots 04 toward a **jet-in-crossflow FPV validation case** that is
closer to StanShock's intended scramjet-style workflow than the earlier
standalone autoignition benchmark.

The reference paper is:

> Micka, D. J.; Driscoll, J. F. _Stratified jet flames in a heated (1390 K) air
> cross-flow with autoignition_. Combustion and Flame 159 (2012) 1205–1214.

## Why this case

This paper is a good match for a pseudo-1D JICF example because it reports
reduced, streamwise quantities that StanShock can compare honestly:

- the axial heat-release profile `q(x)`
- the flame liftoff distance
- the flame length

The paper explicitly emphasizes that heat-release distribution is important for
ramjet/scramjet applications and provides `q(x)` plots and flame-length data for
a pure-hydrogen JICF case.

## Shipped case

This example is configured around **Case 2** from Table 1 of the paper:

- fuel: pure hydrogen
- crossflow static temperature: **1413 K**
- fuel static temperature: **247 K**
- air velocity: **487 m/s**
- fuel velocity: **1198 m/s**
- air density: **0.62 kg/m^3**
- fuel density: **0.504 kg/m^3**
- jet diameter: **2.49 mm**
- test-section height: **25.4 mm**
- test-section width: **38.1 mm**

## Important modeling note

The paper reports an experimentally inferred heat-release profile `q(x)` from
chemiluminescence. StanShock does not produce that quantity directly in the same
way.

So this example compares a **normalized heat-release proxy**:

- the local transported progress-variable source is converted to a line quantity
  by multiplying by cross-sectional area
- the resulting profile is normalized so that its integral is 1
- comparison is therefore made against **`q/Q` shape**, plus:
  - liftoff distance
  - `x90`: the axial distance where 90% of cumulative model heat release has
    occurred

This keeps the comparison honest while still using the most useful paper
observable.

## Required user step

Generate the flamelet table first and place it at:

```text
h2_table/fpv_table.h5
```

Use the provided `h2_table/input.toml` as the starting point.

## Optional reference data

To activate the comparison plots and error metrics, add:

- `reference_data/case2_q_over_Q.csv`
  - columns: `x_mm,q_over_Q`
- `reference_data/case2_scalar_metrics.csv`
  - columns: `metric,value`
  - rows such as:
    - `liftoff_mm,<value>`
    - `x90_mm,<value>`

The example will still run without these files.

## Outputs

The script writes:

- `figures/case2_q_over_Q_comparison.png`
- `figures/case2_state_profiles.png`
- `figures/xt/...`
- `output/case2_state.csv`
- `output/case2_profiles.csv`
- `output/case2_q_over_Q.csv`
- `output/case2_summary.csv`

## Caveats

This is an **FPV-only** case for now. It is meant to answer:

> _How sufficient is the current JICF/FPV modeling for a scramjet-relevant,
> autoignition-assisted hydrogen JICF flame?_

It is not yet a direct FPV-vs-FRC JICF benchmark.
