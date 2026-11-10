# reference_data

Add digitized Micka & Driscoll Case 2 comparison data here.

## case2_q_over_Q.csv

Expected columns:

```text
x_mm,q_over_Q
```

These should come from a digitized version of the hydrogen heat-release profile
(Case 2) in the paper.

## case2_scalar_metrics.csv

Expected columns:

```text
metric,value
```

Suggested rows:

```text
liftoff_mm,<value>
x90_mm,<value>
```

where:

- `liftoff_mm` is the flame liftoff distance inferred from the paper
- `x90_mm` is the axial distance where 90% of total heat release has occurred
