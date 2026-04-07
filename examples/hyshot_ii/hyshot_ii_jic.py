from __future__ import annotations

from pathlib import Path

from case_setup import Hyshot2Interface

from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.plot import XTDiagram

# Output data directories
fig_dir = Path("./figures")
fig_dir.mkdir(exist_ok=True)
(fig_dir / "anim").mkdir(exist_ok=True)

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# Initialize and run the simulation
sim = Hyshot2Interface(chemistry="FPV", inflow="constant")
ss = sim.case
ss.verbose = True
ss.output_every = 100
ss.plot_state_interval = 100

plot_variables = [
    "density",
    "velocity",
    "pressure",
    "temperature",
    "mixture fraction",
    "progress variable",
    "mach",
]

# Update CSV writer initialization to match plot_state_interval
csv_writer = CSVWriter(
    combustor=ss,
    filename=output_dir
    / "data.csv",  # Will become test_00000.csv, test_00001.csv, etc.
    interval=100,  # Same as plot_state_interval=100
)
ss.csv_writers = [csv_writer]

ss.plot_variables = plot_variables
# ss.plot_variables = [
#     "density",
#     "velocity",
#     "pressure",
#     "temperature",
#     "mach",
#     ["Y_H2", "Y_OH", "Y_H2O"],
# ]
ss.xt_diagrams = [XTDiagram(ss, variable, skip_steps=10) for variable in plot_variables]
ss.advance_simulation(ss.injectors[0].t_inj[-1])
for diagram in ss.xt_diagrams:
    diagram.plot(figdir=fig_dir)

# code.interact(local=locals())
