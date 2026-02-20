from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from stanshock.system.backend import Array
from stanshock.models.boundary_layer import BoundaryLayer

class CSVWriter:
    """
    Writes simulation data to CSV file at specified intervals.
    Creates a new numbered file for each output.
    """
    def __init__(
        self,
        combustor,
        filename: str | Path,
        interval: int = 100,  # Write every 'interval' iterations (like plot_state_interval)
        wall_temperature: float | None = None,
        heated_perimeter_fraction: float = 1.0,
        use_adiabatic_wall_temperature: bool = True,
        recovery_model: str = "turbulent_Pr13",
        recovery_factor_constant: float = 0.88,
    ) -> None:
        """
        Initialize CSV writer.
        
        Args:
            combustor: Combustor instance
            filename: Base output CSV file path (will be numbered: filename_00000.csv, etc.)
            interval: Write every 'interval' iterations (0 = never)
            wall_temperature: Wall temperature for heat flux calculation
        """
        self.combustor = combustor
        self.base_filename = Path(filename)
        self.interval = interval
        self.wall_temperature = wall_temperature
        self.output_counter = 0

        self.heated_perimeter_fraction = float(heated_perimeter_fraction)
        self.use_adiabatic_wall_temperature = bool(use_adiabatic_wall_temperature)
        self.recovery_model = str(recovery_model)
        self.recovery_factor_constant = float(recovery_factor_constant)
        
        # Create boundary layer instance for heat flux calculation
        self.boundary_layer = BoundaryLayer(
            wall_temperature=wall_temperature,
            geometry=combustor.geometry,
            physics=combustor.physics,
            heated_perimeter_fraction=self.heated_perimeter_fraction,
            use_adiabatic_wall_temperature=self.use_adiabatic_wall_temperature,
            recovery_model=self.recovery_model,
            recovery_factor_constant=self.recovery_factor_constant,
        )
        
    def update(self, iteration: int) -> None:
        """Update CSV with current state if iteration matches interval."""
        # Same condition as plot_state in combustor.py
        if self.interval <= 0:
            return
        if iteration % self.interval != 0:
            return
            
        self.write_current_state()
    
    def write_current_state(self) -> None:
        """Write current state to a new numbered CSV file."""
        state = self.combustor.state
        physics = self.combustor.physics
        geometry = self.combustor.geometry
        
        # Get interior cell indices (exclude ghost cells)
        idx = geometry.idx_cells
        
        # Get cell centers (excluding ghost cells)
        x = geometry.xc[idx]
        
        # Get basic flow properties (for interior cells only)
        pressure = physics.get_pressure(state)[idx]
        temperature = physics.get_temperature(state)[idx]
        velocity = state.velocity[idx]
        sound_speed = physics.get_sound_speed(state)[idx]
        mach = velocity / sound_speed
        
        # Calculate wall heat flux
        wall_heat_flux = self.calculate_wall_heat_flux(state)
        
        # Get species mass fractions if available
        species_data = self.get_species_mass_fractions(state)
        
        # Prepare data rows
        rows = []
        for i in range(len(x)):
            row = {
                'x [m]': x[i],
                'pressure [Pa]': pressure[i],
                'temperature [K]': temperature[i],
                'Mach': mach[i],
                'velocity [m/s]': velocity[i],
                'wall_heat_flux [W/m^2]': wall_heat_flux[i],
            }
            
            # Add species data if available
            for species_name, values in species_data.items():
                if i < len(values):
                    row[species_name] = values[i]
                else:
                    row[species_name] = 0.0
            
            rows.append(row)
        
        # Create numbered filename
        stem = self.base_filename.stem
        suffix = self.base_filename.suffix
        parent = self.base_filename.parent
        filename = parent / f"{stem}_{self.output_counter:05d}{suffix}"
        
        # Write to CSV
        self._write_rows(rows, filename)
        self.output_counter += 1
    
    def calculate_wall_heat_flux(self, state) -> Array:
        """Return wall heat flux q'' [W/m^2] at cell centers (interior cells only).

        This is intentionally consistent with BoundaryLayer.source_implementation()
        in the patched scramjet-oriented model:

            q'' = St * rho * |U| * cp * (T_drive - T_w)

        where T_drive is either the static temperature T or the adiabatic-wall
        (recovery) temperature T_aw.
        """

        # If no wall temperature is defined, treat as adiabatic (no heat flux).
        if self.wall_temperature is None:
            idx = self.combustor.geometry.idx_cells
            return np.zeros_like(self.combustor.geometry.xc[idx])

        geometry = self.combustor.geometry
        physics = self.combustor.physics

        # Interior cells
        idx = geometry.idx_cells
        x = geometry.xc[idx]
        t = float(self.combustor.t)

        # Geometry (consistent with BoundaryLayer)
        A = geometry.area(t, x)
        P = geometry.perimeter(t, x)
        Dh = geometry.hydraulic_diameter(t, x)
        _ = A, P  # (not needed for q'' itself, but available if you want P/A in the CSV)

        # Flow/transport
        T = physics.get_temperature(state)[idx]
        mu = physics.get_mu(state)[idx]
        rho = state.density[idx]
        U = state.velocity[idx]
        a = physics.get_sound_speed(state)[idx]

        Re = np.abs(rho * U * Dh / mu)
        Mach = np.abs(U / a)

        # Cf table expects T/Tw
        Tw = max(float(self.wall_temperature), 1e-12)
        T_Tw = T / Tw
        cf = self.boundary_layer.skin_friction_coefficient(Re, Mach, T_Tw)

        cp = physics.get_cp(state)[idx]
        k = physics.get_thermal_conductivity(state)[idx]
        Pr = cp * mu / k

        St = self.boundary_layer.get_stanton_number(Re=Re, Pr=Pr, cf=cf, Dh=Dh)

        if self.use_adiabatic_wall_temperature:
            # Adiabatic-wall (recovery) temperature:
            #   T_aw = T * (1 + r * (gamma-1)/2 * M^2)
            gamma = getattr(physics, "gamma", 1.4)
            if self.recovery_model == "turbulent_Pr13":
                r = Pr ** (1.0 / 3.0)
            elif self.recovery_model == "constant":
                r = np.full_like(Pr, self.recovery_factor_constant)
            else:
                # Fall back to the common turbulent recovery default
                r = Pr ** (1.0 / 3.0)
            T_drive = T * (1.0 + r * 0.5 * (gamma - 1.0) * Mach**2)
        else:
            T_drive = T

        q_w = St * rho * np.abs(U) * cp * (T_drive - self.wall_temperature)
        return q_w
    
    def get_species_mass_fractions(self, state):
        """Get species mass fractions if available."""
        species_data = {}
        
        # Get interior cell indices
        idx = self.combustor.geometry.idx_cells
        
        # Try to get species from FPV table
        if hasattr(self.combustor.physics, 'get_mass_fractions'):
            try:
                mass_fractions = self.combustor.physics.get_mass_fractions(state)
                # Get common species names
                species_names = ['Y_O2', 'Y_H2O', 'Y_O', 'Y_H2', 'Y_H']
                
                for name in species_names:
                    species_short = name[2:]  # Remove 'Y_' prefix
                    if hasattr(state, species_short.lower()):
                        # Get array and select interior cells
                        full_array = getattr(state, species_short.lower())
                        if hasattr(full_array, '__len__') and len(full_array) > len(idx):
                            # Array includes ghost cells
                            species_data[name] = full_array[idx]
                        else:
                            species_data[name] = full_array
            except Exception as e:
                # If we can't get species, that's OK
                pass
                
        return species_data
    
    def _write_rows(self, rows: list[dict], filename: Path) -> None:
        """Write rows to CSV file."""
        if not rows:
            return
            
        fieldnames = list(rows[0].keys())
        
        # Ensure output directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Ensure output directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Write new file (overwrites if exists)
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)