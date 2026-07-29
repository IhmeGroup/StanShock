from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from inlet_moc.char_net import CharNet
from inlet_moc.initial_domain import initialize_domain
from inlet_moc.planar_inlet import PlanarInlet
from inlet_moc.rotational_solvers import NoWallIntersectionError, RotationalSolveError
from inlet_moc.shock_solvers.char_shock import (
    DetachedShockError,
    InsufficientCharPtsLError,
    SubsonicFlowError,
)
from inlet_moc.solver_events import get_next_event, handle_event

if TYPE_CHECKING:
    from inlet_moc.processing.triangulated_solution import TriangulatedSolution


class MOCSolution:
    def __init__(
        self,
        inlet: PlanarInlet,
        Mach: float,
        theta: float,
        T_amb: float,
        p_amb: float,
        N_idl: int,
        gamma: float = 1.4,
        R: float = 287.50,
        max_iters: int = 20,
        tol: float = 1e-6,
        x_stop: float | None = None,
        verbose: bool = False,
        plot_during_solve: bool = False,
        figdir: str | Path = "./01_figs",
        case_name: str | None = None,
    ) -> None:
        self.inlet = inlet
        self.x_prog = float(self.inlet.get_infl0()[1][0])
        self.M_init = Mach
        self.theta_init = theta

        self.T_amb = T_amb
        self.p_amb = p_amb
        self.gamma = gamma
        self.R = R

        self.N_idl = N_idl
        self.max_iters = int(max_iters)
        self.tol = tol
        self.x_stop = (
            max(float(self.inlet.centerbody.x_max), float(self.inlet.cowl.x_max))
            if x_stop is None
            else float(x_stop)
        )

        self.plot_during_solve = bool(plot_during_solve)
        self.verbose = bool(verbose)
        self.case_name = None if case_name is None else str(case_name).strip() or None

        self.cells = []
        self.nets = []
        self.leading_shock = None
        self.events = []
        self.next_net = None
        self.figdir = Path(figdir)
        self.error_debug_net: CharNet | None = None
        self.error_plot_path: Path | None = None
        self.solve_stopped_early = False
        self.solve_stop_reason: str | None = None
        self.x_start: float | None = None

        self.point_mesh: TriangulatedSolution | None = None
        self._streamtube: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[MOCSolution] {message}")

    def _store_net(self, net: CharNet) -> CharNet:
        if not any(existing is net for existing in self.nets):
            self.nets.append(net)
        self.error_debug_net = net
        return net

    def _net_progress(self, net: CharNet) -> float:
        corner_mask = net.point_mask("corner")
        if np.count_nonzero(corner_mask) >= 3:
            x_corner = np.sort(np.asarray(net.x[corner_mask], dtype=float))
            self.x_prog = max(self.x_prog, float(x_corner[1:-1][-1]))
        return float(self.x_prog)

    def _stop_reached(self) -> bool:
        return bool(self.x_prog >= (self.x_stop - self.tol))

    def x_final(self) -> float:
        return float(min(self.x_prog, self.x_stop))

    @property
    def streamtube(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._streamtube is None:
            msg = "streamtube is unavailable until processing computes it."
            raise RuntimeError(msg)
        return self._streamtube

    def save_last_progress_plot(
        self,
        error_message: str | None = None,
        *,
        highlight_net: CharNet | None = None,
        highlight_ids: bool = False,
        output_path: str | Path | None = None,
    ) -> Path:
        from inlet_moc.plot.solution import save_error_state_plot

        self.error_plot_path = save_error_state_plot(
            self,
            subtitle=error_message,
            highlight_net=highlight_net,
            highlight_ids=highlight_ids,
            output_path=output_path,
        )
        self._log(f"Saved error-state plot {self.error_plot_path}.")
        return self.error_plot_path

    def case_filename(self, filename: str) -> str:
        if self.case_name is None:
            return filename
        return f"{self.case_name}_{filename}"

    def _finish_net_L(self, net: CharNet, message: str) -> float:
        x_prog = self._net_progress(net)
        self.error_debug_net = net
        self._log(message.format(net_num=len(self.nets), x_prog=x_prog))
        return x_prog

    def _stop_at_x_stop(self, net: CharNet) -> bool:
        if not self._stop_reached():
            return False
        self.solve_stopped_early = True
        self.solve_stop_reason = f"Reached x_final={self.x_final():.3e}:."
        self._log(self.solve_stop_reason)
        self.next_net = net
        return True

    def _stop_on_supported_solver_limit(self, err: Exception) -> None:
        self.solve_stopped_early = True
        self.solve_stop_reason = str(err)
        self._log(f"[warning]: {err} Solve stopped.")
        self.save_last_progress_plot(str(err), highlight_net=self.error_debug_net)

    def _register_incomplete_net(self, net: CharNet) -> int:
        self._store_net(net)
        net_num = len(self.nets)
        print(f"[warning] Net {net_num} not fully solved; proceeding to next event.")
        return net_num

    def solve_inlet(self) -> None:
        try:
            self._log("Building and solving initial net.")
            active_net = self.initialize_domain()
            self.error_debug_net = active_net
            initial_net_incomplete = False
            try:
                active_net.solve_net()
                self._store_net(active_net)
            except (NoWallIntersectionError, RotationalSolveError):
                self._register_incomplete_net(active_net)
                initial_net_incomplete = True

            net_num = len(self.nets)
            next_event = get_next_event(active_net, self.inlet, self.tol)

            if initial_net_incomplete:
                self._log(f"Net {net_num} partially solved. x_prog={self.x_prog:.6g}")
            elif next_event is None:
                self._log(f"Net {net_num} completed. No shock event was detected.")
            else:
                self._log(f"Net {net_num} completed. Awaiting first event.")

            if next_event is None:
                self.next_net = None
                return

            self._log(f"First event detected: family={next_event.family}.")

            while next_event is not None:
                self.events.append(next_event)
                self._log(
                    f"Handling event {len(self.events)}: family={next_event.family}."
                )
                try:
                    active_net, downstream_net, reflected_event = handle_event(
                        inlet=self.inlet,
                        max_iters=self.max_iters,
                        tol=self.tol,
                        plot=self.plot_during_solve,
                        net_im1=active_net,
                        event_i=next_event,
                        figdir=self.figdir,
                    )
                except (
                    NoWallIntersectionError,
                    RotationalSolveError,
                    DetachedShockError,
                    InsufficientCharPtsLError,
                    SubsonicFlowError,
                ) as err:
                    self.error_debug_net = getattr(err, "current_net", active_net)
                    raise

                self._finish_net_L(
                    active_net,
                    "Net {net_num} finalized. x_prog={x_prog:.6g}",
                )
                if self._stop_at_x_stop(active_net):
                    return

                active_net = self._store_net(downstream_net)
                next_event = reflected_event
                if next_event is None:
                    next_event = get_next_event(active_net, self.inlet, self.tol)

            self.next_net = active_net
            return

        except (
            NoWallIntersectionError,
            RotationalSolveError,
            DetachedShockError,
            InsufficientCharPtsLError,
            SubsonicFlowError,
        ) as err:
            self._stop_on_supported_solver_limit(err)
            return
        except Exception:
            raise

    def initialize_domain(self) -> CharNet:
        domain = initialize_domain(
            self.inlet,
            self.M_init,
            self.theta_init,
            self.T_amb,
            self.p_amb,
            self.N_idl,
            self.gamma,
            self.R,
            self.max_iters,
            self.tol,
        )
        self.leading_shock = domain.leading_shock
        self.cells.extend(domain.cells)
        self.x_start = domain.x_start
        return domain.net
