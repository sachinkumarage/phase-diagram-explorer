from dataclasses import dataclass

import numpy as np

from phase_diagram_explorer.equilibrium.equilibrium import (
    _hull_for_temperature,
    _resolve_from_hull,
)


@dataclass
class PhaseDiagram:
    T_grid: np.ndarray
    x_grid: np.ndarray
    phase_labels: np.ndarray
    phase_fractions: np.ndarray


def compute_diagram(
    system: dict,
    T_range: tuple[float, float],
    n_T: int = 200,
    n_x: int = 200,
    n_points: int = 500,
) -> PhaseDiagram:
    """Compute a binary T-x phase diagram over a temperature and composition grid.

    For each temperature, every phase's Gibbs energy curve and the resulting
    lower convex hull are evaluated exactly once and reused across the whole
    composition grid at that temperature, rather than recomputed per
    composition point.

    phase_labels[i, j] is a tuple of the stable phase name(s) at
    (T_grid[i], x_grid[j]); phase_fractions[i, j] is the matching dict of
    phase name to phase fraction.
    """
    T_min, T_max = T_range
    T_grid = np.linspace(T_min, T_max, n_T)
    x_grid = np.linspace(0.0, 1.0, n_x)

    phase_labels = np.empty((n_T, n_x), dtype=object)
    phase_fractions = np.empty((n_T, n_x), dtype=object)

    for i, T in enumerate(T_grid):
        hull_x, hull_G, hull_labels = _hull_for_temperature(system, T, n_points=n_points)
        for j, x in enumerate(x_grid):
            result = _resolve_from_hull(system, T, x, hull_x, hull_G, hull_labels)
            phase_labels[i, j] = tuple(result.stable_phases)
            phase_fractions[i, j] = dict(result.phase_fractions)

    return PhaseDiagram(
        T_grid=T_grid,
        x_grid=x_grid,
        phase_labels=phase_labels,
        phase_fractions=phase_fractions,
    )
