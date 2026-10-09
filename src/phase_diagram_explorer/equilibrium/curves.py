import numpy as np

from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.thermo.sublattice import SublatticePhase


def evaluate_phase_curves(system: dict, T: float, n_points: int = 500):
    """Evaluate molar Gibbs energy for every phase in `system` on a shared grid.

    `system` maps phase name to a phase model (a SublatticePhase, including
    SolutionPhase and StoichiometricPhase). Stoichiometric phases have a
    fixed composition and are represented as a single point marker at the
    nearest grid index, with NaN elsewhere; other phases are NaN outside
    their composition range.

    Returns (x, curves): x is the shared composition grid, shape (n_points,);
    curves maps phase name to a (n_points,) array of Gibbs energies.
    """
    x = np.linspace(0.0, 1.0, n_points)
    curves = {}

    for name, phase in system.items():
        if isinstance(phase, StoichiometricPhase):
            curves[name] = _stoichiometric_curve(phase, T, x)
        elif isinstance(phase, SublatticePhase):
            curve = np.asarray(phase.molar_gibbs(T, x), dtype=float)
            curves[name] = np.where(np.isfinite(curve), curve, np.nan)
        else:
            raise TypeError(f"unsupported phase type for {name!r}: {type(phase)!r}")

    return x, curves


def _stoichiometric_curve(phase: StoichiometricPhase, T: float, x: np.ndarray) -> np.ndarray:
    curve = np.full_like(x, np.nan)
    index = int(np.argmin(np.abs(x - phase.composition)))
    curve[index] = phase.molar_gibbs(T)
    return curve


def _cross(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)


def lower_convex_hull(x: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower convex hull of the (x, G) point cloud, sorted by x.

    Non-finite points (e.g. the NaN filler in stoichiometric phase curves) are
    ignored, so curves from evaluate_phase_curves can be concatenated and
    passed in directly. When multiple points share the same x, only the
    lowest G is kept before computing the hull via a monotone chain.
    """
    x = np.asarray(x, dtype=float)
    G = np.asarray(G, dtype=float)

    mask = np.isfinite(x) & np.isfinite(G)
    x = x[mask]
    G = G[mask]

    x, inverse = np.unique(x, return_inverse=True)
    G_min = np.full(x.shape, np.inf)
    np.minimum.at(G_min, inverse, G)
    G = G_min

    hull_x: list[float] = []
    hull_G: list[float] = []

    for xi, gi in zip(x, G):
        while len(hull_x) >= 2 and _cross(
            hull_x[-2], hull_G[-2], hull_x[-1], hull_G[-1], xi, gi
        ) <= 0:
            hull_x.pop()
            hull_G.pop()
        hull_x.append(xi)
        hull_G.append(gi)

    return np.array(hull_x), np.array(hull_G)
