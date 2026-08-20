from dataclasses import dataclass, field

import numpy as np

from phase_diagram_explorer.equilibrium.curves import _cross, evaluate_phase_curves
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

COMPOSITION_TOLERANCE = 1e-6


@dataclass
class EquilibriumResult:
    T: float
    x_overall: float
    stable_phases: list[str]
    phase_compositions: dict[str, float] = field(default_factory=dict)
    phase_fractions: dict[str, float] = field(default_factory=dict)
    total_gibbs: float = 0.0


def _labeled_lower_hull(x: np.ndarray, G: np.ndarray, labels: list[str]):
    """Lower convex hull of (x, G), keeping the owning phase label at each hull point.

    At each unique x, only the lowest G (and its label) is kept before running
    the hull, so this implements the standard lower-envelope-then-hull method:
    the envelope over all phases is taken first, then its convex hull gives
    the true global tie-lines.
    """
    x = np.asarray(x, dtype=float)
    G = np.asarray(G, dtype=float)
    labels = np.asarray(labels, dtype=object)

    order = np.argsort(x, kind="stable")
    x, G, labels = x[order], G[order], labels[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    G_min = np.full(unique_x.shape, np.inf)
    label_min = np.empty(unique_x.shape, dtype=object)
    for i in range(len(x)):
        j = inverse[i]
        if G[i] < G_min[j]:
            G_min[j] = G[i]
            label_min[j] = labels[i]

    hull_x: list[float] = []
    hull_G: list[float] = []
    hull_labels: list[str] = []

    for xi, gi, li in zip(unique_x, G_min, label_min):
        while len(hull_x) >= 2 and _cross(
            hull_x[-2], hull_G[-2], hull_x[-1], hull_G[-1], xi, gi
        ) <= 0:
            hull_x.pop()
            hull_G.pop()
            hull_labels.pop()
        hull_x.append(xi)
        hull_G.append(gi)
        hull_labels.append(li)

    return np.array(hull_x), np.array(hull_G), hull_labels


def _phase_composition(system: dict, label: str, grid_x: float) -> float:
    phase = system[label]
    if isinstance(phase, StoichiometricPhase):
        return phase.composition
    return grid_x


def compute_equilibrium(
    system: dict, T: float, x_overall: float, n_points: int = 500
) -> EquilibriumResult:
    """Compute the stable phase(s), compositions, and fractions at (T, x_overall).

    Builds the global lower convex hull over every phase's Gibbs energy curve
    (the common-tangent construction), then reads off the equilibrium state
    for x_overall from the hull segment it falls on: a single stable phase if
    x_overall lands on a segment belonging to one phase, or two phases tied
    together by a common tangent (a two-phase tie line) otherwise, with phase
    fractions from the lever rule.
    """
    x, curves = evaluate_phase_curves(system, T, n_points=n_points)

    all_x: list[np.ndarray] = []
    all_G: list[np.ndarray] = []
    all_labels: list[str] = []
    for name, curve in curves.items():
        finite = np.isfinite(curve)
        all_x.append(x[finite])
        all_G.append(curve[finite])
        all_labels.extend([name] * int(finite.sum()))

    hull_x, hull_G, hull_labels = _labeled_lower_hull(
        np.concatenate(all_x), np.concatenate(all_G), all_labels
    )

    if x_overall < hull_x[0] - COMPOSITION_TOLERANCE or x_overall > hull_x[-1] + COMPOSITION_TOLERANCE:
        raise ValueError(
            f"x_overall={x_overall} is outside the valid composition range "
            f"[{hull_x[0]}, {hull_x[-1]}]"
        )

    idx = int(np.searchsorted(hull_x, x_overall))
    idx = min(max(idx, 1), len(hull_x) - 1)
    x_left, x_right = hull_x[idx - 1], hull_x[idx]
    G_left, G_right = hull_G[idx - 1], hull_G[idx]
    label_left, label_right = hull_labels[idx - 1], hull_labels[idx]

    if label_left == label_right:
        phase = system[label_left]
        if isinstance(phase, StoichiometricPhase):
            total_gibbs = phase.molar_gibbs(T)
            composition = phase.composition
        else:
            total_gibbs = float(phase.molar_gibbs(T, x_overall))
            composition = x_overall

        return EquilibriumResult(
            T=T,
            x_overall=x_overall,
            stable_phases=[label_left],
            phase_compositions={label_left: composition},
            phase_fractions={label_left: 1.0},
            total_gibbs=total_gibbs,
        )

    # two-phase region: the hull segment between x_left and x_right is the
    # common tangent line, so its endpoints are the equilibrium phase
    # compositions; snap stoichiometric endpoints to their exact composition.
    x_left = _phase_composition(system, label_left, x_left)
    x_right = _phase_composition(system, label_right, x_right)

    if x_right - x_left < COMPOSITION_TOLERANCE:
        fraction_right = 0.0
    else:
        fraction_right = (x_overall - x_left) / (x_right - x_left)
    fraction_right = min(max(fraction_right, 0.0), 1.0)
    fraction_left = 1.0 - fraction_right

    total_gibbs = fraction_left * G_left + fraction_right * G_right

    return EquilibriumResult(
        T=T,
        x_overall=x_overall,
        stable_phases=[label_left, label_right],
        phase_compositions={label_left: x_left, label_right: x_right},
        phase_fractions={label_left: fraction_left, label_right: fraction_right},
        total_gibbs=total_gibbs,
    )
