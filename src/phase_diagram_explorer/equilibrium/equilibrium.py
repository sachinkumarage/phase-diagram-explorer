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


COMPOSITION_SET_SEPARATOR = "#"
# A tie line joining two points of the same phase is a miscibility gap only
# if the envelope points it bridges lie at least this far (J/mol) above it;
# smaller deviations are round-off on a convex curve.
MISCIBILITY_GAP_TOLERANCE = 1e-6
# x_overall this close to a hull vertex is resolved as that vertex's phase alone.
VERTEX_TOLERANCE = 1e-12


def composition_set_label(phase_name: str, index: int) -> str:
    return f"{phase_name}{COMPOSITION_SET_SEPARATOR}{index}"


def base_phase_name(label: str) -> str:
    """Phase name without a composition set suffix: "FCC_A1#2" -> "FCC_A1"."""
    name, separator, index = label.rpartition(COMPOSITION_SET_SEPARATOR)
    return name if separator and index.isdigit() else label


def _number_composition_sets(
    env_x: np.ndarray, env_G: np.ndarray, env_labels: np.ndarray, hull_idx: list[int]
) -> list[str]:
    """Hull vertex labels, with composition sets numbered for phases that
    have a miscibility gap.

    A hull segment between two vertices of the same phase that bridges
    envelope points lying above it is a tie line between two tangent points
    on that one phase's Gibbs curve. Each such phase's disjoint stretches on
    the hull are then labelled PHASE#1, PHASE#2, ... from left to right;
    phases without a gap keep their plain name.
    """
    hull_labels = [env_labels[i] for i in hull_idx]
    gap_after = [False] * len(hull_idx)
    gapped_phases: set[str] = set()

    for k in range(len(hull_idx) - 1):
        i, j = hull_idx[k], hull_idx[k + 1]
        if hull_labels[k] != hull_labels[k + 1] or j - i < 2:
            continue
        bridged_x = env_x[i + 1:j]
        tie_line = env_G[i] + (env_G[j] - env_G[i]) * (bridged_x - env_x[i]) / (env_x[j] - env_x[i])
        if np.max(env_G[i + 1:j] - tie_line) > MISCIBILITY_GAP_TOLERANCE:
            gap_after[k] = True
            gapped_phases.add(hull_labels[k])

    if not gapped_phases:
        return hull_labels

    numbered: list[str] = []
    set_count: dict[str, int] = {}
    for k, label in enumerate(hull_labels):
        if label not in gapped_phases:
            numbered.append(label)
            continue
        if k == 0 or hull_labels[k - 1] != label or gap_after[k - 1]:
            set_count[label] = set_count.get(label, 0) + 1
        numbered.append(composition_set_label(label, set_count[label]))
    return numbered


def _labeled_lower_hull(x: np.ndarray, G: np.ndarray, labels: list[str]):
    """Lower convex hull of (x, G), keeping the owning phase label at each hull point.

    At each unique x, only the lowest G (and its label) is kept before running
    the hull, so this implements the standard lower-envelope-then-hull method:
    the envelope over all phases is taken first, then its convex hull gives
    the true global tie-lines. Phases with a miscibility gap get numbered
    composition set labels (see _number_composition_sets).
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

    hull_idx: list[int] = []
    for i in range(len(unique_x)):
        while len(hull_idx) >= 2 and _cross(
            unique_x[hull_idx[-2]], G_min[hull_idx[-2]],
            unique_x[hull_idx[-1]], G_min[hull_idx[-1]],
            unique_x[i], G_min[i],
        ) <= 0:
            hull_idx.pop()
        hull_idx.append(i)

    hull_labels = _number_composition_sets(unique_x, G_min, label_min, hull_idx)
    return unique_x[hull_idx], G_min[hull_idx], hull_labels


def _phase_composition(system: dict, label: str, grid_x: float) -> float:
    phase = system[base_phase_name(label)]
    if isinstance(phase, StoichiometricPhase):
        return phase.composition
    return grid_x


def _hull_for_temperature(system: dict, T: float, n_points: int = 500):
    """Evaluate every phase's Gibbs energy curve at T and build the labeled hull.

    Solution phases contribute their curve on the shared composition grid;
    stoichiometric phases contribute one point at their exact composition
    (not the nearest grid point).

    This is the expensive step (one Gibbs energy evaluation per phase per grid
    point) and depends only on T, not on composition, so callers that need
    equilibrium at many compositions for the same T should compute this once
    and reuse it via _resolve_from_hull.
    """
    x, curves = evaluate_phase_curves(system, T, n_points=n_points)

    all_x: list[np.ndarray] = []
    all_G: list[np.ndarray] = []
    all_labels: list[str] = []
    for name, curve in curves.items():
        phase = system[name]
        if isinstance(phase, StoichiometricPhase):
            all_x.append(np.array([phase.composition]))
            all_G.append(np.array([phase.molar_gibbs(T)], dtype=float))
            all_labels.append(name)
            continue
        finite = np.isfinite(curve)
        all_x.append(x[finite])
        all_G.append(curve[finite])
        all_labels.extend([name] * int(finite.sum()))

    return _labeled_lower_hull(
        np.concatenate(all_x), np.concatenate(all_G), all_labels
    )


def _resolve_from_hull(
    system: dict,
    T: float,
    x_overall: float,
    hull_x: np.ndarray,
    hull_G: np.ndarray,
    hull_labels: list[str],
) -> EquilibriumResult:
    """Read off the equilibrium state at x_overall from a precomputed hull."""
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

    # Exactly on a hull vertex (e.g. a stoichiometric compound at its own
    # composition): that vertex's phase alone, not a tie line ending there.
    nearest = int(np.argmin(np.abs(hull_x - x_overall)))
    if abs(hull_x[nearest] - x_overall) <= VERTEX_TOLERANCE:
        label_left = label_right = hull_labels[nearest]

    if label_left == label_right:
        phase = system[base_phase_name(label_left)]
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
    hull_x, hull_G, hull_labels = _hull_for_temperature(system, T, n_points=n_points)
    return _resolve_from_hull(system, T, x_overall, hull_x, hull_G, hull_labels)
