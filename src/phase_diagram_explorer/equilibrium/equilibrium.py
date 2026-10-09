from dataclasses import dataclass, field

import numpy as np

from scipy.optimize import minimize_scalar

from phase_diagram_explorer.equilibrium.curves import _cross, evaluate_phase_curves
from phase_diagram_explorer.equilibrium.tangent import common_tangent, gibbs, tangent_line
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


@dataclass
class PhaseField:
    """One composition interval of a PhaseAssemblage: a single-phase stretch
    (one label) or a two-phase tie line (two labels, lower-x phase first)."""

    phases: tuple[str, ...]
    x_min: float
    x_max: float

    @property
    def is_two_phase(self) -> bool:
        return len(self.phases) == 2

    @property
    def base_phases(self) -> tuple[str, ...]:
        return tuple(base_phase_name(phase) for phase in self.phases)


@dataclass
class PhaseAssemblage:
    """The equilibrium state of a binary system across all compositions at
    one temperature: fields tile [0, 1] in composition order, and tie-line
    endpoints are exact common tangents (not grid points)."""

    T: float
    fields: list[PhaseField]

    @property
    def key(self) -> tuple[tuple[str, ...], ...]:
        """Ordered field structure by base phase name, ignoring compositions."""
        return tuple(field.base_phases for field in self.fields)

    @property
    def tie_lines(self) -> list[PhaseField]:
        return [field for field in self.fields if field.is_two_phase]


# Envelope points sampled inside a tie line when checking that no other phase
# lies below it; geometric spacing resolves fields hugging the tie-line ends.
STABILITY_SAMPLES = 201
STABILITY_EDGE_SAMPLES = 40
STABILITY_TOLERANCE = 1e-6  # J/mol
MAX_TIE_SPLITS = 4


def _sample_inside(x_lo: float, x_hi: float) -> np.ndarray:
    width = x_hi - x_lo
    edge = np.geomspace(1e-7, 0.5, STABILITY_EDGE_SAMPLES) * width
    x = np.concatenate([np.linspace(x_lo, x_hi, STABILITY_SAMPLES), x_lo + edge, x_hi - edge])
    x = np.unique(x)
    return x[(x > x_lo) & (x < x_hi) & (x > 0.0) & (x < 1.0)]


def _deepest_phase_below(system: dict, T: float, tie: PhaseField):
    """(phase name, composition, depth) of the phase lying furthest below
    the tie line's tangent, or None if every other phase lies above it."""
    left, right = (system[name] for name in tie.base_phases)
    slope, intercept = tangent_line(left, right, T, tie.x_min, tie.x_max)
    x = _sample_inside(tie.x_min, tie.x_max)
    best = None

    for name, phase in system.items():
        if name in tie.base_phases:
            continue
        if isinstance(phase, StoichiometricPhase):
            if not tie.x_min < phase.composition < tie.x_max:
                continue
            x_q = phase.composition
            depth = float(phase.molar_gibbs(T)) - (slope * x_q + intercept)
        else:
            distance = np.asarray(phase.molar_gibbs(T, x), dtype=float) - (slope * x + intercept)
            k = int(np.argmin(distance))
            lo, hi = x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]
            refined = minimize_scalar(
                lambda xi: float(phase.molar_gibbs(T, xi)) - (slope * xi + intercept),
                bounds=(lo, hi), method="bounded", options={"xatol": 1e-12},
            )
            x_q, depth = (refined.x, refined.fun) if refined.fun < distance[k] else (x[k], distance[k])
        if depth < -STABILITY_TOLERANCE and (best is None or depth < best[2]):
            best = (name, float(x_q), float(depth))
    return best


def _stable_tie_lines(system: dict, T: float, tie: PhaseField, depth: int = 0) -> list[PhaseField]:
    """Refine a hull tie line to the exact common tangent, splitting it if
    another phase lies below it (a field too narrow for the sampled hull)."""
    left, right = (system[name] for name in tie.base_phases)
    x_min, x_max = common_tangent(left, right, T, tie.x_min, tie.x_max)
    tie = PhaseField(tie.phases, x_min, x_max)

    below = _deepest_phase_below(system, T, tie) if depth < MAX_TIE_SPLITS else None
    if below is None:
        return [tie]
    name, x_q, _ = below
    return (
        _stable_tie_lines(system, T, PhaseField((tie.phases[0], name), tie.x_min, x_q), depth + 1)
        + [PhaseField((name,), x_q, x_q)]
        + _stable_tie_lines(system, T, PhaseField((name, tie.phases[1]), x_q, tie.x_max), depth + 1)
    )


def _drop_unstable_middle_phases(system: dict, T: float, fields: list[PhaseField]) -> list[PhaseField]:
    """Merge tie lines P1+P2, P2+P3 into P1+P3 where the refined tangents
    show P2 is not actually stable between them (the tangents cross, or
    their slopes decrease), which the sampled hull cannot always tell."""
    changed = True
    while changed:
        changed = False
        ties = [k for k, field in enumerate(fields) if field.is_two_phase]
        for a, b in zip(ties, ties[1:]):
            left, right = fields[a], fields[b]
            if any(f.is_two_phase for f in fields[a + 1:b]):
                continue
            slope_left, _ = tangent_line(*(system[n] for n in left.base_phases), T, left.x_min, left.x_max)
            slope_right, _ = tangent_line(*(system[n] for n in right.base_phases), T, right.x_min, right.x_max)
            if left.x_max <= right.x_min + VERTEX_TOLERANCE and slope_left <= slope_right:
                continue
            merged = PhaseField((left.phases[0], right.phases[1]), left.x_min, right.x_max)
            fields = fields[:a] + _stable_tie_lines(system, T, merged) + fields[b + 1:]
            changed = True
            break
    return fields


def phase_assemblage(system: dict, T: float, n_points: int = 500) -> PhaseAssemblage:
    """Equilibrium fields across the whole composition range at T.

    The sampled convex hull identifies which fields exist; each tie line is
    then refined to the exact common tangent and checked against every other
    phase, so the result is independent of any composition grid.
    """
    hull_x, _, hull_labels = _hull_for_temperature(system, T, n_points=n_points)

    # Runs of consecutive hull vertices with the same label are single-phase
    # stretches; the segment between two runs is a tie line.
    runs: list[tuple[str, float, float]] = []
    for x, label in zip(hull_x, hull_labels):
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1], float(x))
        else:
            runs.append((label, float(x), float(x)))

    fields: list[PhaseField] = []
    for k, (label, x_lo, x_hi) in enumerate(runs):
        fields.append(PhaseField((label,), x_lo, x_hi))
        if k + 1 < len(runs):
            next_label, next_lo, _ = runs[k + 1]
            for item in _stable_tie_lines(system, T, PhaseField((label, next_label), x_hi, next_lo)):
                fields.append(item)

    fields = _drop_unstable_middle_phases(system, T, fields)

    # Single-phase stretches span exactly between their neighbouring tie lines.
    for k, field in enumerate(fields):
        if field.is_two_phase:
            continue
        x_lo = fields[k - 1].x_max if k > 0 else 0.0
        x_hi = fields[k + 1].x_min if k + 1 < len(fields) else 1.0
        if x_lo > x_hi:
            x_lo = x_hi = (x_lo + x_hi) / 2.0
        field.x_min, field.x_max = x_lo, x_hi

    return PhaseAssemblage(T=T, fields=_label_composition_sets(fields))


def _label_composition_sets(fields: list[PhaseField]) -> list[PhaseField]:
    """Final composition set labels, after tie lines were split or merged.

    Fields alternate single-phase stretch / tie line. A phase with a tie line
    to itself has a miscibility gap: its stretches are numbered PHASE#1,
    PHASE#2, ... left to right, and every tie line takes the labels of the
    stretches on either side of it.
    """
    gapped = {f.base_phases[0] for f in fields if f.is_two_phase and f.base_phases[0] == f.base_phases[1]}
    counts: dict[str, int] = {}
    singles: list[str] = []
    for f in fields:
        if f.is_two_phase:
            continue
        base = f.base_phases[0]
        if base in gapped:
            counts[base] = counts.get(base, 0) + 1
            singles.append(composition_set_label(base, counts[base]))
        else:
            singles.append(base)

    labelled: list[PhaseField] = []
    single_index = 0
    for f in fields:
        if f.is_two_phase:
            labelled.append(PhaseField((singles[single_index - 1], singles[single_index]), f.x_min, f.x_max))
        else:
            labelled.append(PhaseField((singles[single_index],), f.x_min, f.x_max))
            single_index += 1
    return labelled


def equilibrium_from_assemblage(system: dict, assemblage: PhaseAssemblage, x_overall: float) -> EquilibriumResult:
    """Read off the equilibrium at x_overall from a PhaseAssemblage."""
    T = assemblage.T
    for field in assemblage.fields:
        if field.is_two_phase or not field.x_min - VERTEX_TOLERANCE <= x_overall <= field.x_max + VERTEX_TOLERANCE:
            continue
        label = field.phases[0]
        phase = system[base_phase_name(label)]
        if isinstance(phase, StoichiometricPhase):
            composition, total_gibbs = phase.composition, float(phase.molar_gibbs(T))
        else:
            composition, total_gibbs = x_overall, float(phase.molar_gibbs(T, x_overall))
        return EquilibriumResult(
            T=T, x_overall=x_overall, stable_phases=[label],
            phase_compositions={label: composition}, phase_fractions={label: 1.0},
            total_gibbs=total_gibbs,
        )

    for field in assemblage.tie_lines:
        if not field.x_min < x_overall < field.x_max:
            continue
        label_left, label_right = field.phases
        left, right = (system[name] for name in field.base_phases)
        fraction_right = (x_overall - field.x_min) / (field.x_max - field.x_min)
        fraction_left = 1.0 - fraction_right
        total_gibbs = (
            fraction_left * gibbs(left, T, field.x_min) + fraction_right * gibbs(right, T, field.x_max)
        )
        return EquilibriumResult(
            T=T, x_overall=x_overall, stable_phases=[label_left, label_right],
            phase_compositions={label_left: field.x_min, label_right: field.x_max},
            phase_fractions={label_left: fraction_left, label_right: fraction_right},
            total_gibbs=total_gibbs,
        )

    raise ValueError(f"x_overall={x_overall} is not covered by the phase assemblage at T={T}")


def compute_equilibrium(
    system: dict, T: float, x_overall: float, n_points: int = 500
) -> EquilibriumResult:
    """Compute the stable phase(s), compositions, and fractions at (T, x_overall).

    Builds the global lower convex hull over every phase's Gibbs energy curve
    (the common-tangent construction) to find which fields exist, refines
    every tie line to the exact common tangent (see phase_assemblage), then
    reads off the equilibrium state for x_overall: a single stable phase, or
    two phases joined by a tie line with phase fractions from the lever rule.
    """
    if x_overall < -COMPOSITION_TOLERANCE or x_overall > 1.0 + COMPOSITION_TOLERANCE:
        raise ValueError(f"x_overall={x_overall} is outside the valid composition range [0, 1]")
    x_overall = min(max(x_overall, 0.0), 1.0)
    return equilibrium_from_assemblage(system, phase_assemblage(system, T, n_points=n_points), x_overall)
