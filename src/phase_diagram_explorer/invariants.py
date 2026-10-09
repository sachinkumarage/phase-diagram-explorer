from dataclasses import dataclass

import numpy as np

from phase_diagram_explorer.diagram import PhaseDiagram, compute_diagram
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium

EUTECTIC = "eutectic"
PERITECTIC = "peritectic"
EUTECTOID = "eutectoid"
PERITECTOID = "peritectoid"


@dataclass
class InvariantReaction:
    temperature: float
    type: str
    phases: tuple[str, str, str]
    composition: dict[str, float]


@dataclass
class _TwoPhaseSegment:
    """A contiguous run of grid points in one temperature row where the same
    two phases (or composition sets) coexist, `left` being the lower-x one."""

    left: str
    right: str
    x_min: float
    x_max: float
    last_index: int

    @property
    def base_phases(self) -> tuple[str, str]:
        return base_phase_name(self.left), base_phase_name(self.right)

    def overlaps(self, x_min: float, x_max: float) -> bool:
        return self.x_min <= x_max and self.x_max >= x_min


def _row_two_phase_segments(phase_labels_row, x_grid) -> list[_TwoPhaseSegment]:
    """Contiguous two-phase segments of one row, ordered by composition.

    Labels of a two-phase grid point are ordered by composition (from the
    hull), so each segment knows which phase lies on which side. Two separate
    fields of the same phase pair (e.g. LIQUID+FCC on both sides of a
    compound) stay separate segments.
    """
    segments: list[_TwoPhaseSegment] = []
    for j, label in enumerate(phase_labels_row):
        if len(label) != 2:
            continue
        x = float(x_grid[j])
        last = segments[-1] if segments else None
        if last is not None and last.last_index == j - 1 and (last.left, last.right) == tuple(label):
            last.x_max = x
            last.last_index = j
        else:
            segments.append(_TwoPhaseSegment(label[0], label[1], x, x, j))
    return segments


def _default_liquid_phases(system: dict) -> set[str]:
    return {name for name in system if "liquid" in name.lower()}


def _refine_composition(
    system: dict, T: float, segment: _TwoPhaseSegment, n_points: int
) -> dict[str, float]:
    x_mid = (segment.x_min + segment.x_max) / 2.0
    result = compute_equilibrium(system, T, x_mid, n_points=n_points)
    return result.phase_compositions


def _classify(phases: tuple[str, str, str], is_eutectic_type: bool, liquid_phases: set[str]) -> str:
    involves_liquid = any(base_phase_name(phase) in liquid_phases for phase in phases)
    if is_eutectic_type:
        return EUTECTIC if involves_liquid else EUTECTOID
    return PERITECTIC if involves_liquid else PERITECTOID


def _find_reactions(
    segments_with_p2: list[_TwoPhaseSegment],
    segments_without_p2: list[_TwoPhaseSegment],
    T_with_p2: float,
    T_without_p2: float,
    system: dict,
    liquid_phases: set[str],
    n_points: int,
    is_eutectic_type: bool,
) -> list[InvariantReaction]:
    """Find reactions where phase p2, flanked by adjacent two-phase segments
    P1+P2 and P2+P3 in the `segments_with_p2` row, is replaced by a P1+P3
    segment spanning p2's composition in the other row.

    Phases are compared by base name, so P1 and P3 may be two composition
    sets of one phase (e.g. LIQUID -> FCC#1 + FCC#2). The reaction is labelled
    with the composition set names from the row where P1 and P3 coexist.
    """
    reactions = []
    for left, right in zip(segments_with_p2, segments_with_p2[1:]):
        if left.right != right.left:
            continue
        p2 = left.right
        outer = (base_phase_name(left.left), base_phase_name(right.right))
        # The P1+P3 field must span the composition where P2 is stable in
        # this row, between the two flanking fields.
        window = (left.x_max, right.x_min)

        combined = [
            s for s in segments_without_p2 if s.base_phases == outer and s.overlaps(*window)
        ]
        if not combined or any(
            s.base_phases == outer and s.overlaps(*window) for s in segments_with_p2
        ):
            continue
        p1, p3 = combined[0].left, combined[0].right

        left_composition = _refine_composition(system, T_with_p2, left, n_points)
        right_composition = _refine_composition(system, T_with_p2, right, n_points)
        x1 = left_composition[left.left]
        x3 = right_composition[right.right]
        x2 = (left_composition[p2] + right_composition[p2]) / 2.0

        reactions.append(
            InvariantReaction(
                temperature=(T_with_p2 + T_without_p2) / 2.0,
                type=_classify((p1, p2, p3), is_eutectic_type, liquid_phases),
                phases=(p1, p2, p3),
                composition={p1: x1, p2: x2, p3: x3},
            )
        )
    return reactions


def detect_invariants(
    diagram: PhaseDiagram,
    system: dict,
    liquid_phases: set[str] | None = None,
    n_points: int = 500,
) -> list[InvariantReaction]:
    """Detect three-phase invariant reactions from a computed PhaseDiagram.

    Scans consecutive temperature rows for a change in which two-phase
    fields are present. A phase P2 flanked by adjacent two-phase fields P1+P2
    and P2+P3 in one row, replaced by a P1+P3 field over the same composition
    window in the adjacent row, marks an invariant reaction. P1 and P3 may be
    composition sets of one phase (PHASE#1, PHASE#2): if P2 is stable in the hotter
    row and absent from that composition window in the colder row, P2
    decomposes into P1+P3 on cooling (a eutectic-type reaction). If P2
    instead appears only in the colder row, P1+P3 combine into P2 on cooling
    (a peritectic-type reaction). Whether P2 is a liquid phase (from
    `liquid_phases`, defaulting to phase names containing "liquid") decides
    the eutectic/eutectoid or peritectic/peritectoid label. The reaction
    temperature is the midpoint of the two straddling rows, and phase
    compositions are refined via compute_equilibrium rather than read
    directly off the (coarser) diagram grid.
    """
    if liquid_phases is None:
        liquid_phases = _default_liquid_phases(system)

    reactions: list[InvariantReaction] = []
    T_grid = diagram.T_grid
    x_grid = diagram.x_grid

    for i in range(len(T_grid) - 1):
        T_lower, T_upper = T_grid[i], T_grid[i + 1]
        segments_lower = _row_two_phase_segments(diagram.phase_labels[i], x_grid)
        segments_upper = _row_two_phase_segments(diagram.phase_labels[i + 1], x_grid)

        reactions.extend(
            _find_reactions(
                segments_upper, segments_lower, T_upper, T_lower,
                system, liquid_phases, n_points, is_eutectic_type=True,
            )
        )
        reactions.extend(
            _find_reactions(
                segments_lower, segments_upper, T_lower, T_upper,
                system, liquid_phases, n_points, is_eutectic_type=False,
            )
        )

    return reactions


INVARIANT_T_STEP_K = 2.0
INVARIANT_N_X = 201


def detect_invariants_over_range(
    system: dict,
    T_range: tuple[float, float],
    T_step: float = INVARIANT_T_STEP_K,
    n_x: int = INVARIANT_N_X,
    n_points: int = 500,
    liquid_phases: set[str] | None = None,
) -> list[InvariantReaction]:
    """Detect invariant reactions over a full temperature range on a dedicated
    grid with spacing of at most `T_step`, so the result does not depend on
    the grid used to display a diagram. Reaction temperatures are accurate to
    about T_step / 2.
    """
    T_min, T_max = T_range
    n_T = max(int(np.ceil((T_max - T_min) / T_step)) + 1, 2)
    diagram = compute_diagram(system, T_range=(T_min, T_max), n_T=n_T, n_x=n_x, n_points=n_points)
    return detect_invariants(diagram, system, liquid_phases=liquid_phases, n_points=n_points)
