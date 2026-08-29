from dataclasses import dataclass

from phase_diagram_explorer.diagram import PhaseDiagram
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium

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


def _row_two_phase_fields(phase_labels_row, x_grid) -> dict[frozenset, list[float]]:
    """Map each two-phase field (frozenset of phase names) in a row to its
    (x_min, x_max) composition extent."""
    fields: dict[frozenset, list[float]] = {}
    for j, label in enumerate(phase_labels_row):
        if len(label) == 2:
            key = frozenset(label)
            x = x_grid[j]
            if key not in fields:
                fields[key] = [x, x]
            else:
                fields[key][0] = min(fields[key][0], x)
                fields[key][1] = max(fields[key][1], x)
    return fields


def _default_liquid_phases(system: dict) -> set[str]:
    return {name for name in system if "liquid" in name.lower()}


def _shared_phase_pairs(fields: dict[frozenset, list[float]]):
    """Yield (p2, left_key, right_key) for pairs of two-phase fields that
    share a common phase p2, ordered left-to-right by composition."""
    keys = list(fields.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            shared = keys[a] & keys[b]
            if len(shared) != 1:
                continue
            p2 = next(iter(shared))
            key_a, key_b = keys[a], keys[b]
            if sum(fields[key_a]) <= sum(fields[key_b]):
                yield p2, key_a, key_b
            else:
                yield p2, key_b, key_a


def _refine_composition(
    system: dict, T: float, x_range: list[float], n_points: int
) -> dict[str, float]:
    x_mid = (x_range[0] + x_range[1]) / 2.0
    result = compute_equilibrium(system, T, x_mid, n_points=n_points)
    return result.phase_compositions


def _classify(phases: tuple[str, str, str], is_eutectic_type: bool, liquid_phases: set[str]) -> str:
    involves_liquid = any(phase in liquid_phases for phase in phases)
    if is_eutectic_type:
        return EUTECTIC if involves_liquid else EUTECTOID
    return PERITECTIC if involves_liquid else PERITECTOID


def _find_reactions(
    fields_with_p2: dict[frozenset, list[float]],
    fields_without_p2: dict[frozenset, list[float]],
    T_with_p2: float,
    T_without_p2: float,
    system: dict,
    liquid_phases: set[str],
    n_points: int,
    is_eutectic_type: bool,
) -> list[InvariantReaction]:
    """Find reactions where phase p2, flanked by two two-phase fields in the
    `fields_with_p2` row, is replaced by their combined field in the other row.
    """
    reactions = []
    for p2, left_key, right_key in _shared_phase_pairs(fields_with_p2):
        p1 = next(iter(left_key - {p2}))
        p3 = next(iter(right_key - {p2}))
        combined_key = frozenset({p1, p3})
        if combined_key not in fields_without_p2 or combined_key in fields_with_p2:
            continue

        left_composition = _refine_composition(system, T_with_p2, fields_with_p2[left_key], n_points)
        right_composition = _refine_composition(system, T_with_p2, fields_with_p2[right_key], n_points)
        x1 = left_composition[p1]
        x3 = right_composition[p3]
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
    fields are present. A phase P2 flanked by two-phase fields {P1,P2} and
    {P2,P3} in one row, replaced by the combined field {P1,P3} in the
    adjacent row, marks an invariant reaction: if P2 is stable in the hotter
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
        fields_lower = _row_two_phase_fields(diagram.phase_labels[i], x_grid)
        fields_upper = _row_two_phase_fields(diagram.phase_labels[i + 1], x_grid)

        reactions.extend(
            _find_reactions(
                fields_upper, fields_lower, T_upper, T_lower,
                system, liquid_phases, n_points, is_eutectic_type=True,
            )
        )
        reactions.extend(
            _find_reactions(
                fields_lower, fields_upper, T_lower, T_upper,
                system, liquid_phases, n_points, is_eutectic_type=False,
            )
        )

    return reactions
