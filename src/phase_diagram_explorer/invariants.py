"""Three-phase invariant reactions, located by root-finding.

Candidates are found where the equilibrium phase assemblage (the ordered
sequence of single- and two-phase fields across composition) changes between
two temperatures. Each candidate is then refined by root-finding on the
three-phase condition: the temperature at which the reacting phase P2 just
touches the common tangent of P1 and P3. No composition grid is involved, so
results do not depend on the resolution of any displayed diagram.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from phase_diagram_explorer.diagram import PhaseDiagram
from phase_diagram_explorer.equilibrium.equilibrium import (
    PhaseAssemblage,
    PhaseField,
    base_phase_name,
    phase_assemblage,
)
from phase_diagram_explorer.equilibrium.tangent import common_tangent, tangent_line
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

EUTECTIC = "eutectic"
PERITECTIC = "peritectic"
EUTECTOID = "eutectoid"
PERITECTOID = "peritectoid"

INVARIANT_T_STEP_K = 5.0
# Assemblage changes are bracketed to this width before root-finding.
EVENT_BRACKET_K = 1e-3
ROOT_TOLERANCE_K = 1e-6
# If a bracket does not straddle the root (the sampled hull can see a field
# appear slightly early), it is widened by these amounts in turn.
BRACKET_EXPANSIONS_K = (0.01, 0.1, 1.0, 5.0)
# Reactions between the same phases closer than this are one reaction.
DUPLICATE_T_K = 0.5
OVERLAP_TOLERANCE = 1e-9
REACTING_PHASE_SAMPLES = 401


@dataclass
class InvariantReaction:
    temperature: float
    type: str
    phases: tuple[str, str, str]
    composition: dict[str, float]


AssemblageFunction = Callable[[float], PhaseAssemblage]


def assemblage_function(system: dict, n_points: int = 500) -> AssemblageFunction:
    """Memoised phase_assemblage(system, T) for one system."""
    cache: dict[float, PhaseAssemblage] = {}

    def assemble(T: float) -> PhaseAssemblage:
        T = float(T)
        if T not in cache:
            cache[T] = phase_assemblage(system, T, n_points=n_points)
        return cache[T]

    return assemble


def assemblage_changes(
    levels, assemble: AssemblageFunction, bracket: float = EVENT_BRACKET_K
) -> list[tuple[PhaseAssemblage, PhaseAssemblage]]:
    """(lower, upper) assemblage pairs no more than `bracket` apart in T,
    bracketing every change of field structure between consecutive levels.

    Intervals whose ends differ are bisected, so several changes inside one
    interval are separated.
    """
    changes: list[tuple[PhaseAssemblage, PhaseAssemblage]] = []

    def bisect(lower: PhaseAssemblage, upper: PhaseAssemblage) -> None:
        if lower.key == upper.key:
            return
        if upper.T - lower.T <= bracket:
            changes.append((lower, upper))
            return
        middle = assemble((lower.T + upper.T) / 2.0)
        bisect(lower, middle)
        bisect(middle, upper)

    levels = sorted(float(T) for T in levels)
    for T_lower, T_upper in zip(levels, levels[1:]):
        bisect(assemble(T_lower), assemble(T_upper))
    return changes


def _default_liquid_phases(system: dict) -> set[str]:
    return {name for name in system if "liquid" in name.lower()}


def _classify(phases: tuple[str, str, str], is_eutectic_type: bool, liquid_phases: set[str]) -> str:
    involves_liquid = any(base_phase_name(phase) in liquid_phases for phase in phases)
    if is_eutectic_type:
        return EUTECTIC if involves_liquid else EUTECTOID
    return PERITECTIC if involves_liquid else PERITECTOID


def _overlaps(field: PhaseField, x_lo: float, x_hi: float) -> bool:
    return field.x_min <= x_hi + OVERLAP_TOLERANCE and field.x_max >= x_lo - OVERLAP_TOLERANCE


def _reaction_candidates(with_p2: PhaseAssemblage, without_p2: PhaseAssemblage):
    """(p2 label, P1+P3 tie line) for each phase P2 that is flanked by tie
    lines P1+P2 and P2+P3 in `with_p2` and replaced by a P1+P3 tie line
    spanning its composition in `without_p2`. P1 and P3 are compared by base
    phase name, so they may be composition sets of one phase."""
    ties = with_p2.tie_lines
    for left, right in zip(ties, ties[1:]):
        if left.phases[1] != right.phases[0]:
            continue
        outer = (left.base_phases[0], right.base_phases[1])
        window = (left.x_max, right.x_min)
        if any(t.base_phases == outer and _overlaps(t, *window) for t in ties):
            continue
        combined = [t for t in without_p2.tie_lines if t.base_phases == outer and _overlaps(t, *window)]
        if combined:
            yield left.phases[1], combined[0]


def _three_phase_state(system: dict, T: float, p2: str, tie13: PhaseField):
    """(distance, x1, x2, x3): how far P2 lies below (negative) or above
    (positive) the P1+P3 common tangent at T, and the compositions."""
    phase1, phase3 = (system[name] for name in tie13.base_phases)
    phase2 = system[base_phase_name(p2)]
    x1, x3 = common_tangent(phase1, phase3, T, tie13.x_min, tie13.x_max)
    slope, intercept = tangent_line(phase1, phase3, T, x1, x3)

    if isinstance(phase2, StoichiometricPhase):
        x2 = phase2.composition
        return float(phase2.molar_gibbs(T)) - (slope * x2 + intercept), x1, x2, x3

    def distance(x):
        return np.asarray(phase2.molar_gibbs(T, x), dtype=float) - (slope * np.asarray(x) + intercept)

    x = np.linspace(x1, x3, REACTING_PHASE_SAMPLES)[1:-1]
    k = int(np.argmin(distance(x)))
    refined = minimize_scalar(
        lambda xi: float(distance(xi)), bounds=(x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]),
        method="bounded", options={"xatol": 1e-12},
    )
    return float(refined.fun), x1, float(refined.x), x3


def _solve_reaction(system: dict, lower: PhaseAssemblage, upper: PhaseAssemblage, p2: str, tie13: PhaseField):
    """Temperature and compositions where P2 touches the P1+P3 tangent."""
    def g(T: float) -> float:
        return _three_phase_state(system, T, p2, tie13)[0]

    T_root = None
    for expansion in (0.0, *BRACKET_EXPANSIONS_K):
        T_a, T_b = lower.T - expansion, upper.T + expansion
        g_a, g_b = g(T_a), g(T_b)
        if g_a == 0.0:
            T_root = T_a
        elif g_b == 0.0:
            T_root = T_b
        elif np.sign(g_a) != np.sign(g_b):
            T_root = brentq(g, T_a, T_b, xtol=ROOT_TOLERANCE_K)
        if T_root is not None:
            break
    if T_root is None:
        T_root = (lower.T + upper.T) / 2.0

    _, x1, x2, x3 = _three_phase_state(system, T_root, p2, tie13)
    return float(T_root), x1, x2, x3


def _deduplicate(reactions: list[InvariantReaction]) -> list[InvariantReaction]:
    unique: list[InvariantReaction] = []
    for reaction in sorted(reactions, key=lambda r: r.temperature):
        bases = sorted(base_phase_name(phase) for phase in reaction.phases)
        if any(
            sorted(base_phase_name(phase) for phase in kept.phases) == bases
            and abs(kept.temperature - reaction.temperature) < DUPLICATE_T_K
            for kept in unique
        ):
            continue
        unique.append(reaction)
    return unique


def find_invariants(
    system: dict,
    levels,
    liquid_phases: set[str] | None = None,
    n_points: int = 500,
    assemble: AssemblageFunction | None = None,
) -> list[InvariantReaction]:
    """Invariant reactions between the given temperature levels.

    Levels only need to be fine enough that no field appears and disappears
    again between two of them; the reported temperatures and compositions
    come from root-finding and do not depend on the levels.
    """
    if liquid_phases is None:
        liquid_phases = _default_liquid_phases(system)
    if assemble is None:
        assemble = assemblage_function(system, n_points)

    reactions: list[InvariantReaction] = []
    for lower, upper in assemblage_changes(levels, assemble):
        for with_p2, without_p2, is_eutectic_type in ((upper, lower, True), (lower, upper, False)):
            for p2, tie13 in _reaction_candidates(with_p2, without_p2):
                T, x1, x2, x3 = _solve_reaction(system, lower, upper, p2, tie13)
                p1, p3 = tie13.phases
                reactions.append(
                    InvariantReaction(
                        temperature=T,
                        type=_classify((p1, p2, p3), is_eutectic_type, liquid_phases),
                        phases=(p1, p2, p3),
                        composition={p1: x1, p2: x2, p3: x3},
                    )
                )
    return _deduplicate(reactions)


def detect_invariants(
    diagram: PhaseDiagram,
    system: dict,
    liquid_phases: set[str] | None = None,
    n_points: int = 500,
) -> list[InvariantReaction]:
    """Detect three-phase invariant reactions within a computed PhaseDiagram's
    temperature range.

    Only the diagram's temperature levels are used, to look for changes in
    the phase assemblage; its composition grid and cell labels are not. A
    phase P2 flanked by tie lines P1+P2 and P2+P3 at one temperature and
    replaced by a P1+P3 tie line at another marks a reaction: if P2 is
    present at the higher temperature, P2 decomposes into P1+P3 on cooling
    (eutectic-type), otherwise P1+P3 form P2 on cooling (peritectic-type).
    Whether a liquid phase takes part (from `liquid_phases`, defaulting to
    phase names containing "liquid") decides eutectic/eutectoid and
    peritectic/peritectoid. The temperature is found by root-finding to
    within ROOT_TOLERANCE_K, and all three phase compositions are the exact
    common-tangent compositions at that temperature.
    """
    return find_invariants(system, diagram.T_grid, liquid_phases=liquid_phases, n_points=n_points)


def scan_levels(T_range: tuple[float, float], T_step: float) -> np.ndarray:
    T_min, T_max = T_range
    n_T = max(int(np.ceil((T_max - T_min) / T_step)) + 1, 2)
    return np.linspace(T_min, T_max, n_T)


def detect_invariants_over_range(
    system: dict,
    T_range: tuple[float, float],
    T_step: float = INVARIANT_T_STEP_K,
    n_points: int = 500,
    liquid_phases: set[str] | None = None,
) -> list[InvariantReaction]:
    """Detect invariant reactions over a full temperature range, scanning
    levels at most `T_step` apart. Results do not depend on any displayed
    diagram's range or grid."""
    return find_invariants(system, scan_levels(T_range, T_step), liquid_phases=liquid_phases, n_points=n_points)
