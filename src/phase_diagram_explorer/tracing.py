"""Phase boundary tracing.

Boundaries are traced from the equilibrium phase assemblage (exact tie-line
endpoint compositions) at a set of temperature levels, never from labelled
grid cells, so a two-phase field narrower than any composition grid step is
still traced. Levels are adaptive: changes of field structure are bracketed
by bisection, invariant temperatures get levels just above and below, and
intervals where a boundary moves far or bends sharply are subdivided.

Each traced field is a strip of (T, x_min, x_max) samples, so phase regions
are polygons whose edges are the traced boundary lines. Neighbouring fields
share the same edge samples, so the polygons tile the T-x area without
overlapping.
"""
from dataclasses import dataclass, field as dataclass_field

import numpy as np
from matplotlib.path import Path as PolygonPath

from phase_diagram_explorer.equilibrium.equilibrium import PhaseAssemblage
from phase_diagram_explorer.invariants import (
    InvariantReaction,
    assemblage_changes,
    assemblage_function,
    find_invariants,
)

DEFAULT_LEVELS = 80
# Levels placed this far above and below each invariant temperature (K).
INVARIANT_OFFSET_K = 1e-4
# Subdivide an interval if a boundary moves more than this in x across it...
MAX_BOUNDARY_STEP = 0.01
# ...or bends more than this (deviation from the straight line through its
# neighbours), down to intervals of MIN_STEP_K.
MAX_BOUNDARY_BEND = 0.001
MIN_STEP_K = 0.05
MAX_REFINEMENT_ROUNDS = 8
OVERLAP_TOLERANCE = 1e-9


@dataclass
class TracedField:
    """A phase field traced over a contiguous temperature range: at each
    level T[k] it spans compositions x_min[k]..x_max[k]."""

    phases: tuple[str, ...]  # base phase names, lower-x phase first
    T: list[float] = dataclass_field(default_factory=list)
    x_min: list[float] = dataclass_field(default_factory=list)
    x_max: list[float] = dataclass_field(default_factory=list)

    @property
    def is_two_phase(self) -> bool:
        return len(self.phases) == 2

    @property
    def label(self) -> str:
        """Textbook label: plain phase names, liquid first ("LIQUID + FCC_AG")."""
        if not self.is_two_phase:
            return self.phases[0]
        ordered = sorted(self.phases, key=lambda name: "liquid" not in name.lower())
        return " + ".join(ordered)

    def polygon(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed outline (x, T): the x_min edge upwards, the x_max edge down."""
        x = np.concatenate([self.x_min, self.x_max[::-1]])
        T = np.concatenate([self.T, self.T[::-1]])
        return x, T

    @property
    def area(self) -> float:
        x, T = self.polygon()
        return float(0.5 * abs(np.dot(x, np.roll(T, 1)) - np.dot(T, np.roll(x, 1))))

    def label_position(self) -> tuple[float, float]:
        """Area centroid of the polygon, or the middle of its widest level if
        the centroid falls outside a non-convex outline."""
        x, T = self.polygon()
        cross = x * np.roll(T, -1) - np.roll(x, -1) * T
        signed_area = cross.sum() / 2.0
        if abs(signed_area) > 1e-12:
            cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * signed_area)
            cT = ((T + np.roll(T, -1)) * cross).sum() / (6.0 * signed_area)
            if PolygonPath(np.column_stack([x, T])).contains_point((cx, cT)):
                return float(cx), float(cT)
        widths = np.asarray(self.x_max) - np.asarray(self.x_min)
        k = int(np.argmax(widths))
        return float((self.x_min[k] + self.x_max[k]) / 2.0), float(self.T[k])


@dataclass
class BoundaryLine:
    """A traced phase boundary: composition x[k] at temperature T[k]."""

    name: str
    T: np.ndarray
    x: np.ndarray


@dataclass
class TracedDiagram:
    T_range: tuple[float, float]
    fields: list[TracedField]
    invariants: list[InvariantReaction]
    levels: np.ndarray

    @property
    def boundaries(self) -> list[BoundaryLine]:
        """Both edges of every two-phase field, plus each compound line."""
        lines = []
        for traced in self.fields:
            T = np.asarray(traced.T)
            if traced.is_two_phase:
                left, right = traced.phases
                lines.append(BoundaryLine(f"{left} / {traced.label}", T, np.asarray(traced.x_min)))
                lines.append(BoundaryLine(f"{traced.label} / {right}", T, np.asarray(traced.x_max)))
            elif np.allclose(traced.x_min, traced.x_max) and len(T) > 1:
                lines.append(BoundaryLine(traced.phases[0], T, np.asarray(traced.x_min)))
        return lines

    @property
    def regions(self) -> list[TracedField]:
        """Fields with non-zero area (excludes line compounds)."""
        return [traced for traced in self.fields if traced.area > 0.0]


def _match_fields(previous: PhaseAssemblage, current: PhaseAssemblage) -> dict[int, int]:
    """Map current field index -> previous field index for fields that
    continue from one level to the next."""
    if previous.key == current.key:
        return {i: i for i in range(len(current.fields))}

    def overlapping(field_a, fields_b):
        return [
            j for j, field_b in enumerate(fields_b)
            if field_b.base_phases == field_a.base_phases
            and field_b.x_min <= field_a.x_max + OVERLAP_TOLERANCE
            and field_b.x_max >= field_a.x_min - OVERLAP_TOLERANCE
        ]

    matches = {}
    for i, field in enumerate(current.fields):
        candidates = overlapping(field, previous.fields)
        if len(candidates) == 1 and len(overlapping(previous.fields[candidates[0]], current.fields)) == 1:
            matches[i] = candidates[0]
    return matches


def _track(assemblages: list[PhaseAssemblage]) -> list[TracedField]:
    traced: list[TracedField] = []
    open_fields: list[TracedField] = []
    for k, assemblage in enumerate(assemblages):
        matches = _match_fields(assemblages[k - 1], assemblage) if k > 0 else {}
        current: list[TracedField] = []
        for i, field in enumerate(assemblage.fields):
            if i in matches:
                strip = open_fields[matches[i]]
            else:
                strip = TracedField(field.base_phases)
                traced.append(strip)
            strip.T.append(assemblage.T)
            strip.x_min.append(field.x_min)
            strip.x_max.append(field.x_max)
            current.append(strip)
        open_fields = current
    return traced


def _intervals_to_refine(assemblages: list[PhaseAssemblage]) -> set[float]:
    """Midpoints of intervals where a boundary moves or bends too much."""
    new_levels: set[float] = set()
    for k in range(len(assemblages) - 1):
        lower, upper = assemblages[k], assemblages[k + 1]
        if upper.T - lower.T <= MIN_STEP_K or lower.key != upper.key:
            continue
        for a, b in zip(lower.fields, upper.fields):
            if max(abs(a.x_min - b.x_min), abs(a.x_max - b.x_max)) > MAX_BOUNDARY_STEP:
                new_levels.add((lower.T + upper.T) / 2.0)
                break
        if k == 0 or assemblages[k - 1].key != lower.key:
            continue
        below = assemblages[k - 1]
        weight = (lower.T - below.T) / (upper.T - below.T)
        for c, a, b in zip(below.fields, lower.fields, upper.fields):
            bend = max(
                abs(a.x_min - (c.x_min + weight * (b.x_min - c.x_min))),
                abs(a.x_max - (c.x_max + weight * (b.x_max - c.x_max))),
            )
            if bend > MAX_BOUNDARY_BEND:
                if lower.T - below.T > MIN_STEP_K:
                    new_levels.add((below.T + lower.T) / 2.0)
                new_levels.add((lower.T + upper.T) / 2.0)
                break
    return new_levels


def trace_diagram(
    system: dict,
    T_range: tuple[float, float],
    invariants: list[InvariantReaction] | None = None,
    n_levels: int = DEFAULT_LEVELS,
    n_points: int = 500,
) -> TracedDiagram:
    """Trace every phase field of a binary system over T_range.

    `invariants` (e.g. found over the system's full range) are reused if
    given; otherwise they are found over T_range. Only those inside T_range
    are kept on the result.
    """
    T_min, T_max = float(T_range[0]), float(T_range[1])
    assemble = assemblage_function(system, n_points)
    if invariants is None:
        invariants = find_invariants(system, np.linspace(T_min, T_max, n_levels), n_points=n_points, assemble=assemble)
    invariants = [r for r in invariants if T_min <= r.temperature <= T_max]

    levels = set(np.linspace(T_min, T_max, max(n_levels, 2)).tolist())
    for reaction in invariants:
        for T in (reaction.temperature - INVARIANT_OFFSET_K, reaction.temperature + INVARIANT_OFFSET_K):
            if T_min <= T <= T_max:
                levels.add(T)

    for _ in range(MAX_REFINEMENT_ROUNDS):
        for lower, upper in assemblage_changes(levels, assemble):
            levels.update((lower.T, upper.T))
        assemblages = [assemble(T) for T in sorted(levels)]
        new_levels = _intervals_to_refine(assemblages) - levels
        if not new_levels:
            break
        levels.update(new_levels)

    assemblages = [assemble(T) for T in sorted(levels)]
    return TracedDiagram(
        T_range=(T_min, T_max),
        fields=_track(assemblages),
        invariants=invariants,
        levels=np.array(sorted(levels)),
    )
