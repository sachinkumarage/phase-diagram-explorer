"""Point equilibrium read from a traced diagram, without the database.

A TracedDiagram stores, for every phase field, its composition interval
[x_min(T), x_max(T)] at a set of temperature levels. The equilibrium at
(T, x) is the field containing x at T, with its edges interpolated linearly
between the two neighbouring levels: a single phase, or two phases with the
tie-line ends as compositions and fractions from the lever rule. Labels
follow compute_equilibrium: a phase with a miscibility gap at T is numbered
PHASE#1, PHASE#2, ... from left to right.

The accuracy is that of the stored levels (see precomputed.TABLE_STEP_K);
total Gibbs energies are not available (NaN).
"""
import math

import numpy as np

from phase_diagram_explorer.equilibrium.equilibrium import EquilibriumResult, composition_set_label
from phase_diagram_explorer.tracing import TracedDiagram

# Tolerance (mole fraction) when matching x to a field edge, and (K) when
# matching T to a field's temperature span.
X_TOLERANCE = 1e-12
T_TOLERANCE = 1e-9


def _fields_at(traced: TracedDiagram, T: float):
    """(phases, x_min, x_max) of every field present at T, sorted by x."""
    found = []
    for field in traced.fields:
        Ts = field.T
        if not Ts[0] - T_TOLERANCE <= T <= Ts[-1] + T_TOLERANCE:
            continue
        if len(Ts) == 1:
            lo, hi = field.x_min[0], field.x_max[0]
        else:
            lo = float(np.interp(T, Ts, field.x_min))
            hi = float(np.interp(T, Ts, field.x_max))
        found.append((field.phases, lo, hi))
    return sorted(found, key=lambda item: (item[1], item[2]))


def _labels(fields) -> list[tuple[str, ...]]:
    """Composition-set labels for each field, numbered as compute_equilibrium does."""
    gapped = {phases[0] for phases, _, _ in fields if len(phases) == 2 and phases[0] == phases[1]}
    counts: dict[str, int] = {}
    labelled = []
    for phases, _, _ in fields:
        if len(phases) == 1:
            phase = phases[0]
            if phase in gapped:
                counts[phase] = counts.get(phase, 0) + 1
                labelled.append((composition_set_label(phase, counts[phase]),))
            else:
                labelled.append((phase,))
        else:
            labelled.append(phases)
    # a tie line takes the labels of the single-phase stretches on either side
    for k, (phases, _, _) in enumerate(fields):
        if len(phases) == 2:
            left = next((labelled[j][0] for j in range(k - 1, -1, -1) if len(labelled[j]) == 1), phases[0])
            right = next((labelled[j][0] for j in range(k + 1, len(fields)) if len(labelled[j]) == 1), phases[1])
            if phases[0] == phases[1] and left == right:
                left, right = composition_set_label(phases[0], 1), composition_set_label(phases[0], 2)
            labelled[k] = (left, right)
    return labelled


def equilibrium_from_trace(traced: TracedDiagram, T: float, x: float) -> EquilibriumResult:
    """Equilibrium at (T, x) from a traced diagram (see module docstring)."""
    T_min, T_max = traced.T_range
    if not T_min - T_TOLERANCE <= T <= T_max + T_TOLERANCE:
        raise ValueError(f"T = {T} K is outside the traced range {traced.T_range}")
    fields = _fields_at(traced, T)
    labels = _labels(fields)

    best = None
    for (phases, lo, hi), label in zip(fields, labels):
        if len(phases) == 2 and lo < x < hi:
            best = (label, lo, hi)
            break
        if len(phases) == 1 and lo - X_TOLERANCE <= x <= hi + X_TOLERANCE and best is None:
            best = (label, lo, hi)
    if best is None:
        raise ValueError(f"no traced field contains x = {x} at T = {T} K")

    label, lo, hi = best
    if len(label) == 1:
        composition = x if hi > lo else lo
        return EquilibriumResult(
            T=T, x_overall=x, stable_phases=[label[0]], phase_compositions={label[0]: composition},
            phase_fractions={label[0]: 1.0}, total_gibbs=math.nan,
        )
    fraction_right = (x - lo) / (hi - lo)
    return EquilibriumResult(
        T=T, x_overall=x, stable_phases=list(label),
        phase_compositions={label[0]: lo, label[1]: hi},
        phase_fractions={label[0]: 1.0 - fraction_right, label[1]: fraction_right},
        total_gibbs=math.nan,
    )
