"""Solidus, liquidus and melting (mushy) range of an alloy.

For an overall composition x, the solidus is the lowest temperature at
which a liquid phase is stable and the liquidus the lowest temperature at
which only liquid is stable. Both are found by scanning temperature levels
and refining each change by bisection, so they do not depend on any
displayed diagram. The melting range is liquidus - solidus: zero at a pure
element or the eutectic composition.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium

SCAN_STEP_K = 5.0
TOLERANCE_K = 1e-3


@dataclass(frozen=True)
class MeltingRange:
    solidus: float | None  # K; None if liquid is stable throughout or nowhere in the range
    liquidus: float | None

    @property
    def mushy_range(self) -> float | None:
        if self.solidus is None or self.liquidus is None:
            return None
        return self.liquidus - self.solidus


def _is_liquid(label: str, liquid_phases: set[str]) -> bool:
    return base_phase_name(label) in liquid_phases


def melting_range(
    system: dict | None,
    x: float,
    T_range: tuple[float, float],
    liquid_phases: set[str] | None = None,
    n_points: int = 500,
    equilibrium=None,
) -> MeltingRange:
    """Solidus and liquidus at composition x within T_range.

    Equilibria come from compute_equilibrium on `system`, or from
    `equilibrium(T, x)` if given (e.g. lookup.equilibrium_from_trace, which
    needs no database). Liquid phases default to names containing "liquid".
    """
    if equilibrium is None:
        def equilibrium(T, x):
            return compute_equilibrium(system, T, x, n_points=n_points)

    def is_liquid(label: str) -> bool:
        if liquid_phases is not None:
            return _is_liquid(label, liquid_phases)
        return "liquid" in base_phase_name(label).lower()

    def state(T: float) -> tuple[bool, bool]:
        phases = equilibrium(T, x).stable_phases
        any_liquid = any(is_liquid(p) for p in phases)
        return any_liquid, any_liquid and all(is_liquid(p) for p in phases)

    T_min, T_max = T_range
    levels = np.linspace(T_min, T_max, max(int(np.ceil((T_max - T_min) / SCAN_STEP_K)) + 1, 2))
    states = [state(T) for T in levels]

    def first_change(index: int):
        for k in range(len(levels) - 1):
            if not states[k][index] and states[k + 1][index]:
                return brentq(
                    lambda T: 1.0 if state(T)[index] else -1.0, levels[k], levels[k + 1], xtol=TOLERANCE_K
                )
        return None

    return MeltingRange(solidus=first_change(0), liquidus=first_change(1))
