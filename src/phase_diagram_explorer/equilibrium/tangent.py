"""Common tangent (tie line) refinement.

The convex hull of sampled Gibbs curves locates tie lines to within one
sampling step. The functions here refine a tie line to the exact common
tangent of the two phases' Gibbs energy curves, so equilibrium compositions
do not depend on how finely the curves were sampled.
"""
import numpy as np
from scipy.optimize import minimize_scalar, root
from scipy.special import expit

from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

# Compositions are solved in logit space over each phase's composition
# range, kept this far (relative to the range) from its ends.
COMPOSITION_FLOOR = 1e-14
TANGENT_RESIDUAL_TOLERANCE = 1e-6  # J/mol
# Phases with internal degrees of freedom: a coarse search whose best point
# lies more than this (J/mol) above the line is taken as above it without
# refinement.
COARSE_MARGIN = 200.0
COARSE_SAMPLES = 41


def _logit(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    u = (x - lo) / (hi - lo)
    u = min(max(u, COMPOSITION_FLOOR), 1.0 - COMPOSITION_FLOOR)
    return float(np.log(u / (1.0 - u)))


def _expit(u: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo + (hi - lo) * min(max(float(expit(u)), COMPOSITION_FLOOR), 1.0 - COMPOSITION_FLOOR)


def lowest_point(phase, T: float, slope: float, intercept: float, x: np.ndarray, refine: bool = True):
    """(x, distance) where `phase`'s Gibbs energy lies furthest below (or
    least above) the line slope*x + intercept, searching the sorted samples
    `x` and refining between the neighbours of the best one."""
    if isinstance(phase, StoichiometricPhase):
        x_q = phase.composition
        return x_q, float(phase.molar_gibbs(T)) - (slope * x_q + intercept)
    lo, hi = phase.composition_range
    x = x[(x >= lo) & (x <= hi)]
    if len(x) == 0:
        return None
    if getattr(phase, "has_internal_freedom", False) and len(x) > COARSE_SAMPLES:
        # each evaluation is a minimisation: search a coarse subset first
        coarse = x[np.linspace(0, len(x) - 1, COARSE_SAMPLES).astype(int)]
        distance = np.asarray(phase.molar_gibbs(T, coarse), dtype=float) - (slope * coarse + intercept)
        k = int(np.argmin(distance))
        if not refine or distance[k] > COARSE_MARGIN:
            return float(coarse[k]), float(distance[k])
        x = x[(x >= coarse[max(k - 1, 0)]) & (x <= coarse[min(k + 1, len(coarse) - 1)])]
    distance = np.asarray(phase.molar_gibbs(T, x), dtype=float) - (slope * x + intercept)
    k = int(np.argmin(distance))
    if not refine or len(x) < 2:
        return float(x[k]), float(distance[k])
    refined = minimize_scalar(
        lambda xi: float(phase.molar_gibbs(T, xi)) - (slope * xi + intercept),
        bounds=(x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]), method="bounded", options={"xatol": 1e-12},
    )
    if refined.fun < distance[k]:
        return float(refined.x), float(refined.fun)
    return float(x[k]), float(distance[k])


def gibbs(phase, T: float, x: float) -> float:
    if isinstance(phase, StoichiometricPhase):
        return float(phase.molar_gibbs(T))
    return float(phase.molar_gibbs(T, x))


def tangent_line(phase_left, phase_right, T: float, x_left: float, x_right: float):
    """(slope, intercept) of the straight line through both phases' Gibbs
    energies at x_left and x_right."""
    G_left = gibbs(phase_left, T, x_left)
    G_right = gibbs(phase_right, T, x_right)
    slope = (G_right - G_left) / (x_right - x_left)
    return slope, G_left - slope * x_left


def common_tangent(phase_left, phase_right, T: float, x_left: float, x_right: float) -> tuple[float, float]:
    """Refine a tie line between two phases (or two composition sets of one
    phase) to their exact common tangent at T, starting from the estimate
    (x_left, x_right).

    Stoichiometric phases keep their fixed composition. Returns the estimate
    unchanged if the solver does not converge to a valid, distinct pair.
    """
    left_free = not isinstance(phase_left, StoichiometricPhase)
    right_free = not isinstance(phase_right, StoichiometricPhase)
    range_left = phase_left.composition_range if left_free else (0.0, 1.0)
    range_right = phase_right.composition_range if right_free else (0.0, 1.0)
    if isinstance(phase_left, StoichiometricPhase):
        x_left = phase_left.composition
    if isinstance(phase_right, StoichiometricPhase):
        x_right = phase_right.composition
    if not (left_free or right_free):
        return x_left, x_right

    scale = max(abs(gibbs(phase_left, T, x_left)), abs(gibbs(phase_right, T, x_right)), 1.0)

    def unpack(u):
        u = list(u)
        xl = _expit(u.pop(0), *range_left) if left_free else x_left
        xr = _expit(u.pop(0), *range_right) if right_free else x_right
        return xl, xr

    def residual(u):
        xl, xr = unpack(u)
        if xr - xl == 0.0:
            return [1.0] * len(u)
        slope, intercept = tangent_line(phase_left, phase_right, T, xl, xr)
        equations = []
        if left_free:
            equations.append((phase_left.molar_gibbs_derivative(T, xl) - slope) / scale)
        if right_free:
            equations.append((phase_right.molar_gibbs_derivative(T, xr) - slope) / scale)
        return equations

    start = ([_logit(x_left, *range_left)] if left_free else []) + (
        [_logit(x_right, *range_right)] if right_free else []
    )
    solution = root(residual, start, method="hybr", options={"xtol": 1e-13})
    xl, xr = unpack(solution.x)

    valid = (
        np.all(np.isfinite([xl, xr]))
        and xr - xl > 1e-9
        and max(abs(r) for r in residual(solution.x)) * scale < TANGENT_RESIDUAL_TOLERANCE
    )
    return (xl, xr) if valid else (x_left, x_right)
