"""Common tangent (tie line) refinement.

The convex hull of sampled Gibbs curves locates tie lines to within one
sampling step. The functions here refine a tie line to the exact common
tangent of the two phases' Gibbs energy curves, so equilibrium compositions
do not depend on how finely the curves were sampled.
"""
import numpy as np
from scipy.optimize import root
from scipy.special import expit

from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

# Compositions are solved in logit space, kept this far from 0 and 1.
COMPOSITION_FLOOR = 1e-14
TANGENT_RESIDUAL_TOLERANCE = 1e-6  # J/mol


def _logit(x: float) -> float:
    x = min(max(x, COMPOSITION_FLOOR), 1.0 - COMPOSITION_FLOOR)
    return float(np.log(x / (1.0 - x)))


def _expit(u: float) -> float:
    return min(max(float(expit(u)), COMPOSITION_FLOOR), 1.0 - COMPOSITION_FLOOR)


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
    left_free = isinstance(phase_left, SolutionPhase)
    right_free = isinstance(phase_right, SolutionPhase)
    if isinstance(phase_left, StoichiometricPhase):
        x_left = phase_left.composition
    if isinstance(phase_right, StoichiometricPhase):
        x_right = phase_right.composition
    if not (left_free or right_free):
        return x_left, x_right

    scale = max(abs(gibbs(phase_left, T, x_left)), abs(gibbs(phase_right, T, x_right)), 1.0)

    def unpack(u):
        u = list(u)
        xl = _expit(u.pop(0)) if left_free else x_left
        xr = _expit(u.pop(0)) if right_free else x_right
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

    start = ([_logit(x_left)] if left_free else []) + ([_logit(x_right)] if right_free else [])
    solution = root(residual, start, method="hybr", options={"xtol": 1e-13})
    xl, xr = unpack(solution.x)

    valid = (
        np.all(np.isfinite([xl, xr]))
        and xr - xl > 1e-9
        and max(abs(r) for r in residual(solution.x)) * scale < TANGENT_RESIDUAL_TOLERANCE
    )
    return (xl, xr) if valid else (x_left, x_right)
