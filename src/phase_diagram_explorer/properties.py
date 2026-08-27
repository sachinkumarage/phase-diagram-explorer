import numpy as np

from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

DERIVATIVE_STEP = 1e-5


def _validate_composition(x: float | np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr <= 0.0) or np.any(x_arr >= 1.0):
        raise ValueError("composition must be strictly between 0 and 1")
    return x_arr


def _molar_gibbs_derivative(phase: SolutionPhase, T: float, x: np.ndarray) -> np.ndarray:
    """dG_m/dx at fixed T, via central finite difference, x = mole fraction of B."""
    h = np.minimum(DERIVATIVE_STEP, np.minimum(x, 1.0 - x) / 2.0)
    g_plus = np.asarray(phase.molar_gibbs(T, x + h), dtype=float)
    g_minus = np.asarray(phase.molar_gibbs(T, x - h), dtype=float)
    return (g_plus - g_minus) / (2.0 * h)


def chemical_potential(T: float, x: float | np.ndarray, phase: SolutionPhase) -> float | np.ndarray:
    """Partial molar Gibbs energy of B in `phase` at (T, x).

    Uses the tangent-intercept construction: for molar Gibbs energy G_m(x)
    (x = mole fraction of B), mu_B = G_m + (1 - x) * dG_m/dx.
    """
    x_arr = _validate_composition(x)
    g = np.asarray(phase.molar_gibbs(T, x_arr), dtype=float)
    dg_dx = _molar_gibbs_derivative(phase, T, x_arr)

    mu_b = g + (1.0 - x_arr) * dg_dx

    if np.ndim(x) == 0:
        return float(mu_b)
    return mu_b


def activity(T: float, x: float | np.ndarray, phase: SolutionPhase) -> float | np.ndarray:
    """Activity of B in `phase` at (T, x), relative to phase.gibbs_b as the
    pure-component reference state.

    a_B = exp((mu_B - G_B_reference(T)) / (R*T)); for an ideal solution
    (no Redlich-Kister excess), this reduces exactly to a_B = x_B.
    """
    mu_b = chemical_potential(T, x, phase)
    g_b_ref = float(phase.gibbs_b.G(T))
    a_b = np.exp((np.asarray(mu_b, dtype=float) - g_b_ref) / (GAS_CONSTANT * T))

    if np.ndim(x) == 0:
        return float(a_b)
    return a_b


def driving_force(
    T: float, x: float | np.ndarray, phase: SolutionPhase | StoichiometricPhase, system: dict
) -> float:
    """Driving force for `phase` relative to the current stable equilibrium
    of `system` at (T, x).

    Positive when `phase`'s own Gibbs energy lies below the equilibrium tie
    line at that composition (favored to form); negative when it lies above.
    Stoichiometric phases are evaluated at their fixed composition,
    regardless of the x passed in.
    """
    if isinstance(phase, StoichiometricPhase):
        x_eval = phase.composition
        phase_gibbs = phase.molar_gibbs(T)
    else:
        x_eval = x
        phase_gibbs = float(phase.molar_gibbs(T, x))

    result = compute_equilibrium(system, T, x_eval)
    return result.total_gibbs - phase_gibbs
