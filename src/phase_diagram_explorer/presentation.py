"""Phase fractions in the basis that matches the composition unit, and the
lever-rule working shown by the app.

In at% the app shows mole fractions of the phases; in wt% it shows mass
fractions,

    w_i = f_i M_i / sum_j f_j M_j,    M_i = (1 - x_i) M_A + x_i M_B,

where f_i are the mole fractions (per mole of atoms) and M_i the molar mass
of phase i from its composition x_i. The lever rule then holds in the same
unit: f(left) = (c_right - c0) / (c_right - c_left), with c in at% for mole
fractions and in wt% for mass fractions.
"""
from phase_diagram_explorer.equilibrium.equilibrium import EquilibriumResult
from phase_diagram_explorer.units import WEIGHT_PERCENT, composition_to_display

MOLE = "mol"
MASS = "mass"


def fraction_basis(composition_unit: str) -> str:
    return MASS if composition_unit == WEIGHT_PERCENT else MOLE


def fraction_header(composition_unit: str) -> str:
    return f"phase fraction ({fraction_basis(composition_unit)})"


def phase_fractions(result: EquilibriumResult, composition_unit: str, masses) -> dict[str, float]:
    """Phase fractions of `result` as mole fractions (at%) or mass fractions (wt%)."""
    if fraction_basis(composition_unit) == MOLE:
        return dict(result.phase_fractions)
    mass_a, mass_b = masses
    weights = {
        phase: fraction * ((1.0 - result.phase_compositions[phase]) * mass_a + result.phase_compositions[phase] * mass_b)
        for phase, fraction in result.phase_fractions.items()
    }
    total = sum(weights.values())
    return {phase: weight / total for phase, weight in weights.items()}


def lever_rule_lines(
    result: EquilibriumResult, composition_unit: str, masses, display_names: dict[str, str] | None = None
) -> list[str]:
    """The lever rule for a two-phase result, written out with the numbers in
    the current composition unit (empty for a single phase)."""
    if len(result.stable_phases) != 2:
        return []
    names = display_names or {}
    left, right = sorted(result.stable_phases, key=lambda p: result.phase_compositions[p])

    def shown(x):
        return composition_to_display(x, composition_unit, *masses)

    symbol = "w" if fraction_basis(composition_unit) == MASS else "x"
    c0, c_left, c_right = shown(result.x_overall), shown(result.phase_compositions[left]), shown(result.phase_compositions[right])
    fractions = phase_fractions(result, composition_unit, masses)
    n_left, n_right = names.get(left, left), names.get(right, right)
    s0, s_left, s_right = f"{symbol}0", f"{symbol}_{n_left}", f"{symbol}_{n_right}"
    return [
        f"{s0} = {c0:.2f} {composition_unit}, {s_left} = {c_left:.2f} {composition_unit}, "
        f"{s_right} = {c_right:.2f} {composition_unit}",
        f"f({n_left}) = ({s_right} − {s0}) / ({s_right} − {s_left}) = "
        f"({c_right:.2f} − {c0:.2f}) / ({c_right:.2f} − {c_left:.2f}) = {fractions[left]:.3f}",
        f"f({n_right}) = ({s0} − {s_left}) / ({s_right} − {s_left}) = "
        f"({c0:.2f} − {c_left:.2f}) / ({c_right:.2f} − {c_left:.2f}) = {fractions[right]:.3f}",
    ]
