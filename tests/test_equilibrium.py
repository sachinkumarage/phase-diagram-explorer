import math

import numpy as np
import pytest

from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase


def _mirrored_ideal_system(offset: float):
    """Two ideal solution phases, mirror images of each other under x -> 1-x.

    A_RICH has G_A=0, G_B=offset; B_RICH has G_A=offset, G_B=0. Their common
    tangent has a closed-form solution by symmetry (see below), giving a
    known two-phase tie-line independent of the hull implementation.
    """
    a_rich = SolutionPhase(PureElementGibbs(a=0.0), PureElementGibbs(a=offset))
    b_rich = SolutionPhase(PureElementGibbs(a=offset), PureElementGibbs(a=0.0))
    return {"A_RICH": a_rich, "B_RICH": b_rich}


def _analytical_tie_line(offset: float, T: float):
    """Closed-form common tangent for _mirrored_ideal_system.

    G1(x) = offset*x + RT[(1-x)ln(1-x) + x ln x]        (A_RICH)
    G2(x) = offset*(1-x) + RT[(1-x)ln(1-x) + x ln x]     (B_RICH)
    G1'(x) = offset + RT*ln(x/(1-x))
    G2'(x) = -offset + RT*ln(x/(1-x))

    By the x -> 1-x mirror symmetry between the two curves, the common
    tangent touches G1 at x1 and G2 at x2 = 1 - x1. Setting G1'(x1) =
    G2'(1 - x1) and solving gives the sigmoid closed form below; the
    intercept-matching condition is then satisfied automatically by the
    same symmetry.
    """
    x1 = 1.0 / (1.0 + math.exp(offset / (GAS_CONSTANT * T)))
    x2 = 1.0 - x1
    return x1, x2


def test_analytical_tie_line_is_a_genuine_common_tangent():
    # sanity-check the closed-form formula itself: the line through
    # (x1, G1(x1)) and (x2, G2(x2)) must have slope equal to both curves'
    # derivatives at their respective tangent points.
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)
    x1, x2 = _analytical_tie_line(offset, T)

    G1 = system["A_RICH"].molar_gibbs(T, x1)
    G2 = system["B_RICH"].molar_gibbs(T, x2)
    tie_slope = (G2 - G1) / (x2 - x1)

    def slope1(x):
        return offset + GAS_CONSTANT * T * math.log(x / (1 - x))

    def slope2(x):
        return -offset + GAS_CONSTANT * T * math.log(x / (1 - x))

    assert slope1(x1) == pytest.approx(tie_slope, abs=1e-6)
    assert slope2(x2) == pytest.approx(tie_slope, abs=1e-6)


def test_two_phase_region_matches_analytical_tie_line():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)
    x1, x2 = _analytical_tie_line(offset, T)

    x_overall = 0.5  # midpoint of the tie-line by symmetry
    result = compute_equilibrium(system, T, x_overall, n_points=4001)

    assert set(result.stable_phases) == {"A_RICH", "B_RICH"}
    assert result.phase_compositions["A_RICH"] == pytest.approx(x1, abs=0.01)
    assert result.phase_compositions["B_RICH"] == pytest.approx(x2, abs=0.01)


def test_phase_fractions_sum_to_one_in_two_phase_region():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)

    result = compute_equilibrium(system, T, 0.5, n_points=4001)

    assert sum(result.phase_fractions.values()) == pytest.approx(1.0)


def test_lever_rule_gives_equal_fractions_at_symmetric_midpoint():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)

    # by symmetry the tie-line midpoint is exactly x=0.5, so the overall
    # composition x=0.5 sits exactly halfway between the two phases
    result = compute_equilibrium(system, T, 0.5, n_points=4001)

    assert result.phase_fractions["A_RICH"] == pytest.approx(0.5, abs=0.02)
    assert result.phase_fractions["B_RICH"] == pytest.approx(0.5, abs=0.02)


def test_lever_rule_recovers_overall_composition():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)

    x_overall = 0.4
    result = compute_equilibrium(system, T, x_overall, n_points=4001)

    x_a = result.phase_compositions["A_RICH"]
    x_b = result.phase_compositions["B_RICH"]
    f_a = result.phase_fractions["A_RICH"]
    f_b = result.phase_fractions["B_RICH"]

    assert f_a * x_a + f_b * x_b == pytest.approx(x_overall, abs=1e-6)


def test_single_phase_region_near_pure_a():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)
    x1, _ = _analytical_tie_line(offset, T)

    x_overall = x1 / 2.0  # well inside the single-phase A_RICH region
    result = compute_equilibrium(system, T, x_overall, n_points=4001)

    assert result.stable_phases == ["A_RICH"]
    assert result.phase_fractions == {"A_RICH": 1.0}
    assert result.phase_compositions["A_RICH"] == pytest.approx(x_overall)
    assert sum(result.phase_fractions.values()) == pytest.approx(1.0)


def test_single_phase_region_near_pure_b():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)
    _, x2 = _analytical_tie_line(offset, T)

    x_overall = (x2 + 1.0) / 2.0  # well inside the single-phase B_RICH region
    result = compute_equilibrium(system, T, x_overall, n_points=4001)

    assert result.stable_phases == ["B_RICH"]
    assert result.phase_fractions == {"B_RICH": 1.0}
    assert sum(result.phase_fractions.values()) == pytest.approx(1.0)


def test_total_gibbs_matches_lever_rule_interpolation():
    offset = 5000.0
    T = 500.0
    system = _mirrored_ideal_system(offset)

    result = compute_equilibrium(system, T, 0.5, n_points=4001)

    x_a = result.phase_compositions["A_RICH"]
    x_b = result.phase_compositions["B_RICH"]
    G_a = system["A_RICH"].molar_gibbs(T, x_a)
    G_b = system["B_RICH"].molar_gibbs(T, x_b)
    f_a = result.phase_fractions["A_RICH"]
    f_b = result.phase_fractions["B_RICH"]

    assert result.total_gibbs == pytest.approx(f_a * G_a + f_b * G_b, abs=1.0)


def test_handles_stoichiometric_compound_as_stable_single_phase():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    compound = StoichiometricPhase(
        gibbs_a, gibbs_b, m=1, n=1, g_form=PureElementGibbs(a=-10000.0)
    )
    a_solution = SolutionPhase(PureElementGibbs(a=0.0), PureElementGibbs(a=2000.0))
    b_solution = SolutionPhase(PureElementGibbs(a=2000.0), PureElementGibbs(a=0.0))
    system = {"A_SOL": a_solution, "AB": compound, "B_SOL": b_solution}

    T = 500.0
    result = compute_equilibrium(system, T, compound.composition, n_points=2001)

    assert "AB" in result.stable_phases
    assert result.phase_compositions["AB"] == pytest.approx(compound.composition)
    assert sum(result.phase_fractions.values()) == pytest.approx(1.0)


def test_out_of_range_composition_raises():
    system = _mirrored_ideal_system(5000.0)
    with pytest.raises(ValueError):
        compute_equilibrium(system, 500.0, 1.5, n_points=101)
