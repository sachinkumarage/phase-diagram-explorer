import math

import numpy as np
import pytest

from phase_diagram_explorer.properties import activity, chemical_potential, driving_force
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase


def _ideal_phase(g_a=0.0, g_b=0.0):
    return SolutionPhase(PureElementGibbs(a=g_a), PureElementGibbs(a=g_b))


@pytest.mark.parametrize("x", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_ideal_solution_activity_equals_composition(x):
    phase = _ideal_phase()
    T = 800.0

    assert activity(T, x, phase) == pytest.approx(x, abs=1e-6)


def test_ideal_solution_activity_with_nonzero_reference_energies():
    # activity is relative to the phase's own pure-B reference, so shifting
    # both endpoints should not change the ideal a_B = x_B result
    phase = _ideal_phase(g_a=1500.0, g_b=-2500.0)
    T = 600.0

    for x in (0.2, 0.5, 0.8):
        assert activity(T, x, phase) == pytest.approx(x, abs=1e-6)


def test_activity_array_input_matches_scalar_evaluations():
    phase = _ideal_phase()
    T = 900.0
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    result = activity(T, x, phase)

    assert isinstance(result, np.ndarray)
    for xi, ai in zip(x, result):
        assert ai == pytest.approx(xi, abs=1e-6)


def test_chemical_potential_matches_ideal_analytical_formula():
    # for an ideal solution, mu_B = G_B(T) + R*T*ln(x_B) exactly
    g_b_value = 500.0
    phase = _ideal_phase(g_a=0.0, g_b=g_b_value)
    T = 700.0
    x = 0.4

    expected = g_b_value + GAS_CONSTANT * T * math.log(x)
    assert chemical_potential(T, x, phase) == pytest.approx(expected, abs=1e-3)


def test_activity_raises_at_pure_component_boundaries():
    phase = _ideal_phase()
    with pytest.raises(ValueError):
        activity(500.0, 0.0, phase)
    with pytest.raises(ValueError):
        activity(500.0, 1.0, phase)


def test_activity_deviates_from_composition_with_positive_interaction():
    # a positive Redlich-Kister L0 (unfavorable mixing) should push a_B
    # above the ideal value x_B for x_B < 1
    phase = SolutionPhase(PureElementGibbs(a=0.0), PureElementGibbs(a=0.0), L=[5000.0])
    T = 600.0
    x = 0.3

    assert activity(T, x, phase) > x


def test_driving_force_is_zero_when_phase_is_the_only_stable_phase():
    phase = _ideal_phase()
    system = {"ONLY": phase}
    T = 700.0

    for x in (0.2, 0.5, 0.8):
        assert driving_force(T, x, phase, system) == pytest.approx(0.0, abs=1e-6)


def test_driving_force_positive_for_a_phase_more_stable_than_equilibrium():
    # system stable phase is the plain ideal solution; a hypothetical phase
    # with a large negative offset lies below it everywhere, so it has a
    # positive driving force to form
    stable_phase = _ideal_phase()
    system = {"STABLE": stable_phase}
    favorable_phase = _ideal_phase(g_a=-10000.0, g_b=-10000.0)

    T = 700.0
    x = 0.5
    assert driving_force(T, x, favorable_phase, system) > 0


def test_driving_force_negative_for_a_phase_less_stable_than_equilibrium():
    stable_phase = _ideal_phase()
    system = {"STABLE": stable_phase}
    unfavorable_phase = _ideal_phase(g_a=10000.0, g_b=10000.0)

    T = 700.0
    x = 0.5
    assert driving_force(T, x, unfavorable_phase, system) < 0
