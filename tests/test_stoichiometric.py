import math

import numpy as np
import pytest

from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase


def test_molar_gibbs_matches_analytical_formula():
    gibbs_a = PureElementGibbs(a=100.0, b=-2.0)
    gibbs_b = PureElementGibbs(a=200.0, b=-3.0)
    g_form = PureElementGibbs(a=-500.0, b=0.1, c=-0.01)
    phase = StoichiometricPhase(gibbs_a, gibbs_b, m=2, n=1, g_form=g_form)

    T = 600.0
    expected = 2 * gibbs_a.G(T) + 1 * gibbs_b.G(T) + g_form.G(T)

    assert phase.molar_gibbs(T) == pytest.approx(expected)


def test_default_g_form_is_zero():
    gibbs_a = PureElementGibbs(a=100.0, b=-2.0)
    gibbs_b = PureElementGibbs(a=200.0, b=-3.0)
    phase = StoichiometricPhase(gibbs_a, gibbs_b, m=1, n=1)

    T = 500.0
    expected = gibbs_a.G(T) + gibbs_b.G(T)

    assert phase.molar_gibbs(T) == pytest.approx(expected)


def test_composition_property_for_a2b():
    gibbs_a = PureElementGibbs(a=100.0)
    gibbs_b = PureElementGibbs(a=200.0)
    phase = StoichiometricPhase(gibbs_a, gibbs_b, m=2, n=1)

    assert phase.composition == pytest.approx(1.0 / 3.0)


def test_array_temperature_input():
    gibbs_a = PureElementGibbs(a=100.0, b=-2.0)
    gibbs_b = PureElementGibbs(a=200.0, b=-3.0)
    g_form = PureElementGibbs(a=-500.0, b=0.1)
    phase = StoichiometricPhase(gibbs_a, gibbs_b, m=2, n=1, g_form=g_form)

    temperatures = np.array([400.0, 600.0, 800.0])
    result = phase.molar_gibbs(temperatures)

    assert isinstance(result, np.ndarray)
    for T, value in zip(temperatures, result):
        expected = 2 * gibbs_a.G(T) + 1 * gibbs_b.G(T) + g_form.G(T)
        assert value == pytest.approx(expected)


@pytest.mark.parametrize("m,n", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_raises_for_invalid_composition(m, n):
    gibbs_a = PureElementGibbs(a=100.0)
    gibbs_b = PureElementGibbs(a=200.0)

    with pytest.raises(ValueError):
        StoichiometricPhase(gibbs_a, gibbs_b, m=m, n=n)


def test_temperature_range_validation_is_inherited_from_g_form():
    gibbs_a = PureElementGibbs(a=100.0)
    gibbs_b = PureElementGibbs(a=200.0)
    g_form = PureElementGibbs(a=-500.0, T_min=298.15, T_max=1000.0)
    phase = StoichiometricPhase(gibbs_a, gibbs_b, m=1, n=1, g_form=g_form)

    with pytest.raises(ValueError):
        phase.molar_gibbs(1500.0)
