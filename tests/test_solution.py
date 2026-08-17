import math

import numpy as np
import pytest

from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase


def test_ideal_mixing_entropy_at_x_half():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    phase = SolutionPhase(gibbs_a, gibbs_b)

    T = 1000.0
    result = phase.molar_gibbs(T, 0.5)

    expected = GAS_CONSTANT * T * (0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert result == pytest.approx(expected)


def test_ideal_mixing_at_x_half_matches_r_t_ln_half():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    phase = SolutionPhase(gibbs_a, gibbs_b)

    T = 1200.0
    result = phase.molar_gibbs(T, 0.5)

    expected = GAS_CONSTANT * T * math.log(0.5)
    assert result == pytest.approx(expected)


def test_full_expression_with_redlich_kister_matches_analytical_formula():
    gibbs_a = PureElementGibbs(a=1000.0, b=-5.0)
    gibbs_b = PureElementGibbs(a=1200.0, b=-6.0)
    phase = SolutionPhase(gibbs_a, gibbs_b, L=[500.0, -100.0])

    T = 800.0
    x_b = 0.3
    x_a = 1.0 - x_b

    expected = (
        x_a * gibbs_a.G(T)
        + x_b * gibbs_b.G(T)
        + GAS_CONSTANT * T * (x_a * math.log(x_a) + x_b * math.log(x_b))
        + x_a * x_b * (500.0 + (-100.0) * (x_a - x_b))
    )

    assert phase.molar_gibbs(T, x_b) == pytest.approx(expected)


def test_array_composition_input_matches_scalar_evaluations():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    phase = SolutionPhase(gibbs_a, gibbs_b)

    T = 1000.0
    compositions = np.array([0.1, 0.5, 0.9])
    result = phase.molar_gibbs(T, compositions)

    assert isinstance(result, np.ndarray)
    for x_b, value in zip(compositions, result):
        x_a = 1.0 - x_b
        expected = GAS_CONSTANT * T * (x_a * math.log(x_a) + x_b * math.log(x_b))
        assert value == pytest.approx(expected)


def test_pure_a_limit_at_x_zero():
    gibbs_a = PureElementGibbs(a=1000.0, b=-5.0)
    gibbs_b = PureElementGibbs(a=1200.0, b=-6.0)
    phase = SolutionPhase(gibbs_a, gibbs_b, L=[500.0])

    T = 700.0
    result = phase.molar_gibbs(T, 0.0)

    assert result == pytest.approx(gibbs_a.G(T))


def test_pure_b_limit_at_x_one():
    gibbs_a = PureElementGibbs(a=1000.0, b=-5.0)
    gibbs_b = PureElementGibbs(a=1200.0, b=-6.0)
    phase = SolutionPhase(gibbs_a, gibbs_b, L=[500.0])

    T = 700.0
    result = phase.molar_gibbs(T, 1.0)

    assert result == pytest.approx(gibbs_b.G(T))


def test_scalar_composition_returns_float():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    phase = SolutionPhase(gibbs_a, gibbs_b)

    result = phase.molar_gibbs(1000.0, 0.5)

    assert isinstance(result, float)
