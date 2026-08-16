import json
import math

import numpy as np
import pytest

from phase_diagram_explorer.thermo.pure import PureElementGibbs


def test_constant_term_only():
    gibbs = PureElementGibbs(a=1234.5)
    assert gibbs.G(300.0) == pytest.approx(1234.5)
    assert gibbs.G(900.0) == pytest.approx(1234.5)


def test_linear_term():
    gibbs = PureElementGibbs(a=100.0, b=2.0)
    assert gibbs.G(500.0) == pytest.approx(100.0 + 2.0 * 500.0)


def test_full_polynomial_matches_analytical_formula():
    a, b, c, d, e, f = 100.0, 2.0, -3.0, 0.001, 1e-7, 50.0
    gibbs = PureElementGibbs(a=a, b=b, c=c, d=d, e=e, f=f)

    T = 500.0
    expected = a + b * T + c * T * math.log(T) + d * T**2 + e * T**3 + f / T

    assert gibbs.G(T) == pytest.approx(expected)


def test_scalar_input_returns_float():
    gibbs = PureElementGibbs(a=1.0, b=1.0)
    result = gibbs.G(300.0)
    assert isinstance(result, float)


def test_array_input_returns_array_matching_scalar_evaluations():
    a, b, c, d, e, f = 100.0, 2.0, -3.0, 0.001, 1e-7, 50.0
    gibbs = PureElementGibbs(a=a, b=b, c=c, d=d, e=e, f=f)

    temperatures = np.array([300.0, 500.0, 800.0])
    result = gibbs.G(temperatures)

    assert isinstance(result, np.ndarray)
    for T, value in zip(temperatures, result):
        expected = a + b * T + c * T * math.log(T) + d * T**2 + e * T**3 + f / T
        assert value == pytest.approx(expected)


def test_raises_below_t_min():
    gibbs = PureElementGibbs(a=1.0, T_min=298.15, T_max=1000.0)
    with pytest.raises(ValueError):
        gibbs.G(100.0)


def test_raises_above_t_max():
    gibbs = PureElementGibbs(a=1.0, T_min=298.15, T_max=1000.0)
    with pytest.raises(ValueError):
        gibbs.G(1500.0)


def test_raises_when_any_array_element_out_of_range():
    gibbs = PureElementGibbs(a=1.0, T_min=298.15, T_max=1000.0)
    with pytest.raises(ValueError):
        gibbs.G(np.array([300.0, 1200.0]))


def test_no_range_restriction_when_bounds_unset():
    gibbs = PureElementGibbs(a=1.0)
    assert gibbs.G(1e6) == pytest.approx(1.0)


def test_from_json_parses_coefficients_via_model_layer(tmp_path):
    coefficients = {
        "a": 100.0,
        "b": 2.0,
        "c": -3.0,
        "d": 0.001,
        "e": 1e-7,
        "f": 50.0,
        "T_min": 298.15,
        "T_max": 1000.0,
    }
    path = tmp_path / "coefficients.json"
    path.write_text(json.dumps(coefficients))

    gibbs = PureElementGibbs.from_json(path)

    T = 500.0
    expected = (
        coefficients["a"]
        + coefficients["b"] * T
        + coefficients["c"] * T * math.log(T)
        + coefficients["d"] * T**2
        + coefficients["e"] * T**3
        + coefficients["f"] / T
    )
    assert gibbs.G(T) == pytest.approx(expected)
    assert gibbs.T_min == 298.15
    assert gibbs.T_max == 1000.0
