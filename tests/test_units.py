import numpy as np
import pytest

from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    CELSIUS,
    KELVIN,
    WEIGHT_PERCENT,
    celsius_to_kelvin,
    composition_from_display,
    composition_label,
    composition_to_display,
    kelvin_to_celsius,
    mole_to_weight_fraction,
    temperature_from_display,
    temperature_label,
    temperature_to_display,
    weight_to_mole_fraction,
)

FE = 55.845
C = 12.011
AG = 107.8682
CU = 63.546


def test_kelvin_celsius_reference_points():
    assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert celsius_to_kelvin(779.0) == pytest.approx(1052.15)


def test_temperature_round_trip():
    T = np.linspace(200.0, 2000.0, 37)
    for unit in (KELVIN, CELSIUS):
        assert temperature_from_display(temperature_to_display(T, unit), unit) == pytest.approx(T)


def test_scalar_in_scalar_out():
    assert isinstance(kelvin_to_celsius(1000.0), float)
    assert isinstance(mole_to_weight_fraction(0.5, AG, CU), float)
    assert isinstance(composition_to_display(0.5, WEIGHT_PERCENT, AG, CU), float)


def test_mole_weight_round_trip():
    x = np.linspace(0.0, 1.0, 51)
    assert weight_to_mole_fraction(mole_to_weight_fraction(x, AG, CU), AG, CU) == pytest.approx(x)
    for unit in (ATOMIC_PERCENT, WEIGHT_PERCENT):
        displayed = composition_to_display(x, unit, AG, CU)
        assert composition_from_display(displayed, unit, AG, CU) == pytest.approx(x)


def test_weight_fraction_endpoints():
    assert mole_to_weight_fraction(0.0, AG, CU) == pytest.approx(0.0)
    assert mole_to_weight_fraction(1.0, AG, CU) == pytest.approx(1.0)


def test_fe_c_eutectoid_composition():
    # 0.76 wt% C in Fe with M(Fe) = 55.845, M(C) = 12.011 is 3.44 at% C
    at_percent_c = composition_to_display(
        composition_from_display(0.76, WEIGHT_PERCENT, FE, C), ATOMIC_PERCENT
    )
    assert at_percent_c == pytest.approx(3.44, abs=0.01)


def test_weight_percent_needs_atomic_masses():
    with pytest.raises(ValueError, match="atomic masses"):
        composition_to_display(0.5, WEIGHT_PERCENT)


def test_labels_include_units():
    assert temperature_label(KELVIN) == "Temperature (K)"
    assert temperature_label(CELSIUS) == "Temperature (°C)"
    assert composition_label("Cu", ATOMIC_PERCENT) == "x(Cu) (at%)"
    assert composition_label("Cu", WEIGHT_PERCENT) == "w(Cu) (wt%)"
