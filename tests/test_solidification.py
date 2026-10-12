"""Solidus, liquidus and melting range (solidification.py), on the
synthetic symmetric eutectic used in test_invariants."""
import pytest

from phase_diagram_explorer.invariants import detect_invariants_over_range
from phase_diagram_explorer.solidification import melting_range
from test_invariants import FUSION_ENTHALPY, MELTING_T, _eutectic_system


@pytest.fixture(scope="module")
def system():
    return _eutectic_system()


def test_pure_end_melts_at_one_temperature(system):
    melting = melting_range(system, 1e-9, (400.0, 900.0))
    assert melting.solidus == pytest.approx(MELTING_T, abs=0.01)
    assert melting.mushy_range == pytest.approx(0.0, abs=0.01)
    assert FUSION_ENTHALPY > 0


def test_eutectic_composition_has_no_melting_range(system):
    (reaction,) = detect_invariants_over_range(system, (400.0, 800.0))
    melting = melting_range(system, 0.5, (400.0, 900.0))
    assert melting.solidus == pytest.approx(reaction.temperature, abs=0.01)
    assert melting.mushy_range == pytest.approx(0.0, abs=0.01)


def test_hypoeutectic_alloy_has_a_melting_range(system):
    (reaction,) = detect_invariants_over_range(system, (400.0, 800.0))
    melting = melting_range(system, 0.3, (400.0, 900.0))
    assert melting.solidus == pytest.approx(reaction.temperature, abs=0.01)
    assert reaction.temperature < melting.liquidus < MELTING_T
    assert melting.mushy_range == pytest.approx(melting.liquidus - melting.solidus)


def test_no_liquid_in_range(system):
    melting = melting_range(system, 0.3, (300.0, 400.0))
    assert melting.solidus is None and melting.liquidus is None and melting.mushy_range is None
