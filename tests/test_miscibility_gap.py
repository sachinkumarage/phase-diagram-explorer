"""Miscibility gap (composition set) handling, validated against the
analytical symmetric regular solution in tests/fixtures/regular_solution.json:

    G_m = RT (x ln x + (1-x) ln(1-x)) + L0 x (1-x)

For L0 > 2RT the gap is symmetric about x = 0.5, the binodal satisfies
ln(x/(1-x)) = L0 (2x-1) / (RT), and the critical temperature is L0 / (2R).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants, detect_invariants_over_range
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT
from phase_diagram_explorer.visualization import _phase_field_ids, plot_diagram

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REGULAR_SOLUTION = FIXTURES_DIR / "regular_solution.json"
GAP_EUTECTIC = FIXTURES_DIR / "gap_eutectic.json"

N_POINTS = 2001
GRID_SPACING = 1.0 / (N_POINTS - 1)


def _L0() -> float:
    return json.loads(REGULAR_SOLUTION.read_text())["phases"][0]["interaction_parameters"][0]


def _critical_temperature() -> float:
    return _L0() / (2.0 * GAS_CONSTANT)


def _analytical_binodal(T: float) -> float:
    """A-rich binodal composition: the non-trivial root of
    ln(x/(1-x)) = L0 (2x-1) / (RT) below x = 0.5."""
    L0 = _L0()
    return brentq(lambda x: np.log(x / (1.0 - x)) - L0 * (2.0 * x - 1.0) / (GAS_CONSTANT * T), 1e-12, 0.5 - 1e-9)


@pytest.fixture(scope="module")
def regular_solution():
    return build_system(load_system(REGULAR_SOLUTION))


def test_fixture_is_inside_the_gap_regime():
    # L0 > 2RT at every temperature checked below
    assert _L0() > 2.0 * GAS_CONSTANT * 1150.0


@pytest.mark.parametrize("T", [800.0, 1000.0, 1150.0])
def test_binodal_matches_analytical_solution(regular_solution, T):
    result = compute_equilibrium(regular_solution, T, 0.5, n_points=N_POINTS)

    assert result.stable_phases == ["ALPHA#1", "ALPHA#2"]
    x1 = result.phase_compositions["ALPHA#1"]
    x2 = result.phase_compositions["ALPHA#2"]
    x_binodal = _analytical_binodal(T)

    assert x1 == pytest.approx(x_binodal, abs=GRID_SPACING)
    assert x2 == pytest.approx(1.0 - x_binodal, abs=GRID_SPACING)


@pytest.mark.parametrize("T", [800.0, 1000.0, 1150.0])
def test_gap_is_symmetric_about_one_half(regular_solution, T):
    result = compute_equilibrium(regular_solution, T, 0.5, n_points=N_POINTS)
    x1 = result.phase_compositions["ALPHA#1"]
    x2 = result.phase_compositions["ALPHA#2"]

    assert x1 + x2 == pytest.approx(1.0, abs=1e-9)
    assert result.phase_fractions["ALPHA#1"] == pytest.approx(0.5, abs=1e-9)


def test_lever_rule_inside_gap(regular_solution):
    result = compute_equilibrium(regular_solution, 1000.0, 0.3, n_points=N_POINTS)
    x1, x2 = result.phase_compositions["ALPHA#1"], result.phase_compositions["ALPHA#2"]
    f1, f2 = result.phase_fractions["ALPHA#1"], result.phase_fractions["ALPHA#2"]

    assert f1 + f2 == pytest.approx(1.0)
    assert f1 * x1 + f2 * x2 == pytest.approx(0.3)


def test_critical_temperature_equals_L0_over_2R(regular_solution):
    T_c = _critical_temperature()

    def has_gap(T: float) -> bool:
        return len(compute_equilibrium(regular_solution, T, 0.5, n_points=N_POINTS).stable_phases) == 2

    lower, upper = 1000.0, 1400.0
    assert has_gap(lower) and not has_gap(upper)
    while upper - lower > 0.05:
        middle = (lower + upper) / 2.0
        if has_gap(middle):
            lower = middle
        else:
            upper = middle

    assert (lower + upper) / 2.0 == pytest.approx(T_c, abs=1.0)


def test_single_composition_set_outside_gap(regular_solution):
    above = compute_equilibrium(regular_solution, _critical_temperature() + 20.0, 0.5, n_points=N_POINTS)
    assert above.stable_phases == ["ALPHA"]

    beside = compute_equilibrium(regular_solution, 800.0, 0.02, n_points=N_POINTS)
    assert len(beside.stable_phases) == 1
    assert base_phase_name(beside.stable_phases[0]) == "ALPHA"


def test_base_phase_name():
    assert base_phase_name("FCC_A1#2") == "FCC_A1"
    assert base_phase_name("FCC_A1") == "FCC_A1"
    assert base_phase_name("PHASE#X") == "PHASE#X"


def test_diagram_has_gap_field_and_continuous_single_phase_field(regular_solution):
    diagram = compute_diagram(regular_solution, T_range=(800.0, 1300.0), n_T=26, n_x=101, n_points=1001)
    labels = set(diagram.phase_labels.flat)
    assert ("ALPHA#1", "ALPHA#2") in labels

    _, field_names = _phase_field_ids(diagram.phase_labels)
    # ALPHA above Tc and ALPHA#1 / ALPHA#2 beside the gap are one field
    assert sorted(field_names) == ["ALPHA", "ALPHA#1+ALPHA#2"]
    assert detect_invariants(diagram, regular_solution, n_points=1001) == []

    figure = plot_diagram(diagram, "A-B", dependent_symbol="B")
    assert {trace.name for trace in figure.data} >= {"ALPHA", "ALPHA#1+ALPHA#2"}


def test_eutectic_into_two_composition_sets():
    system = build_system(load_system(GAP_EUTECTIC))
    reactions = detect_invariants_over_range(system, (600.0, 1100.0))

    assert len(reactions) == 1
    reaction = reactions[0]
    assert reaction.type == EUTECTIC
    assert reaction.phases == ("ALPHA#1", "LIQUID", "ALPHA#2")
    assert reaction.composition["LIQUID"] == pytest.approx(0.5, abs=0.01)
    assert reaction.composition["ALPHA#1"] + reaction.composition["ALPHA#2"] == pytest.approx(1.0, abs=0.01)

    below = compute_equilibrium(system, reaction.temperature - 5.0, 0.5, n_points=1001)
    assert below.stable_phases == ["ALPHA#1", "ALPHA#2"]
    above = compute_equilibrium(system, reaction.temperature + 5.0, 0.5, n_points=1001)
    assert above.stable_phases == ["LIQUID"]
