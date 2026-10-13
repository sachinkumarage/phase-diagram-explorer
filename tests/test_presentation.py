"""Phase fractions in the basis of the composition unit, and the lever rule."""
from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.lookup import equilibrium_from_trace
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.precomputed import DEFAULT_POINTS, load
from phase_diagram_explorer.presentation import fraction_header, lever_rule_lines, phase_fractions
from phase_diagram_explorer.units import ATOMIC_PERCENT, WEIGHT_PERCENT, composition_from_display, composition_to_display

ROOT = Path(__file__).resolve().parents[1]

T = 474.95  # K
W0 = 36.4  # wt% Sn
LEVER_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def pb_sn():
    definition = load_system(ROOT / "data" / "systems" / "pb_sn.meta.json")
    masses = (definition.base_element.atomic_mass, definition.dependent_element.atomic_mass)
    return definition, build_system(definition), masses


@pytest.fixture(scope="module")
def results(pb_sn):
    _, system, masses = pb_sn
    x0 = composition_from_display(W0, WEIGHT_PERCENT, *masses)
    live = compute_equilibrium(system, T, x0, n_points=DEFAULT_POINTS)
    table = load(ROOT / "data" / "precomputed" / "pb_sn.npz").table
    return {"live": live, "fallback": equilibrium_from_trace(table, T, x0)}


def test_headers():
    assert fraction_header(ATOMIC_PERCENT) == "phase fraction (mol)"
    assert fraction_header(WEIGHT_PERCENT) == "phase fraction (mass)"


@pytest.mark.parametrize("path", ["live", "fallback"])
def test_mass_fractions_satisfy_the_lever_rule_in_weight_fraction(pb_sn, results, path):
    _, _, masses = pb_sn
    result = results[path]
    assert sorted(result.stable_phases) == ["FCC_A1", "LIQUID"]
    w = {phase: composition_to_display(x, WEIGHT_PERCENT, *masses) / 100 for phase, x in result.phase_compositions.items()}
    w0 = composition_to_display(result.x_overall, WEIGHT_PERCENT, *masses) / 100
    fractions = phase_fractions(result, WEIGHT_PERCENT, masses)
    assert fractions["FCC_A1"] == pytest.approx((w["LIQUID"] - w0) / (w["LIQUID"] - w["FCC_A1"]), abs=LEVER_TOLERANCE)
    assert sum(fractions.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("path", ["live", "fallback"])
def test_mole_fractions_satisfy_the_lever_rule_in_mole_fraction(results, path):
    result = results[path]
    x = result.phase_compositions
    fractions = phase_fractions(result, ATOMIC_PERCENT, (None, None))
    assert fractions["FCC_A1"] == pytest.approx(
        (x["LIQUID"] - result.x_overall) / (x["LIQUID"] - x["FCC_A1"]), abs=LEVER_TOLERANCE
    )


@pytest.mark.parametrize("unit", [ATOMIC_PERCENT, WEIGHT_PERCENT])
def test_fallback_fractions_match_the_live_path(pb_sn, results, unit):
    _, _, masses = pb_sn
    live = phase_fractions(results["live"], unit, masses)
    fallback = phase_fractions(results["fallback"], unit, masses)
    assert fallback.keys() == live.keys()
    for phase in live:
        assert fallback[phase] == pytest.approx(live[phase], abs=LEVER_TOLERANCE)


def test_mass_and_mole_fractions_differ_for_pb_sn(pb_sn, results):
    _, _, masses = pb_sn
    mole = phase_fractions(results["live"], ATOMIC_PERCENT, masses)
    mass = phase_fractions(results["live"], WEIGHT_PERCENT, masses)
    assert abs(mole["FCC_A1"] - mass["FCC_A1"]) > 0.01


@pytest.mark.parametrize("unit, symbol", [(ATOMIC_PERCENT, "x"), (WEIGHT_PERCENT, "w")])
def test_lever_rule_lines(pb_sn, results, unit, symbol):
    _, _, masses = pb_sn
    result = results["live"]
    lines = lever_rule_lines(result, unit, masses, {"LIQUID": "L", "FCC_A1": "(Pb)"})
    fractions = phase_fractions(result, unit, masses)
    assert lines[0].startswith(f"{symbol}0 = ")
    assert lines[1].startswith(f"f((Pb)) = ({symbol}_L − {symbol}0) / ({symbol}_L − {symbol}_(Pb)) = ")
    assert lines[1].endswith(f"= {fractions['FCC_A1']:.3f}")
    assert lines[2].endswith(f"= {fractions['LIQUID']:.3f}")


def test_no_lever_rule_for_one_phase(pb_sn):
    _, system, masses = pb_sn
    assert lever_rule_lines(compute_equilibrium(system, 700.0, 0.5), WEIGHT_PERCENT, masses) == []
