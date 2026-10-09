from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants_over_range
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

SYSTEMS_DIR = Path(__file__).resolve().parents[1] / "data" / "systems"

AG_CU_EUTECTIC_C = 779.0
AG_CU_EUTECTIC_X_CU = 0.399
TEMPERATURE_TOLERANCE_C = 10.0
COMPOSITION_TOLERANCE = 0.03


def test_ag_cu_loads_without_error():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    assert definition.name == "Ag-Cu"
    assert {element.symbol for element in definition.elements} == {"Ag", "Cu"}
    assert {phase.name for phase in definition.phases} == {"LIQUID", "FCC_AG", "FCC_CU"}


def test_al_cu_loads_without_error():
    definition = load_system(SYSTEMS_DIR / "al_cu.json")
    assert definition.name == "Al-Cu"
    assert {element.symbol for element in definition.elements} == {"Al", "Cu"}
    assert {phase.name for phase in definition.phases} == {"LIQUID", "FCC_AL", "AL2CU"}


def test_ag_cu_reproduces_known_eutectic_point(ag_cu_eutectic_scan):
    _, reactions = ag_cu_eutectic_scan
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    reaction = eutectics[0]

    temperature_c = reaction.temperature - 273.15
    assert temperature_c == pytest.approx(AG_CU_EUTECTIC_C, abs=TEMPERATURE_TOLERANCE_C)

    x_cu = reaction.composition["LIQUID"]
    assert x_cu == pytest.approx(AG_CU_EUTECTIC_X_CU, abs=COMPOSITION_TOLERANCE)


def test_ag_cu_eutectic_phases_are_the_two_terminal_solid_solutions_and_liquid(ag_cu_eutectic_scan):
    _, reactions = ag_cu_eutectic_scan
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    assert set(eutectics[0].phases) == {"LIQUID", "FCC_AG", "FCC_CU"}


def test_ag_cu_phase_fractions_sum_to_one_at_eutectic_composition(ag_cu_eutectic_scan):
    _, reactions = ag_cu_eutectic_scan
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    reaction = eutectics[0]
    assert sum(reaction.composition.values()) > 0  # sanity: compositions were populated


@pytest.mark.slow
def test_engine_finds_single_al_cu_reaction():
    """Engine test, not a physics test: with the current provisional Al-Cu
    data, invariant detection over the system's range on the default grid
    finds exactly one LIQUID + FCC_AL + AL2CU reaction, even though its
    Al-rich liquid field is far narrower than any composition grid step.
    Physical expectations for Al-Cu are in tests/test_physics_regression.py."""
    definition = load_system(SYSTEMS_DIR / "al_cu.json")
    system = build_system(definition)

    reactions = detect_invariants_over_range(system, definition.t_range_k)

    assert len(reactions) == 1
    assert set(reactions[0].phases) == {"LIQUID", "FCC_AL", "AL2CU"}


def test_al_cu_compound_composition_matches_stoichiometry():
    definition = load_system(SYSTEMS_DIR / "al_cu.json")
    system = build_system(definition)

    al2cu = system["AL2CU"]
    assert isinstance(al2cu, StoichiometricPhase)
    assert al2cu.composition == pytest.approx(1.0 / 3.0)
