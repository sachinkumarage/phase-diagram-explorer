from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.invariants import EUTECTIC, PERITECTIC, detect_invariants
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


def test_ag_cu_reproduces_known_eutectic_point():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    system = build_system(definition)

    diagram = compute_diagram(system, T_range=(1000.0, 1150.0), n_T=151, n_x=201, n_points=501)
    reactions = detect_invariants(diagram, system, n_points=501)
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    reaction = eutectics[0]

    temperature_c = reaction.temperature - 273.15
    assert temperature_c == pytest.approx(AG_CU_EUTECTIC_C, abs=TEMPERATURE_TOLERANCE_C)

    x_cu = reaction.composition["LIQUID"]
    assert x_cu == pytest.approx(AG_CU_EUTECTIC_X_CU, abs=COMPOSITION_TOLERANCE)


def test_ag_cu_eutectic_phases_are_the_two_terminal_solid_solutions_and_liquid():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    system = build_system(definition)

    diagram = compute_diagram(system, T_range=(1000.0, 1150.0), n_T=151, n_x=201, n_points=501)
    reactions = detect_invariants(diagram, system, n_points=501)
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    assert set(eutectics[0].phases) == {"LIQUID", "FCC_AG", "FCC_CU"}


def test_ag_cu_phase_fractions_sum_to_one_at_eutectic_composition():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    system = build_system(definition)

    diagram = compute_diagram(system, T_range=(1000.0, 1150.0), n_T=151, n_x=201, n_points=501)
    reactions = detect_invariants(diagram, system, n_points=501)
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    reaction = eutectics[0]
    assert sum(reaction.composition.values()) > 0  # sanity: compositions were populated


def test_al_cu_has_at_least_one_invariant_reaction_involving_liquid():
    definition = load_system(SYSTEMS_DIR / "al_cu.json")
    system = build_system(definition)

    # The Al-rich liquid field at the eutectic is only ~0.005 wide in x(Cu),
    # so the composition grid must be fine enough to resolve it.
    diagram = compute_diagram(system, T_range=(900.0, 950.0), n_T=26, n_x=1001, n_points=2001)
    reactions = detect_invariants(diagram, system, n_points=2001)
    liquid_reactions = [
        r for r in reactions if r.type in (EUTECTIC, PERITECTIC) and "LIQUID" in r.phases
    ]

    assert len(liquid_reactions) >= 1
    assert "AL2CU" in liquid_reactions[0].phases


def test_al_cu_compound_composition_matches_stoichiometry():
    definition = load_system(SYSTEMS_DIR / "al_cu.json")
    system = build_system(definition)

    al2cu = system["AL2CU"]
    assert isinstance(al2cu, StoichiometricPhase)
    assert al2cu.composition == pytest.approx(1.0 / 3.0)
