"""Binary subsystems of multicomponent TDB databases (synthetic A-B-C
fixture tests/fixtures/abc_ternary.tdb, not literature data)."""
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.tdb.convert import definition_from_database, extract_subsystem
from phase_diagram_explorer.tdb.parser import parse_tdb, read_tdb
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "abc_ternary.tdb"

# The A-B subsystem written out by hand: what extraction must give.
AB_BINARY = """
ELEMENT /-   ELECTRON_GAS  0.0 0.0 0.0 !
ELEMENT VA   VACUUM        0.0 0.0 0.0 !
ELEMENT A    FCC_A1        40.0 0.0 0.0 !
ELEMENT B    FCC_A1        60.0 0.0 0.0 !
FUNCTION GA 300 -5.0*T; 3000 N !
FUNCTION GB 300 -6.0*T; 3000 N !
PHASE LIQUID % 1 1.0 !
CONSTITUENT LIQUID :A,B: !
PARAMETER G(LIQUID,A;0) 300 9000-8.0*T+GA; 3000 N !
PARAMETER G(LIQUID,B;0) 300 11000-8.5*T+GB; 3000 N !
PARAMETER L(LIQUID,A,B;0) 300 -4000; 3000 N !
PARAMETER L(LIQUID,A,B;1) 300 1500; 3000 N !
PHASE FCC_A1 % 2 1 1 !
CONSTITUENT FCC_A1 :A,B:VA: !
PARAMETER G(FCC_A1,A:VA;0) 300 +GA; 3000 N !
PARAMETER G(FCC_A1,B:VA;0) 300 +GB; 3000 N !
PARAMETER L(FCC_A1,A,B:VA;0) 300 12000-2*T; 3000 N !
PHASE SIGMA % 2 2 1 !
CONSTITUENT SIGMA :A,B:A: !
PARAMETER G(SIGMA,A:A;0) 300 5000+3*GA; 3000 N !
PARAMETER G(SIGMA,B:A;0) 300 -6000+2*GB+GA; 3000 N !
PARAMETER L(SIGMA,A,B:*;0) 300 7000; 3000 N !
"""


def test_load_system_selects_the_binary_subsystem():
    definition = load_system(FIXTURE, elements=["A", "B"])
    assert [element.symbol for element in definition.elements] == ["A", "B"]
    assert [phase.name for phase in definition.phases] == ["LIQUID", "FCC_A1", "SIGMA"]
    assert definition.phases[2].constituents == [["A", "B"], ["A"]]


def test_dropped_phases_and_parameters_are_reported():
    subsystem = load_system(FIXTURE, elements=["A", "B"]).metadata["subsystem"]
    assert subsystem["elements"] == ["A", "B"]
    assert subsystem["dropped_elements"] == ["C"]
    assert subsystem["dropped_phases"] == ["BC2"]
    assert subsystem["filtered_constituents"] == {"LIQUID": ["C"], "FCC_A1": ["C"], "SIGMA": ["C"]}
    dropped = [text.split(" (line")[0] for text in subsystem["dropped_parameters"]]
    assert dropped == [
        "G(LIQUID,C;0)",
        "L(LIQUID,A,C;0)",
        "L(LIQUID,B,C;0)",
        "L(LIQUID,A,B,C;0)",  # ternary interaction
        "G(FCC_A1,C:VA;0)",
        "L(FCC_A1,A,C:VA;1)",
        "G(SIGMA,A:C;0)",
        "G(SIGMA,B:C;0)",
    ]
    assert all("(line " in text for text in subsystem["dropped_parameters"])


@pytest.mark.parametrize("elements", [["A", "B"], ["B", "A"]])
def test_extracted_subsystem_equals_the_binary_written_by_hand(elements):
    extracted = build_system(load_system(FIXTURE, elements=elements))
    binary = build_system(definition_from_database(parse_tdb(AB_BINARY), elements=elements))
    assert list(extracted) == list(binary)
    rng = np.random.default_rng(0)
    for name in binary:
        model_a, model_b = extracted[name].model, binary[name].model
        assert model_a.columns == model_b.columns
        Y = rng.random((8, model_a.n_columns)) + 0.05
        for T in (500.0, 1200.0):
            assert np.allclose(model_a.molar_gibbs(T, Y), model_b.molar_gibbs(T, Y), rtol=1e-14, atol=0)


def test_other_binaries_of_the_same_database():
    a_c = load_system(FIXTURE, elements=["A", "C"])
    assert [phase.name for phase in a_c.phases] == ["LIQUID", "FCC_A1", "SIGMA"]
    assert a_c.metadata["subsystem"]["dropped_phases"] == ["BC2"]
    b_c = build_system(load_system(FIXTURE, elements=["B", "C"]))
    assert "BC2" in b_c and isinstance(b_c["BC2"], StoichiometricPhase)
    assert b_c["BC2"].composition == pytest.approx(2.0 / 3.0)


def test_multicomponent_database_needs_the_elements():
    with pytest.raises(ValueError, match="needs elements"):
        load_system(FIXTURE)
    with pytest.raises(ValueError, match="not in the database"):
        load_system(FIXTURE, elements=["A", "D"])


def test_extraction_leaves_the_database_unchanged():
    database = read_tdb(FIXTURE)
    reduced, _ = extract_subsystem(database, ["A", "B"])
    assert set(database.phases) == {"LIQUID", "FCC_A1", "SIGMA", "BC2"}
    assert set(reduced.phases) == {"LIQUID", "FCC_A1", "SIGMA"}
    assert database.phases["LIQUID"].constituents == [["A", "B", "C"]]
