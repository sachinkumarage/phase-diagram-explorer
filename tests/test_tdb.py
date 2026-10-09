"""TDB parsing, conversion to binary systems, export and round trips."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.builder import build_system, is_computable
from phase_diagram_explorer.models import load_system, system_files
from phase_diagram_explorer.tdb.convert import definition_from_database
from phase_diagram_explorer.tdb.parser import parse_tdb, read_tdb
from phase_diagram_explorer.tdb.writer import to_tdb, write_metadata, write_tdb
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT, DATABASE_GAS_CONSTANT
from phase_diagram_explorer.thermo.magnetic import magnetic_gibbs
from phase_diagram_explorer.thermo.sublattice import MINIMISE
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = ROOT / "data" / "systems"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
JSON_SYSTEMS = [
    SYSTEMS_DIR / "ag_cu.json",
    SYSTEMS_DIR / "al_cu.json",
    FIXTURES_DIR / "regular_solution.json",
    FIXTURES_DIR / "gap_eutectic.json",
]
# Committed TDB fixtures and the JSON systems they were exported from.
TDB_FIXTURES = {
    "ag_cu_provisional.tdb": SYSTEMS_DIR / "ag_cu.json",
    "al_cu_provisional.tdb": SYSTEMS_DIR / "al_cu.json",
    "regular_solution.tdb": FIXTURES_DIR / "regular_solution.json",
    "gap_eutectic.tdb": FIXTURES_DIR / "gap_eutectic.json",
}

HEADER = """
ELEMENT /-   ELECTRON_GAS 0 0 0 !
ELEMENT VA   VACUUM 0 0 0 !
ELEMENT AG   FCC_A1 107.87 5745 42.551 !
ELEMENT CU   FCC_A1 63.546 5004.1 33.15 !
TYPE_DEFINITION % SEQ * !
DEFINE_SYSTEM_DEFAULT ELEMENT 2 !
DEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !
"""


def gibbs_values(system: dict, phase: str, points) -> np.ndarray:
    model = system[phase]
    if isinstance(model, StoichiometricPhase):
        return np.array([model.molar_gibbs(T) for T, _ in points])
    return np.array([model.molar_gibbs(T, x) for T, x in points])


def sample_points(definition, n: int = 20, seed: int = 0):
    T_min, T_max = definition.t_range_k
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(T_min, T_max, n), rng.uniform(0.001, 0.999, n)))


# --- parser ----------------------------------------------------------------


def _line_of(text: str, start: str) -> int:
    return next(k for k, line in enumerate(text.splitlines(), start=1) if line.startswith(start))


def test_comments_continuations_and_abbreviated_keywords():
    text = (
        HEADER
        + """
$ a comment line ! with a bang that is ignored
FUNC GHSERAG  298.15 -7209.512+118.200733*T   $ trailing comment
     -23.8463314*T*LN(T);
     3000 N REF0 !
PHASE LIQUID:L % 1 1.0 !  CONST LIQUID:L :AG,CU: !
PARA G(LIQUID,AG;0) 298.15 +11025.076-8.89102*T+GHSERAG#;
     3000 N !
"""
    )
    database = parse_tdb(text)
    assert set(database.elements) == {"/-", "VA", "AG", "CU"}
    assert database.elements["AG"].mass == pytest.approx(107.87)
    assert set(database.functions) == {"GHSERAG"}
    assert database.functions["GHSERAG"].intervals[0].expression.endswith("-23.8463314*T*LN(T)")
    assert database.phases["LIQUID"].constituents == [["AG", "CU"]]
    (parameter,) = database.parameters
    assert (parameter.kind, parameter.phase, parameter.constituents, parameter.order) == ("G", "LIQUID", [["AG"]], 0)
    assert parameter.line == _line_of(text, "PARA G(LIQUID")
    assert len(database.defaults) == 2


@pytest.mark.parametrize(
    ("command", "keyword"),
    [
        ("ADD_CONSTITUENT LIQUID :AG: !", "ADD_CONSTITUENT"),
        ("TABLE GX 298.15 1 !", "TABLE"),
        ("FTP_FILE X !", "FTP_FILE"),
    ],
)
def test_thermodynamic_keywords_we_cannot_evaluate_name_keyword_and_line(command, keyword):
    text = HEADER + "\n\n" + command + "\n"
    line = text.splitlines().index(command) + 1
    with pytest.raises(NotImplementedError, match=rf"'{keyword}' on line {line}"):
        parse_tdb(text)


def test_metadata_keywords_are_stored_and_never_raise():
    text = HEADER + """
DATABASE_INFO 'Ag-Cu test database ! with a bang and $ a dollar' !
VERSION_DATE 2026-10-10 !
ASSESSED_SYSTEMS AG-CU(;G5 MGSP:MGSP) !
REFERENCE_FILE refs.txt !
SPECIES CU2 CU2 !
LIST_OF_REFERENCES
 NUMBER  SOURCE
 REF1  'First author, Journal 1 (2001) 1-10.'
 REF2  'Second author, Journal 2 (2002)
        20-30 (two lines).'
!
ADD_REFERENCES
 REF3  'Added reference.'
!
"""
    database = parse_tdb(text)
    assert database.metadata["references"] == {
        "REF1": "First author, Journal 1 (2001) 1-10.",
        "REF2": "Second author, Journal 2 (2002) 20-30 (two lines).",
        "REF3": "Added reference.",
    }
    assert database.metadata["database_info"] == "Ag-Cu test database ! with a bang and $ a dollar"
    assert database.metadata["version_date"] == "2026-10-10"
    assert database.metadata["assessed_systems"].startswith("AG-CU")
    assert database.species == {"CU2": "CU2"}
    definition = definition_from_database(database, elements=["Ag", "Cu"])
    assert definition.metadata["references"]["REF2"].endswith("(two lines).")


def test_metadata_only_keywords_log_a_warning(caplog):
    with caplog.at_level("WARNING"):
        database = parse_tdb(HEADER + "ZERO_VOLUME_SPECIES VA !\nDATABASE_REFERENCE_LIST x !\n")
    assert "ZERO_VOLUME_SPECIES" in caplog.text and "DATABASE_REFERENCE_LIST" in caplog.text
    assert len(database.metadata["ignored"]) == 2


def test_temperature_limits_and_parameter_aliases():
    database = parse_tdb(
        HEADER + "TEMP-LIM 200 4000 !\nFUNCTION F +5*T; ,, N !\n"
        "PHASE BCC % 1 1 !\nCONSTITUENT BCC :AG,CU: !\nPARAMETER BM(BCC,CU;0) 1 2.2; 10 N !\n"
    )
    assert [(i.T_min, i.T_max) for i in database.functions["F"].intervals] == [(200.0, 4000.0)]
    assert database.parameters[0].kind == "BMAGN"


def test_quote_is_an_ordinary_type_definition_code():
    database = parse_tdb(HEADER + "TYPE_DEFINITION ' GES A_P_D X MAGNETIC -3.0 0.28 !\nPHASE X %' 1 1 !\n")
    assert database.type_definitions["'"].magnetic_factors() == (-3.0, 0.28)


def test_species_constituents_are_not_supported():
    with pytest.raises(NotImplementedError, match="species CU2"):
        definition_from_database(
            parse_tdb(HEADER + "SPECIES CU2 CU2 !\nPHASE X % 1 1 !\nCONSTITUENT X :AG,CU2: !\n"), elements=["Ag", "Cu"]
        )


def test_unsupported_parameter_type_names_line():
    text = HEADER + "PHASE LIQUID % 1 1 !\nCONSTITUENT LIQUID :AG,CU: !\nPARAMETER MQ(LIQUID&AG,AG;0) 1 0; 10 N !\n"
    with pytest.raises(NotImplementedError, match=f"'MQ' on line {_line_of(text, 'PARAMETER MQ')}"):
        parse_tdb(text)


def test_unterminated_command_is_an_error():
    with pytest.raises(ValueError, match="not terminated"):
        parse_tdb(HEADER + "PHASE LIQUID % 1 1\n")


def test_constituent_sublattice_count_must_match_phase():
    with pytest.raises(ValueError, match="sublattices"):
        parse_tdb(HEADER + "PHASE X % 2 1 1 !\nCONSTITUENT X :AG,CU: !\n")


# --- conversion ------------------------------------------------------------

SUBSTITUTIONAL = HEADER + """
FUNCTION GHSERAG 298.15 -7209.512+118.200733*T-23.8463314*T*LN(T); 3000 N !
FUNCTION GHSERCU 298.15 -7770.458+130.485235*T-24.112392*T*LN(T); 3000 N !
PHASE FCC_A1 % 2 1 1 !
CONSTITUENT FCC_A1 :AG,CU:VA: !
PARAMETER G(FCC_A1,AG:VA;0) 298.15 +GHSERAG; 3000 N !
PARAMETER G(FCC_A1,CU:VA;0) 298.15 +GHSERCU; 3000 N !
PARAMETER L(FCC_A1,CU,AG:VA;0) 298.15 +33819.1-8.1236*T; 3000 N !
PARAMETER L(FCC_A1,CU,AG:VA;1) 298.15 -5601.9+1.32997*T; 3000 N !
PHASE LIQUID % 1 2.0 !
CONSTITUENT LIQUID :AG,CU: !
PARAMETER G(LIQUID,AG;0) 298.15 +2*GHSERAG; 3000 N !
PARAMETER G(LIQUID,CU;0) 298.15 +2*GHSERCU; 3000 N !
PARAMETER L(LIQUID,AG,CU;1) 298.15 -2000; 3000 N !
"""


def _ghser(a, b, c, T):
    return a + b * T + c * T * np.log(T)


def test_substitutional_phase_with_vacancy_sublattice_and_reversed_interaction():
    definition = definition_from_database(parse_tdb(SUBSTITUTIONAL), elements=["Ag", "Cu"])
    system = build_system(definition)
    T, x = 900.0, 0.3  # x = mole fraction of Cu
    g_ag = _ghser(-7209.512, 118.200733, -23.8463314, T)
    g_cu = _ghser(-7770.458, 130.485235, -24.112392, T)
    L0, L1 = 33819.1 - 8.1236 * T, -5601.9 + 1.32997 * T
    # The TDB lists the interaction as (CU,AG), so L1 multiplies (x_Cu - x_Ag).
    expected = (
        (1 - x) * g_ag + x * g_cu
        + DATABASE_GAS_CONSTANT * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
        + x * (1 - x) * (L0 + L1 * (x - (1 - x)))
    )
    assert system["FCC_A1"].molar_gibbs(T, x) == pytest.approx(expected, rel=1e-12)


def test_site_count_gives_energies_per_mole_of_atoms():
    system = build_system(definition_from_database(parse_tdb(SUBSTITUTIONAL), elements=["Ag", "Cu"]))
    T, x = 900.0, 0.3
    g_ag = _ghser(-7209.512, 118.200733, -23.8463314, T)
    g_cu = _ghser(-7770.458, 130.485235, -24.112392, T)
    # LIQUID has 2 sites: G and L parameters are per formula unit of 2 atoms.
    expected = (
        (1 - x) * g_ag + x * g_cu
        + DATABASE_GAS_CONSTANT * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
        + x * (1 - x) * (-1000.0) * ((1 - x) - x)
    )
    assert system["LIQUID"].molar_gibbs(T, x) == pytest.approx(expected, rel=1e-12)


def test_dependent_element_sets_the_composition_axis():
    database = parse_tdb(SUBSTITUTIONAL)
    cu_axis = build_system(definition_from_database(database, dependent_element="Cu"))
    ag_axis = build_system(definition_from_database(database, elements=["Cu", "Ag"]))
    for x in (0.1, 0.4, 0.8):
        assert ag_axis["FCC_A1"].molar_gibbs(900.0, 1 - x) == pytest.approx(cu_axis["FCC_A1"].molar_gibbs(900.0, x))


def test_element_order_is_required(tmp_path):
    path = tmp_path / "agcu.tdb"
    path.write_text(SUBSTITUTIONAL)
    with pytest.raises(ValueError, match="element order is not defined"):
        load_system(path)
    with pytest.raises(ValueError, match="does not match"):
        load_system(path, elements=["Ag", "Al"])

    definition = load_system(path, dependent_element="Cu")
    assert definition.gas_constant == DATABASE_GAS_CONSTANT
    assert load_system(path, dependent_element="Cu", gas_constant=8.3145).gas_constant == 8.3145
    assert [e.symbol for e in definition.elements] == ["Ag", "Cu"]
    assert definition.elements[0].atomic_mass == pytest.approx(107.87)
    assert definition.name == "Ag-Cu"


def test_metadata_file_supplies_order_and_descriptive_fields(tmp_path):
    path = tmp_path / "agcu.tdb"
    path.write_text(SUBSTITUTIONAL)
    (tmp_path / "agcu.meta.json").write_text(
        json.dumps(
            {
                "name": "Ag-Cu test",
                "elements": ["Cu", "Ag"],
                "element_names": {"Ag": "Silver"},
                "t_range_k": [600, 1300],
                "_status": "provisional",
                "_status_reason": "test",
                "gas_constant": 8.314,
            }
        )
    )
    definition = load_system(path)
    assert definition.name == "Ag-Cu test"
    assert definition.base_element.symbol == "Cu"
    assert definition.dependent_element.name == "Silver"
    assert definition.t_range_k == (600.0, 1300.0)
    assert definition.is_provisional
    assert definition.gas_constant == 8.314

    # an argument overrides the metadata
    assert load_system(path, elements=["Ag", "Cu"]).dependent_element.symbol == "Cu"


MAGNETIC = HEADER + """
TYPE_DEFINITION & GES A_P_D FCC_A1 MAGNETIC -3.0 2.80000E-01 !
PHASE FCC_A1 %& 1 1 !
CONSTITUENT FCC_A1 :AG,CU: !
PARAMETER G(FCC_A1,AG;0) 298.15 0; 3000 N !
PARAMETER G(FCC_A1,CU;0) 298.15 0; 3000 N !
PARAMETER TC(FCC_A1,CU;0) 298.15 -201; 3000 N !
PARAMETER BMAGN(FCC_A1,CU;0) 298.15 -0.5; 3000 N !
"""


def test_magnetic_phase_is_evaluated_with_the_ihj_model():
    definition = definition_from_database(parse_tdb(MAGNETIC), elements=["Ag", "Cu"])
    (phase,) = definition.phases
    assert [p.kind for p in phase.magnetic_parameters] == ["TC", "BMAGN"]
    assert phase.type_definitions == ["TYPE_DEFINITION & GES A_P_D FCC_A1 MAGNETIC -3.0 2.80000E-01"]
    assert (phase.magnetic.afm_factor, phase.magnetic.structure_factor) == (-3.0, 0.28)
    assert phase.is_magnetic and is_computable(definition)

    fcc = build_system(definition)["FCC_A1"]
    T, x = 300.0, 0.6  # above the Neel temperature 201 * 0.6 / 3 = 40 K
    TC, beta = -201.0 * x, -0.5 * x
    G_mag = float(magnetic_gibbs(T, TC, beta, fcc.model.magnetic, DATABASE_GAS_CONSTANT)[0])
    ideal = DATABASE_GAS_CONSTANT * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
    assert G_mag < 0.0
    assert fcc.molar_gibbs(T, x) == pytest.approx(ideal + G_mag, rel=1e-12)


def test_magnetic_parameters_without_magnetic_type_definition_are_ignored(caplog):
    text = MAGNETIC.replace("TYPE_DEFINITION & GES A_P_D FCC_A1 MAGNETIC -3.0 2.80000E-01 !", "")
    definition = definition_from_database(parse_tdb(text), elements=["Ag", "Cu"])
    with caplog.at_level("WARNING"):
        fcc = build_system(definition)["FCC_A1"]
    assert "no MAGNETIC type definition" in caplog.text
    assert fcc.molar_gibbs(300.0, 0.5) == pytest.approx(DATABASE_GAS_CONSTANT * 300.0 * np.log(0.5))


def test_other_type_definitions_raise():
    text = MAGNETIC.replace("PHASE FCC_A1 %& 1 1 !", "TYPE_DEFINITION D GES A_P_D FCC_A1 DIS_PART FCC4 !\nPHASE FCC_A1 %&D 1 1 !")
    with pytest.raises(NotImplementedError, match="DIS_PART"):
        build_system(definition_from_database(parse_tdb(text), elements=["Ag", "Cu"]))


SUBLATTICE = HEADER + """
PHASE SIGMA % 3 8 4 18 !
CONSTITUENT SIGMA :AG,CU:AG:AG,CU: !
PARAMETER G(SIGMA,AG:AG:AG;0) 298.15 +1000; 3000 N !
PARAMETER G(SIGMA,CU:AG:CU;0) 298.15 +2000; 3000 N !
PARAMETER G(SIGMA,CU:AG:AG;0) 298.15 +5000; 3000 N !
PARAMETER G(SIGMA,AG:AG:CU;0) 298.15 +5000; 3000 N !
"""


def test_multi_sublattice_phases_are_built():
    definition = definition_from_database(parse_tdb(SUBLATTICE), elements=["Ag", "Cu"])
    (phase,) = definition.phases
    assert phase.model_type == "sublattice"
    assert phase.sublattice_sites == [8.0, 4.0, 18.0]
    assert phase.constituents == [["AG", "CU"], ["AG"], ["AG", "CU"]]
    assert len(phase.raw_parameters) == 4
    sigma = build_system(definition)["SIGMA"]
    assert sigma.mode == MINIMISE
    assert sigma.composition_range == pytest.approx((0.0, 26.0 / 30.0))
    assert np.isfinite(sigma.molar_gibbs(800.0, 0.4))


def test_compound_from_sublattices_is_stoichiometric():
    text = HEADER + """
PHASE AG2CU % 3 2 1 1 !
CONSTITUENT AG2CU :AG:CU:VA: !
PARAMETER G(AG2CU,AG:CU:VA;0) 298.15 -30000+3*T; 3000 N !
"""
    definition = definition_from_database(parse_tdb(text), elements=["Ag", "Cu"])
    compound = build_system(definition)["AG2CU"]
    assert isinstance(compound, StoichiometricPhase)
    assert compound.composition == pytest.approx(1.0 / 3.0)
    assert compound.molar_gibbs(700.0) == pytest.approx((-30000 + 3 * 700.0) / 3.0)


# --- export and round trips -------------------------------------------------


@pytest.mark.parametrize("path", JSON_SYSTEMS, ids=lambda path: path.stem)
def test_json_to_tdb_round_trip_gives_identical_gibbs_energies(tmp_path, path):
    original = load_system(path)
    tdb_path = tmp_path / f"{path.stem}.tdb"
    write_tdb(original, tdb_path)
    write_metadata(original, tdb_path)  # carries the element order and the gas constant
    reloaded = load_system(tdb_path)
    assert reloaded.gas_constant == CODATA_GAS_CONSTANT

    assert [e.symbol for e in reloaded.elements] == [e.symbol for e in original.elements]
    assert [e.atomic_mass for e in reloaded.elements] == [e.atomic_mass for e in original.elements]
    system, system_back = build_system(original), build_system(reloaded)
    assert list(system_back) == list(system)
    points = sample_points(original)
    for phase in system:
        expected = gibbs_values(system, phase, points)
        assert np.max(np.abs(gibbs_values(system_back, phase, points) / expected - 1.0)) < 1e-10, phase


def test_tdb_round_trip_is_stable(tmp_path):
    """TDB -> load -> TDB -> load keeps the same energies (expressions and
    FUNCTIONs are written back)."""
    first = definition_from_database(parse_tdb(SUBSTITUTIONAL), elements=["Ag", "Cu"])
    path = tmp_path / "again.tdb"
    write_tdb(first, path)
    second = load_system(path, elements=["Ag", "Cu"])
    assert set(second.functions) == {"GHSERAG", "GHSERCU"}
    points = [(T, x) for T in (400.0, 900.0, 1400.0) for x in (0.1, 0.5, 0.9)]
    for phase in ("FCC_A1", "LIQUID"):
        expected = gibbs_values(build_system(first), phase, points)
        assert np.allclose(gibbs_values(build_system(second), phase, points), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(("fixture", "source"), TDB_FIXTURES.items(), ids=list(TDB_FIXTURES))
def test_committed_tdb_fixtures_match_their_json_sources(fixture, source):
    definition = load_system(FIXTURES_DIR / fixture)
    original = load_system(source)
    assert definition.name == original.name
    assert definition.t_range_k == original.t_range_k
    assert definition.status == original.status
    assert [e.symbol for e in definition.elements] == [e.symbol for e in original.elements]

    system, expected_system = build_system(definition), build_system(original)
    points = sample_points(original, seed=1)
    for phase in expected_system:
        expected = gibbs_values(expected_system, phase, points)
        assert np.max(np.abs(gibbs_values(system, phase, points) / expected - 1.0)) < 1e-10, phase


@pytest.mark.parametrize("fixture", ["ag_cu_provisional.tdb", "al_cu_provisional.tdb"])
def test_provisional_fixtures_are_labelled(fixture):
    text = (FIXTURES_DIR / fixture).read_text()
    assert "TEST FIXTURE" in text and "NOT LITERATURE DATA" in text
    assert load_system(FIXTURES_DIR / fixture).is_provisional


def test_written_tdb_reads_with_tdb_conventions(tmp_path):
    text = to_tdb(load_system(SYSTEMS_DIR / "al_cu.json"))
    assert "PHASE AL2CU % 2 2.0 1.0 !" in text
    # the per-atom formation energy (-12000 J/mol) per formula unit of 3 atoms
    assert "PARAMETER G(AL2CU,AL:CU;0) 1.0 -36000.0; 10000.0 N !" in text
    database = parse_tdb(text)
    assert set(database.phases) == {"LIQUID", "FCC_AL", "AL2CU"}


def test_write_tdb_writes_sublattice_and_magnetic_phases(tmp_path):
    for text in (SUBLATTICE, MAGNETIC):
        first = definition_from_database(parse_tdb(text), elements=["Ag", "Cu"])
        path = tmp_path / "phase.tdb"
        write_tdb(first, path)
        second = load_system(path, elements=["Ag", "Cu"])
        assert second.phases[0].magnetic == first.phases[0].magnetic
        a, b = build_system(first), build_system(second)
        for name in a:
            Y = np.random.default_rng(0).random((5, a[name].model.n_columns))
            assert np.allclose(a[name].model.molar_gibbs(600.0, Y), b[name].model.molar_gibbs(600.0, Y), rtol=1e-14)


def test_write_tdb_rejects_unsupported_models():
    text = MAGNETIC.replace("PHASE FCC_A1 %& 1 1 !", "TYPE_DEFINITION D GES A_P_D FCC_A1 DIS_PART FCC4 !\nPHASE FCC_A1 %&D 1 1 !")
    with pytest.raises(NotImplementedError, match="write_tdb"):
        to_tdb(definition_from_database(parse_tdb(text), elements=["Ag", "Cu"]))


def test_system_files_lists_tdb_but_not_metadata(tmp_path):
    shutil.copy(FIXTURES_DIR / "ag_cu_provisional.tdb", tmp_path)
    shutil.copy(FIXTURES_DIR / "ag_cu_provisional.meta.json", tmp_path)
    shutil.copy(SYSTEMS_DIR / "al_cu.json", tmp_path)
    assert [path.name for path in system_files(tmp_path)] == ["ag_cu_provisional.tdb", "al_cu.json"]
    assert all(is_computable(load_system(path)) for path in system_files(tmp_path))


def test_write_metadata(tmp_path):
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    path = write_metadata(definition, tmp_path / "x.tdb")
    assert path.name == "x.meta.json"
    data = json.loads(path.read_text())
    assert data["elements"] == ["Ag", "Cu"] and data["_status"] == "provisional"


def test_read_tdb_from_file():
    database = read_tdb(FIXTURES_DIR / "ag_cu_provisional.tdb")
    assert set(database.phases) == {"LIQUID", "FCC_AG", "FCC_CU"}
