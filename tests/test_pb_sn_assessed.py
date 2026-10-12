"""The assessed Pb-Sn system (data/systems/pb_sn.meta.json): the Pb-Sn
subsystem of the NIST solder database,

    Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics -
    Solder Systems, NIST Standard Reference Database 139, National Institute
    of Standards and Technology, https://doi.org/10.18434/T4759N,

with the Pb-Sn description of H.J. Fecht, M.-X. Zhang, Y.A. Chang,
J.H. Perepezko, Metall. Trans. A 20A (1989) 795-803, and the unary data of
A.T. Dinsdale, CALPHAD 15 (1991) 317-425. The composition axis is x(Sn).

The database file is downloaded from NIST (scripts/fetch_tdb.py) and is not
part of this repository.
"""
from pathlib import Path

import pytest
from scipy.optimize import brentq

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants_over_range
from phase_diagram_explorer.models import default_systems_dir, load_system, system_files, system_name
from phase_diagram_explorer.solidification import melting_range
from phase_diagram_explorer.units import mole_to_weight_fraction, weight_to_mole_fraction

ROOT = Path(__file__).resolve().parents[1]
PB_SN = ROOT / "data" / "systems" / "pb_sn.meta.json"
M_PB, M_SN = 207.2, 118.69  # molar masses in the database file (ELEMENT lines)

# a) NIST's calculated Pb-Sn eutectic (https://www.metallurgy.nist.gov/phase/solder/pbsn.html):
#    182.2 C; liquid 62.13, (Pb) 18.91 and (Sn) 96.59 mass % Sn.
#    182.2 C = 455.35 K. Mass % Sn w converts to the mole fraction
#        x(Sn) = (w / M_Sn) / (w / M_Sn + (100 - w) / M_Pb)
#    with the file's molar masses M_Sn = 118.69 and M_Pb = 207.2 g/mol:
#        liquid 62.13: 0.523464 / (0.523464 + 0.182770) = 0.74120
#        (Pb)   18.91: 0.159323 / (0.159323 + 0.391361) = 0.28932
#        (Sn)   96.59: 0.813801 / (0.813801 + 0.016458) = 0.98018
NIST_EUTECTIC_K, NIST_TOLERANCE_K = 455.35, 0.2
NIST_WT_SN = {"liquid": 62.13, "pb_rich": 18.91, "sn_rich": 96.59}
NIST_X = {"liquid": 0.74120, "pb_rich": 0.28932, "sn_rich": 0.98018}
NIST_X_TOLERANCE = 0.0005

# b) Experimental values (mass %)
EXPERIMENTAL_EUTECTIC_C, EXPERIMENTAL_TOLERANCE_C = 183.0, 2.0
EXPERIMENTAL_LIQUID_WT_SN = (61.9, 1.5)
EXPERIMENTAL_MAX_SN_IN_PB = (19.0, 2.0)  # wt% Sn in (Pb) at the eutectic
EXPERIMENTAL_MAX_PB_IN_SN = (2.5, 1.5)  # wt% Pb in (Sn) at the eutectic
PB_MELTING_K, SN_MELTING_K, MELTING_TOLERANCE_K = 600.61, 505.08, 0.1
DETECTION_TOLERANCE_K = 0.05


def wt_sn(x: float) -> float:
    return 100.0 * mole_to_weight_fraction(x, M_PB, M_SN)


@pytest.fixture(scope="module")
def pb_sn():
    definition = load_system(PB_SN)
    return definition, build_system(definition)


@pytest.fixture(scope="module")
def eutectic(pb_sn):
    definition, system = pb_sn
    reactions = detect_invariants_over_range(system, definition.t_range_k)
    assert len(reactions) == 1
    (reaction,) = reactions
    by_base = {base_phase_name(p): x for p, x in reaction.composition.items()}
    return reaction, {"liquid": by_base["LIQUID"], "pb_rich": by_base["FCC_A1"], "sn_rich": by_base["BCT_A5"]}


def test_system_definition(pb_sn):
    definition, _ = pb_sn
    assert definition.name == "Pb-Sn"
    assert [e.symbol for e in definition.elements] == ["Pb", "Sn"]
    assert definition.dependent_element.symbol == "Sn"  # composition axis x(Sn)
    assert [e.atomic_mass for e in definition.elements] == [M_PB, M_SN]
    assert definition.t_range_k == (300.0, 650.0)
    assert definition.display == {"temperature_unit": "°C", "composition_unit": "wt%"}
    assert not definition.is_provisional
    assert definition.data_source == "nist_solder"
    assert [c.text for c in definition.citations] == [
        "Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST Standard "
        "Reference Database 139, National Institute of Standards and Technology, https://doi.org/10.18434/T4759N "
        "(Accessed 12 October 2026).",
        "H.J. Fecht, M.-X. Zhang, Y.A. Chang, J.H. Perepezko, Metall. Trans. A 20A (1989) 795-803.",
        "A.T. Dinsdale, CALPHAD 15 (1991) 317-425.",
        "NIST Pb-Sn page: https://www.metallurgy.nist.gov/phase/solder/pbsn.html",
    ]


def test_nist_conversions():
    for phase, w in NIST_WT_SN.items():
        assert weight_to_mole_fraction(w / 100.0, M_PB, M_SN) == pytest.approx(NIST_X[phase], abs=5e-6)


# d) exactly one invariant reaction, a eutectic


def test_exactly_one_eutectic(eutectic):
    reaction, _ = eutectic
    assert reaction.type == EUTECTIC
    assert [base_phase_name(p) for p in reaction.phases] == ["FCC_A1", "LIQUID", "BCT_A5"]


# a) NIST's own calculation from the same database


def test_eutectic_temperature_matches_nist(eutectic):
    reaction, _ = eutectic
    assert reaction.temperature == pytest.approx(NIST_EUTECTIC_K, abs=NIST_TOLERANCE_K)


@pytest.mark.parametrize("phase", ["liquid", "pb_rich", "sn_rich"])
def test_eutectic_compositions_match_nist(eutectic, phase):
    _, x = eutectic
    assert x[phase] == pytest.approx(NIST_X[phase], abs=NIST_X_TOLERANCE)


# b) experimental values


def test_eutectic_matches_experiment(eutectic):
    reaction, x = eutectic
    assert reaction.temperature - 273.15 == pytest.approx(EXPERIMENTAL_EUTECTIC_C, abs=EXPERIMENTAL_TOLERANCE_C)
    value, tolerance = EXPERIMENTAL_LIQUID_WT_SN
    assert wt_sn(x["liquid"]) == pytest.approx(value, abs=tolerance)


def test_terminal_solubilities_match_experiment(eutectic):
    _, x = eutectic
    value, tolerance = EXPERIMENTAL_MAX_SN_IN_PB
    assert wt_sn(x["pb_rich"]) == pytest.approx(value, abs=tolerance)
    value, tolerance = EXPERIMENTAL_MAX_PB_IN_SN
    assert 100.0 - wt_sn(x["sn_rich"]) == pytest.approx(value, abs=tolerance)


def _pure_crossing(system, x, solid, T_m):
    return brentq(lambda T: system["LIQUID"].molar_gibbs(T, x) - system[solid].molar_gibbs(T, x), T_m - 20, T_m + 20, xtol=1e-9)


@pytest.mark.parametrize(
    ("element", "x", "solid", "T_m"), [("Pb", 0.0, "FCC_A1", PB_MELTING_K), ("Sn", 1.0, "BCT_A5", SN_MELTING_K)]
)
def test_melting_points(pb_sn, element, x, solid, T_m):
    _, system = pb_sn
    crossing = _pure_crossing(system, x, solid, T_m)
    assert crossing == pytest.approx(T_m, abs=MELTING_TOLERANCE_K)
    x_inside = 1e-9 if x == 0.0 else 1.0 - 1e-9
    detected = brentq(
        lambda T: 1.0 if "LIQUID" in compute_equilibrium(system, T, x_inside).stable_phases else -1.0,
        T_m - 20, T_m + 20, xtol=1e-5,
    )
    assert detected == pytest.approx(crossing, abs=DETECTION_TOLERANCE_K)


# e) the app uses the NIST data; pycalphad's Pb-Sn test database is a test fixture only


def test_app_uses_the_nist_pb_sn():
    names = {system_name(path): path for path in system_files(default_systems_dir())}
    assert names["pb_sn"] == PB_SN
    assert not [path for path in names.values() if path.suffix == ".tdb"]
    assert "pbsn.tdb" not in {path.name for path in (ROOT / "data").rglob("*")}


# industry context: 63Sn-37Pb solder


def test_63sn_37pb_is_near_eutectic(pb_sn, eutectic):
    definition, system = pb_sn
    reaction, _ = eutectic
    x = weight_to_mole_fraction(0.63, M_PB, M_SN)
    melting = melting_range(system, x, definition.t_range_k)
    assert melting.solidus == pytest.approx(reaction.temperature, abs=0.01)
    assert 0.0 < melting.mushy_range < 2.0  # about 1 K: near-eutectic
    off_eutectic = melting_range(system, weight_to_mole_fraction(0.40, M_PB, M_SN), definition.t_range_k)
    assert off_eutectic.mushy_range > 30.0
