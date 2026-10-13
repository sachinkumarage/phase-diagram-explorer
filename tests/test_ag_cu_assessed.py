"""The assessed Ag-Cu system (data/systems/ag_cu.meta.json): the Ag-Cu
subsystem of the NIST solder database,

    Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics -
    Solder Systems, NIST Standard Reference Database 139, National Institute
    of Standards and Technology, https://doi.org/10.18434/T4759N,

with the Ag-Cu description of F.H. Hayes, H.L. Lukas, G. Effenberg,
G. Petzow, Z. Metallkd. 77 (1986) 749-754, and the unary data of
A.T. Dinsdale, CALPHAD 15 (1991) 317-425.

The database file is downloaded from NIST (scripts/fetch_tdb.py) and is not
part of this repository.
"""
from pathlib import Path

import pytest
from scipy.optimize import brentq

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants_over_range
from phase_diagram_explorer.models import load_system

ROOT = Path(__file__).resolve().parents[1]
AG_CU = ROOT / "data" / "systems" / "ag_cu.meta.json"

# a) NIST's calculated Ag-Cu eutectic (https://www.metallurgy.nist.gov/phase/solder/agcu.html):
#    779.1 C; liquid 28.41, (Ag) 8.27 and (Cu) 92.53 mass % Cu.
#    779.1 C = 1052.25 K. Mass % Cu w converts to the mole fraction
#        x(Cu) = (w / M_Cu) / (w / M_Cu + (100 - w) / M_Ag)
#    with the database's molar masses M_Cu = 63.546 and M_Ag = 107.87 g/mol:
#        liquid 28.41: 0.447078 / (0.447078 + 0.663669) = 0.40250
#        (Ag)    8.27: 0.130142 / (0.130142 + 0.850375) = 0.13273
#        (Cu)   92.53: 1.456111 / (1.456111 + 0.069250) = 0.95460
#    (M_Ag = 107.8682 changes none of these at five decimals.)
NIST_EUTECTIC_K = 1052.25
NIST_TOLERANCE_K = 0.2
NIST_X = {"liquid": 0.40250, "ag_rich": 0.13273, "cu_rich": 0.95460}
NIST_X_TOLERANCE = 0.0005

# b) Experimental values
EXPERIMENTAL_EUTECTIC_K, EXPERIMENTAL_TOLERANCE_K = 1052.0, 3.0
EXPERIMENTAL_X = {"liquid": (0.399, 0.010), "ag_rich": (0.141, 0.015), "cu_rich": (0.950, 0.010)}
AG_MELTING_K, CU_MELTING_K, MELTING_TOLERANCE_K = 1234.93, 1357.77, 0.1
# Melting detected from equilibria near a pure end must agree with the pure
# LIQUID/FCC_A1 Gibbs energy crossing to within this.
DETECTION_TOLERANCE_K = 0.05


def _load():
    definition = load_system(AG_CU)
    return definition, build_system(definition)


@pytest.fixture(scope="module")
def ag_cu():
    return _load()


@pytest.fixture(scope="module")
def eutectic(ag_cu):
    definition, system = ag_cu
    reactions = detect_invariants_over_range(system, definition.t_range_k)
    assert len(reactions) == 1
    (reaction,) = reactions
    phases = {base_phase_name(p): p for p in reaction.phases}
    solids = sorted(
        (x for p, x in reaction.composition.items() if base_phase_name(p) == "FCC_A1"),
    )
    return reaction, {"liquid": reaction.composition[phases["LIQUID"]], "ag_rich": solids[0], "cu_rich": solids[1]}


def test_system_definition(ag_cu):
    definition, system = ag_cu
    assert definition.name == "Ag-Cu"
    assert [e.symbol for e in definition.elements] == ["Ag", "Cu"]
    assert definition.dependent_element.symbol == "Cu"
    assert definition.t_range_k == (500.0, 1450.0)
    assert not definition.is_provisional and definition.status is None
    assert definition.data_source == "nist_solder"
    assert [c.text for c in definition.citations] == [
        "Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST Standard "
        "Reference Database 139, National Institute of Standards and Technology, https://doi.org/10.18434/T4759N "
        "(Accessed 12 October 2026).",
        "F.H. Hayes, H.L. Lukas, G. Effenberg, G. Petzow, Z. Metallkd. 77 (1986) 749-754.",
        "A.T. Dinsdale, CALPHAD 15 (1991) 317-425.",
    ]
    assert set(system) == {"LIQUID", "FCC_A1", "BCC_A2", "HCP_A3", "BCT_A5", "DO3", "EPSILON"}
    assert "CU6SN5" in definition.metadata["subsystem"]["dropped_phases"]


# d) exactly one invariant reaction


def test_exactly_one_invariant_reaction(eutectic):
    reaction, _ = eutectic
    assert reaction.type == EUTECTIC
    assert [base_phase_name(p) for p in reaction.phases] == ["FCC_A1", "LIQUID", "FCC_A1"]


# a) NIST's own calculation from the same database


def test_eutectic_temperature_matches_nist(eutectic):
    reaction, _ = eutectic
    assert reaction.temperature == pytest.approx(NIST_EUTECTIC_K, abs=NIST_TOLERANCE_K)


@pytest.mark.parametrize("phase", ["liquid", "ag_rich", "cu_rich"])
def test_eutectic_compositions_match_nist(eutectic, phase):
    _, x = eutectic
    assert x[phase] == pytest.approx(NIST_X[phase], abs=NIST_X_TOLERANCE)


# b) experimental values


def test_eutectic_temperature_matches_experiment(eutectic):
    reaction, _ = eutectic
    assert reaction.temperature == pytest.approx(EXPERIMENTAL_EUTECTIC_K, abs=EXPERIMENTAL_TOLERANCE_K)


@pytest.mark.parametrize("phase", ["liquid", "ag_rich", "cu_rich"])
def test_eutectic_compositions_match_experiment(eutectic, phase):
    _, x = eutectic
    value, tolerance = EXPERIMENTAL_X[phase]
    assert x[phase] == pytest.approx(value, abs=tolerance)


def _pure_crossing(system, x: float, T_low: float, T_high: float) -> float:
    liquid, fcc = system["LIQUID"], system["FCC_A1"]
    return brentq(lambda T: liquid.molar_gibbs(T, x) - fcc.molar_gibbs(T, x), T_low, T_high, xtol=1e-9)


def _detected_melting(system, x: float, T_low: float, T_high: float) -> float:
    """Lowest temperature at which LIQUID is stable at composition x."""
    def liquid(T):
        return 1.0 if "LIQUID" in compute_equilibrium(system, T, x).stable_phases else -1.0

    return brentq(liquid, T_low, T_high, xtol=1e-4)


@pytest.mark.parametrize(("element", "x", "T_m"), [("Ag", 0.0, AG_MELTING_K), ("Cu", 1.0, CU_MELTING_K)])
def test_pure_element_gibbs_crossing(ag_cu, element, x, T_m):
    """LIQUID and FCC_A1 Gibbs energies of the pure element cross at its
    melting point (and FCC_A1 is the stable solid there)."""
    _, system = ag_cu
    T = _pure_crossing(system, x, T_m - 20.0, T_m + 20.0)
    assert T == pytest.approx(T_m, abs=MELTING_TOLERANCE_K)
    others = [name for name in ("BCC_A2", "HCP_A3", "BCT_A5", "DO3", "EPSILON") if name in system]
    for name in others:
        phase = system[name]
        if hasattr(phase, "composition") and phase.composition != x:
            continue
        G = phase.molar_gibbs(T) if hasattr(phase, "composition") else phase.molar_gibbs(T, x)
        assert G > system["FCC_A1"].molar_gibbs(T, x)


@pytest.mark.parametrize(
    ("element", "x", "T_m"), [("Ag", 1e-9, AG_MELTING_K), ("Cu", 1.0 - 1e-9, CU_MELTING_K)]
)
def test_melting_detected_next_to_the_pure_elements(ag_cu, element, x, T_m):
    """Equilibria just inside each pure end find LIQUID at the melting point:
    the narrow two-phase field next to a pure element is resolved (it was
    reported up to 1.7 K too early before 0.3.0)."""
    _, system = ag_cu
    detected = _detected_melting(system, x, T_m - 20.0, T_m + 20.0)
    crossing = _pure_crossing(system, round(x), T_m - 20.0, T_m + 20.0)
    assert detected == pytest.approx(crossing, abs=DETECTION_TOLERANCE_K)
    assert detected == pytest.approx(T_m, abs=MELTING_TOLERANCE_K)


@pytest.mark.parametrize("T", [300.0, 400.0, 525.0, 600.0])
def test_low_temperature_solvus_is_a_common_tangent(ag_cu, T):
    """Below about 600 K both FCC_A1 composition sets lie within the
    outermost few cells of the Gibbs-curve grid (spacing 0.002). Regression
    test: they used to be reported at the grid points 0.002004 and 1.0
    (4.4e-4 from pycalphad at 525 K); now both satisfy the common-tangent
    condition."""
    _, system = ag_cu
    result = compute_equilibrium(system, T, 0.5)
    x1, x2 = (result.phase_compositions[p] for p in result.stable_phases)
    assert 0.0 < x1 < 0.01 and 0.99 < x2 < 1.0
    assert (x1, x2) != (pytest.approx(1.0 / 499), 1.0)
    fcc = system["FCC_A1"]
    slope = (fcc.molar_gibbs(T, x2) - fcc.molar_gibbs(T, x1)) / (x2 - x1)
    assert fcc.molar_gibbs_derivative(T, x1) == pytest.approx(slope, abs=1e-3)
    assert fcc.molar_gibbs_derivative(T, x2) == pytest.approx(slope, abs=1e-3)
    if T == 525.0:  # pycalphad on the same file: 0.0024831, 0.9996607
        assert (x1, x2) == (pytest.approx(0.0024831, abs=1e-6), pytest.approx(0.9996607, abs=1e-6))


def test_gap_tangent_is_not_the_trivial_solution(ag_cu):
    """At 504.25 K the hull estimate (0.002004, 1.0) needs the fallback
    starting points; a start inside the gap used to converge to two nearby
    points near the spinodal (0.113679, 0.113681), which also satisfy the
    tangent equations. Both composition sets must be near the pure ends."""
    from phase_diagram_explorer.equilibrium.equilibrium import phase_assemblage

    _, system = ag_cu
    (tie,) = [f for f in phase_assemblage(system, 504.25).fields if f.is_two_phase]
    assert tie.phases == ("FCC_A1#1", "FCC_A1#2")
    assert tie.x_min < 0.003 and tie.x_max > 0.999
