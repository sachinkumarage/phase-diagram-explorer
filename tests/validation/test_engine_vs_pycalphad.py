"""Cross-validation of the equilibrium engine against pycalphad, with both
engines reading the same TDB files.

Systems (see phase_diagram_explorer.validation.CASES):

- tests/fixtures: the provisional Ag-Cu and Al-Cu exports, the synthetic
  regular solution and gap eutectic, the synthetic interstitial system
  (A)1(C,VA)1 + (A)1(C,VA)3 + graphite-like C + liquid, and the synthetic
  magnetic system (ferro- and antiferromagnetic BCC/FCC with
  composition-dependent TC and BMAGN). None of these is literature data.
- Literature databases from pycalphad's test databases:
  - pbsn.tdb, Pb-Sn: T.L. Ngai and Y.A. Chang, CALPHAD 5 (1981) 267-276
    (TDB file by T. Abe, K. Hashimoto and Y. Sawada, NIMS 2011).
  - cumg.tdb, Cu-Mg: P. Liang et al., CALPHAD 22 (1998) 527-544, parameters
    of C.A. Coughanowr et al., Z. Metallkde. 82 (1991) 574-581, parameter set
    3 (TDB file by M. Palumbo, T. Abe and K. Hashimoto, NIMS 2008). Sublattice
    phases, including the Laves phase (CU,MG)2(CU,MG)1 with an internal degree
    of freedom and wildcard parameters.
  - cfe_broshe.tdb, Fe-C: B. Hallstedt et al., CALPHAD 34 (2010) 129-133, with
    the equation of state of E. Brosh et al., CALPHAD 31 (2007) 173-185, and
    A.T. Dinsdale, CALPHAD 15 (1991) 317-425. Interstitial and magnetic
    phases, carbides and pressure-dependent functions.

Tolerances: same stable phases; phase fractions and compositions within
1e-3; invariant temperatures within 0.5 K; Gibbs energies at the same site
fractions and the same gas constant within 1e-6 J/mol.

Needs the optional dependency: pip install -e ".[validation]". Run with
pytest -m validation. The report is regenerated with
python -m phase_diagram_explorer.validation.
"""
import numpy as np
import pytest

pycalphad = pytest.importorskip(
    "pycalphad", reason='pycalphad is not installed; install it with: pip install -e ".[validation]"'
)

from phase_diagram_explorer.builder import build_system  # noqa: E402
from phase_diagram_explorer.validation import (  # noqa: E402
    CASES,
    PYCALPHAD_GAS_CONSTANT,
    PycalphadSystem,
    compare_grid,
    compare_invariants,
    gap_eutectic_temperature,
    gibbs_differences,
    our_equilibrium,
    pycalphad_binodal,
)

pytestmark = pytest.mark.validation

PHASE_FRACTION_TOLERANCE = 1e-3
COMPOSITION_TOLERANCE = 1e-3
INVARIANT_TOLERANCE_K = 0.5
GIBBS_TOLERANCE = 1e-6  # J/mol
BINODAL_TOLERANCE = 1e-3
EXPECTED_INVARIANTS = {
    "Ag-Cu (provisional)": [("FCC_AG", "LIQUID", "FCC_CU")],
    "Al-Cu (provisional)": [("FCC_AL", "LIQUID", "AL2CU")],
    "Regular solution": [],
    "Gap eutectic": [("ALPHA#1", "LIQUID", "ALPHA#2")],
    "Interstitial (synthetic)": [("BCC_A2", "FCC_A1", "GRAPHITE"), ("FCC_A1", "LIQUID", "GRAPHITE")],
    "Magnetic (synthetic)": [("BCC_A2", "FCC_A1", "LIQUID")],
    "Pb-Sn": [("FCC_A1", "LIQUID", "BCT_A5")],
    "Cu-Mg": [("CUMG2", "LIQUID", "HCP_A3"), ("CU2MG", "LIQUID", "CUMG2"), ("FCC_A1", "LIQUID", "CU2MG")],
    "Fe-C": [("BCC_A2", "FCC_A1", "GRAPHITE"), ("FCC_A1", "LIQUID", "GRAPHITE"), ("BCC_A2", "FCC_A1", "LIQUID")],
}


@pytest.fixture(scope="module", params=list(CASES), ids=lambda name: CASES[name].file_name)
def case(request):
    case = CASES[request.param]
    if not case.path.exists():
        pytest.skip(f"{case.path} is not installed")
    return case


@pytest.fixture(scope="module")
def grid(case):
    return compare_grid(case)


def test_thirty_points_per_system(grid):
    assert len(grid) == 30


def test_same_stable_phases(grid):
    differing = [(c.T, c.x, c.ours, c.pycalphad) for c in grid if not c.same_phases]
    assert differing == []


def test_phase_fractions_agree(grid):
    assert max(c.fraction_error for c in grid) < PHASE_FRACTION_TOLERANCE


def test_phase_compositions_agree(grid):
    assert max(c.composition_error for c in grid) < COMPOSITION_TOLERANCE


def test_grid_covers_two_phase_fields(grid):
    assert any(len(c.ours) == 2 for c in grid)


def test_gibbs_energies_agree_at_the_same_gas_constant(case):
    """Every phase at random site fractions, both engines with R = 8.3145:
    the sublattice, magnetic and expression models must agree exactly."""
    differences = gibbs_differences(case)
    assert max(differences.values()) < GIBBS_TOLERANCE, differences


def test_invariant_temperatures_agree(case):
    comparisons = compare_invariants(case)
    assert [item.reaction.phases for item in comparisons] == EXPECTED_INVARIANTS[case.name]
    for item in comparisons:
        assert item.difference < INVARIANT_TOLERANCE_K, (item.reaction, item.pycalphad_temperature)


def test_gap_eutectic_matches_the_analytical_temperature():
    definition = CASES["Gap eutectic"].definition()
    (item,) = compare_invariants(CASES["Gap eutectic"])
    assert item.reaction.temperature == pytest.approx(gap_eutectic_temperature(definition.gas_constant), abs=0.01)


def test_default_gas_constant_of_tdb_systems():
    assert CASES["Fe-C"].definition().gas_constant == 8.31451
    assert CASES["Ag-Cu (provisional)"].definition().gas_constant == 8.314462618
    assert PYCALPHAD_GAS_CONSTANT == float(pycalphad.variables.R)


@pytest.fixture(scope="module")
def regular_solution():
    case = CASES["Regular solution"]
    definition = case.definition()
    return case, build_system(definition), PycalphadSystem(case.path, definition)


@pytest.mark.parametrize("T", [700.0, 800.0, 1000.0, 1150.0])
def test_miscibility_gap_binodal(regular_solution, T):
    _, system, reference = regular_solution
    x1, x2 = pycalphad_binodal(reference, T)
    ours = our_equilibrium(system, T, 0.5)
    assert [state.phase for state in ours] == ["ALPHA", "ALPHA"]
    assert ours[0].x == pytest.approx(x1, abs=BINODAL_TOLERANCE)
    assert ours[1].x == pytest.approx(x2, abs=BINODAL_TOLERANCE)


def test_traced_binodal(regular_solution, traced_diagram):
    """The binodal as drawn (traced boundaries) agrees as well."""
    case, _, reference = regular_solution
    traced = traced_diagram(case.path)
    (gap,) = [field for field in traced.regions if field.is_two_phase]
    for T in (700.0, 900.0, 1100.0):
        x1, x2 = pycalphad_binodal(reference, T)
        assert np.interp(T, gap.T, gap.x_min) == pytest.approx(x1, abs=BINODAL_TOLERANCE)
        assert np.interp(T, gap.T, gap.x_max) == pytest.approx(x2, abs=BINODAL_TOLERANCE)
