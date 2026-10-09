"""Cross-validation of the equilibrium engine against pycalphad, with both
engines reading the same TDB files.

Needs the optional dependency: pip install -e ".[validation]". Run with
pytest -m validation. The comparison report is regenerated with
python -m phase_diagram_explorer.validation.
"""
from pathlib import Path

import numpy as np
import pytest

pycalphad = pytest.importorskip(
    "pycalphad", reason='pycalphad is not installed; install it with: pip install -e ".[validation]"'
)

from phase_diagram_explorer.builder import build_system  # noqa: E402
from phase_diagram_explorer.invariants import detect_invariants_over_range  # noqa: E402
from phase_diagram_explorer.models import load_system  # noqa: E402
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT  # noqa: E402
from phase_diagram_explorer.validation import (  # noqa: E402
    FIXTURES,
    FIXTURES_DIR,
    PycalphadSystem,
    LITERATURE_T_RANGE_K,
    compare_grid,
    gap_eutectic_temperature,
    literature_tdb,
    our_equilibrium,
    pycalphad_binodal,
    pycalphad_phase_boundary_temperature,
)

pytestmark = pytest.mark.validation

PHASE_FRACTION_TOLERANCE = 1e-3
COMPOSITION_TOLERANCE = 1e-3
EUTECTIC_TOLERANCE_K = 0.5
BINODAL_TOLERANCE = 1e-3
# A literature assessment shipped with pycalphad's test databases
# (Pb-Sn, Ngai and Chang, CALPHAD 5 (1981) 267-276).
PBSN_TDB = literature_tdb()


@pytest.fixture(scope="module", params=list(FIXTURES.values()), ids=lambda name: Path(name).stem)
def grid_comparison(request):
    return compare_grid(FIXTURES_DIR / request.param)


def test_thirty_points_per_system(grid_comparison):
    assert len(grid_comparison) == 30


def test_same_stable_phases(grid_comparison):
    differing = [(c.T, c.x, c.ours, c.pycalphad) for c in grid_comparison if not c.same_phases]
    assert differing == []


def test_phase_fractions_agree(grid_comparison):
    assert max(c.fraction_error for c in grid_comparison) < PHASE_FRACTION_TOLERANCE


def test_phase_compositions_agree(grid_comparison):
    assert max(c.composition_error for c in grid_comparison) < COMPOSITION_TOLERANCE


def test_grid_covers_two_phase_fields(grid_comparison):
    assert any(len(c.ours) == 2 for c in grid_comparison)


def _eutectic_check(tdb_path: Path, x: float, T_low: float, T_high: float, **load_kwargs):
    definition = load_system(tdb_path, **load_kwargs)
    T_range = definition.t_range_k or (T_low, T_high)
    (reaction,) = detect_invariants_over_range(build_system(definition), T_range)
    T_reference = pycalphad_phase_boundary_temperature(
        PycalphadSystem(tdb_path, definition), x, "LIQUID", T_low, T_high, tolerance=0.01
    )
    return reaction, T_reference


def test_ag_cu_eutectic_temperature():
    """Our root-found eutectic vs the temperature at which pycalphad's
    LIQUID appears on heating inside the eutectic line (x(Cu) = 0.4)."""
    reaction, T_reference = _eutectic_check(FIXTURES_DIR / FIXTURES["Ag-Cu (provisional)"], 0.4, 1000.0, 1100.0)
    assert set(reaction.phases) == {"FCC_AG", "LIQUID", "FCC_CU"}
    assert reaction.temperature == pytest.approx(T_reference, abs=EUTECTIC_TOLERANCE_K)


def test_gap_eutectic_temperature():
    reaction, T_reference = _eutectic_check(FIXTURES_DIR / FIXTURES["Gap eutectic"], 0.5, 600.0, 1000.0)
    assert reaction.phases == ("ALPHA#1", "LIQUID", "ALPHA#2")
    assert reaction.temperature == pytest.approx(T_reference, abs=EUTECTIC_TOLERANCE_K)
    # and with the analytical value of the symmetric system
    assert reaction.temperature == pytest.approx(gap_eutectic_temperature(GAS_CONSTANT), abs=0.01)


@pytest.fixture(scope="module")
def regular_solution():
    path = FIXTURES_DIR / FIXTURES["Regular solution (miscibility gap)"]
    definition = load_system(path)
    return definition, build_system(definition), PycalphadSystem(path, definition)


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
    definition, _, reference = regular_solution
    traced = traced_diagram(FIXTURES_DIR / FIXTURES["Regular solution (miscibility gap)"])
    (gap,) = [field for field in traced.regions if field.is_two_phase]
    for T in (700.0, 900.0, 1100.0):
        x1, x2 = pycalphad_binodal(reference, T)
        assert np.interp(T, gap.T, gap.x_min) == pytest.approx(x1, abs=BINODAL_TOLERANCE)
        assert np.interp(T, gap.T, gap.x_max) == pytest.approx(x2, abs=BINODAL_TOLERANCE)


# --- a literature TDB ---------------------------------------------------------


@pytest.fixture(scope="module")
def pbsn():
    if not PBSN_TDB.exists():
        pytest.skip("pycalphad's Pb-Sn test database is not installed")
    definition = load_system(PBSN_TDB, elements=["Pb", "Sn"])
    return definition, build_system(definition), PycalphadSystem(PBSN_TDB, definition)


def test_literature_tdb_gibbs_energies(pbsn):
    """Piecewise unary functions with T**7 and T**(-9) terms, nested
    FUNCTION references and Redlich-Kister parameters evaluate as in
    pycalphad. The only difference is pycalphad's gas constant (8.3145),
    which scales the ideal mixing term."""
    _, system, reference = pbsn
    R_pycalphad = float(pycalphad.variables.R)
    for phase in system:
        for T in (350.0, 500.0, 700.0, 1500.0):
            result = pycalphad.calculate(
                reference.database, reference.components, phase, T=T, P=101325.0, N=1.0, pdens=20
            )
            x = result.X.sel(component="SN").values.ravel()
            G = result.GM.values.ravel()
            inside = (x > 1e-6) & (x < 1 - 1e-6)
            x, G = x[inside], G[inside]
            mixing = T * (x * np.log(x) + (1 - x) * np.log(1 - x))
            ours = system[phase].molar_gibbs(T, x)
            assert np.max(np.abs(ours - (G + (GAS_CONSTANT - R_pycalphad) * mixing))) < 1e-6, (phase, T)


def test_literature_tdb_equilibria(pbsn):
    comparisons = compare_grid(PBSN_TDB, LITERATURE_T_RANGE_K, elements=["Pb", "Sn"])
    assert len(comparisons) == 30
    assert all(c.same_phases for c in comparisons)
    assert max(c.fraction_error for c in comparisons) < PHASE_FRACTION_TOLERANCE
    assert max(c.composition_error for c in comparisons) < COMPOSITION_TOLERANCE


def test_literature_tdb_eutectic(pbsn):
    reaction, T_reference = _eutectic_check(PBSN_TDB, 0.6, 400.0, 500.0, elements=["Pb", "Sn"])
    assert set(reaction.phases) == {"FCC_A1", "LIQUID", "BCT_A5"}
    assert reaction.temperature == pytest.approx(T_reference, abs=EUTECTIC_TOLERANCE_K)
