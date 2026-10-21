"""Fe-C against pycalphad on the same file (cfe_broshe.tdb from pycalphad
0.11.2, as fetched by scripts/fetch_tdb.py), in both diagram modes with the
same phases: invariant temperatures within 0.01 K and phase compositions
within 1e-4 in mole fraction."""
import pytest

pytest.importorskip("pycalphad", reason='pycalphad is not installed; install it with: pip install -e ".[validation]"')

from phase_diagram_explorer import fe_c  # noqa: E402

pytestmark = pytest.mark.validation

T_TOLERANCE_K = 0.01
X_TOLERANCE = 1e-4


@pytest.fixture(scope="module", params=fe_c.MODES)
def comparison(request):
    mode = request.param
    reference = fe_c.pycalphad_reference(mode)
    return mode, [(reaction, *fe_c.pycalphad_invariant(reference, reaction)) for reaction in fe_c.invariants(mode)]


def test_same_reactions(comparison):
    mode, rows = comparison
    assert sorted(reaction.type for reaction, _, _ in rows) == ["eutectic", "eutectoid", "peritectic"]


def test_invariant_temperatures(comparison):
    _, rows = comparison
    for reaction, T_pycalphad, _ in rows:
        assert reaction.temperature == pytest.approx(T_pycalphad, abs=T_TOLERANCE_K), reaction


def test_invariant_compositions(comparison):
    _, rows = comparison
    for reaction, _, compositions in rows:
        for phase, x_pycalphad in compositions.items():
            assert fe_c.composition(reaction, phase) == pytest.approx(x_pycalphad, abs=X_TOLERANCE), (reaction, phase)
