"""Point equilibrium from the precomputed equilibrium table (lookup.py),
which the app uses when a database cannot be downloaded. It must match
compute_equilibrium (which needs the database) to within 1e-4 in phase
compositions and fractions, with the same phases."""
from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.lookup import equilibrium_from_trace
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.precomputed import load

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-4

# 20 (T, x) points per system: inside the miscibility gap or two-phase solid
# region, around the eutectic, next to the pure ends, and in other fields.
POINTS = {
    "ag_cu": [
        (800.0, 0.5), (600.0, 0.3), (1000.0, 0.7),             # FCC_A1#1 + FCC_A1#2 (gap)
        (1051.9, 0.40), (1052.6, 0.40), (1052.0, 0.2), (1052.5, 0.9),  # near the eutectic (1052.28 K)
        (900.0, 0.001), (1200.0, 0.0005), (1234.0, 0.002),     # Ag end
        (900.0, 0.999), (1357.0, 0.9995), (1350.0, 0.998),     # Cu end
        (1100.0, 0.3), (1150.0, 0.6), (1300.0, 0.5),            # liquid, L + FCC
        (1200.0, 0.1), (1300.0, 0.85), (700.0, 0.05), (1440.0, 0.5),
    ],
    "pb_sn": [
        (400.0, 0.5), (350.0, 0.2), (450.0, 0.8),               # FCC_A1 + BCT_A5
        (455.2, 0.74), (455.6, 0.74), (455.0, 0.5), (455.8, 0.6),  # near the eutectic (455.39 K)
        (500.0, 0.001), (600.0, 0.0005), (320.0, 0.01),         # Pb end
        (480.0, 0.999), (504.9, 0.9995), (350.0, 0.998),        # Sn end
        (500.0, 0.5), (550.0, 0.3), (600.0, 0.6),               # liquid, L + FCC
        (480.0, 0.2), (490.0, 0.9), (330.0, 0.05), (640.0, 0.5),
    ],
}


@pytest.fixture(scope="module", params=list(POINTS))
def setup(request):
    name = request.param
    system = build_system(load_system(ROOT / "data" / "systems" / f"{name}.meta.json"))
    table = load(ROOT / "data" / "precomputed" / f"{name}.npz").table
    return name, system, table


def test_twenty_points_per_system(setup):
    name, _, _ = setup
    assert len(POINTS[name]) == 20


def test_lookup_matches_compute_equilibrium(setup):
    name, system, table = setup
    for T, x in POINTS[name]:
        expected = compute_equilibrium(system, T, x)
        found = equilibrium_from_trace(table, T, x)
        assert found.stable_phases == expected.stable_phases, (name, T, x)
        for phase in expected.stable_phases:
            assert found.phase_compositions[phase] == pytest.approx(expected.phase_compositions[phase], abs=TOLERANCE), (T, x, phase)
            assert found.phase_fractions[phase] == pytest.approx(expected.phase_fractions[phase], abs=TOLERANCE), (T, x, phase)


def test_points_cover_the_required_regions(setup):
    name, system, _ = setup
    phases = [tuple(compute_equilibrium(system, T, x).stable_phases) for T, x in POINTS[name]]
    assert sum(len(p) == 2 for p in phases) >= 8
    assert sum(len(p) == 1 for p in phases) >= 6
    if name == "ag_cu":
        assert ("FCC_A1#1", "FCC_A1#2") in phases


def test_lookup_outside_the_table_range_raises(setup):
    _, _, table = setup
    with pytest.raises(ValueError, match="outside the traced range"):
        equilibrium_from_trace(table, table.T_range[1] + 10.0, 0.5)


@pytest.mark.parametrize("name", list(POINTS))
def test_stored_gap_tie_lines_are_never_degenerate(name):
    """Regression test: a failed tangent refinement could fall back to the
    trivial solution of the gap equations (two nearby points near the
    spinodal), e.g. a 2e-6 wide FCC_A1#1 + FCC_A1#2 tie line at 504.25 K in
    Ag-Cu, where the real one spans 0.0018-0.9998."""
    precomputed = load(ROOT / "data" / "precomputed" / f"{name}.npz")
    for traced in (precomputed.traced, precomputed.table):
        for field in traced.fields:
            if field.is_two_phase and field.phases[0] == field.phases[1]:
                widths = [hi - lo for lo, hi in zip(field.x_min, field.x_max)]
                assert min(widths) > 0.5, (name, field.phases)
