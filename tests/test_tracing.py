"""Phase boundary tracing, root-found invariants and vector export.

Tests that trace a full system range are marked slow.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as PolygonPath
from scipy.optimize import brentq

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.export import export_figure, invariants_csv, publication_figure
from phase_diagram_explorer.invariants import (
    InvariantReaction,
    _deduplicate,
    detect_invariants,
    detect_invariants_over_range,
)
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT
from phase_diagram_explorer.tracing import trace_diagram

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = ROOT / "data" / "systems"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
SYSTEMS = [
    SYSTEMS_DIR / "ag_cu.json",
    SYSTEMS_DIR / "al_cu.json",
    FIXTURES_DIR / "regular_solution.json",
    FIXTURES_DIR / "gap_eutectic.json",
]


def _load(path: Path):
    definition = load_system(path)
    return definition, build_system(definition)


@pytest.fixture(params=SYSTEMS, ids=lambda path: path.stem)
def traced_system(request, traced_diagram):
    definition, system = _load(request.param)
    return definition, system, traced_diagram(request.param)


# --- invariants ------------------------------------------------------------


@pytest.mark.parametrize("name", ["ag_cu.json", "al_cu.json"])
def test_invariants_do_not_depend_on_composition_grid(name):
    _, system = _load(SYSTEMS_DIR / name)
    coarse = detect_invariants(compute_diagram(system, (800.0, 1100.0), n_T=31, n_x=101), system)
    fine = detect_invariants(compute_diagram(system, (800.0, 1100.0), n_T=31, n_x=1001), system)

    assert len(coarse) == len(fine) == 1
    assert fine[0].phases == coarse[0].phases
    assert fine[0].temperature == pytest.approx(coarse[0].temperature, abs=0.1)
    for phase, x in coarse[0].composition.items():
        assert fine[0].composition[phase] == pytest.approx(x, abs=1e-4)


@pytest.mark.parametrize("name", ["ag_cu.json", "al_cu.json"])
def test_invariants_do_not_depend_on_scan_levels(name):
    _, system = _load(SYSTEMS_DIR / name)
    coarse = detect_invariants_over_range(system, (800.0, 1100.0), T_step=25.0)
    fine = detect_invariants_over_range(system, (800.0, 1100.0), T_step=2.0)

    assert len(coarse) == len(fine) == 1
    assert fine[0].temperature == pytest.approx(coarse[0].temperature, abs=0.1)
    for phase, x in coarse[0].composition.items():
        assert fine[0].composition[phase] == pytest.approx(x, abs=1e-4)


def test_ag_cu_eutectic_is_root_found_to_within_0_1_K():
    """Bisect directly on whether LIQUID is stable at the eutectic
    composition and compare with the reported temperature."""
    _, system = _load(SYSTEMS_DIR / "ag_cu.json")
    (reaction,) = detect_invariants_over_range(system, (1000.0, 1100.0))
    x_liquid = reaction.composition["LIQUID"]

    def liquid_present(T: float) -> float:
        return 1.0 if "LIQUID" in compute_equilibrium(system, T, x_liquid).stable_phases else -1.0

    T_check = brentq(liquid_present, reaction.temperature - 2.0, reaction.temperature + 2.0, xtol=1e-3)
    assert reaction.temperature == pytest.approx(T_check, abs=0.1)


def test_invariant_reports_all_three_phase_compositions():
    _, system = _load(SYSTEMS_DIR / "ag_cu.json")
    (reaction,) = detect_invariants_over_range(system, (1000.0, 1100.0))

    assert set(reaction.composition) == {"FCC_AG", "LIQUID", "FCC_CU"}
    x = reaction.composition
    assert 0.0 < x["FCC_AG"] < x["LIQUID"] < x["FCC_CU"] < 1.0

    # all three phases share one tangent line at the invariant temperature
    below = compute_equilibrium(system, reaction.temperature - 0.01, x["LIQUID"])
    assert below.phase_compositions["FCC_AG"] == pytest.approx(x["FCC_AG"], abs=1e-4)
    assert below.phase_compositions["FCC_CU"] == pytest.approx(x["FCC_CU"], abs=1e-4)


def test_duplicate_reactions_are_merged():
    def reaction(T: float) -> InvariantReaction:
        return InvariantReaction(T, "eutectic", ("A", "L", "B"), {"A": 0.1, "L": 0.5, "B": 0.9})

    assert len(_deduplicate([reaction(1000.0), reaction(1000.3)])) == 1
    assert len(_deduplicate([reaction(1000.0), reaction(1001.0)])) == 2


# --- traced boundaries -----------------------------------------------------


@pytest.mark.slow
def test_boundaries_have_no_nan_gaps(traced_system):
    _, _, traced = traced_system
    assert traced.boundaries
    for boundary in traced.boundaries:
        assert np.all(np.isfinite(boundary.x)), boundary.name
        assert np.all(np.isfinite(boundary.T)), boundary.name
        assert np.all(np.diff(boundary.T) > 0), boundary.name


@pytest.mark.slow
def test_boundaries_meet_invariant_lines(traced_system):
    _, _, traced = traced_system
    for reaction in traced.invariants:
        compositions = list(reaction.composition.values())
        ends = [
            (field.x_min[k], field.x_max[k])
            for field in traced.fields if field.is_two_phase
            for k in (0, -1)
            if abs(field.T[k] - reaction.temperature) < 1e-3
        ]
        # two fields end above the invariant and one starts below it
        assert len(ends) >= 3
        for x_min, x_max in ends:
            for x in (x_min, x_max):
                assert min(abs(x - c) for c in compositions) < 1e-4


@pytest.mark.slow
def test_phase_regions_tile_the_diagram(traced_system):
    _, _, traced = traced_system
    T_min, T_max = traced.T_range
    total_area = T_max - T_min

    assert sum(region.area for region in traced.regions) == pytest.approx(total_area, rel=0.005)

    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(0.0, 1.0, 20000), rng.uniform(T_min, T_max, 20000)])
    containing = np.zeros(len(points), dtype=int)
    for region in traced.regions:
        x, T = region.polygon()
        containing += PolygonPath(np.column_stack([x, T])).contains_points(points)
    assert containing.max() <= 1, "phase-field polygons overlap"
    assert (containing == 1).mean() > 0.995


@pytest.mark.slow
def test_narrow_al_cu_liquid_field_is_traced(traced_diagram):
    """The Al-rich LIQUID + FCC_AL field is narrower than any display grid
    step and must still be traced."""
    traced = traced_diagram(SYSTEMS_DIR / "al_cu.json")
    narrow = [field for field in traced.regions if set(field.phases) == {"LIQUID", "FCC_AL"}]

    assert len(narrow) == 1
    widths = np.asarray(narrow[0].x_max) - np.asarray(narrow[0].x_min)
    assert widths.max() < 1.0 / 150
    assert narrow[0].T[0] == pytest.approx(traced.invariants[0].temperature, abs=1e-3)


@pytest.mark.slow
def test_levels_are_refined_near_invariants():
    definition, system = _load(SYSTEMS_DIR / "ag_cu.json")
    traced = trace_diagram(system, definition.t_range_k, n_levels=20)
    T_inv = traced.invariants[0].temperature

    near = traced.levels[np.abs(traced.levels - T_inv) < 0.01]
    assert len(near) >= 2
    assert np.min(np.diff(traced.levels)) < 0.01
    assert np.max(np.diff(traced.levels)) <= (definition.t_range_k[1] - definition.t_range_k[0]) / 19 + 1e-9


@pytest.mark.slow
def test_traced_binodal_matches_analytical_solution(traced_diagram):
    path = FIXTURES_DIR / "regular_solution.json"
    L0 = json.loads(path.read_text())["phases"][0]["interaction_parameters"][0]
    T_c = L0 / (2.0 * GAS_CONSTANT)
    traced = traced_diagram(path)
    (gap,) = [field for field in traced.regions if field.is_two_phase]

    for T in (700.0, 850.0, 1000.0, 1100.0, 1180.0):
        x_binodal = brentq(
            lambda x: np.log(x / (1.0 - x)) - L0 * (2.0 * x - 1.0) / (GAS_CONSTANT * T), 1e-12, 0.5 - 1e-9
        )
        assert np.interp(T, gap.T, gap.x_min) == pytest.approx(x_binodal, abs=1e-3)
        assert np.interp(T, gap.T, gap.x_max) == pytest.approx(1.0 - x_binodal, abs=1e-3)

    assert gap.T[-1] == pytest.approx(T_c, abs=1.0)


# --- export ----------------------------------------------------------------


@pytest.fixture
def ag_cu_traced(traced_diagram):
    return load_system(SYSTEMS_DIR / "ag_cu.json"), traced_diagram(SYSTEMS_DIR / "ag_cu.json")


@pytest.mark.slow
def test_svg_export_is_vector(ag_cu_traced):
    definition, traced = ag_cu_traced
    svg = export_figure(traced, definition.name, "svg", dependent_symbol="Cu", footnote="Data source: test")
    assert svg.lstrip().startswith(b"<?xml")
    assert b"<svg" in svg
    assert b"<image" not in svg


@pytest.mark.slow
def test_pdf_export_is_vector(ag_cu_traced):
    definition, traced = ag_cu_traced
    pdf = export_figure(traced, definition.name, "pdf", dependent_symbol="Cu", footnote="Data source: test")
    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf


@pytest.mark.slow
def test_export_figure_content(ag_cu_traced):
    definition, traced = ag_cu_traced
    figure = publication_figure(
        traced, definition.name, dependent_symbol="Cu", temperature_unit="°C", footnote="Data source: ag_cu.json"
    )
    (ax,) = figure.axes
    assert ax.get_xlabel() == "x(Cu) (at%)"
    assert ax.get_ylabel() == "Temperature (°C)"
    texts = [text.get_text() for text in ax.texts]
    T_inv = traced.invariants[0].temperature - 273.15
    assert f"{T_inv:.1f} °C" in texts
    assert {"LIQUID", "FCC_AG", "FCC_CU"} <= set(texts)
    assert any("Data source: ag_cu.json" in text.get_text() for text in figure.texts)
    assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)


@pytest.mark.slow
def test_export_rejects_raster_formats(ag_cu_traced):
    definition, traced = ag_cu_traced
    with pytest.raises(ValueError, match="unsupported export format"):
        export_figure(traced, definition.name, "png")


def test_invariants_csv_keeps_full_precision():
    _, system = _load(SYSTEMS_DIR / "ag_cu.json")
    (reaction,) = detect_invariants_over_range(system, (1000.0, 1100.0))
    lines = invariants_csv([reaction], "Cu").splitlines()

    assert lines[0] == "type,temperature_K,phase_1,x_Cu_1,phase_2,x_Cu_2,phase_3,x_Cu_3"
    fields = lines[1].split(",")
    assert fields[0] == "eutectic"
    assert float(fields[1]) == reaction.temperature
    for k, phase in enumerate(reaction.phases):
        assert fields[2 + 2 * k] == phase
        assert float(fields[3 + 2 * k]) == reaction.composition[phase]
