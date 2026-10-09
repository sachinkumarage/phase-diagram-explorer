import json
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.builder import build_system, is_computable
from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants_over_range
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

SYSTEMS_DIR = Path(__file__).resolve().parents[1] / "data" / "systems"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _write_reordered(tmp_path: Path, name: str) -> Path:
    data = json.loads((SYSTEMS_DIR / name).read_text())
    data["phases"] = list(reversed(data["phases"]))
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_dependent_element_is_second_listed_element():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    assert definition.base_element.symbol == "Ag"
    assert definition.dependent_element.symbol == "Cu"


def test_end_members_are_mapped_by_element_symbol():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    system = build_system(definition)

    fcc_ag = system["FCC_AG"]
    assert isinstance(fcc_ag, SolutionPhase)
    # x = 0 is pure Ag, x = 1 is pure Cu
    assert fcc_ag.molar_gibbs(800.0, 0.0) == pytest.approx(definition.phases[1].end_members["Ag"].a)
    assert fcc_ag.molar_gibbs(800.0, 1.0) == pytest.approx(definition.phases[1].end_members["Cu"].a)


@pytest.mark.parametrize("name", ["ag_cu.json", "al_cu.json"])
def test_reordering_phases_in_json_gives_identical_diagram(tmp_path, name):
    original = build_system(load_system(SYSTEMS_DIR / name))
    reordered = build_system(load_system(_write_reordered(tmp_path, name)))
    assert list(reordered) != list(original)

    kwargs = dict(T_range=(600.0, 1400.0), n_T=41, n_x=101, n_points=501)
    diagram_original = compute_diagram(original, **kwargs)
    diagram_reordered = compute_diagram(reordered, **kwargs)

    assert np.array_equal(diagram_original.phase_labels, diagram_reordered.phase_labels)
    for fractions_original, fractions_reordered in zip(
        diagram_original.phase_fractions.flat, diagram_reordered.phase_fractions.flat
    ):
        assert fractions_reordered == pytest.approx(fractions_original)


def test_reordering_phases_in_json_gives_identical_invariants(tmp_path):
    original = build_system(load_system(SYSTEMS_DIR / "ag_cu.json"))
    reordered = build_system(load_system(_write_reordered(tmp_path, "ag_cu.json")))

    reactions_original = detect_invariants_over_range(original, (1000.0, 1100.0))
    reactions_reordered = detect_invariants_over_range(reordered, (1000.0, 1100.0))

    assert [(r.type, r.temperature, r.phases) for r in reactions_reordered] == [
        (r.type, r.temperature, r.phases) for r in reactions_original
    ]


def test_al2cu_is_placed_at_its_site_ratio_composition():
    system = build_system(load_system(SYSTEMS_DIR / "al_cu.json"))
    al2cu = system["AL2CU"]

    assert isinstance(al2cu, StoichiometricPhase)
    assert al2cu.composition == pytest.approx(1.0 / 3.0)

    # Tie lines on both sides of the compound end at x(Cu) = 1/3 exactly.
    for x_overall in (0.30, 0.36):
        result = compute_equilibrium(system, 700.0, x_overall, n_points=501)
        assert "AL2CU" in result.stable_phases
        assert result.phase_compositions["AL2CU"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("n_points", [500, 501, 1000])
def test_al2cu_is_single_phase_at_exactly_one_third(n_points):
    """The compound is a hull point at its exact composition, whatever the grid."""
    system = build_system(load_system(SYSTEMS_DIR / "al_cu.json"))
    result = compute_equilibrium(system, 700.0, 1.0 / 3.0, n_points=n_points)

    assert result.stable_phases == ["AL2CU"]
    assert result.phase_fractions["AL2CU"] == pytest.approx(1.0, abs=1e-9)
    assert result.phase_compositions["AL2CU"] == 1.0 / 3.0
    assert result.total_gibbs == pytest.approx(system["AL2CU"].molar_gibbs(700.0))


@pytest.mark.slow
def test_ag_cu_invariants_cover_the_full_system_range():
    definition = load_system(SYSTEMS_DIR / "ag_cu.json")
    reactions = detect_invariants_over_range(build_system(definition), definition.t_range_k)
    eutectics = [r for r in reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    assert eutectics[0].temperature == pytest.approx(1052.0, abs=5.0)


def test_ag_cu_default_range_includes_both_melting_points():
    T_min, T_max = load_system(SYSTEMS_DIR / "ag_cu.json").t_range_k
    assert T_min < 1052.0
    assert T_max > 1357.77


def test_phase_without_gibbs_data_is_rejected():
    with pytest.raises(ValueError, match="no end_members"):
        build_system(load_system(FIXTURES_DIR / "example.json"))


def test_is_computable():
    assert is_computable(load_system(SYSTEMS_DIR / "ag_cu.json"))
    assert is_computable(load_system(SYSTEMS_DIR / "al_cu.json"))
    assert not is_computable(load_system(FIXTURES_DIR / "example.json"))


def test_shipped_systems_all_have_complete_gibbs_data():
    paths = sorted(SYSTEMS_DIR.glob("*.json"))
    assert paths
    assert all(is_computable(load_system(path)) for path in paths)
