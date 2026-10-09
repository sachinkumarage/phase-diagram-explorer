"""Every system that existed in 0.2.0 gives identical results with the
sublattice (CEF) engine: equilibria at 63 (T, x) points per system to 1e-10,
and invariant reactions to the root-finding tolerance.

The reference values in tests/fixtures/reference_equilibria_0.2.0.json were
computed with version 0.2.0 before the solution and stoichiometric models
became special cases of the sublattice model.
"""
import json
from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.invariants import ROOT_TOLERANCE_K, detect_invariants_over_range
from phase_diagram_explorer.models import load_system

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests" / "fixtures" / "reference_equilibria_0.2.0.json").read_text())
SYSTEMS = list(REFERENCE["systems"])
TOLERANCE = 1e-10


def test_reference_is_from_0_2_0():
    assert REFERENCE["generated_with"] == "0.2.0"
    assert len(SYSTEMS) == 8


@pytest.mark.parametrize("path", SYSTEMS)
def test_equilibria_are_identical(path):
    system = build_system(load_system(ROOT / path))
    for point in REFERENCE["systems"][path]["points"]:
        result = compute_equilibrium(system, point["T"], point["x"])
        context = (path, point["T"], point["x"])
        assert result.stable_phases == point["phases"], context
        for phase in result.stable_phases:
            assert result.phase_fractions[phase] == pytest.approx(point["fractions"][phase], abs=TOLERANCE), context
            assert result.phase_compositions[phase] == pytest.approx(point["compositions"][phase], abs=TOLERANCE), context
        assert result.total_gibbs == pytest.approx(point["total_gibbs"], rel=TOLERANCE, abs=TOLERANCE), context


@pytest.mark.slow
@pytest.mark.parametrize("path", SYSTEMS)
def test_invariants_are_identical(path):
    definition = load_system(ROOT / path)
    reactions = detect_invariants_over_range(build_system(definition), definition.t_range_k)
    expected = REFERENCE["systems"][path]["invariants"]
    assert len(reactions) == len(expected)
    for reaction, reference in zip(reactions, expected):
        assert reaction.type == reference["type"]
        assert list(reaction.phases) == reference["phases"]
        assert reaction.temperature == pytest.approx(reference["temperature"], abs=ROOT_TOLERANCE_K)
        for phase, x in reference["composition"].items():
            assert reaction.composition[phase] == pytest.approx(x, abs=1e-8)
