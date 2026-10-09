"""Session-scoped fixtures sharing expensive computations between tests."""
from pathlib import Path

import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.invariants import detect_invariants
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.tracing import trace_diagram

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = ROOT / "data" / "systems"


@pytest.fixture(scope="session")
def traced_diagram():
    """traced_diagram(path) -> the TracedDiagram of a system file over its
    full t_range_k, traced once per session."""
    cache = {}

    def get(path):
        path = Path(path).resolve()
        if path not in cache:
            definition = load_system(path)
            cache[path] = trace_diagram(build_system(definition), definition.t_range_k)
        return cache[path]

    return get


@pytest.fixture(scope="session")
def ag_cu_eutectic_scan():
    """(system, reactions) for Ag-Cu: invariants detected on a 1000-1150 K
    diagram (151 x 201, 501 Gibbs points)."""
    system = build_system(load_system(SYSTEMS_DIR / "ag_cu.json"))
    diagram = compute_diagram(system, T_range=(1000.0, 1150.0), n_T=151, n_x=201, n_points=501)
    return system, detect_invariants(diagram, system, n_points=501)
