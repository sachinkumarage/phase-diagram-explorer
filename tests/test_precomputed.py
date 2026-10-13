"""Precomputed diagrams for the app (data/precomputed): present, up to date
and identical to a live computation."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.models import default_systems_dir, load_system
from phase_diagram_explorer.precomputed import (
    app_systems,
    compute,
    default_precomputed_dir,
    load,
    precomputed_path,
    save,
    status,
)

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = ROOT / "data" / "systems"
PRECOMPUTED_DIR = ROOT / "data" / "precomputed"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
MAX_BYTES = 1_000_000
APP_SYSTEMS = app_systems(SYSTEMS_DIR)
REGENERATE = "regenerate with: python scripts/precompute.py"


def test_app_systems_are_the_shipped_systems():
    assert [path.name for path in APP_SYSTEMS] == ["ag_cu.meta.json", "al_cu.json", "pb_sn.meta.json"]
    assert default_systems_dir() == SYSTEMS_DIR
    assert default_precomputed_dir() == PRECOMPUTED_DIR


@pytest.mark.parametrize("system_path", APP_SYSTEMS, ids=lambda path: path.stem)
def test_every_app_system_has_an_up_to_date_precomputed_file(system_path):
    """Fails when system data or the package version changed without
    regenerating the precomputed data."""
    precomputed, reason = status(system_path, PRECOMPUTED_DIR)
    assert reason is None, f"{reason}; {REGENERATE}"
    assert precomputed.metadata["system"] == system_path.name


@pytest.mark.parametrize("system_path", APP_SYSTEMS, ids=lambda path: path.stem)
def test_precomputed_files_are_small(system_path):
    assert precomputed_path(system_path, PRECOMPUTED_DIR).stat().st_size < MAX_BYTES


def test_precomputed_metadata():
    precomputed = load(PRECOMPUTED_DIR / "ag_cu.npz")
    assert set(precomputed.metadata) >= {
        "source_sha256", "package_version", "gas_constant", "t_range_k", "computation_time_s", "n_levels", "n_points",
    }
    assert precomputed.metadata["gas_constant"] == 8.31451  # TDB default; Al-Cu (JSON) keeps 8.314462618
    assert precomputed.metadata["t_range_k"] == [500.0, 1450.0]


def _short_range_fixture(tmp_path) -> Path:
    """The regular solution over 1000-1100 K, so that its equilibrium table
    (levels 0.25 K apart) is quick to compute."""
    data = json.loads((FIXTURES_DIR / "regular_solution.json").read_text())
    data["t_range_k"] = [1000.0, 1100.0]
    path = tmp_path / "regular_solution.json"
    path.write_text(json.dumps(data))
    return path


def test_round_trip(tmp_path):
    original = compute(_short_range_fixture(tmp_path))
    loaded = load(save(original, tmp_path / "regular_solution.npz"))
    assert loaded.metadata == original.metadata
    assert loaded.invariants == original.invariants
    assert loaded.default_equilibrium == original.default_equilibrium
    assert [f.phases for f in loaded.traced.fields] == [f.phases for f in original.traced.fields]
    for a, b in zip(loaded.traced.fields, original.traced.fields):
        assert (a.T, a.x_min, a.x_max) == (list(b.T), list(b.x_min), list(b.x_max))
    assert np.array_equal(loaded.traced.levels, original.traced.levels)
    assert np.array_equal(loaded.table.levels, original.table.levels)
    assert loaded.summary == original.summary


def test_changed_data_or_version_makes_a_file_stale(tmp_path, monkeypatch):
    system = _short_range_fixture(tmp_path)
    original_text = system.read_text()
    save(compute(system), precomputed_path(system, tmp_path))
    assert status(system, tmp_path)[1] is None

    data = json.loads(system.read_text())
    data["phases"][0]["interaction_parameters"] = [20001.0]
    system.write_text(json.dumps(data))
    precomputed, reason = status(system, tmp_path)
    assert precomputed is None and "source_sha256" in reason

    system.write_text(original_text)
    old = load(precomputed_path(system, tmp_path))
    save(replace(old, metadata={**old.metadata, "package_version": "0.0.0"}), precomputed_path(system, tmp_path))
    assert "package_version" in status(system, tmp_path)[1]


def test_missing_or_unreadable_files(tmp_path):
    system = FIXTURES_DIR / "regular_solution.json"
    assert "no precomputed file" in status(system, tmp_path)[1]
    precomputed_path(system, tmp_path).write_bytes(b"not an npz file")
    assert "could not be read" in status(system, tmp_path)[1]


@pytest.mark.slow
@pytest.mark.parametrize("system_path", APP_SYSTEMS, ids=lambda path: path.stem)
def test_precomputed_data_equals_a_live_computation(system_path):
    stored = load(precomputed_path(system_path, PRECOMPUTED_DIR))
    live = compute(system_path)
    assert stored.invariants == live.invariants
    assert stored.default_equilibrium == live.default_equilibrium
    assert [f.phases for f in stored.traced.fields] == [f.phases for f in live.traced.fields]
    for a, b in zip(stored.traced.fields, live.traced.fields):
        assert np.allclose([a.T, a.x_min, a.x_max], [b.T, b.x_min, b.x_max], rtol=0, atol=1e-12)


@pytest.mark.parametrize("system_path", APP_SYSTEMS, ids=lambda path: path.stem)
def test_files_hold_what_the_app_needs_without_the_database(system_path):
    precomputed = load(precomputed_path(system_path, PRECOMPUTED_DIR))
    assert precomputed.table is not None and precomputed.metadata["table_step_k"] == 0.25
    definition = precomputed.definition
    live = load_system(system_path)
    assert definition.name == live.name
    assert [e.model_dump() for e in definition.elements] == [e.model_dump() for e in live.elements]
    assert [p.name for p in definition.phases] == [p.name for p in live.phases]
    assert definition.citations == live.citations and definition.display == live.display
    assert definition.is_provisional == live.is_provisional


def test_no_thermodynamic_parameters_are_stored():
    """Precomputed files hold names, elements and results only: no Gibbs
    energy expressions or parameters from the (not redistributed) database."""
    import numpy as np

    for path in PRECOMPUTED_DIR.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            text = " ".join(str(data[key]) for key in data.files if data[key].dtype.kind == "U")
        for marker in ("GHSER", "PARAMETER", "FUNCTION", "*LN(T)", "raw_parameters", "end_members"):
            assert marker not in text, (path.name, marker)


def test_staleness_check_needs_no_database(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE_DIAGRAM_TDB_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("PHASE_DIAGRAM_DISABLE_DOWNLOADS", "1")
    for system_path in APP_SYSTEMS:
        precomputed, reason = status(system_path, PRECOMPUTED_DIR)
        assert reason is None, reason
        assert precomputed.definition.name
