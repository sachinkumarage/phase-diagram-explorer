"""Precomputed diagrams for the app (data/precomputed): present, up to date
and identical to a live computation."""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.models import default_systems_dir
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
    assert [path.name for path in APP_SYSTEMS] == ["ag_cu.json", "al_cu.json"]
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
    assert precomputed.metadata["gas_constant"] == 8.314462618
    assert precomputed.metadata["t_range_k"] == [500.0, 1450.0]


def test_round_trip(tmp_path):
    original = compute(FIXTURES_DIR / "regular_solution.json")
    loaded = load(save(original, tmp_path / "regular_solution.npz"))
    assert loaded.metadata == original.metadata
    assert loaded.invariants == original.invariants
    assert loaded.default_equilibrium == original.default_equilibrium
    assert [f.phases for f in loaded.traced.fields] == [f.phases for f in original.traced.fields]
    for a, b in zip(loaded.traced.fields, original.traced.fields):
        assert (a.T, a.x_min, a.x_max) == (list(b.T), list(b.x_min), list(b.x_max))
    assert np.array_equal(loaded.traced.levels, original.traced.levels)


def test_changed_data_or_version_makes_a_file_stale(tmp_path, monkeypatch):
    system = tmp_path / "regular_solution.json"
    shutil.copy(FIXTURES_DIR / "regular_solution.json", system)
    save(compute(system), precomputed_path(system, tmp_path))
    assert status(system, tmp_path)[1] is None

    data = json.loads(system.read_text())
    data["phases"][0]["interaction_parameters"] = [20001.0]
    system.write_text(json.dumps(data))
    precomputed, reason = status(system, tmp_path)
    assert precomputed is None and "source_sha256" in reason

    system.write_text((FIXTURES_DIR / "regular_solution.json").read_text())
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
