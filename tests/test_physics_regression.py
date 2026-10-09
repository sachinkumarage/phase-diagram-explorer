"""Physics regression tests for the Ag-Cu system as rendered by the Streamlit app.

tests/test_systems.py validates a model built directly from the coefficients in
data/systems/ag_cu.json. The app does not use those coefficients: it builds
placeholder Gibbs curves from phase names and list positions (see
app._build_computable_system), so what the user sees disagrees with the
assessed Ag-Cu diagram. These tests drive the real app through Streamlit's
AppTest harness and are marked xfail(strict=True) until the app is fixed; once
it is, they will XPASS and the markers must be removed. See docs/audit_v0.1.md.

Composition is the mole fraction of the second element listed in the system
JSON ("mole fraction B"). For ag_cu.json that is Cu, so x = x(Cu).
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase_diagram_explorer.models import load_system

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "phase_diagram_explorer" / "app.py"
AG_CU_PATH = ROOT / "data" / "systems" / "ag_cu.json"

AG_MELTING_K = 1234.93
CU_MELTING_K = 1357.77
MELTING_TOLERANCE_K = 2.0

EUTECTIC_K = 1052.0
EUTECTIC_TOLERANCE_K = 5.0
EUTECTIC_X_CU = 0.399
EUTECTIC_X_TOLERANCE = 0.015

T_RANGE_LABEL = "Temperature range (K)"
X_LABEL = "Composition (mole fraction B)"
N_T_LABEL = "Temperature points"
T_SELECTED_LABEL = "Temperature (K) for equilibrium and Gibbs curves"

KNOWN_APP_BUG = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="app ignores ag_cu.json coefficients and maps FCC phases by list position (docs/audit_v0.1.md)",
)


def _slider(at: AppTest, label: str):
    matches = [slider for slider in at.slider if slider.label == label]
    assert len(matches) == 1, f"expected one slider labelled {label!r}"
    return matches[0]


def _run_app(T_range: tuple[float, float], n_T: int | None = None) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.selectbox[0].set_value("ag_cu")
    _slider(at, T_RANGE_LABEL).set_value(T_range)
    if n_T is not None:
        _slider(at, N_T_LABEL).set_value(n_T)
    at.run()
    assert not at.exception
    return at


def _stable_phases_at(T: float, x: float) -> list[str]:
    """Stable phases shown by the app's equilibrium readout at (T, x)."""
    at = _run_app(T_range=(T - 100.0, T + 100.0))
    # The equilibrium temperature slider is recreated when T_range changes,
    # so it can only be set on a second run.
    _slider(at, T_SELECTED_LABEL).set_value(T)
    _slider(at, X_LABEL).set_value(x)
    at.run()
    assert not at.exception

    markdown = [element.value for element in at.markdown]
    stable = markdown[markdown.index("**Stable phases**") + 1]
    return [phase.strip() for phase in stable.split(",")]


def _invariant_markers(at: AppTest) -> list[tuple[float, float, str]]:
    """(x, T, label) of each invariant marker drawn on the phase diagram."""
    figure = json.loads(at.get("plotly_chart")[0].proto.spec)
    for trace in figure["data"]:
        if trace.get("name") == "invariant reactions":
            return list(zip(trace["x"], trace["y"], trace["text"]))
    return []


def test_ag_cu_json_lists_ag_then_cu():
    """Precondition for the tests below: the app's x axis is x(B) = x(Cu)."""
    definition = load_system(AG_CU_PATH)
    assert [element.symbol for element in definition.elements] == ["Ag", "Cu"]


@KNOWN_APP_BUG
def test_ag_melts_at_reference_temperature():
    below = _stable_phases_at(AG_MELTING_K - MELTING_TOLERANCE_K, x=0.0)
    above = _stable_phases_at(AG_MELTING_K + MELTING_TOLERANCE_K, x=0.0)
    assert "LIQUID" not in below
    assert above == ["LIQUID"]


@KNOWN_APP_BUG
def test_cu_melts_at_reference_temperature():
    below = _stable_phases_at(CU_MELTING_K - MELTING_TOLERANCE_K, x=1.0)
    above = _stable_phases_at(CU_MELTING_K + MELTING_TOLERANCE_K, x=1.0)
    assert "LIQUID" not in below
    assert above == ["LIQUID"]


@KNOWN_APP_BUG
def test_eutectic_temperature_and_composition():
    at = _run_app(T_range=(1000.0, 1100.0), n_T=300)

    markers = [marker for marker in _invariant_markers(at) if marker[2].startswith("Eutectic")]
    assert len(markers) == 1
    x_liquid, temperature, _ = markers[0]
    assert temperature == pytest.approx(EUTECTIC_K, abs=EUTECTIC_TOLERANCE_K)
    assert x_liquid == pytest.approx(EUTECTIC_X_CU, abs=EUTECTIC_X_TOLERANCE)


@KNOWN_APP_BUG
def test_ag_rich_terminal_phase_is_on_the_ag_side():
    assert _stable_phases_at(800.0, x=0.0) == ["FCC_AG"]
    assert _stable_phases_at(800.0, x=1.0) == ["FCC_CU"]
