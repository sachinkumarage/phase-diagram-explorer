"""Physics regression tests for the Ag-Cu system as rendered by the Streamlit app.

These drive the real app through Streamlit's AppTest harness and check what a
user sees against the assessed Ag-Cu diagram. They were introduced as strict
xfails in 0.1.1, when the app used placeholder Gibbs curves instead of the
system JSON (see docs/audit_v0.1.md), and pass since 0.1.2.

Composition is the mole fraction of the second element listed in the system
JSON (SystemDefinition.dependent_element). For ag_cu.json that is Cu, so
x = x(Cu); the app shows it in at% by default.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name
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

# Assessed solid solubilities at the eutectic temperature
AG_RICH_FCC_X_CU = 0.141
AG_RICH_FCC_X_TOLERANCE = 0.015
CU_RICH_FCC_X_CU = 0.950
CU_RICH_FCC_X_TOLERANCE = 0.010

AL_CU_EUTECTIC_K = 821.0
AL_CU_EUTECTIC_TOLERANCE_K = 3.0
AL_CU_EUTECTIC_LIQUID_X_CU = 0.173
AL_CU_EUTECTIC_LIQUID_X_TOLERANCE = 0.010
AL_CU_FCC_AL_MAX_X_CU = 0.0248
AL_CU_FCC_AL_MAX_X_TOLERANCE = 0.003

AL_CU_DATA_PENDING = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="provisional Al-Cu data; fixed by assessed data",
)

SOLVUS_DATA_PENDING = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="solvus wrong with simplified parameters; fixed by assessed data",
)

T_RANGE_LABEL = "Temperature range (K)"
X_LABEL = "Composition x(Cu) (at%)"
N_T_LABEL = "Temperature points"
T_SELECTED_LABEL = "Temperature (K) for equilibrium and Gibbs curves"

def _slider(at: AppTest, label: str):
    matches = [slider for slider in at.slider if slider.label == label]
    assert len(matches) == 1, f"expected one slider labelled {label!r}"
    return matches[0]


def _run_app(T_range: tuple[float, float], n_T: int | None = None, system: str = "ag_cu") -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.selectbox[0].set_value(system)
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
    _slider(at, X_LABEL).set_value(100.0 * x)
    at.run()
    assert not at.exception

    markdown = [element.value for element in at.markdown]
    stable = markdown[markdown.index("**Stable phases**") + 1]
    return [phase.strip() for phase in stable.split(",")]


def _invariant_markers(at: AppTest) -> list[tuple[float, float, str]]:
    """(x, T, label) of each invariant marker drawn on the phase diagram, with
    x converted from the app's default at% display to mole fraction."""
    figure = json.loads(at.get("plotly_chart")[0].proto.spec)
    for trace in figure["data"]:
        if trace.get("name") == "invariant reactions":
            return [(x / 100.0, T, text) for x, T, text in zip(trace["x"], trace["y"], trace["text"])]
    return []


def _eutectic_from_table(system: str = "ag_cu") -> tuple[float, dict[str, float]]:
    """(T in K, {phase: x(Cu)}) of the single eutectic in the app's invariant
    reaction table (K and at% by default)."""
    at = _run_app(T_range=(500.0, 1450.0), system=system)
    table = at.table[-1].value
    eutectics = table[table["type"] == "eutectic"]
    assert len(eutectics) == 1

    row = eutectics.iloc[0]
    compositions = {}
    for entry in row["x(Cu) (at%)"].split(", "):
        phase, at_percent = entry.rsplit(" ", 1)
        compositions[phase] = float(at_percent) / 100.0
    return float(row["Temperature (K)"]), compositions


def _eutectic_solid_compositions() -> tuple[float, float]:
    """x(Cu) of the Ag-rich and Cu-rich solid phases at the eutectic."""
    _, compositions = _eutectic_from_table()
    solids = [x for phase, x in compositions.items() if "liquid" not in base_phase_name(phase).lower()]
    assert len(solids) == 2
    return min(solids), max(solids)


def test_ag_cu_json_lists_ag_then_cu():
    """Precondition for the tests below: the app's x axis is x(B) = x(Cu)."""
    definition = load_system(AG_CU_PATH)
    assert [element.symbol for element in definition.elements] == ["Ag", "Cu"]


@pytest.mark.slow
def test_ag_melts_at_reference_temperature():
    below = _stable_phases_at(AG_MELTING_K - MELTING_TOLERANCE_K, x=0.0)
    above = _stable_phases_at(AG_MELTING_K + MELTING_TOLERANCE_K, x=0.0)
    assert "LIQUID" not in below
    assert above == ["LIQUID"]


@pytest.mark.slow
def test_cu_melts_at_reference_temperature():
    below = _stable_phases_at(CU_MELTING_K - MELTING_TOLERANCE_K, x=1.0)
    above = _stable_phases_at(CU_MELTING_K + MELTING_TOLERANCE_K, x=1.0)
    assert "LIQUID" not in below
    assert above == ["LIQUID"]


@pytest.mark.slow
def test_eutectic_temperature_and_composition():
    at = _run_app(T_range=(1000.0, 1100.0), n_T=300)

    markers = [marker for marker in _invariant_markers(at) if marker[2].startswith("Eutectic")]
    assert len(markers) == 1
    x_liquid, temperature, _ = markers[0]
    assert temperature == pytest.approx(EUTECTIC_K, abs=EUTECTIC_TOLERANCE_K)
    assert x_liquid == pytest.approx(EUTECTIC_X_CU, abs=EUTECTIC_X_TOLERANCE)


@pytest.mark.slow
def test_ag_rich_terminal_phase_is_on_the_ag_side():
    assert _stable_phases_at(800.0, x=0.0) == ["FCC_AG"]
    assert _stable_phases_at(800.0, x=1.0) == ["FCC_CU"]


@pytest.mark.slow
@SOLVUS_DATA_PENDING
def test_max_cu_solubility_in_ag_rich_fcc_at_eutectic():
    ag_rich, _ = _eutectic_solid_compositions()
    assert ag_rich == pytest.approx(AG_RICH_FCC_X_CU, abs=AG_RICH_FCC_X_TOLERANCE)


@pytest.mark.slow
@SOLVUS_DATA_PENDING
def test_cu_rich_fcc_composition_at_eutectic():
    _, cu_rich = _eutectic_solid_compositions()
    assert cu_rich == pytest.approx(CU_RICH_FCC_X_CU, abs=CU_RICH_FCC_X_TOLERANCE)


@pytest.mark.slow
@AL_CU_DATA_PENDING
def test_al_cu_eutectic_temperature():
    T, _ = _eutectic_from_table("al_cu")
    assert T == pytest.approx(AL_CU_EUTECTIC_K, abs=AL_CU_EUTECTIC_TOLERANCE_K)


@pytest.mark.slow
@AL_CU_DATA_PENDING
def test_al_cu_eutectic_liquid_composition():
    _, compositions = _eutectic_from_table("al_cu")
    assert compositions["LIQUID"] == pytest.approx(
        AL_CU_EUTECTIC_LIQUID_X_CU, abs=AL_CU_EUTECTIC_LIQUID_X_TOLERANCE
    )


@pytest.mark.slow
@AL_CU_DATA_PENDING
def test_al_cu_max_cu_solubility_in_fcc_al_at_eutectic():
    _, compositions = _eutectic_from_table("al_cu")
    assert compositions["FCC_AL"] == pytest.approx(AL_CU_FCC_AL_MAX_X_CU, abs=AL_CU_FCC_AL_MAX_X_TOLERANCE)
