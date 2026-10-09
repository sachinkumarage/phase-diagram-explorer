"""Tests of the Streamlit app's wiring: where its thermodynamics come from,
how invariant reactions relate to the displayed range, and display units."""
import ast
import base64
import json
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from phase_diagram_explorer import builder
from phase_diagram_explorer.units import CELSIUS_OFFSET

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "phase_diagram_explorer" / "app.py"

THERMO_CLASSES = {"PureElementGibbs", "SolutionPhase", "StoichiometricPhase"}
OUTSIDE_NOTE = "outside displayed temperature range"


def _slider(at: AppTest, label: str):
    matches = [slider for slider in at.slider if slider.label == label]
    assert len(matches) == 1, f"expected one slider labelled {label!r}"
    return matches[0]


def _run(system: str = "ag_cu", **radios) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.selectbox[0].set_value(system)
    for radio in at.radio:
        if radio.label in radios:
            radio.set_value(radios[radio.label])
    at.run()
    assert not at.exception
    return at


def _figure(at: AppTest, index: int = 0) -> dict:
    return json.loads(at.get("plotly_chart")[index].proto.spec)


def _trace(figure: dict, name: str) -> dict | None:
    return next((trace for trace in figure["data"] if trace.get("name") == name), None)


def _values(array) -> list:
    """Plotly serialises numpy arrays as {"dtype", "bdata"} typed arrays."""
    if isinstance(array, dict) and "bdata" in array:
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]).tolist()
    return list(array)


def _invariant_table(at: AppTest):
    return at.table[-1].value


def test_app_does_not_construct_gibbs_energy_models():
    tree = ast.parse(APP_PATH.read_text())
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & THERMO_CLASSES
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "phase_diagram_explorer.thermo.pure" not in imported_modules
    assert "phase_diagram_explorer.thermo.solution" not in imported_modules
    assigned = {
        target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert not assigned & {"MELTING_T", "FUSION_ENTHALPY", "BASE_OFFSET", "OFFSET_STEP"}


@pytest.fixture
def isolated_cache():
    """Keep systems built under a monkeypatched builder out of the shared
    st.cache_data cache used by other app tests."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_app_system_comes_from_build_system(monkeypatch, isolated_cache):
    """Remove LIQUID from whatever build_system returns: if the app's
    equilibrium readout really uses that object, pure Ag stays solid far above
    its melting point."""
    real_build_system = builder.build_system
    calls = []

    def build_system_without_liquid(definition):
        system = real_build_system(definition)
        calls.append(definition.name)
        return {name: phase for name, phase in system.items() if name != "LIQUID"}

    monkeypatch.setattr(builder, "build_system", build_system_without_liquid)

    at = _run()
    _slider(at, "Composition x(Cu) (at%)").set_value(0.0)
    _slider(at, "Temperature (K) for equilibrium and Gibbs curves").set_value(1400.0)
    at.run()
    assert not at.exception

    assert "Ag-Cu" in calls
    markdown = [element.value for element in at.markdown]
    assert markdown[markdown.index("**Stable phases**") + 1] == "FCC_AG"


def test_ag_cu_default_display_range_includes_both_melting_points():
    at = _run()
    T_min, T_max = _slider(at, "Temperature range (K)").value
    assert T_min < 1234.93
    assert T_max > 1357.77


def test_invariants_do_not_depend_on_displayed_range():
    at_wide = _run()
    at_narrow = _run()
    _slider(at_narrow, "Temperature range (K)").set_value((500.0, 900.0))
    _slider(at_narrow, "Temperature points").set_value(20)
    at_narrow.run()
    assert not at_narrow.exception

    wide = _invariant_table(at_wide)
    narrow = _invariant_table(at_narrow)
    columns = ["type", "Temperature (K)", "phases", "x(Cu) (at%)"]
    assert narrow[columns].equals(wide[columns])
    assert list(wide["type"]) == ["eutectic"]


def test_invariant_outside_displayed_range_is_listed_with_note():
    at = _run()
    _slider(at, "Temperature range (K)").set_value((500.0, 900.0))
    at.run()
    assert not at.exception

    table = _invariant_table(at)
    assert list(table["type"]) == ["eutectic"]
    assert list(table["note"]) == [OUTSIDE_NOTE]
    assert _trace(_figure(at), "invariant reactions") is None


@pytest.mark.parametrize("temperature_unit", ["K", "°C"])
@pytest.mark.parametrize("composition_unit", ["at%", "wt%"])
def test_axes_markers_and_tables_share_units(temperature_unit, composition_unit):
    at = _run(**{"Temperature unit": temperature_unit, "Composition unit": composition_unit})
    T_title = f"Temperature ({temperature_unit})"
    x_title = f"{'x' if composition_unit == 'at%' else 'w'}(Cu) ({composition_unit})"

    diagram = _figure(at, 0)
    assert diagram["layout"]["xaxis"]["title"]["text"] == x_title
    assert diagram["layout"]["yaxis"]["title"]["text"] == T_title

    gibbs = _figure(at, 1)
    assert gibbs["layout"]["xaxis"]["title"]["text"] == x_title
    assert gibbs["layout"]["yaxis"]["title"]["text"] == "Molar Gibbs energy (J/mol)"

    T_min_display, _ = _slider(at, f"Temperature range ({temperature_unit})").value
    offset = CELSIUS_OFFSET if temperature_unit == "°C" else 0.0
    assert T_min_display == pytest.approx(500.0 - offset)
    assert _values(_trace(diagram, "phase fields")["y"])[0] == pytest.approx(T_min_display)

    table = _invariant_table(at)
    marker = _trace(diagram, "invariant reactions")
    assert _values(marker["y"])[0] == pytest.approx(table[T_title][0])
    assert marker["text"][0].endswith(f" {temperature_unit}")
    liquid_in_table = next(
        part for part in table[x_title][0].split(", ") if part.startswith("LIQUID ")
    )
    assert _values(marker["x"])[0] == pytest.approx(float(liquid_in_table.split()[1]), abs=0.05)

    compositions = at.table[1].value
    assert x_title in compositions.columns
