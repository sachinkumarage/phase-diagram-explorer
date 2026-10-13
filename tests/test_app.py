"""Tests of the Streamlit app's wiring: where its thermodynamics come from,
how invariant reactions relate to the displayed range, and display units."""
import ast
import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from phase_diagram_explorer import builder
from phase_diagram_explorer.units import CELSIUS_OFFSET

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "phase_diagram_explorer" / "app.py"
SYSTEMS_DIR = ROOT / "data" / "systems"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
PROVISIONAL_BANNER = "Preview: provisional thermodynamic data, not yet validated."

THERMO_CLASSES = {"PureElementGibbs", "SolutionPhase", "StoichiometricPhase"}
OUTSIDE_NOTE = "outside displayed temperature range"


def _slider(at: AppTest, label: str):
    matches = [slider for slider in at.slider if slider.label == label]
    assert len(matches) == 1, f"expected one slider labelled {label!r}"
    return matches[0]


def _systems_dir(monkeypatch, tmp_path: Path, *files: Path) -> None:
    """Point the app at a directory holding copies of `files`."""
    for path in files:
        shutil.copy(path, tmp_path / path.name)
    monkeypatch.setenv("PHASE_DIAGRAM_SYSTEMS_DIR", str(tmp_path))


PREVIEW_LABEL = "Show preview systems (provisional data)"


def _checkbox(at: AppTest, label: str):
    (box,) = [checkbox for checkbox in at.checkbox if checkbox.label == label]
    return box


def _show_preview_systems(at: AppTest) -> None:
    _checkbox(at, PREVIEW_LABEL).check()
    at.run()
    assert not at.exception


def _run(system: str = "ag_cu", **radios) -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    if system not in at.selectbox[0].options:
        _show_preview_systems(at)
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
    st.cache_data and st.cache_resource caches used by other app tests."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


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
    assert markdown[markdown.index("**Stable phases**") + 1] == "FCC_A1"


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
    region_T = [
        T for trace in diagram["data"] if trace.get("fill") == "toself" for T in _values(trace["y"])
    ]
    assert min(region_T) == pytest.approx(T_min_display)
    assert diagram["layout"]["yaxis"]["range"][0] == pytest.approx(T_min_display)

    table = _invariant_table(at)
    marker = _trace(diagram, "invariant reactions")
    assert _values(marker["y"])[0] == pytest.approx(float(table[T_title][0]), abs=0.05)
    assert marker["text"][0].endswith(f" {temperature_unit}")
    liquid_in_table = next(
        part for part in table[x_title][0].split(", ") if part.startswith("LIQUID ")
    )
    assert _values(marker["x"])[0] == pytest.approx(float(liquid_in_table.split()[1]), abs=0.05)

    compositions = at.table[1].value
    assert x_title in compositions.columns


def test_selector_lists_only_systems_with_complete_gibbs_data(monkeypatch, tmp_path):
    _systems_dir(monkeypatch, tmp_path, SYSTEMS_DIR / "ag_cu.meta.json", FIXTURES_DIR / "example.json")
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    assert at.selectbox[0].options == ["ag_cu"]


def test_shipped_selector_has_no_example_system():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    _show_preview_systems(at)
    assert at.selectbox[0].options == ["ag_cu", "al_cu", "pb_sn"]


def test_public_defaults_list_only_validated_systems_with_ag_cu_selected():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    assert _checkbox(at, PREVIEW_LABEL).value is False
    assert at.selectbox[0].options == ["ag_cu", "pb_sn"]
    assert at.selectbox[0].value == "ag_cu"
    assert len(at.warning) == 0

    _show_preview_systems(at)
    assert at.selectbox[0].options == ["ag_cu", "al_cu", "pb_sn"]
    assert at.selectbox[0].value == "ag_cu"
    at.selectbox[0].set_value("al_cu")
    at.run()
    assert [warning.value for warning in at.warning] == [PROVISIONAL_BANNER]


def test_provisional_systems_show_banner():
    at = _run("al_cu")
    assert [warning.value for warning in at.warning] == [PROVISIONAL_BANNER]


def test_assessed_ag_cu_shows_no_banner_and_its_citations():
    at = _run("ag_cu")
    assert len(at.warning) == 0
    markdown = [element.value for element in at.markdown]
    citations = markdown[markdown.index("**Data sources**") + 1]
    assert "[Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems" in citations
    assert "](https://doi.org/10.18434/T4759N)" in citations
    assert "[F.H. Hayes, H.L. Lukas, G. Effenberg, G. Petzow, Z. Metallkd. 77 (1986) 749-754.](" in citations
    assert "[A.T. Dinsdale, CALPHAD 15 (1991) 317-425.](https://doi.org/10.1016/0364-5916(91)90030-N)" in citations


def test_provisional_al_cu_has_no_citations():
    at = _run("al_cu")
    assert "**Data sources**" not in [element.value for element in at.markdown]


def test_validated_systems_show_no_banner(monkeypatch, tmp_path):
    _systems_dir(monkeypatch, tmp_path, FIXTURES_DIR / "regular_solution.json")
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    assert len(at.warning) == 0


def test_composition_sets_in_tables_and_plots(monkeypatch, tmp_path):
    _systems_dir(monkeypatch, tmp_path, FIXTURES_DIR / "regular_solution.json")
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    _slider(at, "Temperature (K) for equilibrium and Gibbs curves").set_value(1000.0)
    at.run()
    assert not at.exception

    markdown = [element.value for element in at.markdown]
    assert markdown[markdown.index("**Stable phases**") + 1] == "ALPHA#1, ALPHA#2"
    assert list(at.table[1].value["phase"]) == ["ALPHA#1", "ALPHA#2"]

    diagram = _figure(at, 0)
    legend = {trace.get("name") for trace in diagram["data"]}
    assert {"ALPHA", "ALPHA#1 + ALPHA#2"} <= legend

    gibbs = _figure(at, 1)
    tangent = _trace(gibbs, "tangent points")
    assert tangent is not None
    x1, x2 = _values(tangent["x"])
    assert x1 + x2 == pytest.approx(100.0, abs=0.2)


def test_diagram_is_drawn_from_traced_polygons_not_a_heatmap():
    at = _run()
    diagram = _figure(at, 0)
    types = {trace.get("type", "scatter") for trace in diagram["data"]}
    assert "heatmap" not in types and "contour" not in types

    regions = [trace for trace in diagram["data"] if trace.get("fill") == "toself"]
    # assessed Ag-Cu: one FCC_A1 phase with a miscibility gap
    assert {trace["name"] for trace in regions} == {"FCC_A1", "LIQUID", "LIQUID + FCC_A1", "FCC_A1#1 + FCC_A1#2"}
    labels = _trace(diagram, "field labels")
    assert set(labels["text"]) == {trace["name"] for trace in regions}


def test_narrow_al_cu_reaction_is_in_invariant_table_at_default_grid():
    at = _run("al_cu")
    table = _invariant_table(at)
    assert list(table["type"]) == ["eutectic"]
    assert set(table["phases"][0].split("+")) == {"FCC_AL", "LIQUID", "AL2CU"}


@pytest.mark.parametrize("fmt", ["SVG", "PDF"])
def test_export_buttons_offer_vector_figures(fmt):
    at = _run()
    labels = [button.proto.label for button in at.get("download_button")]
    assert f"Export figure ({fmt})" in labels


def _decimals(text: str) -> int:
    return len(text.split(".")[1])


@pytest.mark.parametrize("temperature_unit", ["K", "°C"])
def test_display_precision(temperature_unit):
    """T to 0.1 K or °C, compositions to 0.1 at%, fractions to 0.001."""
    at = _run(**{"Temperature unit": temperature_unit})
    T_title = f"Temperature ({temperature_unit})"
    table = _invariant_table(at)
    assert _decimals(table[T_title][0]) == 1
    for part in table["x(Cu) (at%)"][0].split(", "):
        assert _decimals(part.split()[1]) == 1

    fractions, compositions = at.table[0].value, at.table[1].value
    assert all(_decimals(value) == 3 for value in fractions["fraction"])
    assert all(_decimals(value) == 1 for value in compositions["x(Cu) (at%)"])

    diagram = _figure(at, 0)
    boundary = next(trace for trace in diagram["data"] if " / " in trace.get("name", ""))
    assert "%{x:.1f}" in boundary["hovertemplate"] and "%{y:.1f}" in boundary["hovertemplate"]


def test_invariant_csv_has_full_precision():
    at = _run()
    (button,) = [b for b in at.get("download_button") if "CSV" in b.proto.label]
    assert "full precision" in button.proto.label


# --- deployment: entry point, precomputed data, caching --------------------

PRECOMPUTED_NOTE = "Diagram and invariant reactions from precomputed data"
LIVE_NOTICE = "computing the diagram live"
ENTRY_POINT = ROOT / "streamlit_app.py"


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.caption]


def test_root_entry_point_runs_the_app():
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Phase Diagram Explorer"
    assert at.selectbox[0].options == ["ag_cu", "pb_sn"]


BANNERS = {"ag_cu": [], "al_cu": [PROVISIONAL_BANNER], "pb_sn": []}


@pytest.mark.parametrize("system", ["ag_cu", "al_cu", "pb_sn"])
def test_each_system_loads_from_precomputed_data(system):
    at = _run(system)
    assert any(caption.startswith(PRECOMPUTED_NOTE) for caption in _captions(at))
    assert not [info for info in at.info if LIVE_NOTICE in info.value]
    assert [warning.value for warning in at.warning] == BANNERS[system]


def test_switching_systems_and_units_does_not_raise():
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=120).run()
    _show_preview_systems(at)
    for system in ["al_cu", "pb_sn", "ag_cu", "al_cu"]:
        at.selectbox[0].set_value(system)
        at.run()
        assert not at.exception
        for temperature_unit, composition_unit in [("°C", "wt%"), ("K", "at%"), ("°C", "at%")]:
            at.radio[0].set_value(temperature_unit)
            at.radio[1].set_value(composition_unit)
            at.run()
            assert not at.exception
            assert [warning.value for warning in at.warning] == BANNERS[system]
            assert any(caption.startswith(PRECOMPUTED_NOTE) for caption in _captions(at))


def test_missing_precomputed_data_is_computed_live_with_a_notice(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASE_DIAGRAM_PRECOMPUTED_DIR", str(tmp_path))
    at = _run()
    notices = [info.value for info in at.info if LIVE_NOTICE in info.value]
    assert len(notices) == 1 and "no precomputed file ag_cu.npz" in notices[0]
    assert "Diagram computed live for these settings." in _captions(at)
    assert list(_invariant_table(at)["type"]) == ["eutectic"]


def test_stale_precomputed_data_is_not_used(monkeypatch, tmp_path):
    from phase_diagram_explorer.precomputed import load, save

    stale = load(ROOT / "data" / "precomputed" / "ag_cu.npz")
    stale.metadata["source_sha256"] = "0" * 64
    save(stale, tmp_path / "ag_cu.npz")
    monkeypatch.setenv("PHASE_DIAGRAM_PRECOMPUTED_DIR", str(tmp_path))
    at = _run()
    notices = [info.value for info in at.info if LIVE_NOTICE in info.value]
    assert len(notices) == 1 and "out of date (source_sha256 changed)" in notices[0]


def test_other_settings_are_computed_live_without_a_notice():
    at = _run()
    _slider(at, "Temperature points").set_value(40)
    at.run()
    assert not at.exception
    assert "Diagram computed live for these settings." in _captions(at)
    assert not at.info


def test_high_resolution_warns_with_an_estimated_time_before_running():
    at = _run()
    _checkbox(at, "High resolution").check()
    at.run()
    assert not at.exception
    (warning,) = [w.value for w in at.warning if w.value.startswith("High resolution")]
    assert "about" in warning and warning.rstrip(").").endswith("estimate")
    assert _slider(at, "Temperature points").proto.disabled


def test_figure_exports_are_rendered_on_demand():
    """The export buttons are created without rendering the figures (their
    data is a callable), which keeps matplotlib out of the start-up path."""
    at = _run()
    labels = [button.proto.label for button in at.get("download_button")]
    assert {"Export figure (SVG)", "Export figure (PDF)"} <= set(labels)


def _local_registry(monkeypatch, tmp_path, url: str) -> None:
    """Register nist_solder with `url` and an empty download directory."""
    entry = json.loads((ROOT / "data" / "sources.json").read_text())["nist_solder"]
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps({"nist_solder": {**entry, "url": url}}))
    monkeypatch.setenv("PHASE_DIAGRAM_SOURCES", str(registry))
    monkeypatch.setenv("PHASE_DIAGRAM_TDB_DIR", str(tmp_path / "tdb"))


def test_database_is_downloaded_on_first_use(monkeypatch, tmp_path):
    """A missing database is fetched (and verified) when its system is first
    selected; here from a local copy of the NIST file."""
    _local_registry(monkeypatch, tmp_path, (ROOT / "data" / "tdb" / "nist_solder.tdb").as_uri())
    at = _run("ag_cu")
    assert not at.error
    assert (tmp_path / "tdb" / "nist_solder.tdb").exists()
    assert list(_invariant_table(at)["type"]) == ["eutectic"]


GIBBS_NOTICE = "Gibbs energy curves need the NIST database, which could not be downloaded right now."


def _offline(monkeypatch, tmp_path) -> list:
    """No database file, and every download fails (sources.fetch raises)."""
    from phase_diagram_explorer import sources

    attempts = []

    def failing_fetch(entry, *args, **kwargs):
        attempts.append(entry.name)
        raise sources.SourceError(f"could not download {entry.name} from {entry.url} after 3 attempts (simulated)")

    monkeypatch.setattr(sources, "fetch", failing_fetch)
    monkeypatch.setenv("PHASE_DIAGRAM_TDB_DIR", str(tmp_path / "empty"))
    return attempts


def test_failed_download_keeps_the_page_working(monkeypatch, tmp_path):
    """With the database unavailable, the diagram, invariant table, citations
    and point equilibrium come from precomputed data; only the Gibbs energy
    curves section shows a notice."""
    attempts = _offline(monkeypatch, tmp_path)
    at = _run("ag_cu")
    assert attempts == ["nist_solder"]
    assert not at.exception and not at.error

    diagram = _figure(at, 0)
    regions = {trace["name"] for trace in diagram["data"] if trace.get("fill") == "toself"}
    assert regions == {"FCC_A1", "LIQUID", "LIQUID + FCC_A1", "FCC_A1#1 + FCC_A1#2"}
    assert len(at.get("plotly_chart")) == 1  # no Gibbs energy curves
    assert list(_invariant_table(at)["type"]) == ["eutectic"]
    markdown = [element.value for element in at.markdown]
    assert "**Data sources**" in markdown
    assert markdown[markdown.index("**Stable phases**") + 1] == "FCC_A1#1, FCC_A1#2"
    assert list(at.table[1].value["phase"]) == ["FCC_A1#1", "FCC_A1#2"]
    assert [info.value for info in at.info] == [GIBBS_NOTICE]
    assert "Needs the NIST database." in _captions(at)

    # point equilibrium follows the selected (T, x) without the database
    _slider(at, "Temperature (K) for equilibrium and Gibbs curves").set_value(1300.0)
    _slider(at, "Composition x(Cu) (at%)").set_value(10.0)
    at.run()
    assert not at.exception
    markdown = [element.value for element in at.markdown]
    assert markdown[markdown.index("**Stable phases**") + 1] == "LIQUID"
    for temperature_unit, composition_unit in [("°C", "wt%"), ("K", "at%")]:
        at.radio[0].set_value(temperature_unit)
        at.radio[1].set_value(composition_unit)
        at.run()
        assert not at.exception and not at.error


def test_failed_download_is_not_retried_on_every_rerun(monkeypatch, tmp_path):
    attempts = _offline(monkeypatch, tmp_path)
    at = _run("ag_cu")
    _slider(at, "Composition x(Cu) (at%)").set_value(30.0)
    at.run()
    at.selectbox[0].set_value("pb_sn")
    at.run()
    assert attempts == ["nist_solder"]  # cached failure for both systems


def test_pb_sn_without_the_database(monkeypatch, tmp_path):
    _offline(monkeypatch, tmp_path)
    at = _run("pb_sn")
    assert not at.exception and not at.error
    _slider(at, "Composition w(Sn) (wt%)").set_value(63.0)
    at.run()
    infos = [info.value for info in at.info]
    assert GIBBS_NOTICE in infos
    (note,) = [info for info in infos if info.startswith("63Sn-37Pb")]
    assert "solidus 182.2 °C, liquidus 183.2 °C" in note
    assert list(_invariant_table(at)["type"]) == ["eutectic"]


PB_SN_INFO = "63Sn-37Pb solder (63 wt% Sn) is near-eutectic"


def test_pb_sn_opens_in_celsius_and_weight_percent_with_citations_and_info_box():
    at = _run("pb_sn")
    assert [radio.value for radio in at.radio] == ["°C", "wt%"]
    assert len(at.warning) == 0
    markdown = [element.value for element in at.markdown]
    citations = markdown[markdown.index("**Data sources**") + 1]
    assert "[H.J. Fecht, M.-X. Zhang, Y.A. Chang, J.H. Perepezko, Metall. Trans. A 20A (1989) 795-803.](" in citations
    assert "https://www.metallurgy.nist.gov/phase/solder/pbsn.html" in citations
    (info,) = [i.value for i in at.info if i.value.startswith(PB_SN_INFO)]
    assert "Eutectic: 182.2 °C." in info
    assert "melting (mushy) range" in info


def test_pb_sn_info_box_follows_the_selected_alloy():
    at = _run("pb_sn")
    _slider(at, "Composition w(Sn) (wt%)").set_value(63.0)
    at.run()
    (info,) = [i.value for i in at.info if i.value.startswith(PB_SN_INFO)]
    assert "Selected alloy (63.0 wt% Sn): solidus 182.2 °C, liquidus 183.2 °C, melting (mushy) range 0.9 K." in info


def test_other_systems_have_no_info_box():
    for system in ("ag_cu", "al_cu"):
        at = _run(system)
        assert not [i for i in at.info if i.value.startswith(PB_SN_INFO)]


def test_units_default_per_system():
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=120).run()
    at.selectbox[0].set_value("ag_cu")
    at.run()
    assert [radio.value for radio in at.radio] == ["K", "at%"]
    at.selectbox[0].set_value("pb_sn")
    at.run()
    assert [radio.value for radio in at.radio] == ["°C", "wt%"]
