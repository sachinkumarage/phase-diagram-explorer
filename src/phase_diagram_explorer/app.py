"""Streamlit app. Run with "streamlit run streamlit_app.py" from the
repository root (or "streamlit run src/phase_diagram_explorer/app.py").

At the default settings the diagram and invariants come from
data/precomputed (see precomputed.py), so the app starts quickly on a small
cloud machine; anything else is computed live and cached. The equilibrium
at the selected (T, x) is computed live.

If a system's database cannot be downloaded, the page still works from the
precomputed data alone: the diagram, invariant table, citations and unit
toggles, and the equilibrium at the selected (T, x) read from the
precomputed equilibrium table (lookup.py). Only the Gibbs energy curves and
the total Gibbs energy need the database; that section then shows a notice.

Public defaults: only validated systems are listed until "Show preview
systems (provisional data)" is ticked; Ag-Cu is selected first.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from phase_diagram_explorer import __version__
from phase_diagram_explorer.builder import build_system, is_computable
from phase_diagram_explorer.equilibrium.curves import evaluate_phase_curves
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.export import EXPORT_FORMATS, export_figure, invariants_csv
from phase_diagram_explorer.invariants import detect_invariants_over_range
from phase_diagram_explorer.models import (
    SystemDefinition,
    default_systems_dir,
    is_sourced_system,
    load_system,
    system_files,
    system_name,
)
from phase_diagram_explorer.precomputed import (
    DEFAULT_POINTS,
    DEFAULT_X,
    FALLBACK_T_RANGE_K,
    analysis_range,
    default_precomputed_dir,
    source_hash,
    status,
)
from phase_diagram_explorer.solidification import melting_range
from phase_diagram_explorer.lookup import equilibrium_from_trace
from phase_diagram_explorer.sources import SourceError, ensure, registry_path, source, tdb_dir
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.tracing import DEFAULT_LEVELS, trace_diagram
from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    CELSIUS,
    COMPOSITION_UNITS,
    TEMPERATURE_UNITS,
    composition_from_display,
    composition_label,
    composition_to_display,
    temperature_from_display,
    temperature_label,
    temperature_to_display,
)
from phase_diagram_explorer.visualization import COMPOSITION_FORMAT, TEMPERATURE_FORMAT, plot_diagram

PROVISIONAL_BANNER = "Preview: provisional thermodynamic data, not yet validated."
PREVIEW_LABEL = "Show preview systems (provisional data)"
DEFAULT_SYSTEM = "ag_cu"
GIBBS_NOTICE = "Gibbs energy curves need the {name} database, which could not be downloaded right now."
TOTAL_GIBBS_NOTICE = "Needs the {name} database."
# A failed download is not retried for this long (s), so that reruns stay fast.
FAILED_DOWNLOAD_TTL_S = 300
PRECOMPUTED_NOTE = "Diagram and invariant reactions from precomputed data"
LIVE_NOTICE = "Precomputed data not used ({reason}); computing the diagram live."

# Temperature slider limits (K), also the analysis range for systems without "t_range_k".
SLIDER_LIMITS_K = FALLBACK_T_RANGE_K
GIBBS_UNIT = "J/mol"
TANGENT_POINTS = 200
FRACTION_FORMAT = ".3f"
# "High resolution": tracer levels and Gibbs-curve points.
HIGH_RESOLUTION = (240, 2000)


def _system_names(paths: list[Path]) -> list[str]:
    """System names ("ag_cu"), or full file names where two files share one."""
    names = [system_name(path) for path in paths]
    return [name if names.count(name) == 1 else path.name for name, path in zip(names, paths)]


@st.cache_data(show_spinner=False)
def _source_hash(system_path: str, modified: float) -> str:
    return source_hash(system_path)


def _data_hash(system_path: str) -> str:
    return _source_hash(system_path, Path(system_path).stat().st_mtime)


@st.cache_resource(show_spinner="Loading system...")
def _load(system_path: str, data_hash: str):
    """(definition, evaluable system) for a system file, built once per data version."""
    definition = load_system(system_path)
    return definition, build_system(definition)


@st.cache_data(show_spinner=False)
def _is_computable(system_path: str, data_hash: str) -> bool:
    # A system whose database is downloaded is listed without loading it,
    # so that listing the systems never downloads anything.
    return is_sourced_system(system_path) or is_computable(load_system(system_path))


@st.cache_data(show_spinner=False)
def _is_preview(system_path: str, data_hash: str) -> bool:
    """Provisional data, read without loading a sourced system's database."""
    if is_sourced_system(system_path):
        return json.loads(Path(system_path).read_text()).get("_status") == "provisional"
    return load_system(system_path).is_provisional


@st.cache_data(show_spinner=False, ttl=FAILED_DOWNLOAD_TTL_S)
def _failed_download(name: str, directory: str, registry: str) -> str:
    """Download a database (verified), returning "" on success or the error
    message. Results are cached, so a failure is retried only after the TTL."""
    entry = source(name)
    try:
        with st.spinner(f"Downloading {entry.title or name} from {entry.url} (first use only)..."):
            ensure(name, directory)
    except SourceError as error:
        return str(error)
    return ""


def _database_error(system_path: str) -> str | None:
    """None if the system's thermodynamic data are available (downloading a
    sourced system's database on first use), else why not."""
    if not is_sourced_system(system_path):
        return None
    name = json.loads(Path(system_path).read_text())["source"]
    entry = source(name)
    try:
        if (tdb_dir() / entry.file).exists():
            ensure(name, download=False)
            return None
    except SourceError as error:
        return str(error)
    error = _failed_download(name, str(tdb_dir()), str(registry_path()))
    if not error and not (tdb_dir() / entry.file).exists():
        _failed_download.clear()  # cached success, but the file is gone: fetch again
        error = _failed_download(name, str(tdb_dir()), str(registry_path()))
    return error or None


def _database_name(system_path: str) -> str:
    if is_sourced_system(system_path):
        entry = source(json.loads(Path(system_path).read_text())["source"])
        return entry.short_name or entry.name
    return "thermodynamic"


def _show_citations(definition: SystemDefinition) -> None:
    if not definition.citations:
        return
    st.markdown("**Data sources**")
    st.markdown(
        "\n".join(
            f"- [{citation.text}]({citation.url})" if citation.url else f"- {citation.text}"
            for citation in definition.citations
        )
    )


@st.cache_data(show_spinner=False)
def _precomputed(system_path: str, data_hash: str, version: str, directory: str):
    return status(system_path, directory)


@st.cache_data(show_spinner="Computing the melting range...")
def _cached_melting_range(system_path: str, data_hash: str, version: str, x: float):
    definition, system = _load(system_path, data_hash)
    return melting_range(system, x, analysis_range(definition))


@st.cache_data(show_spinner="Computing the melting range...")
def _table_melting_range(system_path: str, data_hash: str, version: str, x: float, _table):
    """Melting range from the precomputed equilibrium table (no database)."""
    return melting_range(None, x, _table.T_range, equilibrium=lambda T, x: equilibrium_from_trace(_table, T, x))


def _default_index(options, preferred: str | None) -> int:
    aliases = {"C": CELSIUS, "°C": CELSIUS}
    preferred = aliases.get(preferred, preferred)
    return list(options).index(preferred) if preferred in options else 0


def _show_industry_note(definition, reactions, melting, to_T, T_unit: str, composition_text: str) -> None:
    """Short context box: the note, the eutectic, and the melting (mushy)
    range of the selected alloy."""
    lines = [definition.industry_note]
    for reaction in reactions:
        if reaction.type == "eutectic":
            lines.append(f"Eutectic: {to_T(reaction.temperature):.1f} {T_unit}.")
    if melting.mushy_range is not None:
        lines.append(
            f"Selected alloy ({composition_text}): solidus {to_T(melting.solidus):.1f} {T_unit}, "
            f"liquidus {to_T(melting.liquidus):.1f} {T_unit}, melting (mushy) range {melting.mushy_range:.1f} K."
        )
    else:
        lines.append(f"Selected alloy ({composition_text}): no solidus and liquidus within the analysis range.")
    st.info("\n\n".join(lines))


@st.cache_data(show_spinner="Detecting invariant reactions...")
def _cached_invariants(system_path: str, data_hash: str, version: str, n_points: int):
    definition, system = _load(system_path, data_hash)
    return detect_invariants_over_range(system, analysis_range(definition), n_points=n_points)


@st.cache_data(show_spinner="Tracing phase boundaries...")
def _cached_trace(
    system_path: str, data_hash: str, version: str, T_min: float, T_max: float, n_T: int, n_points: int, invariants
):
    _, system = _load(system_path, data_hash)
    return trace_diagram(system, (T_min, T_max), invariants=invariants, n_levels=n_T, n_points=n_points)


def _footnote(definition: SystemDefinition, system_path: str) -> str:
    """Short data-source note for exported figures."""
    note = f"Data source: {Path(system_path).name} ({definition.name})."
    if definition.source:
        note += " " + definition.source.split(". ")[0].rstrip(".") + "."
    if definition.is_provisional:
        note += " Provisional thermodynamic data, not yet validated."
    return note


@st.cache_data(show_spinner="Rendering figure...")
def _cached_export(
    system_path: str, data_hash: str, version: str, settings: tuple, T_unit: str, x_unit: str, fmt: str, _traced
) -> bytes:
    definition, _ = _load(system_path, data_hash)
    return export_figure(
        _traced, definition.name, fmt,
        dependent_symbol=definition.dependent_element.symbol, temperature_unit=T_unit, composition_unit=x_unit,
        atomic_masses=(definition.base_element.atomic_mass, definition.dependent_element.atomic_mass),
        footnote=_footnote(definition, system_path),
    )


def estimated_seconds(base_seconds: float, n_T: int, n_points: int) -> float:
    """Rough time for a live computation, scaled from the precomputed run at
    the default settings. Both the Gibbs-curve points and the tracer's
    starting levels raise the work less than linearly (measured for Ag-Cu:
    1.2 s at the defaults, about 4 s at high resolution)."""
    return base_seconds * (n_points / DEFAULT_POINTS) ** 0.5 * (n_T / DEFAULT_LEVELS) ** 0.5


def _tangent_line_figure(system, T: float, x_overall: float, n_points: int, to_x, x_title: str, T_text: str) -> go.Figure:
    x, curves = evaluate_phase_curves(system, T, n_points=n_points)
    x_display = to_x(x)
    fig = go.Figure()

    for name, curve in curves.items():
        finite = np.isfinite(curve)
        if finite.sum() == 1:
            fig.add_trace(
                go.Scatter(x=x_display[finite], y=curve[finite], mode="markers", name=name, marker=dict(size=10))
            )
        else:
            fig.add_trace(go.Scatter(x=x_display, y=curve, mode="lines", name=name))

    result = compute_equilibrium(system, T, x_overall, n_points=n_points)

    if len(result.stable_phases) == 2:
        p1, p2 = result.stable_phases
        x1, x2 = result.phase_compositions[p1], result.phase_compositions[p2]
        phase1, phase2 = system[base_phase_name(p1)], system[base_phase_name(p2)]
        g1 = phase1.molar_gibbs(T) if isinstance(phase1, StoichiometricPhase) else float(phase1.molar_gibbs(T, x1))
        g2 = phase2.molar_gibbs(T) if isinstance(phase2, StoichiometricPhase) else float(phase2.molar_gibbs(T, x2))

        if x2 != x1:
            # The tangent is a straight line in mole fraction; sample it densely
            # so it is drawn correctly on a non-linear (wt%) axis as well.
            slope = (g2 - g1) / (x2 - x1)
            line_x = np.linspace(0.0, 1.0, TANGENT_POINTS)
            line_y = g1 + slope * (line_x - x1)
            fig.add_trace(
                go.Scatter(
                    x=to_x(line_x), y=line_y, mode="lines", name="common tangent",
                    line=dict(color="black", dash="dash"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=to_x(np.array([x1, x2])), y=[g1, g2], mode="markers", name="tangent points",
                marker=dict(color="black", size=10, symbol="x"),
            )
        )

    fig.add_vline(x=to_x(x_overall), line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title=f"Gibbs energy curves at T = {T_text}",
        xaxis_title=x_title,
        yaxis_title=f"Molar Gibbs energy ({GIBBS_UNIT})",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Phase Diagram Explorer", layout="wide")
    st.title("Phase Diagram Explorer")

    systems_dir = default_systems_dir()
    precomputed_dir = str(default_precomputed_dir())
    # Only systems whose every phase can be evaluated are offered (JSON, YAML
    # or TDB with its .meta.json).
    system_paths = [
        path for path in system_files(systems_dir) if _is_computable(str(path), _data_hash(str(path)))
    ]
    if not system_paths:
        st.error(f"No system definitions with complete Gibbs energy data found in {systems_dir}")
        st.stop()

    with st.sidebar:
        show_preview = st.checkbox(PREVIEW_LABEL, value=False)
    if not show_preview:
        system_paths = [path for path in system_paths if not _is_preview(str(path), _data_hash(str(path)))]
    if not system_paths:
        st.info(f"No validated systems are available; tick \"{PREVIEW_LABEL}\" in the sidebar.")
        st.stop()

    system_names = _system_names(system_paths)
    selected_name = st.selectbox(
        "System", system_names, index=system_names.index(DEFAULT_SYSTEM) if DEFAULT_SYSTEM in system_names else 0
    )
    selected_path = str(system_paths[system_names.index(selected_name)])
    data_hash = _data_hash(selected_path)
    precomputed, stale_reason = _precomputed(selected_path, data_hash, __version__, precomputed_dir)
    database_error = _database_error(selected_path)
    if database_error is None:
        definition, system = _load(selected_path, data_hash)
    elif precomputed is not None:
        # The database is unavailable: everything below except the Gibbs
        # energy curves works from the precomputed data.
        definition, system = precomputed.definition, None
    else:
        st.error(
            f"The database for this system could not be loaded, and no precomputed data are available: {database_error}"
        )
        st.stop()
    database_name = _database_name(selected_path)

    if definition.is_provisional:
        st.warning(PROVISIONAL_BANNER)
        if definition.status_reason:
            st.caption(definition.status_reason)

    base = definition.base_element
    dependent = definition.dependent_element
    atomic_masses = (base.atomic_mass, dependent.atomic_mass)
    analysis_range_K = analysis_range(definition)

    st.caption(
        f"Elements: {base.symbol} (x = 0), {dependent.symbol} (x = 1) · "
        f"Phases: {', '.join(phase.name for phase in definition.phases)}"
    )

    with st.sidebar:
        st.header("Units")
        # One widget per system, so that each system opens in its own default units.
        T_unit = st.radio(
            "Temperature unit", TEMPERATURE_UNITS, horizontal=True, key=f"T_unit_{selected_name}",
            index=_default_index(TEMPERATURE_UNITS, definition.display.get("temperature_unit")),
        )
        available_composition_units = COMPOSITION_UNITS if None not in atomic_masses else (ATOMIC_PERCENT,)
        x_unit = st.radio(
            "Composition unit", available_composition_units, horizontal=True, key=f"x_unit_{selected_name}",
            index=_default_index(available_composition_units, definition.display.get("composition_unit")),
        )

    def to_T(T_kelvin):
        return temperature_to_display(T_kelvin, T_unit)

    def to_x(x_b):
        return composition_to_display(x_b, x_unit, *atomic_masses)

    x_title = composition_label(dependent.symbol, x_unit)
    T_title = temperature_label(T_unit)
    default_x_display = round(float(to_x(DEFAULT_X)), 1)

    with st.sidebar:
        st.header("Diagram settings")
        T_range_display = st.slider(
            f"Temperature range ({T_unit})",
            to_T(SLIDER_LIMITS_K[0]), to_T(SLIDER_LIMITS_K[1]),
            (to_T(analysis_range_K[0]), to_T(analysis_range_K[1])),
            step=10.0,
        )
        T_range = (
            temperature_from_display(T_range_display[0], T_unit), temperature_from_display(T_range_display[1], T_unit)
        )
        x_display = st.slider(f"Composition {x_title}", 0.0, 100.0, default_x_display, step=0.1)
        x_overall = min(max(composition_from_display(x_display, x_unit, *atomic_masses), 0.0), 1.0)
        with st.expander("Grid resolution"):
            high_resolution = st.checkbox(
                "High resolution",
                help=f"{HIGH_RESOLUTION[0]} temperature levels and {HIGH_RESOLUTION[1]} Gibbs curve points.",
            )
            n_T = st.slider("Temperature points", 20, 300, DEFAULT_LEVELS, step=10, disabled=high_resolution)
            n_points = st.slider(
                "Gibbs curve resolution", 100, 2000, DEFAULT_POINTS, step=100, disabled=high_resolution
            )
            if high_resolution:
                n_T, n_points = HIGH_RESOLUTION
        T_selected_display = st.slider(
            f"Temperature ({T_unit}) for equilibrium and Gibbs curves",
            T_range_display[0], T_range_display[1], (T_range_display[0] + T_range_display[1]) / 2.0,
        )
        T_selected = temperature_from_display(T_selected_display, T_unit)

    def is_displayed(reaction) -> bool:
        return T_range[0] <= reaction.temperature <= T_range[1]

    default_settings = n_T == DEFAULT_LEVELS and n_points == DEFAULT_POINTS
    full_range = np.allclose(T_range, analysis_range_K, atol=1e-6)

    if precomputed is not None and (default_settings or system is None):
        reactions = precomputed.invariants
    else:
        reactions = _cached_invariants(selected_path, data_hash, __version__, n_points)

    if precomputed is not None and ((default_settings and full_range) or system is None):
        traced = precomputed.traced
        source_note = (
            f"{PRECOMPUTED_NOTE} (version {precomputed.metadata['package_version']}, "
            f"computed in {precomputed.metadata['computation_time_s']:.1f} s)."
        )
        if system is None and not (default_settings and full_range):
            st.info(
                f"Other temperature ranges and resolutions need the {database_name} database, which could not "
                "be downloaded right now; showing the precomputed diagram over the selected range."
            )
    else:
        if precomputed is None:
            st.info(LIVE_NOTICE.format(reason=stale_reason))
        if high_resolution:
            base_seconds = precomputed.metadata["computation_time_s"] if precomputed is not None else 2.0
            st.warning(
                f"High resolution: this computation takes about "
                f"{estimated_seconds(base_seconds, n_T, n_points):.0f} s (scaled from the precomputed run; "
                "slower machines take longer, rough estimate)."
            )
        traced = _cached_trace(
            selected_path, data_hash, __version__, T_range[0], T_range[1], n_T, n_points, reactions
        )
        source_note = "Diagram computed live for these settings."

    st.subheader("Phase diagram")
    figure = plot_diagram(
        traced, definition.name,
        dependent_symbol=dependent.symbol, temperature_unit=T_unit,
        composition_unit=x_unit, atomic_masses=atomic_masses,
    )
    if not np.allclose(traced.T_range, T_range, atol=1e-6):
        figure.update_yaxes(range=[to_T(T_range[0]), to_T(T_range[1])])
    st.plotly_chart(figure, width="stretch")
    st.caption(source_note)
    _show_citations(definition)
    if definition.industry_note:
        if system is not None:
            melting = _cached_melting_range(selected_path, data_hash, __version__, x_overall)
        else:
            melting = _table_melting_range(selected_path, data_hash, __version__, x_overall, precomputed.table)
        _show_industry_note(
            definition, reactions, melting, to_T, T_unit, f"{x_display:.1f} {x_unit} {dependent.symbol}"
        )
    export_columns = st.columns(len(EXPORT_FORMATS))
    settings = (T_range[0], T_range[1], n_T, n_points)
    for column, fmt in zip(export_columns, EXPORT_FORMATS):
        with column:
            st.download_button(
                f"Export figure ({fmt.upper()})",
                # rendered only when clicked
                data=lambda fmt=fmt: _cached_export(
                    selected_path, data_hash, __version__, settings, T_unit, x_unit, fmt, traced
                ),
                file_name=f"{selected_name}_phase_diagram.{fmt}",
                mime="image/svg+xml" if fmt == "svg" else "application/pdf",
            )

    T_selected_text = f"{T_selected_display:.1f} {T_unit}"
    st.subheader(f"Equilibrium at T = {T_selected_text}, {x_title} = {x_display:.1f}")
    if system is not None:
        result = compute_equilibrium(system, T_selected, x_overall, n_points=n_points)
    else:
        try:
            result = equilibrium_from_trace(precomputed.table, T_selected, x_overall)
        except ValueError:
            result = None
            st.info(
                f"Without the {database_name} database, equilibria are available only within "
                f"{to_T(analysis_range_K[0]):.0f}–{to_T(analysis_range_K[1]):.0f} {T_unit}."
            )

    col1, col2 = st.columns(2)
    if result is not None:
        with col1:
            st.markdown("**Stable phases**")
            st.write(", ".join(result.stable_phases))
            st.markdown("**Phase fractions**")
            st.table(
                {
                    "phase": list(result.phase_fractions.keys()),
                    "fraction": [f"{f:{FRACTION_FORMAT}}" for f in result.phase_fractions.values()],
                }
            )
        with col2:
            st.markdown("**Phase compositions**")
            st.table(
                {
                    "phase": list(result.phase_compositions.keys()),
                    x_title: [f"{to_x(x):{COMPOSITION_FORMAT}}" for x in result.phase_compositions.values()],
                }
            )
            st.markdown(f"**Molar Gibbs energy ({GIBBS_UNIT})**")
            if np.isfinite(result.total_gibbs):
                st.write(f"{result.total_gibbs:.2f}")
            else:
                st.caption(TOTAL_GIBBS_NOTICE.format(name=database_name))

    st.subheader("Gibbs energy curves and common tangent")
    if system is not None:
        st.plotly_chart(
            _tangent_line_figure(system, T_selected, x_overall, n_points, to_x, x_title, T_selected_text),
            width="stretch",
        )
    else:
        st.info(GIBBS_NOTICE.format(name=database_name))

    st.subheader("Invariant reactions")
    st.caption(
        f"Detected over {to_T(analysis_range_K[0]):.0f}–{to_T(analysis_range_K[1]):.0f} {T_unit}, "
        "independent of the displayed range; temperatures and all three phase compositions by root-finding."
    )
    if reactions:
        st.table(
            {
                "type": [r.type for r in reactions],
                T_title: [f"{to_T(r.temperature):{TEMPERATURE_FORMAT}}" for r in reactions],
                "phases": ["+".join(r.phases) for r in reactions],
                x_title: [
                    ", ".join(f"{phase} {to_x(x):{COMPOSITION_FORMAT}}" for phase, x in r.composition.items())
                    for r in reactions
                ],
                "note": ["" if is_displayed(r) else "outside displayed temperature range" for r in reactions],
            }
        )
        st.download_button(
            "Download invariant reactions (CSV, full precision)",
            data=invariants_csv(reactions, dependent.symbol),
            file_name=f"{selected_name}_invariants.csv",
            mime="text/csv",
        )
    else:
        st.write("None found.")


if __name__ == "__main__":
    main()
