from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from phase_diagram_explorer.builder import build_system, is_computable
from phase_diagram_explorer.equilibrium.curves import evaluate_phase_curves
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.export import EXPORT_FORMATS, export_figure, invariants_csv
from phase_diagram_explorer.invariants import detect_invariants_over_range
from phase_diagram_explorer.models import SystemDefinition, default_systems_dir, load_system, system_files
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.tracing import trace_diagram
from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    COMPOSITION_UNITS,
    KELVIN,
    TEMPERATURE_UNITS,
    composition_from_display,
    composition_label,
    composition_to_display,
    temperature_from_display,
    temperature_label,
    temperature_to_display,
)
from phase_diagram_explorer.visualization import COMPOSITION_FORMAT, TEMPERATURE_FORMAT, plot_diagram

SYSTEMS_DIR = default_systems_dir()
PROVISIONAL_BANNER = "Preview: provisional thermodynamic data, not yet validated."

# Temperature slider limits (K). Used as the analysis range for systems whose
# JSON has no "t_range_k".
SLIDER_LIMITS_K = (200.0, 2000.0)
GIBBS_UNIT = "J/mol"
TANGENT_POINTS = 200
FRACTION_FORMAT = ".3f"


def _system_names(paths: list[Path]) -> list[str]:
    """File stems, or full file names where two files share a stem."""
    stems = [path.stem for path in paths]
    return [path.stem if stems.count(path.stem) == 1 else path.name for path in paths]


@st.cache_data(show_spinner=False)
def _load_definition(system_path: str) -> SystemDefinition:
    return load_system(system_path)


def _analysis_range(definition: SystemDefinition) -> tuple[float, float]:
    return tuple(definition.t_range_k) if definition.t_range_k else SLIDER_LIMITS_K


@st.cache_data(show_spinner="Detecting invariant reactions...")
def _cached_invariants(system_path: str, n_points: int):
    definition = _load_definition(system_path)
    return detect_invariants_over_range(build_system(definition), _analysis_range(definition), n_points=n_points)


@st.cache_data(show_spinner="Tracing phase boundaries...")
def _cached_trace(system_path: str, T_min: float, T_max: float, n_T: int, n_points: int):
    system = build_system(_load_definition(system_path))
    invariants = _cached_invariants(system_path, n_points)
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
    system_path: str, T_min: float, T_max: float, n_T: int, n_points: int, T_unit: str, x_unit: str, fmt: str
) -> bytes:
    definition = _load_definition(system_path)
    traced = _cached_trace(system_path, T_min, T_max, n_T, n_points)
    return export_figure(
        traced, definition.name, fmt,
        dependent_symbol=definition.dependent_element.symbol, temperature_unit=T_unit, composition_unit=x_unit,
        atomic_masses=(definition.base_element.atomic_mass, definition.dependent_element.atomic_mass),
        footnote=_footnote(definition, system_path),
    )


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


st.set_page_config(page_title="Phase Diagram Explorer", layout="wide")
st.title("Phase Diagram Explorer")

# Only systems whose every phase can be evaluated are offered (JSON, YAML or
# TDB with its .meta.json).
system_paths = [path for path in system_files(SYSTEMS_DIR) if is_computable(_load_definition(str(path)))]
if not system_paths:
    st.error(f"No system definitions with complete Gibbs energy data found in {SYSTEMS_DIR}")
    st.stop()

system_names = _system_names(system_paths)
selected_name = st.selectbox("System", system_names)
selected_path = system_paths[system_names.index(selected_name)]

definition = _load_definition(str(selected_path))
system = build_system(definition)

if definition.is_provisional:
    st.warning(PROVISIONAL_BANNER)
    if definition.status_reason:
        st.caption(definition.status_reason)

base = definition.base_element
dependent = definition.dependent_element
atomic_masses = (base.atomic_mass, dependent.atomic_mass)
analysis_range_K = _analysis_range(definition)

st.caption(
    f"Elements: {base.symbol} (x = 0), {dependent.symbol} (x = 1) · "
    f"Phases: {', '.join(phase.name for phase in definition.phases)}"
)

with st.sidebar:
    st.header("Units")
    T_unit = st.radio("Temperature unit", TEMPERATURE_UNITS, horizontal=True)
    available_composition_units = COMPOSITION_UNITS if None not in atomic_masses else (ATOMIC_PERCENT,)
    x_unit = st.radio("Composition unit", available_composition_units, horizontal=True)


def to_T(T_kelvin):
    return temperature_to_display(T_kelvin, T_unit)


def to_x(x_b):
    return composition_to_display(x_b, x_unit, *atomic_masses)


def is_displayed(reaction) -> bool:
    return T_range[0] <= reaction.temperature <= T_range[1]


x_title = composition_label(dependent.symbol, x_unit)
T_title = temperature_label(T_unit)

with st.sidebar:
    st.header("Diagram settings")
    T_range_display = st.slider(
        f"Temperature range ({T_unit})",
        to_T(SLIDER_LIMITS_K[0]), to_T(SLIDER_LIMITS_K[1]),
        (to_T(analysis_range_K[0]), to_T(analysis_range_K[1])),
        step=10.0,
    )
    T_range = (temperature_from_display(T_range_display[0], T_unit), temperature_from_display(T_range_display[1], T_unit))
    x_display = st.slider(f"Composition {x_title}", 0.0, 100.0, 50.0, step=0.1)
    x_overall = min(max(composition_from_display(x_display, x_unit, *atomic_masses), 0.0), 1.0)
    with st.expander("Grid resolution"):
        n_T = st.slider("Temperature points", 20, 300, 80, step=10)
        n_points = st.slider("Gibbs curve resolution", 100, 2000, 500, step=100)
    T_selected_display = st.slider(
        f"Temperature ({T_unit}) for equilibrium and Gibbs curves",
        T_range_display[0], T_range_display[1], (T_range_display[0] + T_range_display[1]) / 2.0,
    )
    T_selected = temperature_from_display(T_selected_display, T_unit)

reactions = _cached_invariants(str(selected_path), n_points)
traced = _cached_trace(str(selected_path), T_range[0], T_range[1], n_T, n_points)

st.subheader("Phase diagram")
st.plotly_chart(
    plot_diagram(
        traced, definition.name,
        dependent_symbol=dependent.symbol, temperature_unit=T_unit,
        composition_unit=x_unit, atomic_masses=atomic_masses,
    ),
    width='stretch',
)
export_columns = st.columns(len(EXPORT_FORMATS))
for column, fmt in zip(export_columns, EXPORT_FORMATS):
    with column:
        st.download_button(
            f"Export figure ({fmt.upper()})",
            data=_cached_export(str(selected_path), T_range[0], T_range[1], n_T, n_points, T_unit, x_unit, fmt),
            file_name=f"{selected_name}_phase_diagram.{fmt}",
            mime="image/svg+xml" if fmt == "svg" else "application/pdf",
        )

T_selected_text = f"{T_selected_display:.1f} {T_unit}"
st.subheader(f"Equilibrium at T = {T_selected_text}, {x_title} = {x_display:.1f}")
result = compute_equilibrium(system, T_selected, x_overall, n_points=n_points)

col1, col2 = st.columns(2)
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
    st.write(f"{result.total_gibbs:.2f}")

st.subheader("Gibbs energy curves and common tangent")
st.plotly_chart(
    _tangent_line_figure(system, T_selected, x_overall, n_points, to_x, x_title, T_selected_text),
    width='stretch',
)

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
            "note": [
                "" if is_displayed(r) else "outside displayed temperature range" for r in reactions
            ],
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
