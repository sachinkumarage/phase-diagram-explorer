from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.equilibrium.curves import evaluate_phase_curves
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.invariants import detect_invariants
from phase_diagram_explorer.models import SystemDefinition, load_system
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.visualization import plot_diagram

SYSTEMS_DIR = Path(__file__).resolve().parents[2] / "data" / "systems"

BASE_OFFSET = 6000.0
OFFSET_STEP = 3000.0
FUSION_ENTHALPY = 9000.0
MELTING_T = 900.0


def _build_computable_system(definition: SystemDefinition) -> dict[str, SolutionPhase | StoichiometricPhase]:
    """Construct evaluable thermodynamic phase objects for a loaded system definition.

    The system JSON schema records phase names and model types but not raw
    Gibbs energy coefficients, so each phase's curve is generated
    deterministically from its name and position in the phase list: phases
    named "liquid" (case-insensitive) get a simple linear fusion model that
    favors them at high temperature, other solution phases get alternating
    endpoint offsets so consecutive solid phases compete across composition,
    and stoichiometric phases get an increasingly favorable formation energy.
    """
    system: dict[str, SolutionPhase | StoichiometricPhase] = {}
    solution_index = 0
    stoichiometric_index = 0

    for phase in definition.phases:
        if phase.model_type == "solution":
            if "liquid" in phase.name.lower():
                endpoint = PureElementGibbs(a=FUSION_ENTHALPY, b=-FUSION_ENTHALPY / MELTING_T)
                system[phase.name] = SolutionPhase(endpoint, endpoint)
            else:
                offset = BASE_OFFSET + OFFSET_STEP * (solution_index // 2)
                if solution_index % 2 == 0:
                    gibbs_a, gibbs_b = PureElementGibbs(a=0.0), PureElementGibbs(a=offset)
                else:
                    gibbs_a, gibbs_b = PureElementGibbs(a=offset), PureElementGibbs(a=0.0)
                system[phase.name] = SolutionPhase(gibbs_a, gibbs_b)
            solution_index += 1
        elif phase.model_type == "stoichiometric":
            formation = -(5000.0 + 1000.0 * stoichiometric_index)
            system[phase.name] = StoichiometricPhase(
                gibbs_a=PureElementGibbs(a=0.0),
                gibbs_b=PureElementGibbs(a=0.0),
                m=1,
                n=1,
                g_form=PureElementGibbs(a=formation),
            )
            stoichiometric_index += 1
        else:
            st.warning(f"Skipping phase {phase.name!r}: model_type {phase.model_type!r} is not supported.")

    return system


@st.cache_data(show_spinner=False)
def _load_definition(system_path: str) -> SystemDefinition:
    return load_system(system_path)


@st.cache_data(show_spinner="Computing phase diagram...")
def _cached_diagram(system_path: str, T_min: float, T_max: float, n_T: int, n_x: int, n_points: int):
    definition = _load_definition(system_path)
    system = _build_computable_system(definition)
    diagram = compute_diagram(system, T_range=(T_min, T_max), n_T=n_T, n_x=n_x, n_points=n_points)
    reactions = detect_invariants(diagram, system, n_points=n_points)
    return diagram, reactions


def _tangent_line_figure(system, T: float, x_overall: float, n_points: int) -> go.Figure:
    x, curves = evaluate_phase_curves(system, T, n_points=n_points)
    fig = go.Figure()

    for name, curve in curves.items():
        finite = np.isfinite(curve)
        if finite.sum() == 1:
            fig.add_trace(go.Scatter(x=x[finite], y=curve[finite], mode="markers", name=name, marker=dict(size=10)))
        else:
            fig.add_trace(go.Scatter(x=x, y=curve, mode="lines", name=name))

    result = compute_equilibrium(system, T, x_overall, n_points=n_points)

    if len(result.stable_phases) == 2:
        p1, p2 = result.stable_phases
        x1, x2 = result.phase_compositions[p1], result.phase_compositions[p2]
        phase1, phase2 = system[p1], system[p2]
        g1 = phase1.molar_gibbs(T) if isinstance(phase1, StoichiometricPhase) else float(phase1.molar_gibbs(T, x1))
        g2 = phase2.molar_gibbs(T) if isinstance(phase2, StoichiometricPhase) else float(phase2.molar_gibbs(T, x2))

        if x2 != x1:
            slope = (g2 - g1) / (x2 - x1)
            line_x = np.array([0.0, 1.0])
            line_y = g1 + slope * (line_x - x1)
            fig.add_trace(
                go.Scatter(
                    x=line_x, y=line_y, mode="lines", name="common tangent",
                    line=dict(color="black", dash="dash"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[x1, x2], y=[g1, g2], mode="markers", name="tangent points",
                marker=dict(color="black", size=10, symbol="x"),
            )
        )

    fig.add_vline(x=x_overall, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        title=f"Gibbs energy curves at T={T:.1f}",
        xaxis_title="Composition (mole fraction B)",
        yaxis_title="Molar Gibbs energy",
    )
    return fig


st.set_page_config(page_title="Phase Diagram Explorer", layout="wide")
st.title("Phase Diagram Explorer")

system_paths = sorted(SYSTEMS_DIR.glob("*.json"))
if not system_paths:
    st.error(f"No system definitions found in {SYSTEMS_DIR}")
    st.stop()

system_names = [path.stem for path in system_paths]
selected_name = st.selectbox("System", system_names)
selected_path = system_paths[system_names.index(selected_name)]

definition = _load_definition(str(selected_path))
system = _build_computable_system(definition)

st.caption(
    f"Elements: {', '.join(element.symbol for element in definition.elements)} · "
    f"Phases: {', '.join(phase.name for phase in definition.phases)}"
)

with st.sidebar:
    st.header("Diagram settings")
    T_range = st.slider("Temperature range (K)", 200.0, 2000.0, (400.0, 900.0), step=10.0)
    x_overall = st.slider("Composition (mole fraction B)", 0.0, 1.0, 0.5, step=0.01)
    with st.expander("Grid resolution"):
        n_T = st.slider("Temperature points", 20, 300, 80, step=10)
        n_x = st.slider("Composition points", 20, 300, 150, step=10)
        n_points = st.slider("Gibbs curve resolution", 100, 2000, 500, step=100)
    T_selected = st.slider(
        "Temperature (K) for equilibrium and Gibbs curves",
        T_range[0], T_range[1], (T_range[0] + T_range[1]) / 2.0,
    )

diagram, reactions = _cached_diagram(str(selected_path), T_range[0], T_range[1], n_T, n_x, n_points)

st.subheader("Phase diagram")
st.plotly_chart(plot_diagram(diagram, selected_name, invariants=reactions), width='stretch')

st.subheader(f"Equilibrium at T={T_selected:.1f}, x={x_overall:.3f}")
result = compute_equilibrium(system, T_selected, x_overall, n_points=n_points)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Stable phases**")
    st.write(", ".join(result.stable_phases))
    st.markdown("**Phase fractions**")
    st.table({"phase": list(result.phase_fractions.keys()), "fraction": list(result.phase_fractions.values())})
with col2:
    st.markdown("**Phase compositions**")
    st.table(
        {"phase": list(result.phase_compositions.keys()), "composition": list(result.phase_compositions.values())}
    )
    st.markdown("**Total Gibbs energy**")
    st.write(f"{result.total_gibbs:.2f}")

st.subheader("Gibbs energy curves and common tangent")
st.plotly_chart(_tangent_line_figure(system, T_selected, x_overall, n_points), width='stretch')

if reactions:
    st.subheader("Invariant reactions")
    st.table(
        {
            "type": [r.type for r in reactions],
            "temperature": [r.temperature for r in reactions],
            "phases": ["+".join(r.phases) for r in reactions],
        }
    )
