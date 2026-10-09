import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from phase_diagram_explorer.diagram import PhaseDiagram
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name
from phase_diagram_explorer.invariants import InvariantReaction
from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    KELVIN,
    composition_label,
    composition_to_display,
    temperature_label,
    temperature_to_display,
)


def _phase_field_ids(phase_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Assign a stable integer id to each distinct phase-field tuple.

    Returns the id grid (same shape as phase_labels) and the ordered list of
    field names ("ALPHA", "ALPHA+LIQUID" or "ALPHA#1+ALPHA#2") indexed by id.
    A single-phase field is named by its base phase, so a composition set
    (ALPHA#1) and the same phase above its miscibility gap (ALPHA) form one
    continuous field with no boundary drawn between them.
    """
    field_names: list[str] = []
    field_index: dict[str, int] = {}
    ids = np.empty(phase_labels.shape, dtype=float)

    for i in range(phase_labels.shape[0]):
        for j in range(phase_labels.shape[1]):
            phases = phase_labels[i, j]
            name = base_phase_name(phases[0]) if len(phases) == 1 else "+".join(phases)
            if name not in field_index:
                field_index[name] = len(field_names)
                field_names.append(name)
            ids[i, j] = field_index[name]

    return ids, field_names


def _discrete_colorscale(n_fields: int) -> list[list]:
    palette = qualitative.Plotly + qualitative.Set3
    colors = [palette[i % len(palette)] for i in range(max(n_fields, 1))]
    colorscale = []
    for i, color in enumerate(colors):
        colorscale.append([i / n_fields, color])
        colorscale.append([(i + 1) / n_fields, color])
    return colorscale


def _hover_customdata(diagram: PhaseDiagram) -> np.ndarray:
    n_T, n_x = diagram.phase_labels.shape
    customdata = np.empty((n_T, n_x, 2), dtype=object)

    for i in range(n_T):
        for j in range(n_x):
            phases = diagram.phase_labels[i, j]
            fractions = diagram.phase_fractions[i, j]
            customdata[i, j, 0] = "+".join(phases)
            customdata[i, j, 1] = ", ".join(
                f"{name}: {fraction:.2f}" for name, fraction in fractions.items()
            )

    return customdata


def plot_diagram(
    diagram: PhaseDiagram,
    system_name: str,
    invariants: list[InvariantReaction] | None = None,
    *,
    dependent_symbol: str = "B",
    temperature_unit: str = KELVIN,
    composition_unit: str = ATOMIC_PERCENT,
    atomic_masses: tuple[float | None, float | None] = (None, None),
) -> go.Figure:
    """Build an interactive T-x phase diagram figure.

    Phase fields are drawn as a discrete colored grid (a Heatmap), with a
    Contour trace at half-integer levels overlaid to trace crisp phase
    boundary lines between fields. Hovering any point shows temperature,
    composition, the stable phase(s), and their phase fractions. If
    `invariants` is given, each reaction is marked with a labeled point at
    its reacting phase's composition and temperature.

    The diagram is computed in K and mole fraction of the dependent element;
    axes, hover text and markers are all shown in `temperature_unit` and
    `composition_unit` (wt% needs `atomic_masses` of the base and dependent
    element).
    """
    field_ids, field_names = _phase_field_ids(diagram.phase_labels)
    customdata = _hover_customdata(diagram)
    mass_a, mass_b = atomic_masses
    x_display = composition_to_display(diagram.x_grid, composition_unit, mass_a, mass_b)
    T_display = temperature_to_display(diagram.T_grid, temperature_unit)
    x_title = composition_label(dependent_symbol, composition_unit)
    T_title = temperature_label(temperature_unit)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_display,
            y=T_display,
            z=field_ids,
            customdata=customdata,
            colorscale=_discrete_colorscale(len(field_names)),
            zmin=0,
            zmax=len(field_names),
            showscale=False,
            hovertemplate=(
                f"{x_title}: %{{x:.2f}}<br>"
                f"{T_title}: %{{y:.1f}}<br>"
                "Stable phases: %{customdata[0]}<br>"
                "Phase fractions: %{customdata[1]}"
                "<extra></extra>"
            ),
            name="phase fields",
        )
    )

    fig.add_trace(
        go.Contour(
            x=x_display,
            y=T_display,
            z=field_ids,
            contours=dict(
                start=0.5,
                end=len(field_names) - 0.5,
                size=1.0,
                coloring="lines",
            ),
            line=dict(color="black", width=1),
            showscale=False,
            hoverinfo="skip",
            name="phase boundaries",
        )
    )

    for i, name in enumerate(field_names):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=_discrete_colorscale(len(field_names))[2 * i][1]),
                name=name,
                showlegend=True,
            )
        )

    if invariants:
        marker_x = []
        marker_y = []
        marker_text = []
        for reaction in invariants:
            reacting_phase = reaction.phases[1]
            x_marker = composition_to_display(
                reaction.composition[reacting_phase], composition_unit, mass_a, mass_b
            )
            T_marker = temperature_to_display(reaction.temperature, temperature_unit)
            marker_x.append(x_marker)
            marker_y.append(T_marker)
            marker_text.append(f"{reaction.type.capitalize()} {T_marker:.0f} {temperature_unit}")

        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers+text",
                text=marker_text,
                textposition="top center",
                marker=dict(symbol="diamond", size=10, color="black"),
                name="invariant reactions",
                hovertemplate=f"%{{text}}<br>{x_title}: %{{x:.2f}}<br>{T_title}: %{{y:.1f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{system_name} phase diagram",
        xaxis_title=x_title,
        yaxis_title=T_title,
        legend_title="Phase fields",
    )

    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path)
