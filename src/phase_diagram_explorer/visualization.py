import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from phase_diagram_explorer.diagram import PhaseDiagram
from phase_diagram_explorer.invariants import InvariantReaction


def _phase_field_ids(phase_labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Assign a stable integer id to each distinct phase-field tuple.

    Returns the id grid (same shape as phase_labels) and the ordered list of
    field names ("ALPHA" or "ALPHA+LIQUID") indexed by id.
    """
    field_names: list[str] = []
    field_index: dict[tuple, int] = {}
    ids = np.empty(phase_labels.shape, dtype=float)

    for i in range(phase_labels.shape[0]):
        for j in range(phase_labels.shape[1]):
            phases = phase_labels[i, j]
            if phases not in field_index:
                field_index[phases] = len(field_names)
                field_names.append("+".join(phases))
            ids[i, j] = field_index[phases]

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
) -> go.Figure:
    """Build an interactive T-x phase diagram figure.

    Phase fields are drawn as a discrete colored grid (a Heatmap), with a
    Contour trace at half-integer levels overlaid to trace crisp phase
    boundary lines between fields. Hovering any point shows temperature,
    composition, the stable phase(s), and their phase fractions. If
    `invariants` is given, each reaction is marked with a labeled point at
    its reacting phase's composition and temperature.
    """
    field_ids, field_names = _phase_field_ids(diagram.phase_labels)
    customdata = _hover_customdata(diagram)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=diagram.x_grid,
            y=diagram.T_grid,
            z=field_ids,
            customdata=customdata,
            colorscale=_discrete_colorscale(len(field_names)),
            zmin=0,
            zmax=len(field_names),
            showscale=False,
            hovertemplate=(
                "Composition: %{x:.3f}<br>"
                "Temperature: %{y:.1f}<br>"
                "Stable phases: %{customdata[0]}<br>"
                "Phase fractions: %{customdata[1]}"
                "<extra></extra>"
            ),
            name="phase fields",
        )
    )

    fig.add_trace(
        go.Contour(
            x=diagram.x_grid,
            y=diagram.T_grid,
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
            x_marker = reaction.composition[reacting_phase]
            celsius = reaction.temperature - 273.15
            marker_x.append(x_marker)
            marker_y.append(reaction.temperature)
            marker_text.append(f"{reaction.type.capitalize()} {celsius:.0f} C")

        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers+text",
                text=marker_text,
                textposition="top center",
                marker=dict(symbol="diamond", size=10, color="black"),
                name="invariant reactions",
                hovertemplate="%{text}<br>Composition: %{x:.3f}<br>Temperature: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{system_name} phase diagram",
        xaxis_title="Composition (mole fraction B)",
        yaxis_title="Temperature",
        legend_title="Phase fields",
    )

    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path)
