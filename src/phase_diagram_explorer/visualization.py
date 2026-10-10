import plotly.graph_objects as go
from plotly.colors import qualitative

from phase_diagram_explorer.invariants import InvariantReaction
from phase_diagram_explorer.tracing import TracedDiagram
from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    KELVIN,
    composition_label,
    composition_to_display,
    temperature_label,
    temperature_to_display,
)

# Display precision: temperatures to 0.1 K (or 0.1 °C), compositions to
# 0.1 at% or wt%. Full precision stays in the API and the CSV export.
TEMPERATURE_FORMAT = ".1f"
COMPOSITION_FORMAT = ".1f"

# Fields smaller than this fraction of the plotted T-x area get no text label
# (they still show their phases on hover).
MIN_LABELLED_AREA_FRACTION = 0.005
FIELD_OPACITY = 0.35
# Text drawn on the (always white) plot area: dark in light and dark themes.
PLOT_TEXT_COLOR = "#222222"


def field_colors(labels) -> dict[str, str]:
    """A stable colour per field label, in order of first appearance."""
    palette = qualitative.Pastel + qualitative.Set3
    colors: dict[str, str] = {}
    for label in labels:
        if label not in colors:
            colors[label] = palette[len(colors) % len(palette)]
    return colors


def invariant_label(reaction: InvariantReaction, temperature_unit: str) -> str:
    T = temperature_to_display(reaction.temperature, temperature_unit)
    return f"{reaction.type.capitalize()} {T:{TEMPERATURE_FORMAT}} {temperature_unit}"


def plot_diagram(
    traced: TracedDiagram,
    system_name: str,
    *,
    dependent_symbol: str = "B",
    temperature_unit: str = KELVIN,
    composition_unit: str = ATOMIC_PERCENT,
    atomic_masses: tuple[float | None, float | None] = (None, None),
) -> go.Figure:
    """Build an interactive T-x phase diagram from traced boundaries.

    Phase regions are filled polygons bounded by the traced boundary lines
    and invariant lines (hover shows the phases), each labelled at its
    centroid with plain phase names. Boundaries are drawn as lines (hover
    shows T and composition), invariant reactions as horizontal lines across
    their three phase compositions, with a marker at the reacting phase.

    Everything is computed in K and mole fraction of the dependent element;
    axes, hover text and markers are all shown in `temperature_unit` and
    `composition_unit` (wt% needs `atomic_masses` of the base and dependent
    element).
    """
    mass_a, mass_b = atomic_masses

    def to_x(x):
        return composition_to_display(x, composition_unit, mass_a, mass_b)

    def to_T(T):
        return temperature_to_display(T, temperature_unit)

    x_title = composition_label(dependent_symbol, composition_unit)
    T_title = temperature_label(temperature_unit)
    point_hover = f"{x_title}: %{{x:{COMPOSITION_FORMAT}}}<br>{T_title}: %{{y:{TEMPERATURE_FORMAT}}}"

    fig = go.Figure()
    regions = traced.regions
    colors = field_colors(region.label for region in regions)
    total_area = traced.T_range[1] - traced.T_range[0]

    shown_in_legend: set[str] = set()
    for region in regions:
        x, T = region.polygon()
        fig.add_trace(
            go.Scatter(
                x=to_x(x), y=to_T(T), mode="lines", fill="toself",
                fillcolor=colors[region.label], opacity=1.0, line=dict(width=0),
                name=region.label, legendgroup=region.label,
                showlegend=region.label not in shown_in_legend,
                hoveron="fills", hoverinfo="name",
            )
        )
        shown_in_legend.add(region.label)

    for boundary in traced.boundaries:
        fig.add_trace(
            go.Scatter(
                x=to_x(boundary.x), y=to_T(boundary.T), mode="lines",
                line=dict(color="black", width=1.2), name=boundary.name, showlegend=False,
                hovertemplate=f"{boundary.name}<br>{point_hover}<extra></extra>",
            )
        )

    for reaction in traced.invariants:
        compositions = list(reaction.composition.values())
        T = to_T(reaction.temperature)
        fig.add_trace(
            go.Scatter(
                x=[to_x(min(compositions)), to_x(max(compositions))], y=[T, T], mode="lines",
                line=dict(color="black", width=1.2), showlegend=False,
                name=f"{reaction.type} line",
                hovertemplate=f"{invariant_label(reaction, temperature_unit)}<br>{point_hover}<extra></extra>",
            )
        )

    labelled = [r for r in regions if r.area >= MIN_LABELLED_AREA_FRACTION * total_area]
    positions = [r.label_position() for r in labelled]
    fig.add_trace(
        go.Scatter(
            x=[to_x(x) for x, _ in positions], y=[to_T(T) for _, T in positions],
            mode="text", text=[r.label for r in labelled], textfont=dict(size=12, color=PLOT_TEXT_COLOR),
            name="field labels", showlegend=False, hoverinfo="skip",
        )
    )

    if traced.invariants:
        fig.add_trace(
            go.Scatter(
                x=[to_x(r.composition[r.phases[1]]) for r in traced.invariants],
                y=[to_T(r.temperature) for r in traced.invariants],
                mode="markers+text",
                text=[invariant_label(r, temperature_unit) for r in traced.invariants],
                textposition="top center",
                textfont=dict(color=PLOT_TEXT_COLOR),
                marker=dict(symbol="diamond", size=9, color="black"),
                name="invariant reactions",
                hovertemplate=f"%{{text}}<br>{point_hover}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{system_name} phase diagram",
        xaxis_title=x_title,
        yaxis_title=T_title,
        legend_title="Phase fields",
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[to_x(0.0), to_x(1.0)], showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(
        range=[to_T(traced.T_range[0]), to_T(traced.T_range[1])], showline=True, linecolor="black", mirror=True
    )
    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path)
