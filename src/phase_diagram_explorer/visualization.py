import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from phase_diagram_explorer.invariants import InvariantReaction
from phase_diagram_explorer.labels import (
    PhaseNames,
    PlotBox,
    clip_polygon,
    field_names,
    layout_labels,
    polygon_anchor,
    polygon_area,
)
from phase_diagram_explorer.tracing import TracedDiagram
from phase_diagram_explorer.units import (
    ATOMIC_PERCENT,
    KELVIN,
    WEIGHT_PERCENT,
    composition_from_display,
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
# in the exported figure (they still show their phases on hover).
MIN_LABELLED_AREA_FRACTION = 0.005
# Degenerate fields below this area fraction get no label at all; others
# that do not fit are labelled outside with a leader line.
MIN_LEADER_AREA_FRACTION = 1e-5
LABEL_FONT_SIZE = 12
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


def reaction_names(reaction: InvariantReaction, names: PhaseNames) -> list[str]:
    """Display names of an invariant's three phases, in reaction.phases order."""
    return names.names([(phase, reaction.composition[phase]) for phase in reaction.phases], T=reaction.temperature)


def reaction_text(reaction: InvariantReaction, names: PhaseNames) -> str:
    """e.g. "L -> (Ag) + (Cu)"."""
    return names.reaction(reaction.type, reaction_names(reaction, names))


def composition_lines(marks: dict | None, atomic_masses) -> list[tuple[float, str, str]]:
    """(x as mole fraction, left label, right label) of the vertical
    composition lines in a system's diagram_marks (needs atomic masses)."""
    if not marks or None in tuple(atomic_masses):
        return []
    return [
        (float(composition_from_display(line["wt_percent"], WEIGHT_PERCENT, *atomic_masses)),
         line["left_label"], line["right_label"])
        for line in marks.get("composition_lines", [])
    ]


def critical_line_anchors(traced: TracedDiagram, marks: dict | None, T_range_k, x_range) -> list[tuple[str, str, float, float]]:
    """(name, description, x, T) of a point on each marked critical line
    (diagram_marks["critical_lines"]) inside the displayed range, in K and
    mole fraction.

    {"invariant": "eutectoid"}: the line of that invariant reaction;
    {"fields": [[P1, P2], ...], "edge": "x_min" | "x_max", "below_k": T}: that
    edge of the first two-phase field of those phases (at temperatures
    below `below_k`, if given)."""
    anchors = []
    for name, spec in (marks or {}).get("critical_lines", {}).items():
        if "invariant" in spec:
            for reaction in traced.invariants:
                if reaction.type == spec["invariant"] and T_range_k[0] <= reaction.temperature <= T_range_k[1]:
                    lo = max(min(reaction.composition.values()), x_range[0])
                    hi = min(max(reaction.composition.values()), x_range[1])
                    anchors.append((name, f"{name}: {' + '.join(reaction.phases)} ({reaction.type})",
                                    lo + 0.8 * (hi - lo), reaction.temperature))
                    break
            continue
        wanted = [tuple(fields) for fields in spec["fields"]]
        for strip in traced.fields:
            if strip.phases not in wanted:
                continue
            edge = strip.x_max if spec["edge"] == "x_max" else strip.x_min
            levels = [
                k for k, T in enumerate(strip.T)
                if T_range_k[0] <= T <= T_range_k[1] and x_range[0] <= edge[k] <= x_range[1]
                and T < spec.get("below_k", float("inf"))
            ]
            if levels:
                # halfway along the line in temperature
                T_mid = (strip.T[levels[0]] + strip.T[levels[-1]]) / 2.0
                x_mid = float(np.interp(T_mid, [strip.T[k] for k in levels], [edge[k] for k in levels]))
                anchors.append((name, f"{name}: {' + '.join(strip.phases)} boundary", x_mid, T_mid))
                break
    return anchors


def diagram_labels(traced: TracedDiagram, names: PhaseNames, to_x, to_T, T_range_k, temperature_unit: str,
                   plot: PlotBox, min_area_fraction: float = MIN_LEADER_AREA_FRACTION, x_range=(0.0, 1.0),
                   marks: dict | None = None, atomic_masses=(None, None)):
    """Field, invariant and marked-line labels laid out by
    labels.layout_labels, in display units. Fields are clipped to x_range
    (mole fraction) before their labels are placed."""
    x_lo, x_hi = (float(to_x(x)) for x in x_range)
    total_area = (T_range_k[1] - T_range_k[0]) * (x_range[1] - x_range[0])
    candidates = {}
    for region in traced.regions:
        x, T = region.polygon()
        x_clipped, T_clipped = clip_polygon(x, T, x_range)
        area = polygon_area(x_clipped, T_clipped) if len(x_clipped) else 0.0
        if area < min_area_fraction * total_area:
            continue
        x_anchor, T_anchor = polygon_anchor(x_clipped, T_clipped)
        if not T_range_k[0] <= T_anchor <= T_range_k[1]:
            continue
        display, calphad = field_names(names, region, traced)
        item = (display, calphad, (to_x(x_clipped), to_T(T_clipped)), (float(to_x(x_anchor)), float(to_T(T_anchor))))
        # one label per name: a field traced as several strips (e.g. the
        # liquid on both sides of a level where its neighbours change) is
        # labelled once, in its largest part
        if display not in candidates or area > candidates[display][0]:
            candidates[display] = (area, item)
    field_items = [item for _, item in candidates.values()]
    invariant_items = []
    for reaction in traced.invariants:
        if not T_range_k[0] <= reaction.temperature <= T_range_k[1]:
            continue
        compositions = list(reaction.composition.values())
        invariant_items.append((
            invariant_label(reaction, temperature_unit), reaction_text(reaction, names),
            max(float(to_x(min(compositions))), x_lo), min(float(to_x(max(compositions))), x_hi),
            float(to_T(reaction.temperature)), float(to_x(reaction.composition[reaction.phases[1]])),
        ))
    lines = [(to_x(b.x), to_T(b.T)) for b in traced.boundaries]
    lines += [([lo, hi], [T, T]) for _, _, lo, hi, T, _ in invariant_items]
    composition_items = []
    for x_line, left, right in composition_lines(marks, atomic_masses):
        composition_items.append((left, right, float(to_x(x_line))))
        lines.append(([float(to_x(x_line))] * 2, [float(to_T(T_range_k[0])), float(to_T(T_range_k[1]))]))
    line_items = [
        (name, description, (float(to_x(x)), float(to_T(T))))
        for name, description, x, T in critical_line_anchors(traced, marks, T_range_k, x_range)
    ]
    return layout_labels(field_items, invariant_items, lines, plot, line_items, composition_items)


def _beyond_range(region, x_range) -> bool:
    x, _ = region.polygon()
    return bool(
        (x_range[1] < 1.0 and np.min(x) >= x_range[1] - 1e-12) or (x_range[0] > 0.0 and np.max(x) <= x_range[0] + 1e-12)
    )


def plot_diagram(
    traced: TracedDiagram,
    system_name: str,
    *,
    dependent_symbol: str = "B",
    temperature_unit: str = KELVIN,
    composition_unit: str = ATOMIC_PERCENT,
    atomic_masses: tuple[float | None, float | None] = (None, None),
    display_names: dict | None = None,
    T_range_k: tuple[float, float] | None = None,
    x_range: tuple[float, float] | None = None,
    diagram_marks: dict | None = None,
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

    Fields are labelled with textbook names (`display_names`, see
    labels.PhaseNames), placed inside the field or outside with a leader
    line; the legend and hover text show the CALPHAD names as well.
    `T_range_k` is the displayed temperature range (default: the traced one)
    and `x_range` the displayed composition range (mole fraction; default 0
    to 1). `diagram_marks` adds dashed composition lines and labels for
    critical lines (see diagram_labels).
    """
    mass_a, mass_b = atomic_masses
    names = PhaseNames(display_names)
    T_range_k = tuple(T_range_k or traced.T_range)
    x_range = tuple(x_range or (0.0, 1.0))

    def to_x(x):
        return composition_to_display(x, composition_unit, mass_a, mass_b)

    def to_T(T):
        return temperature_to_display(T, temperature_unit)

    x_title = composition_label(dependent_symbol, composition_unit)
    T_title = temperature_label(temperature_unit)
    point_hover = f"{x_title}: %{{x:{COMPOSITION_FORMAT}}}<br>{T_title}: %{{y:{TEMPERATURE_FORMAT}}}"

    fig = go.Figure()
    # Fields lying wholly beyond a cut composition range (Fe-C: past Fe3C at
    # x = 0.25) are not drawn or listed in the legend.
    regions = [region for region in traced.regions if not _beyond_range(region, x_range)]
    legend_names = [names.legend(*field_names(names, region, traced)) for region in regions]
    colors = field_colors(legend_names)

    shown_in_legend: set[str] = set()
    for region, legend_name in zip(regions, legend_names):
        x, T = region.polygon()
        fig.add_trace(
            go.Scatter(
                x=to_x(x), y=to_T(T), mode="lines", fill="toself",
                fillcolor=colors[legend_name], opacity=1.0, line=dict(width=0),
                name=legend_name, legendgroup=legend_name,
                showlegend=legend_name not in shown_in_legend,
                hoveron="fills", hoverinfo="name",
            )
        )
        shown_in_legend.add(legend_name)

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
                hovertemplate=(
                    f"{invariant_label(reaction, temperature_unit)}: {reaction_text(reaction, names)}"
                    f"<br>{' + '.join(reaction.phases)}<br>{point_hover}<extra></extra>"
                ),
            )
        )

    if traced.invariants:
        # Markers only: the temperature labels are placed annotations (below).
        fig.add_trace(
            go.Scatter(
                x=[to_x(r.composition[r.phases[1]]) for r in traced.invariants],
                y=[to_T(r.temperature) for r in traced.invariants],
                mode="markers",
                text=[invariant_label(r, temperature_unit) for r in traced.invariants],
                customdata=[f"{reaction_text(r, names)}<br>{' + '.join(r.phases)}" for r in traced.invariants],
                marker=dict(symbol="diamond", size=9, color="black"),
                name="invariant reactions",
                hovertemplate=f"%{{text}}: %{{customdata}}<br>{point_hover}<extra></extra>",
            )
        )

    for x_line, left, right in composition_lines(diagram_marks, atomic_masses):
        fig.add_trace(
            go.Scatter(
                x=[to_x(x_line)] * 2, y=[to_T(T_range_k[0]), to_T(T_range_k[1])], mode="lines",
                line=dict(color=PLOT_TEXT_COLOR, width=1.0, dash="dash"), showlegend=False,
                name=f"{left} / {right}", hovertemplate=f"{left} / {right}: {x_title} %{{x:{COMPOSITION_FORMAT}}}<extra></extra>",
            )
        )

    plot = PlotBox(
        (float(to_x(x_range[0])), float(to_x(x_range[1]))), (float(to_T(T_range_k[0])), float(to_T(T_range_k[1])))
    )
    labels = diagram_labels(
        traced, names, to_x, to_T, T_range_k, temperature_unit, plot, x_range=x_range, marks=diagram_marks,
        atomic_masses=atomic_masses,
    )
    for label in labels:
        common = dict(
            text=label.text, hovertext=label.hover, font=dict(size=LABEL_FONT_SIZE, color=PLOT_TEXT_COLOR),
            name=f"{label.kind} label", xanchor="center", yanchor="middle",
        )
        if label.leader:
            fig.add_annotation(
                x=label.anchor[0], y=label.anchor[1], ax=label.position[0], ay=label.position[1],
                axref="x", ayref="y", showarrow=True, arrowhead=0, arrowwidth=0.8, arrowcolor=PLOT_TEXT_COLOR,
                standoff=2, **common,
            )
        else:
            fig.add_annotation(x=label.position[0], y=label.position[1], showarrow=False, **common)

    fig.update_layout(
        title=f"{system_name} phase diagram",
        xaxis_title=x_title,
        yaxis_title=T_title,
        legend_title="Phase fields",
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[to_x(x_range[0]), to_x(x_range[1])], showline=True, linecolor="black", mirror=True)
    fig.update_yaxes(
        range=[to_T(T_range_k[0]), to_T(T_range_k[1])], showline=True, linecolor="black", mirror=True
    )
    return fig


def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path)
