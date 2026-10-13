"""Publication-style vector export of a traced phase diagram (SVG, PDF).

Rendered with matplotlib from the same traced boundaries, invariant lines
and field labels as the interactive figure, on a white background. Only
lines, polygons and text are drawn (text as paths in SVG, embedded TrueType
in PDF), so exported files contain no raster images.
"""
import csv
import io
import textwrap

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
from phase_diagram_explorer.labels import PhaseNames, PlotBox
from phase_diagram_explorer.visualization import MIN_LABELLED_AREA_FRACTION, diagram_labels

EXPORT_FORMATS = ("svg", "pdf")


def _pyplot():
    """matplotlib.pyplot with the non-interactive Agg backend, imported on
    first use so that the app starts without loading matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
FOOTNOTE_WIDTH = 110
FIGURE_SIZE_IN = (6.5, 5.2)
LABEL_FONT_SIZE = 9
# The label layout's pixel model (labels.CHAR_WIDTH_PX) is for 12 px text;
# axes sizes in points are scaled by this factor for 9 pt labels.
LAYOUT_SCALE = 1.3
STYLE = {
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def publication_figure(
    traced: TracedDiagram,
    system_name: str,
    *,
    dependent_symbol: str = "B",
    temperature_unit: str = KELVIN,
    composition_unit: str = ATOMIC_PERCENT,
    atomic_masses: tuple[float | None, float | None] = (None, None),
    footnote: str = "",
    display_names: dict | None = None,
):
    plt = _pyplot()
    mass_a, mass_b = atomic_masses

    def to_x(x):
        return composition_to_display(x, composition_unit, mass_a, mass_b)

    def to_T(T):
        return temperature_to_display(T, temperature_unit)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN)
        for boundary in traced.boundaries:
            ax.plot(to_x(boundary.x), to_T(boundary.T), color="black", linewidth=1.0)

        x_right = to_x(1.0)
        for reaction in traced.invariants:
            compositions = list(reaction.composition.values())
            T = to_T(reaction.temperature)
            ax.plot([to_x(min(compositions)), to_x(max(compositions))], [T, T], color="black", linewidth=1.0)

        ax.set_xlim(to_x(0.0), x_right)
        ax.set_ylim(to_T(traced.T_range[0]), to_T(traced.T_range[1]))
        fig.canvas.draw()
        size = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plot = PlotBox(
            (float(to_x(0.0)), float(x_right)), (float(to_T(traced.T_range[0])), float(to_T(traced.T_range[1]))),
            width_px=size.width * 72 * LAYOUT_SCALE, height_px=size.height * 72 * LAYOUT_SCALE,
        )
        labels = diagram_labels(
            traced, PhaseNames(display_names), to_x, to_T, traced.T_range, temperature_unit, plot,
            min_area_fraction=MIN_LABELLED_AREA_FRACTION,
        )
        for label in labels:
            text = f"{label.anchor[1]:.1f} {temperature_unit}" if label.kind == "invariant" else label.text
            if label.leader:
                ax.annotate(
                    text, xy=label.anchor, xytext=label.position, ha="center", va="center", fontsize=LABEL_FONT_SIZE,
                    arrowprops=dict(arrowstyle="-", linewidth=0.6, color="black", shrinkA=0, shrinkB=1),
                )
            else:
                ax.text(*label.position, text, ha="center", va="center", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel(composition_label(dependent_symbol, composition_unit))
        ax.set_ylabel(temperature_label(temperature_unit))
        ax.set_title(system_name)

        if footnote:
            wrapped = "\n".join(textwrap.wrap(footnote, FOOTNOTE_WIDTH))
            fig.text(0.01, 0.01, wrapped, fontsize=6.5, ha="left", va="bottom", color="0.25")
            fig.subplots_adjust(bottom=0.11 + 0.025 * wrapped.count("\n"))
    return fig


def export_figure(traced: TracedDiagram, system_name: str, fmt: str, **kwargs) -> bytes:
    """The publication figure as SVG or PDF bytes."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")
    plt = _pyplot()
    fig = publication_figure(traced, system_name, **kwargs)
    buffer = io.BytesIO()
    with plt.rc_context(STYLE):
        fig.savefig(buffer, format=fmt)
    plt.close(fig)
    return buffer.getvalue()


def invariants_csv(reactions: list[InvariantReaction], dependent_symbol: str = "B") -> str:
    """Invariant reactions as CSV at full precision: temperature in K and
    each phase's composition as the mole fraction of the dependent element
    (shortest round-tripping representation)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["type", "temperature_K", "phase_1", f"x_{dependent_symbol}_1", "phase_2", f"x_{dependent_symbol}_2",
         "phase_3", f"x_{dependent_symbol}_3"]
    )
    for reaction in reactions:
        row = [reaction.type, repr(float(reaction.temperature))]
        for phase in reaction.phases:
            row += [phase, repr(float(reaction.composition[phase]))]
        writer.writerow(row)
    return buffer.getvalue()
