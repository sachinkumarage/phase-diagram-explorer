"""Publication-style vector export of a traced phase diagram (SVG, PDF).

Rendered with matplotlib from the same traced boundaries, invariant lines
and field labels as the interactive figure, on a white background. Only
lines, polygons and text are drawn (text as paths in SVG, embedded TrueType
in PDF), so exported files contain no raster images.
"""
import csv
import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phase_diagram_explorer.invariants import InvariantReaction  # noqa: E402
from phase_diagram_explorer.tracing import TracedDiagram  # noqa: E402
from phase_diagram_explorer.units import (  # noqa: E402
    ATOMIC_PERCENT,
    KELVIN,
    composition_label,
    composition_to_display,
    temperature_label,
    temperature_to_display,
)
from phase_diagram_explorer.visualization import MIN_LABELLED_AREA_FRACTION  # noqa: E402

EXPORT_FORMATS = ("svg", "pdf")
FOOTNOTE_WIDTH = 110
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
):
    mass_a, mass_b = atomic_masses

    def to_x(x):
        return composition_to_display(x, composition_unit, mass_a, mass_b)

    def to_T(T):
        return temperature_to_display(T, temperature_unit)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        for boundary in traced.boundaries:
            ax.plot(to_x(boundary.x), to_T(boundary.T), color="black", linewidth=1.0)

        x_right = to_x(1.0)
        for reaction in traced.invariants:
            compositions = list(reaction.composition.values())
            T = to_T(reaction.temperature)
            ax.plot([to_x(min(compositions)), to_x(max(compositions))], [T, T], color="black", linewidth=1.0)
            ax.annotate(
                f"{T:.1f} {temperature_unit}", xy=(to_x(max(compositions)), T),
                xytext=(4, 2), textcoords="offset points", fontsize=8, ha="left", va="bottom",
            )

        total_area = traced.T_range[1] - traced.T_range[0]
        for region in traced.regions:
            if region.area < MIN_LABELLED_AREA_FRACTION * total_area:
                continue
            x, T = region.label_position()
            ax.text(to_x(x), to_T(T), region.label, ha="center", va="center", fontsize=9)

        ax.set_xlim(to_x(0.0), x_right)
        ax.set_ylim(to_T(traced.T_range[0]), to_T(traced.T_range[1]))
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
