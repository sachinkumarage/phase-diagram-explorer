"""The assessed Fe-C system (data/systems/fe_c.meta.json): its two diagram
modes, reference values, and the comparison with pycalphad behind
docs/validation/fe_c.md and the tests.

Data: pycalphad's test database cfe_broshe.tdb (release 0.11.2), fetched
from the pycalphad repository and verified by SHA256 (data/sources.json).
Hallstedt et al., Calphad 34 (2010) 129-133, with unary data of Dinsdale
(1991) and the equation of state of Brosh et al. (2007). M7C3_D101 and
DIAMOND_A4 are left out (see the .meta.json).

Modes:
- "metastable": the Fe-Fe3C diagram of steel practice (GRAPHITE suspended);
- "stable": the Fe-graphite diagram.

Compositions are mole fractions of C internally; wt% uses the database's
molar masses (Fe 55.847, C 12.011 g/mol):
w = 100 x M_C / (x M_C + (1 - x) M_Fe).

    python -m phase_diagram_explorer.fe_c     # regenerate docs/validation/fe_c.md and its plot
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name
from phase_diagram_explorer.invariants import InvariantReaction, detect_invariants_over_range
from phase_diagram_explorer.models import SystemDefinition, load_system
from phase_diagram_explorer.units import CELSIUS_OFFSET, mole_to_weight_fraction

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "systems" / "fe_c.meta.json"
MODES = ("metastable", "stable")
# Cementite Fe3C: the right-hand end of the displayed diagram.
X_CEMENTITE = 0.25

# This database's values (step 0 of the Fe-C work: this engine and
# pycalphad agree to 0.002 K and 4e-5 in mole fraction). T in °C,
# compositions in wt% C, keyed by the CALPHAD phase.
DATABASE_VALUES = {
    "metastable": {
        "eutectoid": {"T": 726.53, "FCC_A1": 0.7617, "BCC_A2": 0.0191},
        "eutectic": {"T": 1148.16, "LIQUID": 4.3834, "FCC_A1": 2.0541},
        "peritectic": {"T": 1494.61},
    },
    "stable": {
        "eutectoid": {"T": 738.02},
        "eutectic": {"T": 1153.43},
        "peritectic": {"T": 1494.61},
    },
}
DATABASE_T_TOLERANCE_K = 0.05
DATABASE_W_TOLERANCE = 0.0005  # wt%


@dataclass(frozen=True)
class LiteratureValue:
    mode: str
    reaction: str
    quantity: str  # "T" (°C) or a phase name (wt% C)
    value: float
    tolerance: float
    label: str


# Commonly quoted textbook values, not taken from a single assessed source,
# with tolerances that cover the spread between published assessments and
# compilations (see the tests' docstrings).
LITERATURE = (
    LiteratureValue("metastable", "eutectoid", "T", 727.0, 2.0, "Eutectoid T (°C)"),
    LiteratureValue("metastable", "eutectoid", "FCC_A1", 0.76, 0.03, "Eutectoid austenite (wt% C)"),
    LiteratureValue("metastable", "eutectoid", "BCC_A2", 0.022, 0.004, "Max C in ferrite (wt% C)"),
    LiteratureValue("metastable", "eutectic", "T", 1147.0, 2.0, "Eutectic T (°C)"),
    LiteratureValue("metastable", "eutectic", "LIQUID", 4.30, 0.10, "Eutectic liquid (wt% C)"),
    LiteratureValue("metastable", "eutectic", "FCC_A1", 2.14, 0.12, "Max C in austenite (wt% C)"),
    LiteratureValue("metastable", "peritectic", "T", 1495.0, 2.0, "Peritectic T (°C)"),
    LiteratureValue("stable", "eutectoid", "T", 738.0, 2.0, "Stable eutectoid T (°C)"),
    LiteratureValue("stable", "eutectic", "T", 1153.0, 2.0, "Stable eutectic T (°C)"),
)


def definition(include_suspended: bool = False) -> SystemDefinition:
    return load_system(META, include_suspended=include_suspended)


def masses(fe_c: SystemDefinition) -> tuple[float, float]:
    return fe_c.base_element.atomic_mass, fe_c.dependent_element.atomic_mass


def weight_percent(x, fe_c: SystemDefinition):
    return 100.0 * mole_to_weight_fraction(x, *masses(fe_c))


def system(mode: str, include: tuple[str, ...] = ()) -> dict:
    """The evaluable system in `mode`, with the suspended phases in `include`
    put back (e.g. ("M7C3_D101",), for the test that leaving it out changes
    nothing)."""
    full = definition(include_suspended=True).in_mode(mode)
    left_out = set(full.suspended_phases) - set(include)
    return build_system(full.model_copy(update={"phases": [p for p in full.phases if p.name not in left_out]}))


def invariants(mode: str, include: tuple[str, ...] = ()) -> list[InvariantReaction]:
    """Invariant reactions over the analysis range (400-1600 °C)."""
    return detect_invariants_over_range(system(mode, include), definition().t_range_k)


def by_type(reactions: list[InvariantReaction]) -> dict[str, InvariantReaction]:
    return {reaction.type: reaction for reaction in reactions}


def composition(reaction: InvariantReaction, phase: str) -> float:
    """Mole fraction of C of `phase` (a CALPHAD name) in the reaction."""
    (x,) = [x for label, x in reaction.composition.items() if base_phase_name(label) == phase]
    return x


# --- pycalphad ------------------------------------------------------------------

PYCALPHAD_T_TOLERANCE_K = 0.002
TIE_LINE_OFFSET_K = 0.05


def pycalphad_reference(mode: str):
    """pycalphad on the same file with the same phases as `mode`."""
    from phase_diagram_explorer.sources import ensure
    from phase_diagram_explorer.validation import PycalphadSystem

    fe_c = definition()
    reference = PycalphadSystem(ensure("pycalphad_cfe_broshe", download=False), fe_c)
    reference.phases = sorted(phase.name for phase in fe_c.in_mode(mode).phases)
    return reference


def pycalphad_invariant(reference, reaction: InvariantReaction) -> tuple[float, dict[str, float]]:
    """pycalphad's temperature of `reaction` (where its middle phase appears
    or disappears inside the reaction line, bisected) and the three phase
    compositions from its tie lines TIE_LINE_OFFSET_K away, on the side where
    the middle phase exists."""
    from phase_diagram_explorer.validation import pycalphad_transition_temperature

    p1, p2, p3 = (base_phase_name(label) for label in reaction.phases)
    x1, x2, x3 = (reaction.composition[label] for label in reaction.phases)
    T = pycalphad_transition_temperature(
        reference, (x1 + x2) / 2.0, p2, reaction.temperature - 15.0, reaction.temperature + 15.0,
        PYCALPHAD_T_TOLERANCE_K,
    )
    for offset in (TIE_LINE_OFFSET_K, -TIE_LINE_OFFSET_K):
        left = {s.phase: s.x for s in reference.equilibrium(T + offset, (x1 + x2) / 2.0)}
        right = {s.phase: s.x for s in reference.equilibrium(T + offset, (x2 + x3) / 2.0)}
        if set(left) == {p1, p2} and set(right) == {p2, p3}:
            return T, {p1: left[p1], p2: (left[p2] + right[p2]) / 2.0, p3: right[p3]}
    raise ValueError(f"no tie lines {p1}+{p2} and {p2}+{p3} next to {T:.3f} K in pycalphad")


def pycalphad_boundary_points(reference, T_range, n_T: int = 41, n_x: int = 126):
    """(x, T) end points of pycalphad's two-phase tie lines on a grid with
    x(C) <= 0.25."""
    temperatures = np.linspace(T_range[0] + 1.0, T_range[1] - 1.0, n_T)
    xs = np.linspace(0.0005, X_CEMENTITE - 0.0005, n_x)
    states = reference.equilibria(temperatures, xs)
    points = set()
    for i, T in enumerate(temperatures):
        for row in states[i]:
            if len(row) == 2:
                for state in row:
                    if state.x <= X_CEMENTITE + 1e-9:
                        points.add((round(state.x, 6), float(T)))
    return sorted(points)


# --- report ---------------------------------------------------------------------


def _overlay_plot(path: Path, traced_by_mode: dict, references: dict, fe_c: SystemDefinition) -> None:
    from phase_diagram_explorer.export import _pyplot

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        traced = traced_by_mode[mode]
        for boundary in traced.boundaries:
            ax.plot(weight_percent(np.asarray(boundary.x), fe_c), np.asarray(boundary.T) - CELSIUS_OFFSET,
                    color="black", linewidth=0.9)
        for reaction in traced.invariants:
            xs = [min(reaction.composition.values()), min(max(reaction.composition.values()), X_CEMENTITE)]
            ax.plot(weight_percent(np.array(xs), fe_c), [reaction.temperature - CELSIUS_OFFSET] * 2,
                    color="black", linewidth=0.9)
        points = pycalphad_boundary_points(references[mode], fe_c.t_range_k)
        if points:
            x, T = np.array(points).T
            ax.plot(weight_percent(x, fe_c), T - CELSIUS_OFFSET, "o", markersize=2.0, color="tab:red",
                    label="pycalphad tie-line ends")
        ax.set_xlim(0.0, float(weight_percent(X_CEMENTITE, fe_c)))
        ax.set_ylim(fe_c.t_range_k[0] - CELSIUS_OFFSET, fe_c.t_range_k[1] - CELSIUS_OFFSET)
        ax.set_xlabel("w(C) (wt%)")
        ax.set_title(fe_c.mode(mode).label)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_report(docs_dir: Path = ROOT / "docs") -> Path:
    """docs/validation/fe_c.md: this engine vs pycalphad vs this database's
    reference values vs commonly quoted textbook values, for both modes, with
    an overlay plot."""
    import pycalphad

    from phase_diagram_explorer import __version__
    from phase_diagram_explorer.tracing import DEFAULT_LEVELS, trace_diagram

    fe_c = definition()
    lines = [
        "# Fe-C: assessed system vs pycalphad and textbook values",
        "",
        f"Generated by `python -m phase_diagram_explorer.fe_c` with phase-diagram-explorer {__version__} and "
        f"pycalphad {pycalphad.__version__}. The tests in `tests/test_fe_c.py` and "
        "`tests/validation/test_fe_c_vs_pycalphad.py` check these values.",
        "",
        "**Data.** pycalphad's test database `cfe_broshe.tdb` (release 0.11.2), fetched from the pycalphad "
        "repository (`scripts/fetch_tdb.py`, verified by SHA256) and not redistributed here. The file carries no "
        "license text of its own; the pycalphad repository is MIT-licensed.",
        "",
    ]
    lines += [f"- {citation.text}" for citation in fe_c.citations]
    lines += ["", "**Phases left out** (in both modes):", ""]
    lines += [f"- {name}: {reason}" for name, reason in fe_c.suspended_phases.items()]
    lines += [
        "",
        "**Modes.** Metastable Fe-Fe3C (steel practice): GRAPHITE suspended. Stable Fe-graphite: GRAPHITE "
        "included (cementite stays in the system but is never stable). The diagram is shown for "
        "x(C) = 0 to 0.25 (0 to 6.69 wt% C, Fe3C).",
        "",
        f"Compositions: w = 100 x M_C / (x M_C + (1 - x) M_Fe), with the database's M_Fe = "
        f"{masses(fe_c)[0]} and M_C = {masses(fe_c)[1]} g/mol. Temperatures: °C = K - {CELSIUS_OFFSET}.",
        "",
    ]
    traced_by_mode, references = {}, {}
    for mode in MODES:
        reactions = invariants(mode)
        reference = pycalphad_reference(mode)
        references[mode] = reference
        lines += [
            f"## {fe_c.mode(mode).label}",
            "",
            "| Reaction | Quantity | This engine | pycalphad | Database reference | Commonly quoted textbook values |",
            "|---|---|---|---|---|---|",
        ]
        literature = {(v.reaction, v.quantity): v for v in LITERATURE if v.mode == mode}
        for reaction in reactions:
            T_ref, x_ref = pycalphad_invariant(reference, reaction)
            database = DATABASE_VALUES[mode].get(reaction.type, {})
            phases = " + ".join(base_phase_name(p) for p in reaction.phases)
            entry = literature.get((reaction.type, "T"))
            lines.append(
                f"| {reaction.type} ({phases}) | T (°C) | {reaction.temperature - CELSIUS_OFFSET:.3f} | "
                f"{T_ref - CELSIUS_OFFSET:.3f} | {database['T']:.2f} ± {DATABASE_T_TOLERANCE_K} | "
                + (f"{entry.value:g} ± {entry.tolerance:g} |" if entry else "- |")
            )
            for label in reaction.phases:
                phase = base_phase_name(label)
                x = reaction.composition[label]
                if x > X_CEMENTITE + 1e-9:
                    continue
                entry = literature.get((reaction.type, phase))
                stored = database.get(phase)
                lines.append(
                    f"| | {phase} wt% C (x) | {float(weight_percent(x, fe_c)):.4f} ({x:.6f}) | "
                    f"{float(weight_percent(x_ref[phase], fe_c)):.4f} ({x_ref[phase]:.6f}) | "
                    + (f"{stored:.4f} ± {DATABASE_W_TOLERANCE} | " if stored is not None else "- | ")
                    + (f"{entry.value:g} ± {entry.tolerance:g} |" if entry else "- |")
                )
        lines.append("")
        traced_by_mode[mode] = trace_diagram(
            system(mode), fe_c.t_range_k, invariants=reactions, n_levels=DEFAULT_LEVELS
        )
    lines += [
        "Tolerances in the tests: this engine vs pycalphad 0.01 K and 1e-4 in mole fraction; vs the database "
        f"reference values {DATABASE_T_TOLERANCE_K} K and {DATABASE_W_TOLERANCE} wt%; vs the textbook values as "
        "listed (the spread between published assessments and compilations). pycalphad's temperature is where the "
        "reaction's middle phase appears or disappears inside the reaction line (bisected to "
        f"{PYCALPHAD_T_TOLERANCE_K} K); its compositions are from tie lines {TIE_LINE_OFFSET_K} K away.",
        "",
        "**Textbook values.** The last column lists values commonly quoted in materials science textbooks and "
        "course notes. They are not taken from a single assessed source, and no single source has been "
        "confirmed for all of them; they serve as a plausibility check, while the quantitative validation is "
        "against pycalphad on the same database file.",
        "",
        "Leaving out M7C3_D101 changes nothing below x(C) = 0.25: with it, the metastable system has one more "
        "invariant, Fe3C + L + M7C3 at 1210.7 °C and 7.7-8.4 wt% C, beyond Fe3C (checked in `tests/test_fe_c.py`).",
        "",
        "## Phase boundaries",
        "",
        "Lines: boundaries traced by this engine. Points: two-phase tie-line end points from pycalphad "
        "equilibria on the same file (41 temperatures x 126 compositions per mode).",
        "",
        "![Fe-C](../img/fe_c_vs_pycalphad.svg)",
        "",
    ]
    _overlay_plot(docs_dir / "img" / "fe_c_vs_pycalphad.svg", traced_by_mode, references, fe_c)
    path = docs_dir / "validation" / "fe_c.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate docs/validation/fe_c.md")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    print(write_report(parser.parse_args(argv).docs_dir))


if __name__ == "__main__":
    main()
