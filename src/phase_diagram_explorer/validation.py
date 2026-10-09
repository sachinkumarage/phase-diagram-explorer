"""Cross-validation of the equilibrium engine against pycalphad.

pycalphad is an optional dependency (pip install ".[validation]"). Both
engines read the same TDB file: this package through load_system, pycalphad
through its own TDB reader, so the comparison also checks the TDB reader and
writer.

Run as a module to regenerate the comparison report:

    python -m phase_diagram_explorer.validation

which writes docs/validation/engine_vs_pycalphad.md and overlay plots of
both engines' phase boundaries to docs/img/.
"""
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.models import SystemDefinition, load_system
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
# TDB fixtures compared by the validation tests and the report.
FIXTURES = {
    "Ag-Cu (provisional)": "ag_cu_provisional.tdb",
    "Al-Cu (provisional)": "al_cu_provisional.tdb",
    "Regular solution (miscibility gap)": "regular_solution.tdb",
    "Gap eutectic": "gap_eutectic.tdb",
}
# A literature assessment shipped with pycalphad's test databases.
LITERATURE_NAME = "Pb-Sn (Ngai and Chang, CALPHAD 5 (1981) 267; pycalphad test database)"
LITERATURE_T_RANGE_K = (325.0, 625.0)


def literature_tdb() -> Path:
    import pycalphad

    return Path(pycalphad.__file__).parent / "tests" / "databases" / "pbsn.tdb"


PRESSURE_PA = 101325.0
# Validation grid: N_T temperatures x N_X compositions = 30 points per system.
N_T = 6
N_X = 5
GRID_MARGIN_K = 25.0
VALIDATION_X = (0.05, 0.27, 0.5, 0.73, 0.95)
# pycalphad vertices of one phase closer than this in composition are one phase.
SAME_COMPOSITION = 1e-4
MIN_FRACTION = 1e-12


@dataclass(frozen=True)
class PhaseState:
    phase: str  # base phase name
    fraction: float
    x: float  # mole fraction of the dependent element


@dataclass(frozen=True)
class PointComparison:
    T: float
    x: float
    ours: tuple[PhaseState, ...]
    pycalphad: tuple[PhaseState, ...]

    @property
    def same_phases(self) -> bool:
        return [s.phase for s in self.ours] == [s.phase for s in self.pycalphad]

    @property
    def fraction_error(self) -> float:
        return max(abs(a.fraction - b.fraction) for a, b in zip(self.ours, self.pycalphad))

    @property
    def composition_error(self) -> float:
        return max(abs(a.x - b.x) for a, b in zip(self.ours, self.pycalphad))


def _pycalphad():
    import pycalphad

    warnings.filterwarnings("ignore", module="pycalphad")
    return pycalphad


class PycalphadSystem:
    """pycalphad's view of a TDB file, in this package's conventions:
    temperatures in K, compositions as the dependent element's mole
    fraction, phases sorted by composition."""

    def __init__(self, tdb_path: str | Path, definition: SystemDefinition):
        pycalphad = _pycalphad()
        self.variables = pycalphad.variables
        self._equilibrium = pycalphad.equilibrium
        self.database = pycalphad.Database(str(tdb_path))
        self.components = [definition.base_element.symbol.upper(), definition.dependent_element.symbol.upper(), "VA"]
        self.dependent = definition.dependent_element.symbol.upper()
        self.phases = sorted(self.database.phases)

    def equilibria(self, T, x):
        """Equilibrium states on the grid T x x: result[i][j] is the tuple of
        PhaseStates at (T[i], x[j])."""
        v = self.variables
        T, x = np.atleast_1d(T).astype(float), np.atleast_1d(x).astype(float)
        result = self._equilibrium(
            self.database, self.components, self.phases,
            {v.X(self.dependent): x, v.T: T, v.P: PRESSURE_PA, v.N: 1.0},
        )
        names = result.Phase.values[0, 0]
        fractions = result.NP.values[0, 0]
        compositions = result.X.sel(component=self.dependent).values[0, 0]
        return [
            [_merge(names[i, j], fractions[i, j], compositions[i, j]) for j in range(len(x))]
            for i in range(len(T))
        ]

    def equilibrium(self, T: float, x: float) -> tuple[PhaseState, ...]:
        return self.equilibria(T, x)[0][0]

    def has_phase(self, T: float, x: float, phase: str) -> bool:
        return any(state.phase == phase for state in self.equilibrium(T, x))


def _merge(names, fractions, compositions) -> tuple[PhaseState, ...]:
    states: list[PhaseState] = []
    for name, fraction, x in zip(names, fractions, compositions):
        name = str(name)
        if not name or not np.isfinite(fraction) or fraction < MIN_FRACTION:
            continue
        for k, state in enumerate(states):
            if state.phase == name and abs(state.x - x) < SAME_COMPOSITION:
                states[k] = PhaseState(name, state.fraction + float(fraction), state.x)
                break
        else:
            states.append(PhaseState(name, float(fraction), float(x)))
    return tuple(sorted(states, key=lambda state: state.x))


def our_equilibrium(system: dict, T: float, x: float) -> tuple[PhaseState, ...]:
    result = compute_equilibrium(system, T, x)
    return tuple(
        sorted(
            (
                PhaseState(base_phase_name(label), result.phase_fractions[label], result.phase_compositions[label])
                for label in result.stable_phases
            ),
            key=lambda state: state.x,
        )
    )


def validation_grid(T_range: tuple[float, float]) -> list[tuple[float, float]]:
    T_min, T_max = T_range
    temperatures = np.linspace(T_min + GRID_MARGIN_K, T_max - GRID_MARGIN_K, N_T)
    return [(float(T), x) for T in temperatures for x in VALIDATION_X]


def compare_grid(tdb_path: str | Path, T_range: tuple[float, float] | None = None, **load_kwargs) -> list[PointComparison]:
    """Both engines at 30 (T, x) points over T_range (default: the system's
    t_range_k)."""
    definition = load_system(tdb_path, **load_kwargs)
    system = build_system(definition)
    reference = PycalphadSystem(tdb_path, definition)
    points = validation_grid(T_range or definition.t_range_k)
    temperatures = sorted({T for T, _ in points})
    states = reference.equilibria(temperatures, VALIDATION_X)
    return [
        PointComparison(T, x, our_equilibrium(system, T, x), states[temperatures.index(T)][VALIDATION_X.index(x)])
        for T, x in points
    ]


def pycalphad_phase_boundary_temperature(
    reference: PycalphadSystem, x: float, phase: str, T_low: float, T_high: float, tolerance: float = 0.01
) -> float:
    """Temperature at which `phase` appears on heating at composition x,
    by bisection on pycalphad equilibria: absent at T_low, present at T_high."""
    if reference.has_phase(T_low, x, phase) or not reference.has_phase(T_high, x, phase):
        raise ValueError(f"{phase} must be absent at {T_low} K and present at {T_high} K for x = {x}")
    while T_high - T_low > tolerance:
        middle = (T_low + T_high) / 2.0
        if reference.has_phase(middle, x, phase):
            T_high = middle
        else:
            T_low = middle
    return (T_low + T_high) / 2.0


def pycalphad_binodal(reference: PycalphadSystem, T: float, x: float = 0.5) -> tuple[float, float]:
    states = reference.equilibrium(T, x)
    if len(states) != 2 or states[0].phase != states[1].phase:
        raise ValueError(f"no miscibility gap at T = {T} K, x = {x}: {states}")
    return states[0].x, states[1].x


def pycalphad_boundary_points(reference: PycalphadSystem, T_range, n_T: int = 60, n_x: int = 201):
    """(T, x) end points of every two-phase tie line found on a T x x grid."""
    temperatures = np.linspace(T_range[0], T_range[1], n_T)
    states = reference.equilibria(temperatures, np.linspace(0.0005, 0.9995, n_x))
    points = set()
    for T, row in zip(temperatures, states):
        for phases in row:
            if len(phases) == 2:
                points.update((float(T), round(state.x, 6)) for state in phases)
    return sorted(points)


# --- report -------------------------------------------------------------------

PYCALPHAD_R = 8.3145


def gap_eutectic_temperature(R: float, L0: float = 20000.0, H_fus: float = 10000.0, T_m: float = 1000.0) -> float:
    """Analytical eutectic of the symmetric gap-eutectic fixture: where the
    ideal liquid at x = 0.5 meets the horizontal tangent of the regular
    solution's binodal."""
    from scipy.optimize import brentq

    def binodal(T):
        return brentq(lambda x: np.log(x / (1 - x)) - L0 * (2 * x - 1) / (R * T), 1e-12, 0.5 - 1e-9)

    def alpha(T):
        x = binodal(T)
        return R * T * (x * np.log(x) + (1 - x) * np.log(1 - x)) + L0 * x * (1 - x)

    return brentq(lambda T: H_fus * (1 - T / T_m) + R * T * np.log(0.5) - alpha(T), 600.0, 1000.0)



def _overlay_plot(name: str, tdb_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phase_diagram_explorer.tracing import trace_diagram

    definition = load_system(tdb_path)
    traced = trace_diagram(build_system(definition), definition.t_range_k)
    points = pycalphad_boundary_points(PycalphadSystem(tdb_path, definition), definition.t_range_k)

    with plt.rc_context({"svg.fonttype": "path", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        for k, boundary in enumerate(traced.boundaries):
            ax.plot(boundary.x, boundary.T, color="black", linewidth=1.0, label="this engine (traced)" if k == 0 else None)
        for reaction in traced.invariants:
            compositions = list(reaction.composition.values())
            ax.plot([min(compositions), max(compositions)], [reaction.temperature] * 2, color="black", linewidth=1.0)
        if points:
            T, x = zip(*points)
            ax.plot(x, T, linestyle="none", marker="o", markersize=2.2, color="tab:red", label="pycalphad tie-line ends")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*definition.t_range_k)
        ax.set_xlabel(f"x({definition.dependent_element.symbol})")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(output, format="svg")
        plt.close(fig)


def write_report(docs_dir: Path = ROOT / "docs") -> Path:
    import pycalphad

    from phase_diagram_explorer import __version__
    from phase_diagram_explorer.invariants import detect_invariants_over_range

    image_dir = docs_dir / "img"
    report_path = docs_dir / "validation" / "engine_vs_pycalphad.md"
    image_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Engine cross-validation against pycalphad",
        "",
        f"Generated by `python -m phase_diagram_explorer.validation` with phase-diagram-explorer {__version__} "
        f"and pycalphad {pycalphad.__version__}. The tests in `tests/validation/` check the same quantities "
        "(marker `validation`).",
        "",
        "Both engines read the same TDB fixtures in `tests/fixtures/`. The Ag-Cu and Al-Cu fixtures are the "
        "provisional example systems exported with `write_tdb`; they are test fixtures, not literature data. "
        "This comparison checks that the engine solves the model correctly, not that the model is right.",
        "",
        "## Equilibrium at 30 (T, x) points per system",
        "",
        f"A grid of {N_T} temperatures (the system range less {GRID_MARGIN_K:g} K at each end) by compositions "
        f"x = {', '.join(f'{x:g}' for x in VALIDATION_X)}. Phases are compared by name, ordered by composition "
        "(composition sets of one phase compare as two phases of that name). Tolerance: 1e-3 on phase "
        "fractions and on phase compositions.",
        "",
        "| System | Points | Same stable phases | Max phase fraction difference | Max phase composition difference |",
        "|---|---|---|---|---|",
    ]
    grids = {name: (FIXTURES_DIR / file_name, None, {}) for name, file_name in FIXTURES.items()}
    grids[LITERATURE_NAME] = (literature_tdb(), LITERATURE_T_RANGE_K, {"elements": ["Pb", "Sn"]})
    for name, (path, T_range, load_kwargs) in grids.items():
        comparisons = compare_grid(path, T_range, **load_kwargs)
        agreeing = [c for c in comparisons if c.same_phases]
        fraction = max(c.fraction_error for c in agreeing)
        composition = max(c.composition_error for c in agreeing)
        lines.append(
            f"| {name} | {len(comparisons)} | {len(agreeing)}/{len(comparisons)} | {fraction:.1e} | {composition:.1e} |"
        )

    lines += [
        "",
        "The Pb-Sn row uses a literature assessment from pycalphad's own test databases (not copied into this "
        f"repository), over {LITERATURE_T_RANGE_K[0]:g}-{LITERATURE_T_RANGE_K[1]:g} K. Its Gibbs energies agree "
        "with pycalphad's to 1e-6 J/mol once pycalphad's gas constant (8.3145 J/(mol K), against "
        "8.314462618 here) is allowed for in the ideal mixing term.",
        "",
        "## Eutectic temperatures",
        "",
        "This engine root-finds the three-phase condition. The pycalphad value is the temperature at which "
        "LIQUID appears on heating at a composition inside the eutectic line, bisected on pycalphad equilibria "
        "to 0.001 K. Tolerance: 0.5 K.",
        "",
        "| System | x | This engine (K) | pycalphad (K) | Difference (K) |",
        "|---|---|---|---|---|",
    ]
    eutectics = [
        ("Ag-Cu (provisional)", FIXTURES_DIR / FIXTURES["Ag-Cu (provisional)"], 0.4, (1000.0, 1100.0), {}),
        ("Gap eutectic", FIXTURES_DIR / FIXTURES["Gap eutectic"], 0.5, (600.0, 1000.0), {}),
        ("Pb-Sn", literature_tdb(), 0.6, (400.0, 500.0), {"elements": ["Pb", "Sn"]}),
    ]
    for name, path, x, (T_low, T_high), load_kwargs in eutectics:
        definition = load_system(path, **load_kwargs)
        (reaction,) = detect_invariants_over_range(build_system(definition), definition.t_range_k or (T_low, T_high))
        T_reference = pycalphad_phase_boundary_temperature(
            PycalphadSystem(path, definition), x, "LIQUID", T_low, T_high, tolerance=0.001
        )
        lines.append(
            f"| {name} | {x:g} | {reaction.temperature:.3f} | {T_reference:.3f} | "
            f"{abs(reaction.temperature - T_reference):.3f} |"
        )
    lines += [
        "",
        "The gap eutectic is symmetric, so its temperature is also known analytically: the ideal LIQUID at "
        "x = 0.5 meets the horizontal common tangent of the two ALPHA composition sets. That gives "
        + ", ".join(f"{gap_eutectic_temperature(R):.3f} K with R = {R}" for R in (GAS_CONSTANT, PYCALPHAD_R))
        + ". This engine matches it; pycalphad's bisected value is about 0.07 K lower, which the gas constant "
        "accounts for only 0.001 K of.",
    ]

    gap_path = FIXTURES_DIR / FIXTURES["Regular solution (miscibility gap)"]
    gap_definition = load_system(gap_path)
    gap_system = build_system(gap_definition)
    gap_reference = PycalphadSystem(gap_path, gap_definition)
    lines += [
        "",
        "## Miscibility gap binodal (regular solution, L0 = 20000 J/mol)",
        "",
        "Both composition sets at x = 0.5. Tolerance: 1e-3.",
        "",
        "| T (K) | This engine | pycalphad | Max difference |",
        "|---|---|---|---|",
    ]
    for T in (800.0, 1000.0, 1150.0):
        ours = our_equilibrium(gap_system, T, 0.5)
        reference = pycalphad_binodal(gap_reference, T)
        difference = max(abs(ours[0].x - reference[0]), abs(ours[1].x - reference[1]))
        lines.append(
            f"| {T:.0f} | {ours[0].x:.6f}, {ours[1].x:.6f} | {reference[0]:.6f}, {reference[1]:.6f} | {difference:.1e} |"
        )

    lines += [
        "",
        "## Phase boundaries",
        "",
        "Lines: boundaries traced by this engine. Points: two-phase tie-line end points from pycalphad "
        "equilibria on a 60 x 201 (T, x) grid. Fields narrower than the grid spacing (the Al-rich "
        "LIQUID + FCC_AL field in Al-Cu) can be missed by the pycalphad grid.",
        "",
    ]
    for name, file_name in FIXTURES.items():
        image = f"engine_vs_pycalphad_{Path(file_name).stem}.svg"
        _overlay_plot(name, FIXTURES_DIR / file_name, image_dir / image)
        lines += [f"![{name}](../img/{image})", ""]

    report_path.write_text("\n".join(lines))
    return report_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    arguments = parser.parse_args(argv)
    print(write_report(arguments.docs_dir))


if __name__ == "__main__":
    main()
