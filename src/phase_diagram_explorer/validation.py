"""Cross-validation of the equilibrium engine against pycalphad.

pycalphad is an optional dependency (pip install ".[validation]"). Both
engines read the same TDB file: this package through load_system, pycalphad
through its own TDB reader, so the comparison also checks the TDB reader.

Run as a module to regenerate the comparison reports:

    python -m phase_diagram_explorer.validation            # all reports
    python -m phase_diagram_explorer.validation --systems  # only docs/validation/ag_cu.md and pb_sn.md

which write docs/validation/engine_vs_pycalphad.md, ag_cu.md and pb_sn.md, and
overlay plots of both engines' phase boundaries to docs/img/.
"""
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import base_phase_name, compute_equilibrium
from phase_diagram_explorer.invariants import InvariantReaction, detect_invariants_over_range
from phase_diagram_explorer.models import SystemDefinition, load_system

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "tests" / "fixtures"
PRESSURE_PA = 101325.0
PYCALPHAD_GAS_CONSTANT = 8.3145
N_T = 6
GRID_MARGIN_K = 25.0
VALIDATION_X = (0.05, 0.27, 0.5, 0.73, 0.95)
# pycalphad vertices of one phase closer than this in composition are one phase.
SAME_COMPOSITION = 1e-4
MIN_FRACTION = 1e-12
# Invariant temperatures from pycalphad are bisected within this window (K)
# around this engine's value.
INVARIANT_WINDOW_K = 5.0


def pycalphad_database_dir() -> Path:
    import pycalphad

    return Path(pycalphad.__file__).parent / "tests" / "databases"


@dataclass(frozen=True)
class ValidationCase:
    """A TDB file compared by the tests and the report."""

    name: str
    file_name: str
    # "fixture": tests/fixtures; "pycalphad": pycalphad's test databases;
    # "data": a downloaded database in data/tdb (see sources.py).
    location: str = "fixture"
    # For "data": the system file (data/systems/...) this engine loads; pycalphad
    # reads the same TDB file (all its elements) with only the system's two.
    system_file: str | None = None
    elements: tuple[str, str] | None = None
    T_range: tuple[float, float] | None = None
    x_values: tuple[float, ...] = VALIDATION_X
    provenance: str = ""
    models: str = ""

    @property
    def path(self) -> Path:
        if self.location == "data":
            from phase_diagram_explorer.sources import ensure

            return ensure(Path(self.file_name).stem)
        base = FIXTURES_DIR if self.location == "fixture" else pycalphad_database_dir()
        return base / self.file_name

    @property
    def load_kwargs(self) -> dict:
        return {"elements": list(self.elements)} if self.elements else {}

    def definition(self, **kwargs) -> SystemDefinition:
        if self.system_file:
            return load_system(ROOT / self.system_file, **kwargs)
        return load_system(self.path, **self.load_kwargs, **kwargs)

    def temperature_range(self) -> tuple[float, float]:
        return self.T_range or tuple(self.definition().t_range_k)


CASES = {
    case.name: case
    for case in [
        ValidationCase("Ag-Cu (provisional)", "ag_cu_provisional.tdb", models="substitutional"),
        ValidationCase("Al-Cu (provisional)", "al_cu_provisional.tdb", models="substitutional, compound"),
        ValidationCase("Regular solution", "regular_solution.tdb", models="miscibility gap"),
        ValidationCase("Gap eutectic", "gap_eutectic.tdb", models="miscibility gap"),
        ValidationCase(
            "Interstitial (synthetic)", "interstitial.tdb", x_values=(0.002, 0.01, 0.03, 0.2, 0.6),
            models="(A)1(C,VA)1, (A)1(C,VA)3, pure C",
        ),
        ValidationCase(
            "Magnetic (synthetic)", "magnetic.tdb",
            models="ferro- and antiferromagnetic BCC/FCC, composition-dependent TC and BMAGN",
        ),
        ValidationCase(
            "Ag-Cu (NIST)", "nist_solder.tdb", location="data", system_file="data/systems/ag_cu.meta.json",
            provenance="Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST "
            "Standard Reference Database 139, National Institute of Standards and Technology, "
            "https://doi.org/10.18434/T4759N (Accessed 12 October 2026); Ag-Cu: F.H. Hayes, H.L. Lukas, "
            "G. Effenberg, G. Petzow, Z. Metallkd. 77 (1986) 749-754; unaries: A.T. Dinsdale, CALPHAD 15 (1991) "
            "317-425. Downloaded from NIST, not redistributed.",
            models="Ag-Cu-Va subsystem of a six-element database; one FCC_A1 with a miscibility gap",
        ),
        ValidationCase(
            "Pb-Sn (NIST)", "nist_solder.tdb", location="data", system_file="data/systems/pb_sn.meta.json",
            provenance="Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST "
            "Standard Reference Database 139, National Institute of Standards and Technology, "
            "https://doi.org/10.18434/T4759N (Accessed 12 October 2026); Pb-Sn: H.J. Fecht, M.-X. Zhang, "
            "Y.A. Chang, J.H. Perepezko, Metall. Trans. A 20A (1989) 795-803; unaries: A.T. Dinsdale, CALPHAD 15 "
            "(1991) 317-425. Downloaded from NIST, not redistributed.",
            models="Pb-Sn-Va subsystem of a six-element database; substitutional phases, pure-Sn phases",
        ),
        ValidationCase(
            "Pb-Sn", "pbsn.tdb", location="pycalphad", elements=("Pb", "Sn"), T_range=(325.0, 625.0),
            provenance="T.L. Ngai and Y.A. Chang, CALPHAD 5 (1981) 267-276; TDB file by T. Abe, K. Hashimoto "
            "and Y. Sawada (NIMS, 2011).",
            models="substitutional",
        ),
        ValidationCase(
            "Cu-Mg", "cumg.tdb", location="pycalphad", elements=("Cu", "Mg"), T_range=(550.0, 1350.0),
            provenance="P. Liang et al., CALPHAD 22 (1998) 527-544, with the parameters of C.A. Coughanowr et al., "
            "Z. Metallkde. 82 (1991) 574-581 (parameter set 3); TDB file by M. Palumbo, T. Abe and K. Hashimoto "
            "(NIMS, 2008).",
            models="Laves (CU,MG)2(CU,MG)1 with an internal degree of freedom and wildcard parameters, "
            "(CU,MG)1(VA)1, (MG)1(VA)0.5, compound",
        ),
        ValidationCase(
            "Fe-C", "cfe_broshe.tdb", location="pycalphad", elements=("Fe", "C"), T_range=(850.0, 1800.0),
            x_values=(0.002, 0.01, 0.03, 0.12, 0.5),
            provenance="B. Hallstedt et al., CALPHAD 34 (2010) 129-133 (Fe-C), with the equation of state of "
            "E. Brosh et al., CALPHAD 31 (2007) 173-185, and A.T. Dinsdale, CALPHAD 15 (1991) 317-425 "
            "(references as listed in the file).",
            models="interstitial (FE)1(C,VA)1, (FE)1(C,VA)3, (FE)1(C,VA)0.5; magnetic BCC, FCC and carbides; "
            "compounds; pure C; pressure-dependent functions",
        ),
    ]
}


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


@dataclass(frozen=True)
class InvariantComparison:
    reaction: InvariantReaction
    pycalphad_temperature: float
    probe_x: float
    probe_phase: str

    @property
    def difference(self) -> float:
        return abs(self.reaction.temperature - self.pycalphad_temperature)


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
        self._calculate = pycalphad.calculate
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

    def gibbs_at_site_fractions(self, phase: str, T: float, points: np.ndarray) -> np.ndarray:
        """GM of `phase` at rows of site fractions in pycalphad's order
        (sublattice by sublattice, species sorted by name)."""
        result = self._calculate(
            self.database, self.components, phase, T=T, P=PRESSURE_PA, N=1.0, points=np.atleast_2d(points)
        )
        return result.GM.values.ravel()


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


def validation_grid(T_range: tuple[float, float], x_values=VALIDATION_X) -> list[tuple[float, float]]:
    T_min, T_max = T_range
    temperatures = np.linspace(T_min + GRID_MARGIN_K, T_max - GRID_MARGIN_K, N_T)
    return [(float(T), x) for T in temperatures for x in x_values]


def compare_grid(case: ValidationCase) -> list[PointComparison]:
    """Both engines at N_T x len(x_values) (T, x) points."""
    definition = case.definition()
    system = build_system(definition)
    reference = PycalphadSystem(case.path, definition)
    points = validation_grid(case.temperature_range(), case.x_values)
    temperatures = sorted({T for T, _ in points})
    states = reference.equilibria(temperatures, list(case.x_values))
    return [
        PointComparison(T, x, our_equilibrium(system, T, x), states[temperatures.index(T)][case.x_values.index(x)])
        for T, x in points
    ]


def pycalphad_transition_temperature(
    reference: PycalphadSystem, x: float, phase: str, T_low: float, T_high: float, tolerance: float = 0.01
) -> float:
    """Temperature at which `phase` appears or disappears at composition x,
    by bisection on pycalphad equilibria (its presence must differ at T_low
    and T_high)."""
    low = reference.has_phase(T_low, x, phase)
    if low == reference.has_phase(T_high, x, phase):
        raise ValueError(f"{phase} is {'present' if low else 'absent'} at both {T_low} and {T_high} K for x = {x}")
    while T_high - T_low > tolerance:
        middle = (T_low + T_high) / 2.0
        if reference.has_phase(middle, x, phase) == low:
            T_low = middle
        else:
            T_high = middle
    return (T_low + T_high) / 2.0


def pycalphad_phase_boundary_temperature(
    reference: PycalphadSystem, x: float, phase: str, T_low: float, T_high: float, tolerance: float = 0.01
) -> float:
    """Temperature at which `phase` appears on heating at composition x."""
    if reference.has_phase(T_low, x, phase) or not reference.has_phase(T_high, x, phase):
        raise ValueError(f"{phase} must be absent at {T_low} K and present at {T_high} K for x = {x}")
    return pycalphad_transition_temperature(reference, x, phase, T_low, T_high, tolerance)


def compare_invariants(case: ValidationCase, tolerance: float = 0.01) -> list[InvariantComparison]:
    """Every invariant this engine finds over the case's range, against the
    temperature at which pycalphad's reacting phase P2 appears or
    disappears at a composition between P1 and P2 (inside the reaction line)."""
    definition = case.definition()
    reference = PycalphadSystem(case.path, definition)
    comparisons = []
    for reaction in detect_invariants_over_range(build_system(definition), case.temperature_range()):
        p1, p2, _ = reaction.phases
        x = (reaction.composition[p1] + reaction.composition[p2]) / 2.0
        probe = base_phase_name(p2)
        T_reference = pycalphad_transition_temperature(
            reference, x, probe, reaction.temperature - INVARIANT_WINDOW_K, reaction.temperature + INVARIANT_WINDOW_K,
            tolerance,
        )
        comparisons.append(InvariantComparison(reaction, T_reference, x, probe))
    return comparisons


def pycalphad_binodal(reference: PycalphadSystem, T: float, x: float = 0.5) -> tuple[float, float]:
    states = reference.equilibrium(T, x)
    if len(states) != 2 or states[0].phase != states[1].phase:
        raise ValueError(f"no miscibility gap at T = {T} K, x = {x}: {states}")
    return states[0].x, states[1].x


def gibbs_differences(case: ValidationCase, temperatures=(400.0, 900.0, 1400.0), samples: int = 6, seed: int = 0):
    """{phase: max |G_ours - G_pycalphad|} at random site fractions, with this
    engine using pycalphad's gas constant (so the models must agree exactly)."""
    definition = case.definition(gas_constant=PYCALPHAD_GAS_CONSTANT)
    system = build_system(definition)
    reference = PycalphadSystem(case.path, definition)
    rng = np.random.default_rng(seed)
    differences = {}
    for name, phase in system.items():
        model = phase.model
        Y = np.zeros((samples, model.n_columns))
        rows = []
        for r in range(samples):
            row = []
            for s, species in enumerate(model.constituents):
                fractions = rng.dirichlet(np.ones(len(species)))
                for sp, y in zip(species, fractions):
                    Y[r, model.column[(s, sp)]] = y
                row += [Y[r, model.column[(s, sp)]] for sp in sorted(species)]
            rows.append(row)
        worst = 0.0
        for T in temperatures:
            ours = model.molar_gibbs(T, Y)
            theirs = reference.gibbs_at_site_fractions(name, T, np.array(rows))
            worst = max(worst, float(np.max(np.abs(ours - theirs))))
        differences[name] = worst
    return differences


def pycalphad_boundary_points(reference: PycalphadSystem, T_range, n_T: int = 60, n_x: int = 201):
    """((T, x) end points of every two-phase tie line found on a T x x grid,
    number of grid points pycalphad failed to solve). Each temperature is
    solved as one row; if pycalphad raises for a row, the row is halved
    until the failing points are isolated, and those are skipped."""
    temperatures = np.linspace(T_range[0], T_range[1], n_T)
    compositions = np.linspace(0.0005, 0.9995, n_x)
    points = set()
    failed = 0

    def solve(T, x):
        """States at (T, x[k]); a failing block is halved until the failing
        points are isolated and dropped."""
        nonlocal failed
        try:
            return reference.equilibria([T], x)[0]
        except Exception:
            if len(x) == 1:
                failed += 1
                return []
            middle = len(x) // 2
            return solve(T, x[:middle]) + solve(T, x[middle:])

    for T in temperatures:
        row = solve(T, compositions)
        for phases in row:
            if len(phases) == 2:
                points.update((float(T), round(state.x, 6)) for state in phases)
    return sorted(points), failed


# --- report -------------------------------------------------------------------


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


def _overlay_plot(case: ValidationCase, output: Path) -> int:
    """Write the overlay plot; returns the number of grid points pycalphad
    failed to solve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phase_diagram_explorer.tracing import trace_diagram

    definition = case.definition()
    T_range = case.temperature_range()
    traced = trace_diagram(build_system(definition), T_range)
    points, failed = pycalphad_boundary_points(PycalphadSystem(case.path, definition), T_range)

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
        ax.set_ylim(*T_range)
        ax.set_xlabel(f"x({definition.dependent_element.symbol})")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(case.name)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(output, format="svg")
        plt.close(fig)
    return failed


def write_report(docs_dir: Path = ROOT / "docs") -> Path:
    import pycalphad

    from phase_diagram_explorer import __version__

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
        "Both engines read the same TDB files. The Ag-Cu and Al-Cu fixtures are the provisional example systems "
        "exported with `write_tdb`; the other files in `tests/fixtures/` are synthetic. None of them is "
        "literature data. The literature databases are pycalphad's own test databases (not copied into this "
        "repository); their provenance is listed below. This comparison checks that the engine solves the models "
        "correctly, not that any model is right.",
        "",
        "**Gas constant.** This engine uses each system's own R: 8.31451 J/(mol K) for TDB files by default "
        "(the fixtures exported from JSON keep 8.314462618 in their `.meta.json`). pycalphad always uses "
        f"{PYCALPHAD_GAS_CONSTANT}. The R of each comparison is listed; the Gibbs energy check uses pycalphad's R "
        "on both sides.",
        "",
        "## Systems and models",
        "",
        "| System | File | Models exercised |",
        "|---|---|---|",
    ]
    for case in CASES.values():
        where = {
            "fixture": f"`tests/fixtures/{case.file_name}`",
            "pycalphad": f"pycalphad `{case.file_name}`",
            "data": f"`data/tdb/{case.file_name}` (downloaded)",
        }[case.location]
        lines.append(f"| {case.name} | {where} | {case.models} |")

    lines += ["", "### Literature databases", ""]
    for case in CASES.values():
        if case.provenance:
            lines.append(f"- **{case.name}** (`{case.file_name}`): {case.provenance}")

    lines += [
        "",
        "## Gibbs energies at the same site fractions",
        "",
        "Every phase at 6 random site-fraction sets and T = 400, 900 and 1400 K, both engines with "
        f"R = {PYCALPHAD_GAS_CONSTANT}. Maximum |G difference| in J/mol of atoms.",
        "",
        "| System | Phases | Max difference (J/mol) |",
        "|---|---|---|",
    ]
    for case in CASES.values():
        differences = gibbs_differences(case)
        lines.append(f"| {case.name} | {len(differences)} | {max(differences.values()):.1e} |")

    lines += [
        "",
        "## Equilibrium at 30 (T, x) points per system",
        "",
        f"A grid of {N_T} temperatures (the range less {GRID_MARGIN_K:g} K at each end) by five compositions "
        "per system. Phases are compared by name, ordered by composition (composition sets of one phase "
        "compare as two phases of that name). Tolerance: 1e-3 on phase fractions and compositions.",
        "",
        "| System | R here | T range (K) | x | Same stable phases | Max fraction difference | Max composition difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in CASES.values():
        comparisons = compare_grid(case)
        agreeing = [c for c in comparisons if c.same_phases]
        fraction = max(c.fraction_error for c in agreeing)
        composition = max(c.composition_error for c in agreeing)
        T_min, T_max = case.temperature_range()
        lines.append(
            f"| {case.name} | {case.definition().gas_constant} | {T_min:g}-{T_max:g} | "
            f"{', '.join(f'{x:g}' for x in case.x_values)} | {len(agreeing)}/{len(comparisons)} | "
            f"{fraction:.1e} | {composition:.1e} |"
        )

    lines += [
        "",
        "## Invariant reactions",
        "",
        "This engine root-finds the three-phase condition. The pycalphad value is the temperature at which the "
        "reacting phase P2 appears or disappears at a composition between P1 and P2 (inside the reaction line), "
        "bisected on pycalphad equilibria to 0.01 K. Tolerance: 0.5 K.",
        "",
        "| System | Reaction | Phases | This engine (K) | pycalphad (K) | Difference (K) |",
        "|---|---|---|---|---|---|",
    ]
    for case in CASES.values():
        for item in compare_invariants(case):
            reaction = item.reaction
            lines.append(
                f"| {case.name} | {reaction.type} | {' + '.join(reaction.phases)} | {reaction.temperature:.3f} | "
                f"{item.pycalphad_temperature:.3f} | {item.difference:.3f} |"
            )
    lines += [
        "",
        "The gap eutectic is symmetric, so its temperature is also known analytically: the ideal LIQUID at "
        "x = 0.5 meets the horizontal common tangent of the two ALPHA composition sets. That gives "
        + ", ".join(f"{gap_eutectic_temperature(R):.3f} K with R = {R}" for R in (8.314462618, PYCALPHAD_GAS_CONSTANT))
        + ". This engine matches it; pycalphad's bisected value is about 0.07 K lower, which the gas constant "
        "accounts for only 0.001 K of.",
    ]

    gap = CASES["Regular solution"]
    gap_definition = gap.definition()
    gap_system = build_system(gap_definition)
    gap_reference = PycalphadSystem(gap.path, gap_definition)
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
        "equilibria on a 60 x 201 (T, x) grid. Fields narrower than the grid spacing (e.g. the Al-rich "
        "LIQUID + FCC_AL field in Al-Cu) can be missed by the pycalphad grid.",
        "",
    ]
    for case in CASES.values():
        image = f"engine_vs_pycalphad_{Path(case.file_name).stem}.svg"
        failed = _overlay_plot(case, image_dir / image)
        lines += [f"![{case.name}](../img/{image})", ""]
        if failed:
            lines += [
                f"pycalphad raised an error (ZeroDivisionError in its minimiser) at {failed} of the "
                f"{60 * 201} grid points for {case.name}; those points are left out of the plot.",
                "",
            ]

    report_path.write_text("\n".join(lines))
    return report_path


@dataclass(frozen=True)
class SystemReport:
    """Key points of an assessed system: NIST's calculation from the same
    database and, where a confirmed source exists, experiment."""

    case: str
    file_stem: str
    title: str
    nist_page: str
    nist_text: str  # NIST's values as published, and their conversion
    nist: dict  # "T" (K), "left", "liquid", "right" (mole fractions)
    experiment: dict  # same keys plus "<element> melting": (value, tolerance); compositions as mole fractions
    experiment_note: str  # source of the eutectic values, or why there are none
    melting_note: str  # source of the melting points
    melting: tuple  # ((element, x, solid phase), ...)
    probe_x: float  # composition inside the eutectic line for pycalphad's bisection
    weight_masses: tuple | None = None  # (M_base, M_dependent) to also show wt%


MELTING_SOURCES = {
    "Ag": "ITS-90 freezing point of silver, 1234.93 K",
    "Cu": "ITS-90 freezing point of copper, 1357.77 K",
    "Sn": "ITS-90 freezing point of tin, 505.078 K",
    "Pb": "ITS-90 secondary reference point, freezing point of lead, 600.612 K",
}
ITS90_CITATION = (
    "H. Preston-Thomas, The International Temperature Scale of 1990 (ITS-90), Metrologia 27 (1990) 3-10, "
    "https://doi.org/10.1088/0026-1394/27/1/002"
)
SECONDARY_POINTS_CITATION = (
    "R.E. Bedford, G. Bonnier, H. Maas, F. Pavese, Recommended values of temperature on the International "
    "Temperature Scale of 1990 for a selected set of secondary reference points, Metrologia 33 (1996) 133-154, "
    "https://doi.org/10.1088/0026-1394/33/2/3"
)


REPORTS = {
    "ag_cu": SystemReport(
        case="Ag-Cu (NIST)", file_stem="ag_cu", title="Ag-Cu",
        nist_page="https://www.metallurgy.nist.gov/phase/solder/agcu.html",
        nist_text="779.1 °C and 28.41, 8.27 and 92.53 mass % Cu. They are converted with "
        "x(Cu) = (w/63.546) / (w/63.546 + (100 - w)/107.87), the database's molar masses, and 779.1 °C = 1052.25 K.",
        nist={"T": 1052.25, "left": 0.13273, "liquid": 0.40250, "right": 0.95460},
        experiment={
            "T": (1052.0, 3.0), "left": (0.141, 0.015), "liquid": (0.399, 0.010), "right": (0.950, 0.010),
            "Ag melting": (1234.93, 0.1), "Cu melting": (1357.77, 0.1),
        },
        experiment_note="Experiment: the assessed eutectic of P.R. Subramanian and J.H. Perepezko, The Ag-Cu "
        "(silver-copper) system, J. Phase Equilib. 14 (1993) 62-75, https://doi.org/10.1007/BF02652162 "
        "(779.1 °C; liquid 39.9 at.% Cu, (Ag) 14.1 at.% Cu, (Cu) 95.1 at.% Cu).",
        melting_note=f"Reference temperatures: {MELTING_SOURCES['Ag']} and {MELTING_SOURCES['Cu']} "
        f"({ITS90_CITATION}).",
        melting=(("Ag", 0.0, "FCC_A1"), ("Cu", 1.0, "FCC_A1")),
        probe_x=0.4,
    ),
    "pb_sn": SystemReport(
        case="Pb-Sn (NIST)", file_stem="pb_sn", title="Pb-Sn",
        nist_page="https://www.metallurgy.nist.gov/phase/solder/pbsn.html",
        nist_text="182.2 °C and 62.13, 18.91 and 96.59 mass % Sn. They are converted with "
        "x(Sn) = (w/118.69) / (w/118.69 + (100 - w)/207.2), the database's molar masses, and 182.2 °C = 455.35 K.",
        nist={"T": 455.35, "left": 0.28932, "liquid": 0.74120, "right": 0.98018},
        experiment={"Pb melting": (600.61, 0.1), "Sn melting": (505.08, 0.1)},
        experiment_note="No experimental column: the commonly quoted Pb-Sn eutectic values (183 °C, 61.9 wt% Sn) "
        "could not be confirmed against a published assessment, so the eutectic is compared only with NIST's "
        "calculation from the same database and with pycalphad. `tests/test_pb_sn_assessed.py` still checks the "
        "eutectic against those values with wide tolerances (±2 °C, ±1.5 wt%), as a plausibility check only.",
        melting_note=f"Reference temperatures: {MELTING_SOURCES['Sn']} ({ITS90_CITATION}); "
        f"{MELTING_SOURCES['Pb']} ({SECONDARY_POINTS_CITATION}).",
        melting=(("Pb", 0.0, "FCC_A1"), ("Sn", 1.0, "BCT_A5")),
        probe_x=0.5,
        weight_masses=(207.2, 118.69),
    ),
}


def write_system_report(key: str, docs_dir: Path = ROOT / "docs") -> Path:
    """docs/validation/<system>.md: an assessed system against NIST's
    calculation, experiment and pycalphad, with an overlay plot."""
    import pycalphad
    from scipy.optimize import brentq

    from phase_diagram_explorer import __version__
    from phase_diagram_explorer.units import mole_to_weight_fraction

    spec = REPORTS[key]
    case = CASES[spec.case]
    definition = case.definition()
    system = build_system(definition)
    (reaction,) = detect_invariants_over_range(system, definition.t_range_k)
    left, liquid, right = (reaction.composition[p] for p in reaction.phases)
    ours = {"T": reaction.temperature, "left": left, "liquid": liquid, "right": right}
    reference = PycalphadSystem(case.path, definition)
    T_pycalphad = pycalphad_transition_temperature(
        reference, spec.probe_x, "LIQUID", reaction.temperature - INVARIANT_WINDOW_K,
        reaction.temperature + INVARIANT_WINDOW_K, tolerance=0.001,
    )
    below = reference.equilibrium(T_pycalphad - 0.01, spec.probe_x)
    dependent = definition.dependent_element.symbol
    names = list(reaction.phases)  # composition-set labels keep the two FCC_A1 rows apart

    def weight(x):
        if not spec.weight_masses:
            return ""
        return f" ({100.0 * mole_to_weight_fraction(x, *spec.weight_masses):.2f} wt%)"

    lines = [
        f"# {spec.title}: assessed system vs NIST" + (" and experiment" if "T" in spec.experiment else " and pycalphad"),
        "",
        f"Generated by `python -m phase_diagram_explorer.validation --systems` with phase-diagram-explorer "
        f"{__version__} and pycalphad {pycalphad.__version__}. The tests in `tests/test_{spec.file_stem}_assessed.py` "
        "and `tests/validation/` check these values.",
        "",
        f"**Data.** The {spec.title} subsystem of the NIST solder database, downloaded from NIST "
        "(`scripts/fetch_tdb.py`, verified by SHA256) and not redistributed here:",
        "",
    ]
    lines += [f"- {citation.text}" for citation in definition.citations]
    lines += [
        "",
        f"Gas constant here: {definition.gas_constant} J/(mol K); pycalphad uses {PYCALPHAD_GAS_CONSTANT}.",
        "",
        f"## Eutectic L -> {names[0]} + {names[2]}",
        "",
        f"NIST's calculated values ({spec.nist_page}) are {spec.nist_text}",
        "",
    ]
    has_experiment = "T" in spec.experiment
    lines += (
        ["| Quantity | This engine | NIST (same database) | Difference | Experiment | Difference | pycalphad |",
         "|---|---|---|---|---|---|---|"]
        if has_experiment
        else ["| Quantity | This engine | NIST (same database) | Difference | pycalphad |", "|---|---|---|---|---|"]
    )
    rows = [
        ("Eutectic T (K)", "T", f"{T_pycalphad:.3f}"),
        (f"{names[0]} x({dependent})", "left", f"{below[0].x:.5f}"),
        (f"Liquid x({dependent})", "liquid", "-"),
        (f"{names[2]} x({dependent})", "right", f"{below[-1].x:.5f}"),
    ]
    for label, key_, pycalphad_value in rows:
        value, nist = ours[key_], spec.nist[key_]
        digits = 3 if key_ == "T" else 5
        shown = f"{value:.{digits}f}" + ("" if key_ == "T" else weight(value))
        compared = ""
        if has_experiment:
            experiment, tolerance = spec.experiment[key_]
            expected = f"{experiment}" + (f" ± {tolerance}" if tolerance is not None else "")
            compared = f"{expected} | {value - experiment:+.{digits}f} | "
        lines.append(
            f"| {label} | {shown} | {nist:.{digits}f} | {value - nist:+.{digits}f} | {compared}{pycalphad_value} |"
        )
    lines += [
        "",
        "Test tolerances: against NIST ±0.2 K and ±0.0005 in mole fraction"
        + ("; against experiment as listed. " if has_experiment else ". ")
        + f"The pycalphad temperature is where LIQUID appears on heating at x({dependent}) = {spec.probe_x} "
        "(bisected to 0.001 K); its solid compositions are 0.01 K below that.",
    ]
    if spec.experiment_note:
        lines += ["", spec.experiment_note]
    lines += [
        "",
        "## Melting points",
        "",
        "| Element | This engine (LIQUID = solid) | Reference (ITS-90) | Difference |",
        "|---|---|---|---|",
    ]
    for element, x, solid in spec.melting:
        experiment, tolerance = spec.experiment[f"{element} melting"]
        T_m = brentq(
            lambda T: system["LIQUID"].molar_gibbs(T, x) - system[solid].molar_gibbs(T, x),
            experiment - 20, experiment + 20, xtol=1e-9,
        )
        lines.append(f"| {element} ({solid}) | {T_m:.3f} K | {experiment} ± {tolerance} K | {T_m - experiment:+.3f} K |")
    lines += [
        "",
        spec.melting_note,
        "",
        "Melting detected from equilibria at x = 1e-9 and 1 - 1e-9 agrees with these crossings to within 0.05 K.",
        "",
        "## Phase boundaries",
        "",
        "Lines: boundaries traced by this engine. Points: two-phase tie-line end points from pycalphad equilibria "
        "on the same file (60 x 201 grid).",
        "",
        f"![{spec.title}](../img/{spec.file_stem}_nist_vs_pycalphad.svg)",
        "",
    ]
    image_dir = docs_dir / "img"
    image_dir.mkdir(parents=True, exist_ok=True)
    _overlay_plot(case, image_dir / f"{spec.file_stem}_nist_vs_pycalphad.svg")
    path = docs_dir / "validation" / f"{spec.file_stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    parser.add_argument(
        "--systems", action="store_true", help="write only the per-system pages (docs/validation/ag_cu.md, pb_sn.md)"
    )
    arguments = parser.parse_args(argv)
    for key in REPORTS:
        print(write_system_report(key, arguments.docs_dir))
    if not arguments.systems:
        print(write_report(arguments.docs_dir))


if __name__ == "__main__":
    main()
