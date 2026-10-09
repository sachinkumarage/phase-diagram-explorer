"""Write a binary SystemDefinition as a TDB database.

Supported: substitutional solution phases (one sublattice holding both
elements, G end members and Redlich-Kister L parameters) and stoichiometric
compounds (one sublattice per element), with energies given as polynomial
coefficients or piecewise expressions. Numbers are written with full
precision, so reading the file back gives the same Gibbs energies.

Per-mole-of-atoms energies are written per formula unit as TDB expects:
a compound A_m B_n has sublattices of m and n sites, and its G parameter is
(m + n) times the per-atom energy.
"""
import json
import textwrap
from pathlib import Path

from phase_diagram_explorer import __version__
from phase_diagram_explorer.models import (
    GibbsCoefficients,
    Phase,
    PiecewiseExpression,
    SystemDefinition,
    metadata_path,
)

# Temperature range written for coefficient-based energies without one (K).
DEFAULT_T_MIN = 1.0
DEFAULT_T_MAX = 10000.0
COMMENT_WIDTH = 76


def _number(value: float) -> str:
    """Shortest round-tripping representation, with an upper-case exponent."""
    return repr(float(value)).replace("e", "E")


def _signed(value: float) -> str:
    text = _number(value)
    return text if text.startswith("-") else "+" + text


def coefficients_expression(coefficients: GibbsCoefficients, factor: float = 1.0) -> str:
    """a + b*T + c*T*LN(T) + d*T**2 + e*T**3 + f*T**(-1), times `factor`."""
    terms = [
        (coefficients.a, ""),
        (coefficients.b, "*T"),
        (coefficients.c, "*T*LN(T)"),
        (coefficients.d, "*T**2"),
        (coefficients.e, "*T**3"),
        (coefficients.f, "*T**(-1)"),
    ]
    text = "".join(_signed(factor * value) + suffix for value, suffix in terms if value != 0.0)
    return text.lstrip("+") if text else "0.0"


def _ranges(intervals: list[tuple[float, float, str]]) -> str:
    """TDB range syntax: "T_low expr; T_high Y expr; ...; T_high N"."""
    parts = [f"{_number(intervals[0][0])} {intervals[0][2]};"]
    for k, (_, T_max, _) in enumerate(intervals):
        if k + 1 < len(intervals):
            parts.append(f" {_number(T_max)} Y {intervals[k + 1][2]};")
        else:
            parts.append(f" {_number(T_max)} N")
    return "".join(parts)


def _energy(value: GibbsCoefficients | PiecewiseExpression | float, factor: float = 1.0) -> str:
    if isinstance(value, PiecewiseExpression):
        return _ranges(
            [
                (i.T_min, i.T_max, i.expression if factor == 1.0 else f"{_number(factor)}*({i.expression})")
                for i in value.intervals
            ]
        )
    if isinstance(value, GibbsCoefficients):
        T_min = value.T_min if value.T_min is not None else DEFAULT_T_MIN
        T_max = value.T_max if value.T_max is not None else DEFAULT_T_MAX
        return _ranges([(T_min, T_max, coefficients_expression(value, factor))])
    return _ranges([(DEFAULT_T_MIN, DEFAULT_T_MAX, _number(factor * value))])


def _is_zero(value) -> bool:
    return isinstance(value, (int, float)) and value == 0.0


def _phase_lines(phase: Phase, base: str, dependent: str) -> list[str]:
    name = phase.name.upper()
    if phase.is_magnetic or phase.type_definitions:
        raise NotImplementedError(f"write_tdb: phase {phase.name} uses a magnetic or other GES model")
    if phase.model_type == "solution":
        symbols = {base: base.upper(), dependent: dependent.upper()}
        end_members = phase.end_members or {}
        lines = [f"PHASE {name} % 1 1.0 !", f"CONSTITUENT {name} :{symbols[base]},{symbols[dependent]}: !"]
        for symbol in (base, dependent):
            if symbol in end_members:
                lines.append(f"PARAMETER G({name},{symbols[symbol]};0) {_energy(end_members[symbol])} !")
        for order, L_v in enumerate(phase.interaction_parameters):
            if not _is_zero(L_v):
                lines.append(
                    f"PARAMETER L({name},{symbols[base]},{symbols[dependent]};{order}) {_energy(L_v)} !"
                )
        return lines
    if phase.model_type == "stoichiometric":
        stoichiometry = phase.stoichiometry or {}
        sites = [stoichiometry[symbol] for symbol in (base, dependent)]
        total = sum(sites)
        lines = [
            f"PHASE {name} % 2 {_number(sites[0])} {_number(sites[1])} !",
            f"CONSTITUENT {name} :{base.upper()}:{dependent.upper()}: !",
        ]
        if phase.formation is not None:
            lines.append(f"PARAMETER G({name},{base.upper()}:{dependent.upper()};0) {_energy(phase.formation, total)} !")
        return lines
    raise NotImplementedError(
        f"write_tdb supports substitutional solution and stoichiometric phases; {phase.name} is {phase.model_type!r}"
    )


def _comment(label: str, text: str | None) -> list[str]:
    if not text:
        return []
    return ["$ " + line for line in textwrap.wrap(f"{label}: {text}", COMMENT_WIDTH)]


def to_tdb(system: SystemDefinition) -> str:
    """The TDB text for a binary system (see module docstring)."""
    base = system.base_element
    dependent = system.dependent_element
    lines = [
        f"$ {system.name}",
        f"$ Written by phase-diagram-explorer {__version__}.",
        *_comment("Source", system.source),
        *_comment("Status", f"{system.status}. {system.status_reason or ''}".strip() if system.status else None),
        "$",
        "ELEMENT /-   ELECTRON_GAS  0.0 0.0 0.0 !",
        "ELEMENT VA   VACUUM        0.0 0.0 0.0 !",
    ]
    for element in (base, dependent):
        mass = element.atomic_mass if element.atomic_mass is not None else 0.0
        lines.append(f"ELEMENT {element.symbol.upper()} {element.reference_state.upper()} {_number(mass)} 0.0 0.0 !")
    lines.append("")
    for name, function in system.functions.items():
        lines.append(f"FUNCTION {name} {_energy(function)} !")
    if system.functions:
        lines.append("")
    lines += [
        "TYPE_DEFINITION % SEQ * !",
        "DEFINE_SYSTEM_DEFAULT ELEMENT 2 !",
        "DEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !",
    ]
    for phase in system.phases:
        lines.append("")
        lines += _phase_lines(phase, base.symbol, dependent.symbol)
    return "\n".join(lines) + "\n"


def write_tdb(system: SystemDefinition, path: str | Path) -> None:
    Path(path).write_text(to_tdb(system), encoding="utf-8")


def write_metadata(system: SystemDefinition, tdb_path: str | Path) -> Path:
    """Write <name>.meta.json next to a TDB file: everything load_system
    needs that the TDB format does not hold. Returns its path."""
    metadata = {
        "name": system.name,
        "elements": [system.base_element.symbol, system.dependent_element.symbol],
        "element_names": {element.symbol: element.name for element in system.elements},
    }
    if system.t_range_k is not None:
        metadata["t_range_k"] = list(system.t_range_k)
    for key, value in (("_source", system.source), ("_status", system.status), ("_status_reason", system.status_reason)):
        if value is not None:
            metadata[key] = value
    path = metadata_path(tdb_path)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
