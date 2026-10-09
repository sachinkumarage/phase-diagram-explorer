"""Reader for the binary subset of the TDB database format.

Supported keywords: ELEMENT, FUNCTION, PHASE, CONSTITUENT, PARAMETER,
TYPE_DEFINITION, DEFINE_SYSTEM_DEFAULT and DEFAULT_COMMAND. Keywords may be
abbreviated as in Thermo-Calc, word by word ("PARA", "TYPE_DEF",
"DEF_SYS_DEF"). "$" starts a comment that runs to the end of the line, and a
command runs over as many lines as it needs until its terminating "!".

PARAMETER types G and L (Redlich-Kister orders 0..n) are Gibbs energy
parameters; TC and BMAGN are magnetic parameters, read and stored but not
evaluated. Any other keyword or parameter type raises NotImplementedError
naming it and its line number.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from phase_diagram_explorer.models import PiecewiseExpression
from phase_diagram_explorer.tdb.expression import normalise_name, parse_piecewise

KEYWORDS = (
    "ELEMENT",
    "FUNCTION",
    "PHASE",
    "CONSTITUENT",
    "PARAMETER",
    "TYPE_DEFINITION",
    "DEFINE_SYSTEM_DEFAULT",
    "DEFAULT_COMMAND",
)
GIBBS_PARAMETERS = ("G", "L")
MAGNETIC_PARAMETERS = ("TC", "BMAGN")

_PARAMETER = re.compile(
    r"^(?P<kind>[A-Za-z]+)\s*\(\s*(?P<phase>[^,()]+?)\s*,\s*(?P<constituents>[^;()]+?)\s*"
    r"(?:;\s*(?P<order>\d+)\s*)?\)\s*(?P<function>.*)$",
    re.DOTALL,
)


@dataclass
class TdbElement:
    symbol: str
    reference_state: str
    mass: float = 0.0
    H298: float = 0.0
    S298: float = 0.0


@dataclass
class TdbPhase:
    name: str
    types: str
    sites: list[float]
    line: int
    constituents: list[list[str]] | None = None


@dataclass
class TdbParameter:
    kind: str
    phase: str
    constituents: list[list[str]]
    order: int
    function: PiecewiseExpression
    line: int


@dataclass
class TdbTypeDefinition:
    code: str
    text: str
    line: int

    @property
    def is_magnetic(self) -> bool:
        return "MAGNETIC" in self.text.upper()

    @property
    def is_sequential(self) -> bool:
        """A plain "TYPE_DEFINITION % SEQ *" with no model attached."""
        return self.text.split()[:1] == ["SEQ"]


@dataclass
class TdbDatabase:
    elements: dict[str, TdbElement] = field(default_factory=dict)
    functions: dict[str, PiecewiseExpression] = field(default_factory=dict)
    phases: dict[str, TdbPhase] = field(default_factory=dict)
    parameters: list[TdbParameter] = field(default_factory=list)
    type_definitions: dict[str, TdbTypeDefinition] = field(default_factory=dict)
    defaults: list[str] = field(default_factory=list)


class TdbError(ValueError):
    pass


def _resolve_keyword(word: str, line: int) -> str:
    """Full keyword for `word`, which may abbreviate it word by word."""
    word = word.upper()
    parts = word.split("_")
    matches = [
        keyword for keyword in KEYWORDS
        if len(parts) <= len(keyword.split("_"))
        and all(full.startswith(part) and part for part, full in zip(parts, keyword.split("_")))
    ]
    if word in KEYWORDS:
        return word
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise TdbError(f"ambiguous TDB keyword {word!r} on line {line} (could be {', '.join(matches)})")
    raise NotImplementedError(f"unsupported TDB keyword {word!r} on line {line}")


def _commands(text: str):
    """(line number, command text) for each "!"-terminated command."""
    buffer: list[str] = []
    start = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("$", 1)[0]
        while line:
            head, bang, line = line.partition("!")
            if head.strip() and start is None:
                start = number
            buffer.append(head)
            if not bang:
                break
            command = " ".join(buffer).strip()
            if command:
                yield start, command
            buffer, start = [], None
    if " ".join(buffer).strip():
        raise TdbError(f"command starting on line {start} is not terminated with '!'")


def _phase_name(token: str) -> str:
    """Phase name without a ":L"-style suffix."""
    return token.split(":", 1)[0].upper()


def _species(text: str) -> list[str]:
    return [item.strip().rstrip("%").upper() for item in text.split(",") if item.strip()]


def _sublattices(text: str) -> list[list[str]]:
    return [_species(part) for part in text.split(":")]


def _float(token: str, what: str, line: int) -> float:
    try:
        return float(token.upper().replace("D", "E"))
    except ValueError:
        raise TdbError(f"expected a number for {what} on line {line}, got {token!r}") from None


def _element(database: TdbDatabase, arguments: str, line: int) -> None:
    tokens = arguments.split()
    if len(tokens) < 2:
        raise TdbError(f"ELEMENT on line {line} needs a symbol and a reference state")
    numbers = [_float(token, "ELEMENT data", line) for token in tokens[2:5]]
    numbers += [0.0] * (3 - len(numbers))
    symbol = tokens[0].upper()
    database.elements[symbol] = TdbElement(symbol, tokens[1].upper(), *numbers)


def _function(database: TdbDatabase, arguments: str, line: int) -> None:
    name, _, body = arguments.strip().partition(" ")
    if not body.strip():
        raise TdbError(f"FUNCTION on line {line} has no temperature ranges")
    try:
        database.functions[normalise_name(name)] = parse_piecewise(body)
    except ValueError as error:
        raise TdbError(f"FUNCTION {name} on line {line}: {error}") from None


def _type_definition(database: TdbDatabase, arguments: str, line: int) -> None:
    tokens = arguments.split(None, 1)
    if len(tokens) < 2:
        raise TdbError(f"TYPE_DEFINITION on line {line} needs a code and a definition")
    database.type_definitions[tokens[0]] = TdbTypeDefinition(tokens[0], tokens[1], line)


def _phase(database: TdbDatabase, arguments: str, line: int) -> None:
    tokens = arguments.split()
    if len(tokens) < 3:
        raise TdbError(f"PHASE on line {line} needs a name, type codes and a sublattice count")
    name, types = _phase_name(tokens[0]), tokens[1]
    count = int(_float(tokens[2], "the number of sublattices", line))
    sites = [_float(token, "sublattice sites", line) for token in tokens[3:]]
    if count < 1 or len(sites) != count:
        raise TdbError(f"PHASE {name} on line {line} declares {count} sublattices but gives {len(sites)} site counts")
    database.phases[name] = TdbPhase(name, types, sites, line)


def _constituent(database: TdbDatabase, arguments: str, line: int) -> None:
    name_token, _, spec = arguments.strip().partition(" ")
    name = _phase_name(name_token)
    if name not in database.phases:
        raise TdbError(f"CONSTITUENT on line {line} for undeclared phase {name}")
    spec = spec.strip()
    if not (spec.startswith(":") and spec.endswith(":")):
        raise TdbError(f"CONSTITUENT {name} on line {line}: expected ':species,...:species:', got {spec!r}")
    sublattices = _sublattices(spec[1:-1])
    phase = database.phases[name]
    if len(sublattices) != len(phase.sites) or not all(sublattices):
        raise TdbError(
            f"CONSTITUENT {name} on line {line} gives {len(sublattices)} sublattices; the phase has {len(phase.sites)}"
        )
    phase.constituents = sublattices


def _parameter(database: TdbDatabase, arguments: str, line: int) -> None:
    match = _PARAMETER.match(arguments.strip())
    if match is None:
        raise TdbError(f"cannot read PARAMETER on line {line}: {arguments.strip()!r}")
    kind = match.group("kind").upper()
    if kind not in GIBBS_PARAMETERS + MAGNETIC_PARAMETERS:
        raise NotImplementedError(f"unsupported TDB parameter type {kind!r} on line {line}")
    phase = _phase_name(match.group("phase"))
    try:
        function = parse_piecewise(match.group("function"))
    except ValueError as error:
        raise TdbError(f"PARAMETER {kind}({phase},...) on line {line}: {error}") from None
    database.parameters.append(
        TdbParameter(
            kind=kind,
            phase=phase,
            constituents=_sublattices(match.group("constituents")),
            order=int(match.group("order") or 0),
            function=function,
            line=line,
        )
    )


_HANDLERS = {
    "ELEMENT": _element,
    "FUNCTION": _function,
    "PHASE": _phase,
    "CONSTITUENT": _constituent,
    "PARAMETER": _parameter,
    "TYPE_DEFINITION": _type_definition,
}


def parse_tdb(text: str) -> TdbDatabase:
    database = TdbDatabase()
    for line, command in _commands(text):
        word, _, arguments = command.partition(" ")
        keyword = _resolve_keyword(word, line)
        if keyword in ("DEFINE_SYSTEM_DEFAULT", "DEFAULT_COMMAND"):
            database.defaults.append(f"{keyword} {arguments.strip()}")
        else:
            _HANDLERS[keyword](database, arguments, line)

    for parameter in database.parameters:
        if parameter.phase not in database.phases:
            raise TdbError(f"PARAMETER on line {parameter.line} refers to undeclared phase {parameter.phase}")
    return database


def read_tdb(path: str | Path) -> TdbDatabase:
    return parse_tdb(Path(path).read_text(encoding="utf-8", errors="replace"))
