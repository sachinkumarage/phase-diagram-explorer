"""Safe evaluation of TDB arithmetic expressions and piecewise functions.

Expressions are tokenized and parsed by a small recursive-descent parser into
a tree that is evaluated with numpy; nothing is ever passed to eval() or
exec(). The grammar is the TDB subset

    expression := term (("+" | "-") term)*
    term       := unary (("*" | "/") unary)*
    unary      := ("+" | "-") unary | power
    power      := atom ("**" unary)?
    atom       := number | "T" | "P" | "R" | name | call | "(" expression ")"
    call       := ("LN" | "LOG" | "EXP") "(" expression ")"

where `name` is a reference to a FUNCTION (resolved through a FunctionTable),
P is the pressure and R the gas constant of the table (8.31451 J/(mol K)
and 101325 Pa unless the system sets them), and LOG is the natural
logarithm, as in Thermo-Calc. Names are
case-insensitive; a trailing "#" on a function reference is ignored.

A piecewise function is a list of temperature intervals, written in a TDB
file as

    298.15  <expression>; 1234.93  Y  <expression>; 3000  N  <reference> !

As in Thermo-Calc, an omitted lower limit is 298.15 K, an omitted (or ",,")
upper limit is 6000 K (both changed by TEMPERATURE_LIMITS), and a final
limit without Y or N ends the function.
Evaluation outside every interval raises ValueError.
"""
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from phase_diagram_explorer.models import ExpressionInterval, PiecewiseExpression
from phase_diagram_explorer.thermo.constants import DATABASE_GAS_CONSTANT, STANDARD_PRESSURE

CALLS = {"LN": np.log, "LOG": np.log, "EXP": np.exp}
TEMPERATURE = "T"
PRESSURE = "P"
GAS_CONSTANT_NAME = "R"

_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<number>(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)#?"
    r"|(?P<op>\*\*|[-+*/()])"
    r")"
)
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_FIRST_INTERVAL = re.compile(rf"^\s*({_NUMBER})\s+(.*)$", re.DOTALL)
_NEXT_INTERVAL = re.compile(rf"^\s*({_NUMBER}|,,)?\s*(?:([YNyn])(?![A-Za-z0-9_]))?\s*(.*)$", re.DOTALL)


def _to_float(text: str) -> float:
    return float(text.upper().replace("D", "E"))


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens = []
    position = 0
    while position < len(text):
        if text[position:].strip() == "":
            break
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"unexpected character {text[position:].lstrip()[0]!r} in expression {text!r}")
        kind = match.lastgroup
        value = match.group(kind)
        tokens.append((kind, value.upper() if kind == "name" else value))
        position = match.end()
    return tokens


class _Parser:
    """Recursive-descent parser producing nested tuples:
    ("num", value), ("T",), ("P",), ("R",), ("ref", name), ("neg", node),
    ("call", name, node) and (op, left, right) for + - * / **."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.position = 0

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _next(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise ValueError(f"unexpected end of expression {self.text!r}")
        self.position += 1
        return token

    def _accept(self, *ops: str) -> str | None:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] in ops:
            self.position += 1
            return token[1]
        return None

    def _expect(self, op: str) -> None:
        if self._accept(op) is None:
            found = self._peek()
            raise ValueError(f"expected {op!r} but found {found[1] if found else 'end'!r} in {self.text!r}")

    def parse(self):
        if not self.tokens:
            raise ValueError("empty expression")
        node = self._expression()
        if self._peek() is not None:
            raise ValueError(f"unexpected {self._peek()[1]!r} in expression {self.text!r}")
        return node

    def _expression(self):
        node = self._term()
        while (op := self._accept("+", "-")) is not None:
            node = (op, node, self._term())
        return node

    def _term(self):
        node = self._unary()
        while (op := self._accept("*", "/")) is not None:
            node = (op, node, self._unary())
        return node

    def _unary(self):
        if self._accept("-") is not None:
            return ("neg", self._unary())
        if self._accept("+") is not None:
            return self._unary()
        return self._power()

    def _power(self):
        node = self._atom()
        if self._accept("**") is not None:
            node = ("**", node, self._unary())
        return node

    def _atom(self):
        kind, value = self._next()
        if kind == "number":
            return ("num", _to_float(value))
        if kind == "op" and value == "(":
            node = self._expression()
            self._expect(")")
            return node
        if kind == "name":
            if self._accept("(") is not None:
                if value not in CALLS:
                    raise ValueError(f"unknown function call {value!r} in expression {self.text!r}")
                argument = self._expression()
                self._expect(")")
                return ("call", value, argument)
            if value in CALLS:
                raise ValueError(f"{value} must be called with parentheses in {self.text!r}")
            if value == TEMPERATURE:
                return ("T",)
            if value == GAS_CONSTANT_NAME:
                return ("R",)
            if value == PRESSURE:
                return ("P",)
            return ("ref", value)
        raise ValueError(f"unexpected {value!r} in expression {self.text!r}")


def _references(node) -> set[str]:
    if node[0] == "ref":
        return {node[1]}
    found: set[str] = set()
    for child in node[1:]:
        if isinstance(child, tuple):
            found |= _references(child)
    return found


MEMO_SIZE = 4096

_BINARY = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "**": np.power,
}


class Expression:
    """A parsed arithmetic expression in T, R and FUNCTION references."""

    def __init__(self, text: str):
        self.text = text
        self.tree = _Parser(text).parse()
        self.references = frozenset(_references(self.tree))

    def evaluate(self, T: np.ndarray, functions: "FunctionTable | None" = None) -> np.ndarray:
        T = np.asarray(T, dtype=float)

        def walk(node):
            kind = node[0]
            if kind == "num":
                return node[1]
            if kind == "T":
                return T
            if kind == "R":
                return functions.gas_constant if functions is not None else DATABASE_GAS_CONSTANT
            if kind == "P":
                return functions.pressure if functions is not None else STANDARD_PRESSURE
            if kind == "ref":
                if functions is None:
                    raise ValueError(f"expression {self.text!r} references undefined function {node[1]!r}")
                return functions.evaluate(node[1], T)
            if kind == "neg":
                return np.negative(walk(node[1]))
            if kind == "call":
                return CALLS[node[1]](walk(node[2]))
            return _BINARY[kind](walk(node[1]), walk(node[2]))

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = walk(self.tree)
        return np.broadcast_to(np.asarray(value, dtype=float), T.shape).copy()


@dataclass(frozen=True)
class Piecewise:
    """Expressions valid on [T_min, T_max) intervals (the last interval
    includes its upper limit)."""

    intervals: tuple[tuple[float, float, Expression], ...]

    @classmethod
    def from_model(cls, model: PiecewiseExpression) -> "Piecewise":
        return cls(tuple((i.T_min, i.T_max, Expression(i.expression)) for i in model.intervals))

    @property
    def references(self) -> frozenset[str]:
        return frozenset().union(*(expression.references for _, _, expression in self.intervals))

    def evaluate(self, T, functions: "FunctionTable | None" = None) -> np.ndarray:
        T = np.atleast_1d(np.asarray(T, dtype=float))
        result = np.full(T.shape, np.nan)
        covered = np.zeros(T.shape, dtype=bool)
        last = len(self.intervals) - 1
        for k, (T_min, T_max, expression) in enumerate(self.intervals):
            inside = (T >= T_min) & ((T < T_max) if k < last else (T <= T_max)) & ~covered
            if inside.any():
                result[inside] = expression.evaluate(T[inside], functions)
                covered |= inside
        if not covered.all():
            outside = T[~covered]
            limits = ", ".join(f"[{T_min:g}, {T_max:g}]" for T_min, T_max, _ in self.intervals)
            raise ValueError(f"T = {outside[0]:g} K is outside every temperature range ({limits})")
        return result


class FunctionTable:
    """TDB FUNCTIONs by name, with every reference resolved and checked for
    cycles when the table is built."""

    def __init__(
        self,
        functions: Mapping[str, PiecewiseExpression | Piecewise] | None = None,
        gas_constant: float = DATABASE_GAS_CONSTANT,
        pressure: float = STANDARD_PRESSURE,
    ):
        self.gas_constant = float(gas_constant)
        self.pressure = float(pressure)
        self._memo: OrderedDict = OrderedDict()
        self.functions: dict[str, Piecewise] = {}
        for name, function in (functions or {}).items():
            piecewise = function if isinstance(function, Piecewise) else Piecewise.from_model(function)
            self.functions[normalise_name(name)] = piecewise
        for name in self.functions:
            self._check(name, ())

    def _check(self, name: str, chain: tuple[str, ...]) -> None:
        if name in chain:
            raise ValueError(f"circular FUNCTION reference: {' -> '.join(chain + (name,))}")
        for reference in self.functions[name].references:
            if reference not in self.functions:
                raise ValueError(f"FUNCTION {name} references undefined function {reference!r}")
            self._check(reference, chain + (name,))

    def check_references(self, piecewise: Piecewise, context: str) -> None:
        for reference in piecewise.references:
            if reference not in self.functions:
                raise ValueError(f"{context} references undefined function {reference!r}")

    def evaluate(self, name: str, T: np.ndarray) -> np.ndarray:
        name = normalise_name(name)
        if name not in self.functions:
            raise ValueError(f"undefined function {name!r}")
        if np.size(T) != 1:
            return self.functions[name].evaluate(T, self)
        # Memoised for single temperatures: nested FUNCTIONs are evaluated
        # once per temperature however often they are referenced.
        key = (name, float(np.ravel(T)[0]))
        if key not in self._memo:
            self._memo[key] = self.functions[name].evaluate(T, self)
            if len(self._memo) > MEMO_SIZE:
                self._memo.popitem(last=False)
        return np.broadcast_to(self._memo[key], np.shape(np.atleast_1d(T))).reshape(np.shape(T)) if np.ndim(T) else self._memo[key]


class PiecewiseGibbs:
    """A piecewise expression evaluated as a Gibbs energy G(T), with the
    same interface as PureElementGibbs."""

    def __init__(self, model: PiecewiseExpression, functions: FunctionTable | None = None, context: str = "expression"):
        self.piecewise = Piecewise.from_model(model)
        self.functions = functions if functions is not None else FunctionTable()
        self.functions.check_references(self.piecewise, context)
        self._memo: OrderedDict = OrderedDict()

    def G(self, T: float | np.ndarray) -> float | np.ndarray:
        if np.ndim(T) == 0:
            key = float(T)
            if key not in self._memo:
                self._memo[key] = float(self.piecewise.evaluate(T, self.functions)[0])
                if len(self._memo) > MEMO_SIZE:
                    self._memo.popitem(last=False)
            return self._memo[key]
        result = self.piecewise.evaluate(T, self.functions)
        return result.reshape(np.shape(T))


def normalise_name(name: str) -> str:
    return name.strip().rstrip("#").upper()


def parse_piecewise(text: str, default_low: float = 298.15, default_high: float = 6000.0) -> PiecewiseExpression:
    """Parse TDB range syntax "T_low expr; T_high Y expr; T_high N [ref]".

    Every expression is parsed (not evaluated) so syntax errors are reported
    here. The reference after N, if any, is dropped.
    """
    segments = text.split(";")
    if len(segments) < 2:
        raise ValueError(f"piecewise function needs at least one ';'-terminated range: {text.strip()!r}")

    first = _FIRST_INTERVAL.match(segments[0])
    if first is not None:
        T_low, expression = _to_float(first.group(1)), first.group(2).strip()
    else:
        T_low, expression = default_low, segments[0].strip()

    intervals: list[ExpressionInterval] = []
    terminated = False
    for segment in segments[1:]:
        if terminated:
            if segment.strip():
                raise ValueError(f"text after the N terminator: {segment.strip()!r}")
            continue
        limit, flag, rest = _NEXT_INTERVAL.match(segment).groups()
        T_high = default_high if limit in (None, ",,") else _to_float(limit)
        if not T_high > T_low:
            raise ValueError(f"temperature limits must increase: {T_low:g} then {T_high:g}")
        Expression(expression)
        intervals.append(ExpressionInterval(T_min=T_low, T_max=T_high, expression=expression))
        if flag is not None and flag.upper() == "Y":
            T_low, expression = T_high, rest.strip()
        else:
            terminated = True
    if not terminated:
        raise ValueError(f"piecewise function is not terminated with N: {text.strip()!r}")
    return PiecewiseExpression(intervals=intervals)


def _rewrite(model: PiecewiseExpression, template: str) -> PiecewiseExpression:
    return PiecewiseExpression(
        intervals=[
            ExpressionInterval(T_min=i.T_min, T_max=i.T_max, expression=template.format(i.expression))
            for i in model.intervals
        ]
    )


def scale(model: PiecewiseExpression, factor: float) -> PiecewiseExpression:
    """`factor` times a piecewise expression (unchanged if factor == 1)."""
    return model if factor == 1.0 else _rewrite(model, f"{float(factor)!r}*({{}})")


def divide(model: PiecewiseExpression, divisor: float) -> PiecewiseExpression:
    """A piecewise expression divided by `divisor` (unchanged if divisor == 1)."""
    return model if divisor == 1.0 else _rewrite(model, f"({{}})/{float(divisor)!r}")
