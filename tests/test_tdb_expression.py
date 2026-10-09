"""Safe TDB expression parsing and piecewise function evaluation."""
import math

import numpy as np
import pytest

from phase_diagram_explorer.models import PiecewiseExpression
from phase_diagram_explorer.tdb.expression import (
    Expression,
    FunctionTable,
    Piecewise,
    PiecewiseGibbs,
    parse_piecewise,
)
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT

# Unary Ag (SGTE, Dinsdale 1991) as a realistic piecewise function.
GHSERAG = (
    "298.15 -7209.512+118.200733*T-23.8463314*T*LN(T)-.001790585*T**2-3.98587E-07*T**3-12011*T**(-1); "
    "1234.93 Y -15095.252+190.266404*T-33.472*T*LN(T)+1.411773E+29*T**(-9); 3000.00 N REF0"
)


def _ghserag(T: float) -> float:
    if T < 1234.93:
        return (
            -7209.512 + 118.200733 * T - 23.8463314 * T * math.log(T) - 0.001790585 * T**2
            - 3.98587e-07 * T**3 - 12011 / T
        )
    return -15095.252 + 190.266404 * T - 33.472 * T * math.log(T) + 1.411773e29 * T**-9


def evaluate(text: str, T: float = 1000.0, functions: FunctionTable | None = None) -> float:
    return float(Expression(text).evaluate(np.array(T), functions))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1+2*3", 7.0),
        ("(1+2)*3", 9.0),
        ("2**3**2", 512.0),
        ("-T**2", -1.0e6),
        ("-2**2", -4.0),
        ("10/4/5", 0.5),
        ("1-2-3", -4.0),
        ("+5", 5.0),
        ("T**(-1)", 1.0e-3),
        ("2.5E+03", 2500.0),
        ("1.0D-3*T", 1.0),
        (".5*T", 500.0),
        ("LN(T)", math.log(1000.0)),
        ("LOG(T)", math.log(1000.0)),
        ("EXP(-T/1000)", math.exp(-1.0)),
        ("ln(t)", math.log(1000.0)),
        ("R*T", GAS_CONSTANT * 1000.0),
    ],
)
def test_arithmetic(text, expected):
    assert evaluate(text) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo unsafe')",
        "__import__",
        "().__class__",
        "T.real",
        "[T]",
        "lambda: 0",
        "OPEN(T)",
        "LN T",
        "1 +",
        "(1+2",
        "1 2",
        "",
        "T; 1",
    ],
)
def test_malicious_or_malformed_expressions_are_rejected(text):
    with pytest.raises(ValueError):
        evaluate(text, functions=FunctionTable())


def test_unknown_names_are_function_references_and_must_be_defined():
    assert Expression("GHSERAG+1").references == {"GHSERAG"}
    with pytest.raises(ValueError, match="undefined function 'GHSERAG'"):
        evaluate("GHSERAG+1", functions=FunctionTable())


def test_expression_evaluation_does_not_use_eval(monkeypatch):
    import builtins

    def forbidden(*args, **kwargs):
        raise AssertionError("eval/exec must not be used")

    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    assert evaluate("2*T+LN(EXP(3))") == pytest.approx(2003.0)


def test_vectorised_evaluation():
    T = np.array([300.0, 600.0, 900.0])
    assert np.allclose(Expression("3*T+1").evaluate(T), 3 * T + 1)
    assert Expression("42").evaluate(T).shape == (3,)


def test_piecewise_breakpoints_and_terminator():
    model = parse_piecewise(GHSERAG)
    assert [(i.T_min, i.T_max) for i in model.intervals] == [(298.15, 1234.93), (1234.93, 3000.0)]

    gibbs = PiecewiseGibbs(model)
    for T in (298.15, 500.0, 1234.92, 1234.93, 1500.0, 3000.0):
        assert gibbs.G(T) == pytest.approx(_ghserag(T), rel=1e-12)
    T = np.array([400.0, 2000.0])
    assert np.allclose(gibbs.G(T), [_ghserag(400.0), _ghserag(2000.0)], rtol=1e-12)


@pytest.mark.parametrize("T", [298.0, 3000.01, 0.0])
def test_evaluation_outside_all_ranges_raises(T):
    gibbs = PiecewiseGibbs(parse_piecewise(GHSERAG))
    with pytest.raises(ValueError, match="outside every temperature range"):
        gibbs.G(T)


def test_evaluation_outside_ranges_raises_for_arrays():
    gibbs = PiecewiseGibbs(parse_piecewise(GHSERAG))
    with pytest.raises(ValueError, match="outside"):
        gibbs.G(np.array([500.0, 5000.0]))


@pytest.mark.parametrize(
    "text",
    [
        "298.15 T; 500 Y",  # Y without a following range
        "298.15 T; 500 Y 2*T",  # not terminated with N
        "298.15 T",  # no range at all
        "298.15 T; 200 N",  # decreasing limits
        "298.15 T; 500 N; 600 N",  # text after the terminator
        "298.15 T+; 500 N",  # bad expression
    ],
)
def test_malformed_piecewise_functions_are_rejected(text):
    with pytest.raises(ValueError):
        parse_piecewise(text)


def test_thermo_calc_range_defaults():
    model = parse_piecewise("+5*T; ,, N")
    assert [(i.T_min, i.T_max) for i in model.intervals] == [(298.15, 6000.0)]
    model = parse_piecewise("298.15 0.0; 6000.00 01DUP")
    assert [(i.T_min, i.T_max) for i in model.intervals] == [(298.15, 6000.0)]


def test_nested_function_references():
    functions = FunctionTable(
        {
            "GHSERAG": parse_piecewise(GHSERAG),
            "GLIQAG": parse_piecewise("298.15 +11025.076-8.89102*T+GHSERAG#; 3000 N"),
            "GDOUBLE": parse_piecewise("298.15 2*GLIQAG-GHSERAG; 3000 N"),
        }
    )
    T = 1100.0
    assert functions.evaluate("GLIQAG", T)[0] == pytest.approx(11025.076 - 8.89102 * T + _ghserag(T), rel=1e-12)
    assert functions.evaluate("gdouble", T)[0] == pytest.approx(
        2 * (11025.076 - 8.89102 * T + _ghserag(T)) - _ghserag(T), rel=1e-12
    )
    gibbs = PiecewiseGibbs(parse_piecewise("298.15 GDOUBLE+1; 3000 N"), functions)
    assert gibbs.G(T) == pytest.approx(functions.evaluate("GDOUBLE", T)[0] + 1.0)


def test_nested_function_range_is_checked():
    functions = FunctionTable({"F": parse_piecewise("298.15 T; 1000 N")})
    gibbs = PiecewiseGibbs(parse_piecewise("1 F; 5000 N"), functions)
    assert gibbs.G(500.0) == pytest.approx(500.0)
    with pytest.raises(ValueError, match="outside"):
        gibbs.G(2000.0)


def test_circular_and_undefined_function_references_are_rejected():
    with pytest.raises(ValueError, match="circular"):
        FunctionTable({"A": parse_piecewise("1 B; 10 N"), "B": parse_piecewise("1 A; 10 N")})
    with pytest.raises(ValueError, match="undefined function 'MISSING'"):
        FunctionTable({"A": parse_piecewise("1 MISSING; 10 N")})
    with pytest.raises(ValueError, match="undefined function 'MISSING'"):
        PiecewiseGibbs(parse_piecewise("1 MISSING; 10 N"))


def test_piecewise_from_model_round_trips():
    model = parse_piecewise(GHSERAG)
    assert PiecewiseExpression.model_validate(model.model_dump()) == model
    assert Piecewise.from_model(model).references == frozenset()
