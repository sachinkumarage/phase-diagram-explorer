import numpy as np
import pytest

from phase_diagram_explorer.equilibrium.curves import evaluate_phase_curves, lower_convex_hull
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase


def _ideal_phase(g_a_value, g_b_value):
    return SolutionPhase(PureElementGibbs(a=g_a_value), PureElementGibbs(a=g_b_value))


def test_evaluate_phase_curves_returns_grid_and_solution_curve():
    system = {"LIQUID": _ideal_phase(0.0, 0.0)}
    T = 1000.0

    x, curves = evaluate_phase_curves(system, T, n_points=50)

    assert x.shape == (50,)
    assert curves["LIQUID"].shape == (50,)
    assert np.all(np.isfinite(curves["LIQUID"]))


def test_evaluate_phase_curves_represents_stoichiometric_phase_as_narrow_point():
    gibbs_a = PureElementGibbs(a=0.0)
    gibbs_b = PureElementGibbs(a=0.0)
    compound = StoichiometricPhase(
        gibbs_a, gibbs_b, m=1, n=1, g_form=PureElementGibbs(a=-1000.0)
    )
    system = {"AB": compound}
    T = 1000.0

    x, curves = evaluate_phase_curves(system, T, n_points=101)

    curve = curves["AB"]
    finite_mask = np.isfinite(curve)
    assert finite_mask.sum() == 1
    finite_x = x[finite_mask][0]
    assert finite_x == pytest.approx(compound.composition, abs=1.0 / 100)
    assert curve[finite_mask][0] == pytest.approx(compound.molar_gibbs(T))


def test_lower_convex_hull_is_convex_and_sorted():
    x = np.linspace(0.0, 1.0, 50)
    G = (x - 0.5) ** 2

    hull_x, hull_G = lower_convex_hull(x, G)

    assert np.all(np.diff(hull_x) > 0)
    for i in range(1, len(hull_x) - 1):
        cross = (hull_x[i] - hull_x[i - 1]) * (hull_G[i + 1] - hull_G[i - 1]) - (
            hull_G[i] - hull_G[i - 1]
        ) * (hull_x[i + 1] - hull_x[i - 1])
        assert cross >= -1e-9


def test_lower_convex_hull_ignores_nan_points():
    x = np.array([0.0, 0.5, 1.0])
    G = np.array([0.0, np.nan, 0.0])

    hull_x, hull_G = lower_convex_hull(x, G)

    assert list(hull_x) == [0.0, 1.0]


def test_lower_convex_hull_finds_correct_tangent_region_for_two_crossing_ideal_solutions():
    phase_a_rich = _ideal_phase(0.0, 5000.0)
    phase_b_rich = _ideal_phase(5000.0, 0.0)
    system = {"A_RICH": phase_a_rich, "B_RICH": phase_b_rich}
    T = 1000.0

    x, curves = evaluate_phase_curves(system, T, n_points=501)

    combined_x = np.concatenate([x, x])
    combined_G = np.concatenate([curves["A_RICH"], curves["B_RICH"]])

    hull_x, hull_G = lower_convex_hull(combined_x, combined_G)

    # the hull must dip below both individual curves at the midpoint,
    # proving it found a genuine two-phase tie line rather than just the
    # pointwise minimum of the two curves
    tie_line_value = np.interp(0.5, hull_x, hull_G)
    raw_minimum = min(
        np.interp(0.5, x, curves["A_RICH"]),
        np.interp(0.5, x, curves["B_RICH"]),
    )
    assert tie_line_value < raw_minimum - 1.0

    # by symmetry, the endpoints reduce to the lower-energy pure component
    assert hull_x[0] == pytest.approx(0.0, abs=1e-9)
    assert hull_G[0] == pytest.approx(0.0, abs=1e-6)
    assert hull_x[-1] == pytest.approx(1.0, abs=1e-9)
    assert hull_G[-1] == pytest.approx(0.0, abs=1e-6)
