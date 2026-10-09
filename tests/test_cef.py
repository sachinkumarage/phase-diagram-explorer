"""Compound Energy Formalism (sublattice model) and its binary phases."""
import math

import numpy as np
import pytest

from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.equilibrium.equilibrium import compute_equilibrium
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.thermo.cef import CEFModel, CEFParameter
from phase_diagram_explorer.thermo.constants import MIN_SITE_FRACTION
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.thermo.sublattice import DIRECT, FIXED, MINIMISE, SublatticePhase

R = 8.31451
T = 900.0


def P(*sublattices, order=0, value=0.0):
    return CEFParameter(tuple(tuple(s.split(",")) for s in sublattices), order, value)


# --- (A,B)1(C,VA)1 ---------------------------------------------------------------

G_A_VA, G_B_VA, G_A_C, G_B_C = -1000.0, -3000.0, 20000.0, 15000.0
L0_AB, L1_AB = 8000.0, -2500.0


def _abcva():
    return CEFModel(
        [1, 1],
        [["A", "B"], ["C", "VA"]],
        [
            P("A", "VA", value=G_A_VA), P("B", "VA", value=G_B_VA),
            P("A", "C", value=G_A_C), P("B", "C", value=G_B_C),
            P("A,B", "VA", value=L0_AB), P("A,B", "VA", order=1, value=L1_AB),
            P("A,B", "C", value=-4000.0), P("A", "C,VA", value=-7000.0), P("B", "C,VA", order=1, value=3000.0),
        ],
        gas_constant=R,
    )


@pytest.mark.parametrize("y_b", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_all_vacancies_reduces_to_the_substitutional_model(y_b):
    """With the second sublattice all VA, (A,B)1(C,VA)1 is the substitutional
    solution (A,B) with end members G(A:VA), G(B:VA) and L(A,B:VA): the
    vacancy sublattice carries no atoms and no entropy."""
    model = _abcva()
    y_a = 1.0 - y_b
    Y = model.site_fraction_matrix([{"A": y_a, "B": y_b}, {"C": 0.0, "VA": 1.0}])[None, :]
    substitutional = SolutionPhase(PureElementGibbs(a=G_A_VA), PureElementGibbs(a=G_B_VA), L=[L0_AB, L1_AB], gas_constant=R)
    assert model.atoms(Y)[0] == pytest.approx(1.0)
    assert model.molar_gibbs(T, Y)[0] == pytest.approx(substitutional.molar_gibbs(T, y_b), rel=1e-13, abs=1e-9)


def test_ideal_entropy_matches_hand_calculation():
    """No parameters: G = RT sum_s a_s sum_i y ln y per formula unit, per mole
    of atoms divided by N = sum_s a_s (1 - y_VA)."""
    model = CEFModel([1, 3], [["A", "B"], ["C", "VA"]], [], gas_constant=R)
    Y = model.site_fraction_matrix([{"A": 0.3, "B": 0.7}, {"C": 0.2, "VA": 0.8}])[None, :]
    G_formula = R * T * (1 * (0.3 * math.log(0.3) + 0.7 * math.log(0.7)) + 3 * (0.2 * math.log(0.2) + 0.8 * math.log(0.8)))
    atoms = 1 + 3 * 0.2
    assert model.formula_gibbs(T, Y)[0] == pytest.approx(G_formula, rel=1e-14)
    assert model.atoms(Y)[0] == pytest.approx(atoms)
    assert model.molar_gibbs(T, Y)[0] == pytest.approx(G_formula / atoms, rel=1e-14)
    # by hand: 0.3 ln 0.3 + 0.7 ln 0.7 = -0.6108643, 3 (0.2 ln 0.2 + 0.8 ln 0.8) = -1.5012073
    assert G_formula == pytest.approx(-R * T * 2.1120716, rel=1e-7)


def test_surface_of_reference_and_excess_terms():
    model = _abcva()
    yA, yB, yC, yV = 0.6, 0.4, 0.25, 0.75
    Y = model.site_fraction_matrix([{"A": yA, "B": yB}, {"C": yC, "VA": yV}])[None, :]
    reference = yA * yV * G_A_VA + yB * yV * G_B_VA + yA * yC * G_A_C + yB * yC * G_B_C
    excess = (
        yA * yB * yV * (L0_AB + L1_AB * (yA - yB))
        + yA * yB * yC * -4000.0
        + yA * yC * yV * -7000.0
        + yB * yC * yV * 3000.0 * (yC - yV)
    )
    entropy = R * T * (yA * math.log(yA) + yB * math.log(yB) + yC * math.log(yC) + yV * math.log(yV))
    assert model.formula_gibbs(T, Y)[0] == pytest.approx(reference + excess + entropy, rel=1e-13)


def test_gradient_matches_finite_differences():
    model = _abcva()
    Y = model.site_fraction_matrix([{"A": 0.6, "B": 0.4}, {"C": 0.25, "VA": 0.75}])[None, :]
    G, grad = model.formula_gibbs(T, Y, gradient=True)
    for k in range(model.n_columns):
        step = np.zeros_like(Y)
        step[0, k] = 1e-6
        numeric = (model.formula_gibbs(T, Y + step)[0] - model.formula_gibbs(T, Y - step)[0]) / 2e-6
        assert grad[0, k] == pytest.approx(numeric, rel=1e-6, abs=1e-6)
    _, point_grad = model.point_molar_gibbs(T, list(Y[0]), gradient=True)
    assert np.allclose(point_grad, model.molar_gibbs_gradient(T, Y)[1][0], rtol=1e-12)


def test_wildcard_parameter_is_independent_of_that_sublattice():
    with_wildcard = CEFModel([2, 1], [["A", "B"], ["A", "B"]], [P("A,B", "*", value=5000.0)], gas_constant=R)
    for y2 in (0.1, 0.6):
        Y = with_wildcard.site_fraction_matrix([{"A": 0.3, "B": 0.7}, {"A": y2, "B": 1 - y2}])[None, :]
        entropy = with_wildcard.formula_gibbs(T, Y)[0] - 0.3 * 0.7 * 5000.0
        expected = R * T * (2 * (0.3 * math.log(0.3) + 0.7 * math.log(0.7)) + y2 * math.log(y2) + (1 - y2) * math.log(1 - y2))
        assert entropy == pytest.approx(expected, rel=1e-13)


def test_reciprocal_parameters_only_at_order_zero():
    CEFModel([1, 1], [["A", "B"], ["A", "B"]], [P("A,B", "A,B", value=1000.0)])
    with pytest.raises(NotImplementedError, match="order-1"):
        CEFModel([1, 1], [["A", "B"], ["A", "B"]], [P("A,B", "A,B", order=1, value=1000.0)])
    with pytest.raises(ValueError, match="not on sublattice"):
        CEFModel([1, 1], [["A", "B"], ["VA"]], [P("A", "B", value=1.0)])


# --- interstitial phases: composition per mole of atoms -----------------------


def _interstitial(a: float):
    """(A)1(C,VA)a with G(A:VA) = 0, G(A:C) = g1 and L(A:C,VA;0) = L."""
    model = CEFModel(
        [1, a], [["A"], ["C", "VA"]],
        [P("A", "VA", value=0.0), P("A", "C", value=30000.0 - 10.0 * T), P("A", "C,VA", value=-12000.0)],
        gas_constant=R,
    )
    return SublatticePhase(model, "A", "C")


@pytest.mark.parametrize("a", [1.0, 3.0, 0.5])
def test_interstitial_composition_excludes_vacancies(a):
    phase = _interstitial(a)
    assert phase.mode == DIRECT
    assert phase.composition_range == pytest.approx((0.0, a / (1 + a)))
    for y in (0.01, 0.2, 0.7):
        x = a * y / (1 + a * y)
        Y = phase.site_fractions(T, x)[0]
        assert Y[phase.model.column[(1, "C")]] == pytest.approx(y, rel=1e-12)
        assert phase.composition_of(Y[None, :])[0] == pytest.approx(x, rel=1e-14)


@pytest.mark.parametrize("a", [1.0, 3.0])
def test_interstitial_gibbs_energy_per_mole_of_atoms(a):
    phase = _interstitial(a)
    for y in (0.05, 0.3, 0.8):
        x = a * y / (1 + a * y)
        G_formula = (
            y * (30000.0 - 10.0 * T) + y * (1 - y) * -12000.0
            + R * T * a * (y * math.log(y) + (1 - y) * math.log(1 - y))
        )
        assert phase.molar_gibbs(T, x) == pytest.approx(G_formula / (1 + a * y), rel=1e-13)
        h = 1e-7
        numeric = (phase.molar_gibbs(T, x + h) - phase.molar_gibbs(T, x - h)) / (2 * h)
        assert phase.molar_gibbs_derivative(T, x) == pytest.approx(numeric, rel=1e-6)
        assert np.asarray(phase.molar_gibbs_derivative(T, np.array([x])))[0] == pytest.approx(numeric, rel=1e-6)
    assert phase.molar_gibbs(T, a / (1 + a) + 0.01) == np.inf


# --- internal degrees of freedom ---------------------------------------------------


def _laves_like():
    """(A,B)2(A,B)1: one internal degree of freedom, with an ordered A2B
    stabilised by its end member and antisite end members penalised."""
    return CEFModel(
        [2, 1], [["A", "B"], ["A", "B"]],
        [
            P("A", "A", value=3000.0), P("B", "B", value=4500.0),
            P("A", "B", value=-30000.0), P("B", "A", value=40000.0),
            P("A,B", "*", value=8000.0), P("*", "A,B", value=5000.0),
        ],
        gas_constant=R,
    )


def _brute_force_minimum(phase: SublatticePhase, x: float, n: int = 200001) -> float:
    """Minimum over the internal coordinate by dense sampling: y2 (B on the
    second sublattice) runs over its feasible range, y1 follows from x."""
    y2 = np.linspace(0.0, 1.0, n)
    y1 = (3 * x - y2) / 2.0
    keep = (y1 >= 0.0) & (y1 <= 1.0)
    y1, y2 = y1[keep], y2[keep]
    model = phase.model
    Y = np.zeros((len(y1), model.n_columns))
    Y[:, model.column[(0, "A")]], Y[:, model.column[(0, "B")]] = 1 - y1, y1
    Y[:, model.column[(1, "A")]], Y[:, model.column[(1, "B")]] = 1 - y2, y2
    return float(np.min(model.molar_gibbs(T, Y)))


@pytest.mark.parametrize("x", [0.05, 0.2, 1 / 3, 0.45, 0.8])
def test_internal_minimisation_finds_the_minimum(x):
    phase = SublatticePhase(_laves_like(), "A", "B")
    assert phase.mode == MINIMISE and phase.internal_dof == 1
    G = phase.molar_gibbs(T, x)
    brute = _brute_force_minimum(phase, x)
    assert G <= brute + 1e-6
    assert G == pytest.approx(brute, abs=0.05)

    Y = phase.site_fractions(T, x)[0]
    assert np.all(Y > MIN_SITE_FRACTION * 0.999) and np.all(Y < 1.0)
    assert phase.composition_of(Y[None, :])[0] == pytest.approx(x, abs=1e-12)


def test_internal_minimisation_derivative_and_batch():
    phase = SublatticePhase(_laves_like(), "A", "B")
    x = np.array([0.1, 0.3, 0.5, 0.7])
    fresh = SublatticePhase(_laves_like(), "A", "B")
    batch = fresh.molar_gibbs(T, x)
    for k, xi in enumerate(x):
        assert phase.molar_gibbs(T, float(xi)) == pytest.approx(batch[k], rel=1e-12)
        h = 1e-6
        numeric = (phase.molar_gibbs(T, xi + h) - phase.molar_gibbs(T, xi - h)) / (2 * h)
        assert phase.molar_gibbs_derivative(T, float(xi)) == pytest.approx(numeric, rel=1e-5)


def test_two_internal_degrees_of_freedom():
    model = CEFModel(
        [1, 1, 1], [["A", "B"], ["A", "B"], ["A", "B"]],
        [P("A", "A", "A", value=0.0), P("B", "B", "B", value=0.0), P("A", "B", "B", value=-9000.0),
         P("B", "A", "A", value=2000.0), P("A,B", "*", "*", value=6000.0)],
        gas_constant=R,
    )
    phase = SublatticePhase(model, "A", "B")
    assert phase.mode == MINIMISE and phase.internal_dof == 2
    x = 0.55
    G = phase.molar_gibbs(T, x)
    # coarse brute force over two free site fractions
    grid = np.linspace(1e-6, 1 - 1e-6, 401)
    y1, y2 = np.meshgrid(grid, grid)
    y3 = 3 * x - y1 - y2
    keep = (y3 > 0) & (y3 < 1)
    Y = np.zeros((int(keep.sum()), model.n_columns))
    for s, y in enumerate((y1[keep], y2[keep], y3[keep])):
        Y[:, model.column[(s, "A")]], Y[:, model.column[(s, "B")]] = 1 - y, y
    assert G <= float(np.min(model.molar_gibbs(T, Y))) + 1e-6


# --- special cases -------------------------------------------------------------------


def test_solution_and_stoichiometric_phases_are_special_cases():
    solution = SolutionPhase(PureElementGibbs(a=100.0), PureElementGibbs(a=-200.0), L=[5000.0, 1000.0])
    assert isinstance(solution, SublatticePhase) and solution.mode == DIRECT
    assert solution.model.sites == (1.0,)
    compound = StoichiometricPhase(PureElementGibbs(), PureElementGibbs(), 2, 1, PureElementGibbs(a=-9000.0))
    assert isinstance(compound, SublatticePhase) and compound.mode == FIXED
    assert compound.composition == pytest.approx(1 / 3)
    assert compound.molar_gibbs(T) == pytest.approx(-9000.0)


def test_pure_element_phase_is_a_fixed_composition_end_point():
    """(C)1 at x = 1 coexists with an interstitial phase."""
    graphite = StoichiometricPhase.from_model(CEFModel([1], [["C"]], [P("C", value=0.0)]), "A", "C")
    assert graphite.composition == 1.0
    system = {"FCC": _interstitial(1.0), "GRAPHITE": graphite}
    result = compute_equilibrium(system, T, 0.8)
    assert result.stable_phases == ["FCC", "GRAPHITE"]
    assert result.phase_compositions["GRAPHITE"] == 1.0


def test_tdb_fixture_modes():
    system = build_system(load_system("tests/fixtures/interstitial.tdb"))
    assert {name: phase.mode for name, phase in system.items()} == {
        "LIQUID": DIRECT, "FCC_A1": DIRECT, "BCC_A2": DIRECT, "GRAPHITE": FIXED,
    }
    assert system["BCC_A2"].composition_range == pytest.approx((0.0, 0.75))
