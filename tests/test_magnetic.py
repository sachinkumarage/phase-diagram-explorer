"""Inden-Hillert-Jarl magnetic model."""
import numpy as np
import pytest

from phase_diagram_explorer.thermo.cef import CEFModel, CEFParameter
from phase_diagram_explorer.thermo.magnetic import (
    BCC_AFM_FACTOR,
    BCC_STRUCTURE_FACTOR,
    FCC_AFM_FACTOR,
    FCC_STRUCTURE_FACTOR,
    MagneticModel,
    ihj_function,
    magnetic_gibbs,
)
from phase_diagram_explorer.thermo.sublattice import SublatticePhase

R = 8.31451
BCC = MagneticModel(BCC_AFM_FACTOR, BCC_STRUCTURE_FACTOR)
FCC = MagneticModel(FCC_AFM_FACTOR, FCC_STRUCTURE_FACTOR)

# f(tau) worked out by hand from the polynomial (exact rational arithmetic):
#   A = 518/1125 + (11692/15975)(1/p - 1)
#   tau < 1:  f = 1 - [79/(140 p tau) + (474/497)(1/p - 1)(tau^3/6 + tau^9/135 + tau^15/600)] / A
#   tau >= 1: f = -[tau^-5/10 + tau^-15/315 + tau^-25/1500] / A
HAND_VALUES = {
    (0.28, 0.5): -0.7425041143166538,
    (0.28, 1.0): -0.044330073618451496,
    (0.28, 2.0): -0.0013341109551234286,
    (0.40, 0.5): -0.8297381374551478,
    (0.40, 1.0): -0.06663818353770791,
    (0.40, 2.0): -0.0020054722095065028,
}


@pytest.mark.parametrize(("p", "tau"), HAND_VALUES)
def test_f_tau_matches_hand_calculation(p, tau):
    f, _ = ihj_function(tau, p)
    assert float(f) == pytest.approx(HAND_VALUES[(p, tau)], rel=1e-13)


@pytest.mark.parametrize("p", [0.28, 0.40])
def test_f_and_its_derivative_are_continuous_at_tau_1(p):
    eps = 1e-9
    f_below, df_below = ihj_function(1.0 - eps, p)
    f_above, df_above = ihj_function(1.0 + eps, p)
    assert float(f_below) == pytest.approx(float(f_above), abs=1e-8)
    assert float(df_below) == pytest.approx(float(df_above), abs=1e-6)


@pytest.mark.parametrize("p", [0.28, 0.40])
def test_df_dtau_is_the_derivative_of_f(p):
    for tau in (0.2, 0.7, 0.95, 1.05, 1.6, 3.0):
        h = 1e-6
        numeric = (float(ihj_function(tau + h, p)[0]) - float(ihj_function(tau - h, p)[0])) / (2 * h)
        assert float(ihj_function(tau, p)[1]) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("model", [BCC, FCC], ids=["bcc", "fcc"])
def test_magnetic_gibbs_and_entropy_are_continuous_at_TC(model):
    """G_mag(T) and dG_mag/dT are continuous at T = TC."""
    TC, beta = 1043.0, 2.22

    def G(T):
        return float(magnetic_gibbs(T, TC, beta, model, R)[0])

    eps = 1e-7
    assert G(TC - eps) == pytest.approx(G(TC + eps), abs=1e-6)
    h = 1e-4
    slope_below = (G(TC - eps) - G(TC - eps - h)) / h
    slope_above = (G(TC + eps + h) - G(TC + eps)) / h
    assert slope_below == pytest.approx(slope_above, abs=1e-3)


def test_magnetic_gibbs_value():
    T, TC, beta = 800.0, 1043.0, 2.22
    f, _ = ihj_function(T / TC, BCC_STRUCTURE_FACTOR)
    G = float(magnetic_gibbs(T, TC, beta, BCC, R)[0])
    assert G == pytest.approx(R * T * np.log(beta + 1) * float(f), rel=1e-14)


@pytest.mark.parametrize(("model", "factor"), [(BCC, -1.0), (FCC, -3.0)], ids=["bcc", "fcc"])
def test_antiferromagnetic_convention(model, factor):
    """Negative TC and BMAGN are divided by the AFM factor (-1 BCC, -3 FCC)."""
    T = 150.0
    G_afm = float(magnetic_gibbs(T, -300.0, -0.6, model, R)[0])
    G_equivalent = float(magnetic_gibbs(T, -300.0 / factor, -0.6 / factor, model, R)[0])
    assert G_afm == pytest.approx(G_equivalent, rel=1e-14)
    assert G_afm < 0.0


def test_zero_moment_or_curie_temperature_gives_no_contribution():
    assert float(magnetic_gibbs(500.0, 1000.0, 0.0, BCC, R)[0]) == 0.0
    assert float(magnetic_gibbs(500.0, 0.0, 2.0, BCC, R)[0]) == 0.0


def test_magnetic_derivatives_match_finite_differences():
    T, TC, beta = 900.0, 1043.0, 2.22
    _, dG_dTC, dG_dbeta = (float(v) for v in magnetic_gibbs(T, TC, beta, BCC, R))
    h = 1e-4
    numeric_TC = (float(magnetic_gibbs(T, TC + h, beta, BCC, R)[0]) - float(magnetic_gibbs(T, TC - h, beta, BCC, R)[0])) / (2 * h)
    numeric_beta = (float(magnetic_gibbs(T, TC, beta + h, BCC, R)[0]) - float(magnetic_gibbs(T, TC, beta - h, BCC, R)[0])) / (2 * h)
    assert dG_dTC == pytest.approx(numeric_TC, rel=1e-6)
    assert dG_dbeta == pytest.approx(numeric_beta, rel=1e-6)


def _magnetic_bcc():
    """(A,B)1(VA)3 with TC = 1000 x_A - 300 x_B + x_A x_B (400 - 200 (x_A - x_B))."""
    return CEFModel(
        [1, 3],
        [["A", "B"], ["VA"]],
        [CEFParameter((("A",), ("VA",)), 0, 0.0), CEFParameter((("B",), ("VA",)), 0, 0.0)],
        magnetic=BCC,
        curie_temperature=[
            CEFParameter((("A",), ("VA",)), 0, 1000.0),
            CEFParameter((("B",), ("VA",)), 0, -300.0),
            CEFParameter((("A", "B"), ("VA",)), 0, 400.0),
            CEFParameter((("A", "B"), ("VA",)), 1, -200.0),
        ],
        magnetic_moment=[CEFParameter((("A",), ("VA",)), 0, 2.2), CEFParameter((("B",), ("VA",)), 0, -0.6)],
        gas_constant=R,
    )


def test_composition_dependent_curie_temperature():
    model = _magnetic_bcc()
    for x_b in (0.0, 0.3, 0.7, 1.0):
        x_a = 1.0 - x_b
        Y = np.array([[x_a, x_b, 1.0]])
        expected = 1000 * x_a - 300 * x_b + x_a * x_b * (400 - 200 * (x_a - x_b))
        assert model.curie_temperature(500.0, Y)[0] == pytest.approx(expected)
        assert model.magnetic_moment(500.0, Y)[0] == pytest.approx(2.2 * x_a - 0.6 * x_b)


def test_magnetic_phase_gibbs_energy_and_derivative():
    """Ideal mixing plus G_mag with composition-dependent TC and beta; G is
    per mole of atoms (one atom per formula unit here), dG/dx analytic."""
    model = _magnetic_bcc()
    phase = SublatticePhase(model, "A", "B")
    T = 700.0
    for x in (0.1, 0.45, 0.8):
        x_a = 1.0 - x
        TC = 1000 * x_a - 300 * x + x_a * x * (400 - 200 * (x_a - x))
        beta = 2.2 * x_a - 0.6 * x
        G_mag = float(magnetic_gibbs(T, TC, beta, BCC, R)[0])
        ideal = R * T * (x * np.log(x) + x_a * np.log(x_a))
        assert phase.molar_gibbs(T, x) == pytest.approx(ideal + G_mag, rel=1e-12)
        h = 1e-6
        numeric = (phase.molar_gibbs(T, x + h) - phase.molar_gibbs(T, x - h)) / (2 * h)
        assert phase.molar_gibbs_derivative(T, x) == pytest.approx(numeric, rel=1e-6)
