import math

import pytest
from scipy.optimize import brentq

from phase_diagram_explorer.diagram import compute_diagram
from phase_diagram_explorer.invariants import EUTECTIC, detect_invariants
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase

# Synthetic symmetric eutectic system: ALPHA and BETA are mirror-image ideal
# solutions with a large mutual immiscibility offset (as in test_equilibrium),
# and LIQUID is a symmetric ideal solution whose pure-component energies fall
# with temperature via a simple linear fusion model (offset * (1 - T/Tm)),
# so LIQUID becomes favorable in the middle composition range only above the
# eutectic temperature.
OFFSET = 8000.0
FUSION_ENTHALPY = 10000.0
MELTING_T = 800.0


def _eutectic_system():
    alpha = SolutionPhase(PureElementGibbs(a=0.0), PureElementGibbs(a=OFFSET))
    beta = SolutionPhase(PureElementGibbs(a=OFFSET), PureElementGibbs(a=0.0))
    liquid_endpoint = PureElementGibbs(a=FUSION_ENTHALPY, b=-FUSION_ENTHALPY / MELTING_T)
    liquid = SolutionPhase(liquid_endpoint, liquid_endpoint)
    return {"ALPHA": alpha, "BETA": beta, "LIQUID": liquid}


def _x_alpha_tangent(T: float) -> float:
    return 1.0 / (1.0 + math.exp(OFFSET / (GAS_CONSTANT * T)))


def _alpha_beta_tangent_value(T: float) -> float:
    x = _x_alpha_tangent(T)
    return OFFSET * x + GAS_CONSTANT * T * (x * math.log(x) + (1 - x) * math.log(1 - x))


def _liquid_midpoint_value(T: float) -> float:
    return FUSION_ENTHALPY * (1 - T / MELTING_T) + GAS_CONSTANT * T * math.log(0.5)


def _known_eutectic_point():
    """Independent closed-form/root-find ground truth, not derived from the
    diagram or invariant-detection code under test.

    By the x -> 1-x mirror symmetry (ALPHA<->BETA, LIQUID self-symmetric),
    the ALPHA-BETA common tangent line is horizontal, and the eutectic
    temperature is where LIQUID's minimum (at x=0.5) meets that tangent
    line's height.
    """

    def f(T):
        return _liquid_midpoint_value(T) - _alpha_beta_tangent_value(T)

    T_eu = brentq(f, 500.0, 600.0)
    x_alpha = _x_alpha_tangent(T_eu)
    x_beta = 1.0 - x_alpha
    return T_eu, x_alpha, x_beta


def test_known_eutectic_point_is_consistent():
    T_eu, x_alpha, x_beta = _known_eutectic_point()
    assert 400.0 < T_eu < 700.0
    assert 0.0 < x_alpha < 0.5
    assert x_beta == pytest.approx(1.0 - x_alpha)


SCAN_T_RANGE = (400.0, 800.0)
SCAN_N_T = 81


@pytest.fixture(scope="session")
def eutectic_scan_reactions():
    system = _eutectic_system()
    diagram = compute_diagram(system, T_range=SCAN_T_RANGE, n_T=SCAN_N_T, n_x=201, n_points=1001)
    return detect_invariants(diagram, system, n_points=1001)


def test_detects_single_eutectic_reaction_near_known_point(eutectic_scan_reactions):
    T_eu, x_alpha, x_beta = _known_eutectic_point()
    eutectics = [r for r in eutectic_scan_reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    reaction = eutectics[0]

    T_step = (SCAN_T_RANGE[1] - SCAN_T_RANGE[0]) / (SCAN_N_T - 1)
    assert reaction.temperature == pytest.approx(T_eu, abs=T_step)


def test_eutectic_reaction_phases_and_composition(eutectic_scan_reactions):
    T_eu, x_alpha, x_beta = _known_eutectic_point()
    eutectics = [r for r in eutectic_scan_reactions if r.type == EUTECTIC]

    assert len(eutectics) == 1
    reaction = eutectics[0]

    assert set(reaction.phases) == {"ALPHA", "LIQUID", "BETA"}
    assert reaction.composition["ALPHA"] == pytest.approx(x_alpha, abs=0.02)
    assert reaction.composition["BETA"] == pytest.approx(x_beta, abs=0.02)
    assert reaction.composition["LIQUID"] == pytest.approx(0.5, abs=0.02)


def test_no_reactions_detected_away_from_eutectic_temperature():
    system = _eutectic_system()
    # a narrow temperature window entirely above the eutectic point, deep in
    # the region where LIQUID is always stable in the middle: no transition
    # in two-phase field structure should occur
    diagram = compute_diagram(
        system, T_range=(700.0, 750.0), n_T=11, n_x=101, n_points=501
    )
    reactions = detect_invariants(diagram, system, n_points=501)

    assert reactions == []
