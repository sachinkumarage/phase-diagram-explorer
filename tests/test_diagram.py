import math

import numpy as np
import pytest

from phase_diagram_explorer.diagram import PhaseDiagram, compute_diagram
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import GAS_CONSTANT, SolutionPhase


def _mirrored_ideal_system(offset: float):
    a_rich = SolutionPhase(PureElementGibbs(a=0.0), PureElementGibbs(a=offset))
    b_rich = SolutionPhase(PureElementGibbs(a=offset), PureElementGibbs(a=0.0))
    return {"A_RICH": a_rich, "B_RICH": b_rich}


def _analytical_tie_line_x1(offset: float, T: float) -> float:
    return 1.0 / (1.0 + math.exp(offset / (GAS_CONSTANT * T)))


def test_output_shapes():
    system = _mirrored_ideal_system(5000.0)
    diagram = compute_diagram(system, T_range=(400.0, 600.0), n_T=5, n_x=21, n_points=501)

    assert isinstance(diagram, PhaseDiagram)
    assert diagram.T_grid.shape == (5,)
    assert diagram.x_grid.shape == (21,)
    assert diagram.phase_labels.shape == (5, 21)
    assert diagram.phase_fractions.shape == (5, 21)


def test_T_grid_and_x_grid_values():
    system = _mirrored_ideal_system(5000.0)
    diagram = compute_diagram(system, T_range=(400.0, 600.0), n_T=5, n_x=21, n_points=501)

    assert diagram.T_grid[0] == pytest.approx(400.0)
    assert diagram.T_grid[-1] == pytest.approx(600.0)
    assert diagram.x_grid[0] == pytest.approx(0.0)
    assert diagram.x_grid[-1] == pytest.approx(1.0)


def test_single_phase_field_near_pure_a_across_temperature_range():
    offset = 5000.0
    system = _mirrored_ideal_system(offset)
    T_range = (400.0, 600.0)
    diagram = compute_diagram(system, T_range=T_range, n_T=5, n_x=101, n_points=1001)

    # x1(T) increases with T, so the smallest T bounds the single-phase
    # A_RICH region for the whole temperature range
    x1_min = _analytical_tie_line_x1(offset, T_range[0])
    safe_x = x1_min / 2.0
    j = int(np.argmin(np.abs(diagram.x_grid - safe_x)))

    for i in range(diagram.T_grid.shape[0]):
        assert diagram.phase_labels[i, j] == ("A_RICH",)
        assert diagram.phase_fractions[i, j] == {"A_RICH": 1.0}


def test_two_phase_field_at_symmetric_midpoint():
    offset = 5000.0
    system = _mirrored_ideal_system(offset)
    diagram = compute_diagram(system, T_range=(400.0, 600.0), n_T=5, n_x=101, n_points=1001)

    j_mid = int(np.argmin(np.abs(diagram.x_grid - 0.5)))

    for i in range(diagram.T_grid.shape[0]):
        assert set(diagram.phase_labels[i, j_mid]) == {"A_RICH", "B_RICH"}
        assert sum(diagram.phase_fractions[i, j_mid].values()) == pytest.approx(1.0)


def test_phase_fractions_sum_to_one_everywhere():
    system = _mirrored_ideal_system(5000.0)
    diagram = compute_diagram(system, T_range=(400.0, 600.0), n_T=3, n_x=31, n_points=501)

    for i in range(diagram.T_grid.shape[0]):
        for j in range(diagram.x_grid.shape[0]):
            total = sum(diagram.phase_fractions[i, j].values())
            assert total == pytest.approx(1.0, abs=1e-6)


def test_gibbs_curves_evaluated_once_per_temperature_not_per_grid_point(monkeypatch):
    system = _mirrored_ideal_system(5000.0)
    call_counts = {name: 0 for name in system}

    def counting(name, original):
        def wrapper(*args, **kwargs):
            call_counts[name] += 1
            return original(*args, **kwargs)
        return wrapper

    for name, phase in system.items():
        monkeypatch.setattr(phase, "hull_points", counting(name, phase.hull_points))

    n_T, n_x = 4, 25
    compute_diagram(system, T_range=(400.0, 600.0), n_T=n_T, n_x=n_x, n_points=101)

    assert call_counts == {name: n_T for name in system}
