"""Textbook phase names and label placement (labels.py)."""
import json
from pathlib import Path

import numpy as np
import pytest

from phase_diagram_explorer.labels import (
    PhaseNames,
    PlotBox,
    _hits_lines,
    field_names,
    label_box_px,
    layout_labels,
)
from phase_diagram_explorer.precomputed import load
from phase_diagram_explorer.units import CELSIUS, WEIGHT_PERCENT, composition_to_display, temperature_to_display
from phase_diagram_explorer.visualization import diagram_labels, reaction_text

ROOT = Path(__file__).resolve().parents[1]

AG_CU = {"LIQUID": "L", "FCC_A1": {"base_side": "(Ag)", "dependent_side": "(Cu)"}}
PB_SN = {"LIQUID": "L", "FCC_A1": "(Pb)", "BCT_A5": "(Sn)"}


def test_names_come_from_the_side_of_the_gap_not_the_set_number():
    names = PhaseNames(AG_CU)
    assert names.names([("FCC_A1#1", 0.05), ("FCC_A1#2", 0.95)]) == ["(Ag)", "(Cu)"]
    assert names.names([("FCC_A1#2", 0.05), ("FCC_A1#1", 0.95)]) == ["(Ag)", "(Cu)"]
    assert names.name("FCC_A1", 0.1) == "(Ag)"
    assert names.name("FCC_A1", 0.9) == "(Cu)"
    assert names.name("FCC_A1", 0.4, split=0.3) == "(Cu)"
    assert names.name("LIQUID") == "L"
    assert names.name("BCC_A2") == "BCC_A2"


def test_names_without_display_names_are_calphad_labels():
    names = PhaseNames()
    assert not names
    assert names.names([("ALPHA#1", 0.2), ("ALPHA#2", 0.8)]) == ["ALPHA#1", "ALPHA#2"]
    assert names.legend("ALPHA", "ALPHA") == "ALPHA"


def test_legend_and_reaction_strings():
    names = PhaseNames(PB_SN)
    assert names.legend("(Ag)", "FCC_A1") == "(Ag) FCC_A1"
    assert names.legend("L + (Ag)", "LIQUID + FCC_A1") == "L + (Ag) (LIQUID + FCC_A1)"
    assert names.reaction("eutectic", ["(Ag)", "L", "(Cu)"]) == "L -> (Ag) + (Cu)"
    assert names.reaction("peritectic", ["L", "(Cu)", "(Ag)"]) == "L + (Ag) -> (Cu)"


@pytest.fixture(scope="module", params=["ag_cu", "pb_sn"])
def system(request):
    meta = json.loads((ROOT / "data" / "systems" / f"{request.param}.meta.json").read_text())
    precomputed = load(ROOT / "data" / "precomputed" / f"{request.param}.npz")
    return request.param, meta["display_names"], precomputed


def test_display_names_are_stored_in_precomputed_data(system):
    _, display_names, precomputed = system
    assert precomputed.definition.display_names == display_names


def test_field_and_reaction_names(system):
    name, display_names, precomputed = system
    names = PhaseNames(display_names)
    shown = {field_names(names, region, precomputed.traced)[0] for region in precomputed.traced.regions}
    a, b = ("(Ag)", "(Cu)") if name == "ag_cu" else ("(Pb)", "(Sn)")
    assert shown == {"L", a, b, f"L + {a}", f"L + {b}", f"{a} + {b}"}
    (reaction,) = precomputed.invariants
    assert reaction_text(reaction, names) == f"L -> {a} + {b}"


@pytest.mark.parametrize("unit", [None, WEIGHT_PERCENT])
@pytest.mark.parametrize("size", [(760.0, 470.0), (420.0, 300.0)])
def test_labels_stay_inside_the_plot_and_off_boundaries(system, unit, size):
    _, display_names, precomputed = system
    definition = precomputed.definition
    masses = (definition.base_element.atomic_mass, definition.dependent_element.atomic_mass)

    def to_x(x):
        return composition_to_display(x, unit or "at%", *masses)

    def to_T(T):
        return temperature_to_display(T, CELSIUS)

    traced = precomputed.traced
    plot = PlotBox(
        (float(to_x(0.0)), float(to_x(1.0))), (float(to_T(traced.T_range[0])), float(to_T(traced.T_range[1]))), *size
    )
    labels = diagram_labels(traced, PhaseNames(display_names), to_x, to_T, traced.T_range, CELSIUS, plot)
    assert {label.kind for label in labels} == {"field", "invariant"}
    for label in labels:
        left, bottom, right, top = label_box_px(label, plot)
        assert left >= 0 and bottom >= 0 and right <= plot.width_px and top <= plot.height_px, label.text

    boundaries = [plot.to_px(to_x(b.x), to_T(b.T)) for b in traced.boundaries]
    (invariant,) = [label for label in labels if label.kind == "invariant"]
    box = label_box_px(invariant, plot)
    assert box[1] > float(plot.to_px(0.0, invariant.anchor[1])[1])  # above its line
    assert not _hits_lines(box, boundaries)


def test_a_field_too_small_for_its_label_gets_a_leader_line():
    plot = PlotBox((0.0, 100.0), (0.0, 100.0))
    sliver = (np.array([50.0, 51.0, 51.0, 50.0]), np.array([40.0, 40.0, 60.0, 60.0]))
    (label,) = layout_labels([("L + (Ag)", "LIQUID + FCC_A1", sliver, (50.5, 50.0))], [], [], plot)
    assert label.leader
    assert label.anchor == (50.5, 50.0)
    left, bottom, right, top = label_box_px(label, plot)
    assert left >= 0 and right <= plot.width_px


def test_a_field_label_at_the_plot_edge_is_not_clipped():
    plot = PlotBox((0.0, 100.0), (0.0, 100.0))
    edge = (np.array([0.0, 2.0, 2.0, 0.0]), np.array([0.0, 0.0, 100.0, 100.0]))
    (label,) = layout_labels([("(Ag) + (Cu)", "", edge, (1.0, 99.0))], [], [], plot)
    left, bottom, right, top = label_box_px(label, plot)
    assert left >= 0 and bottom >= 0 and right <= plot.width_px and top <= plot.height_px


def test_invariant_label_moves_off_a_crossing_boundary():
    plot = PlotBox((0.0, 100.0), (0.0, 100.0))
    vertical = (np.array([50.0, 50.0]), np.array([0.0, 100.0]))
    (label,) = layout_labels([], [("Eutectic 779.1 °C", "", 20.0, 80.0, 50.0, 50.0)], [vertical], plot)
    box = label_box_px(label, plot)
    assert not _hits_lines(box, [plot.to_px(*vertical)])
    assert box[1] > float(plot.to_px(0.0, 50.0)[1])
