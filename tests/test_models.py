from pathlib import Path

import pytest
from pydantic import ValidationError

from phase_diagram_explorer.models import (
    Composition,
    Element,
    Phase,
    SystemDefinition,
    load_system,
)

EXAMPLE_SYSTEM_PATH = Path(__file__).resolve().parent / "fixtures" / "example.json"
SYSTEMS_DIR = Path(__file__).resolve().parents[1] / "data" / "systems"


def test_composition_valid_sum():
    composition = Composition({"A": 0.4, "B": 0.6})
    assert composition.root["A"] == pytest.approx(0.4)


def test_composition_invalid_sum_raises():
    with pytest.raises(ValidationError):
        Composition({"A": 0.4, "B": 0.4})


def test_element_fields():
    element = Element(symbol="A", name="Element A", reference_state="solid")
    assert element.symbol == "A"


def test_phase_model_type():
    phase = Phase(name="LIQUID", model_type="solution")
    assert phase.model_type == "solution"


def test_phase_invalid_model_type_raises():
    with pytest.raises(ValidationError):
        Phase(name="LIQUID", model_type="gas")


def test_system_definition():
    system = SystemDefinition(
        name="A-B",
        elements=[
            Element(symbol="A", name="Element A", reference_state="solid"),
            Element(symbol="B", name="Element B", reference_state="solid"),
        ],
        phases=[
            Phase(name="LIQUID", model_type="solution"),
            Phase(name="FCC_A1", model_type="solution"),
        ],
    )
    assert len(system.elements) == 2
    assert len(system.phases) == 2


def test_load_system_json():
    system = load_system(EXAMPLE_SYSTEM_PATH)
    assert system.name == "A-B"
    assert {element.symbol for element in system.elements} == {"A", "B"}
    assert {phase.name for phase in system.phases} == {"LIQUID", "FCC_A1"}


def test_load_system_yaml(tmp_path):
    import json

    import yaml

    data = json.loads(EXAMPLE_SYSTEM_PATH.read_text())
    yaml_path = tmp_path / "example.yaml"
    yaml_path.write_text(yaml.safe_dump(data))

    system_from_json = load_system(EXAMPLE_SYSTEM_PATH)
    system_from_yaml = load_system(yaml_path)

    assert system_from_yaml.name == system_from_json.name
    assert len(system_from_yaml.elements) == len(system_from_json.elements)


def test_provisional_status_is_read_from_underscore_fields():
    for name in ("ag_cu.json", "al_cu.json"):
        system = load_system(SYSTEMS_DIR / name)
        assert system.status == "provisional"
        assert system.is_provisional
        assert system.status_reason


def test_status_defaults_to_none():
    system = load_system(EXAMPLE_SYSTEM_PATH)
    assert system.status is None
    assert not system.is_provisional
