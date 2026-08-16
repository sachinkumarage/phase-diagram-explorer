import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, RootModel, model_validator

COMPOSITION_TOLERANCE = 1e-6


class Composition(RootModel[dict[str, float]]):
    @model_validator(mode="after")
    def check_sums_to_one(self) -> "Composition":
        total = sum(self.root.values())
        if abs(total - 1.0) > COMPOSITION_TOLERANCE:
            raise ValueError(f"mole fractions must sum to 1, got {total}")
        return self


class Element(BaseModel):
    symbol: str
    name: str
    reference_state: str


class Phase(BaseModel):
    name: str
    model_type: Literal["pure", "solution", "stoichiometric"]


class SystemDefinition(BaseModel):
    name: str
    elements: list[Element]
    phases: list[Phase]


def load_system(path: str | Path) -> SystemDefinition:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return SystemDefinition.model_validate(data)
