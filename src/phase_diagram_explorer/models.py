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


class GibbsCoefficients(BaseModel):
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    f: float = 0.0
    T_min: float | None = None
    T_max: float | None = None


class Phase(BaseModel):
    name: str
    model_type: Literal["pure", "solution", "stoichiometric"]
    # "solution": Gibbs energy coefficients of each end member (pure component
    # in this phase's own reference structure), keyed by element symbol.
    end_members: dict[str, GibbsCoefficients] | None = None
    # "solution": Redlich-Kister excess parameters L0, L1, L2, ...
    interaction_parameters: list[float] = []
    # "stoichiometric": mole count of each element in the compound, e.g.
    # {"A": 2, "B": 1} for A2B.
    stoichiometry: dict[str, float] | None = None
    # "stoichiometric": Gibbs energy of formation from the pure elements.
    formation: GibbsCoefficients | None = None


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
