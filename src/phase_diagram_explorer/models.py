import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

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
    # Standard atomic weight in g/mol, used for mole <-> weight fraction conversion.
    atomic_mass: float | None = None


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


PROVISIONAL = "provisional"


class SystemDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    # Free-text description of where the data come from.
    source: str | None = Field(default=None, alias="_source")
    # Data status, e.g. "provisional" for parameters not yet validated against
    # assessed data, with a short human-readable reason.
    status: str | None = Field(default=None, alias="_status")
    status_reason: str | None = Field(default=None, alias="_status_reason")
    elements: list[Element]
    phases: list[Phase]
    # Full temperature range (K) over which the system is analysed: invariant
    # reactions are always detected over this range, independent of what is
    # displayed, and it is the default display range.
    t_range_k: tuple[float, float] | None = None

    @model_validator(mode="after")
    def check_t_range(self) -> "SystemDefinition":
        if self.t_range_k is not None:
            T_min, T_max = self.t_range_k
            if not 0.0 < T_min < T_max:
                raise ValueError(f"t_range_k must satisfy 0 < T_min < T_max, got {self.t_range_k}")
        return self

    @property
    def is_provisional(self) -> bool:
        return self.status == PROVISIONAL

    def _binary_elements(self) -> tuple[Element, Element]:
        if len(self.elements) != 2:
            raise ValueError(f"expected a binary system with 2 elements, got {len(self.elements)}")
        return self.elements[0], self.elements[1]

    @property
    def base_element(self) -> Element:
        """First listed element: x = 0 on the composition axis."""
        return self._binary_elements()[0]

    @property
    def dependent_element(self) -> Element:
        """Second listed element: the composition axis is its mole fraction,
        so x = 1 is the pure dependent element."""
        return self._binary_elements()[1]


def load_system(path: str | Path) -> SystemDefinition:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return SystemDefinition.model_validate(data)
