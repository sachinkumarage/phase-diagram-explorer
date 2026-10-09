import json
import os
from pathlib import Path
from typing import Annotated, Literal, Sequence

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


class ExpressionInterval(BaseModel):
    T_min: float
    T_max: float
    # TDB arithmetic expression in T (see tdb/expression.py).
    expression: str


class PiecewiseExpression(BaseModel):
    """A temperature-dependent quantity given as TDB expressions on
    consecutive temperature intervals (from a TDB file)."""

    intervals: list[ExpressionInterval]


# A Gibbs energy given either as polynomial coefficients (JSON systems) or as
# a piecewise TDB expression. The expression form is tried first: it requires
# "intervals", so coefficient dicts always fall through to GibbsCoefficients.
GibbsEnergy = Annotated[PiecewiseExpression | GibbsCoefficients, Field(union_mode="left_to_right")]
Interaction = Annotated[PiecewiseExpression | float, Field(union_mode="left_to_right")]


class RawParameter(BaseModel):
    """A TDB PARAMETER kept as written: for phases or parameter types that
    no model here evaluates yet (multi-sublattice phases, TC, BMAGN)."""

    kind: str
    # Species on each sublattice, interacting species together.
    constituents: list[list[str]]
    order: int = 0
    function: PiecewiseExpression


class Phase(BaseModel):
    name: str
    # "sublattice": a TDB phase stored as read (sublattice_sites,
    # constituents, raw_parameters) that no model here evaluates yet.
    model_type: Literal["pure", "solution", "stoichiometric", "sublattice"]
    # "solution": Gibbs energy of each end member (pure component in this
    # phase's own reference structure) per mole of atoms, keyed by element symbol.
    end_members: dict[str, GibbsEnergy] | None = None
    # "solution": Redlich-Kister excess parameters L0, L1, L2, ... per mole of
    # atoms, multiplying (x_base - x_dependent)^v.
    interaction_parameters: list[Interaction] = []
    # "stoichiometric": mole count of each element in the compound, e.g.
    # {"A": 2, "B": 1} for A2B.
    stoichiometry: dict[str, float] | None = None
    # "stoichiometric": Gibbs energy of the compound per mole of atoms, on the
    # same reference as the end members (JSON systems reference each element
    # to its own stable solid, G = 0, so this is the energy of formation).
    formation: GibbsEnergy | None = None
    # TDB phases: sites and species of each sublattice, as read.
    sublattice_sites: list[float] | None = None
    constituents: list[list[str]] | None = None
    # TDB parameters stored but not evaluated (see RawParameter).
    raw_parameters: list[RawParameter] = []
    magnetic_parameters: list[RawParameter] = []
    # TDB TYPE_DEFINITION commands (other than plain SEQ ones) applying to
    # this phase, e.g. the magnetic AFCC/ABCC definitions.
    type_definitions: list[str] = []

    @property
    def is_magnetic(self) -> bool:
        return bool(self.magnetic_parameters) or any("MAGNETIC" in t.upper() for t in self.type_definitions)


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
    # TDB FUNCTIONs referenced by piecewise expressions, by upper-case name.
    functions: dict[str, PiecewiseExpression] = {}

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


METADATA_SUFFIX = ".meta.json"
TDB_SUFFIX = ".tdb"
SYSTEM_SUFFIXES = (".json", ".yaml", ".yml", TDB_SUFFIX)


def metadata_path(path: str | Path) -> Path:
    """The metadata file of a TDB system: <name>.meta.json next to <name>.tdb."""
    path = Path(path)
    return path.with_name(path.stem + METADATA_SUFFIX)


def load_system(
    path: str | Path,
    *,
    elements: Sequence[str] | None = None,
    dependent_element: str | None = None,
) -> SystemDefinition:
    """Load a binary system from JSON, YAML or a TDB database.

    For a .tdb file the element order (base element first, then the
    dependent element, whose mole fraction is the composition axis) comes
    from `elements` or `dependent_element`, falling back to the fields of
    the same name in <name>.meta.json; see docs/data_format.md. These
    arguments are ignored for JSON and YAML, which list their elements in
    order.
    """
    path = Path(path)
    if path.suffix.lower() == TDB_SUFFIX:
        from phase_diagram_explorer.tdb.convert import load_tdb_system

        return load_tdb_system(path, elements=elements, dependent_element=dependent_element)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return SystemDefinition.model_validate(data)


def system_files(directory: str | Path) -> list[Path]:
    """System definitions in a directory: JSON, YAML and TDB files, sorted,
    excluding TDB metadata files."""
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in SYSTEM_SUFFIXES and not path.name.endswith(METADATA_SUFFIX)
    )


def default_systems_dir() -> Path:
    """$PHASE_DIAGRAM_SYSTEMS_DIR, or data/systems in the source tree."""
    return Path(os.environ.get("PHASE_DIAGRAM_SYSTEMS_DIR", Path(__file__).resolve().parents[2] / "data" / "systems"))
