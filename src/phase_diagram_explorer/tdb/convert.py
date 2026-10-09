"""Convert a parsed TDB database into a binary SystemDefinition.

Each TDB phase becomes one of:

- "solution": one sublattice holding both elements (plus any sublattices
  holding only VA). End-member G parameters and G/L interaction parameters
  are divided by the site count, giving energies per mole of atoms, and
  interaction parameters listed as (dependent, base) have odd orders negated
  so they multiply (x_base - x_dependent)^v.
- "stoichiometric": every sublattice (apart from VA-only ones) holds a
  single element. Its G parameter is divided by the total number of sites.
- "sublattice": anything else (mixing on several sublattices, or a phase of
  only one element). Its sublattices and parameters are stored as read;
  build_system raises NotImplementedError for it.

TC and BMAGN parameters and the GES type definitions a phase uses are
stored on the phase; build_system raises NotImplementedError for magnetic
phases until the magnetic model exists.
"""
import json
from pathlib import Path
from typing import Sequence

from phase_diagram_explorer.models import (
    Element,
    Phase,
    PiecewiseExpression,
    RawParameter,
    SystemDefinition,
    metadata_path,
)
from phase_diagram_explorer.tdb.expression import divide, scale
from phase_diagram_explorer.tdb.parser import (
    GIBBS_PARAMETERS,
    MAGNETIC_PARAMETERS,
    TdbDatabase,
    TdbError,
    TdbParameter,
    TdbPhase,
    read_tdb,
)

VACANCY = "VA"
# ELEMENT entries that are not chemical elements.
NON_ELEMENTS = {"/-", VACANCY}


def _raw(parameter: TdbParameter) -> RawParameter:
    return RawParameter(
        kind=parameter.kind, constituents=parameter.constituents, order=parameter.order, function=parameter.function
    )


def _is_vacancy_sublattice(species: list[str]) -> bool:
    return species == [VACANCY]


def _mixing_sublattice(phase: TdbPhase, parameter: TdbParameter) -> list[str]:
    """The species of a parameter on the phase's one non-VA sublattice."""
    if len(parameter.constituents) != len(phase.sites):
        raise TdbError(
            f"PARAMETER on line {parameter.line} gives {len(parameter.constituents)} sublattices; "
            f"phase {phase.name} has {len(phase.sites)}"
        )
    species = None
    for k, sublattice in enumerate(parameter.constituents):
        if _is_vacancy_sublattice(phase.constituents[k]):
            if sublattice != [VACANCY]:
                raise TdbError(f"PARAMETER on line {parameter.line}: expected VA on sublattice {k + 1} of {phase.name}")
        else:
            species = sublattice
    return species


def _solution_phase(phase: TdbPhase, parameters: list[TdbParameter], base: str, dependent: str, symbols) -> dict:
    (index,) = [k for k, species in enumerate(phase.constituents) if not _is_vacancy_sublattice(species)]
    sites = phase.sites[index]
    end_members: dict[str, PiecewiseExpression] = {}
    interactions: dict[int, PiecewiseExpression] = {}

    for parameter in parameters:
        species = _mixing_sublattice(phase, parameter)
        if len(species) == 1:
            if parameter.order != 0:
                raise TdbError(f"end-member PARAMETER on line {parameter.line} has order {parameter.order}")
            (element,) = species
            if element not in (base, dependent):
                raise TdbError(f"PARAMETER on line {parameter.line}: species {element} is not an element of the system")
            if symbols[element] in end_members:
                raise TdbError(f"duplicate end member {element} of phase {phase.name} on line {parameter.line}")
            end_members[symbols[element]] = divide(parameter.function, sites)
        elif sorted(species) == sorted([base, dependent]):
            if parameter.order in interactions:
                raise TdbError(f"duplicate order-{parameter.order} interaction of {phase.name} on line {parameter.line}")
            sign = 1.0 if species == [base, dependent] or parameter.order % 2 == 0 else -1.0
            interactions[parameter.order] = scale(divide(parameter.function, sites), sign)
        else:
            raise NotImplementedError(
                f"PARAMETER on line {parameter.line}: interaction {','.join(species)} in phase {phase.name} "
                "is not a binary Redlich-Kister parameter"
            )

    orders = range(max(interactions) + 1) if interactions else range(0)
    return {
        "model_type": "solution",
        "end_members": end_members,
        "interaction_parameters": [interactions.get(v, 0.0) for v in orders],
    }


def _stoichiometric_phase(phase: TdbPhase, parameters: list[TdbParameter], symbols) -> dict:
    stoichiometry: dict[str, float] = {}
    for species, sites in zip(phase.constituents, phase.sites):
        if not _is_vacancy_sublattice(species):
            stoichiometry[symbols[species[0]]] = stoichiometry.get(symbols[species[0]], 0.0) + sites
    if len(parameters) != 1 or parameters[0].order != 0:
        lines = ", ".join(str(p.line) for p in parameters) or "none"
        raise TdbError(f"stoichiometric phase {phase.name} needs exactly one G parameter (lines: {lines})")
    (parameter,) = parameters
    if parameter.constituents != phase.constituents:
        raise TdbError(f"PARAMETER on line {parameter.line} does not match the constituents of {phase.name}")
    return {
        "model_type": "stoichiometric",
        "stoichiometry": stoichiometry,
        "formation": divide(parameter.function, sum(stoichiometry.values())),
    }


def _phase(database: TdbDatabase, phase: TdbPhase, base: str, dependent: str, symbols) -> Phase:
    if phase.constituents is None:
        raise TdbError(f"phase {phase.name} (line {phase.line}) has no CONSTITUENT")
    for species in phase.constituents:
        for name in species:
            if name != VACANCY and name not in (base, dependent):
                raise TdbError(f"phase {phase.name} has constituent {name}, which is not an element of the system")

    parameters = [p for p in database.parameters if p.phase == phase.name]
    gibbs = [p for p in parameters if p.kind in GIBBS_PARAMETERS]
    magnetic = [_raw(p) for p in parameters if p.kind in MAGNETIC_PARAMETERS]
    type_definitions = [
        f"TYPE_DEFINITION {definition.code} {definition.text}"
        for code in phase.types
        if (definition := database.type_definitions.get(code)) is not None and not definition.is_sequential
    ]

    mixing = [s for s in phase.constituents if not _is_vacancy_sublattice(s)]
    elements = {name for species in mixing for name in species}
    if len(mixing) == 1 and elements == {base, dependent} and VACANCY not in mixing[0]:
        fields = _solution_phase(phase, gibbs, base, dependent, symbols)
    elif len(mixing) >= 2 and all(len(s) == 1 for s in mixing) and elements == {base, dependent}:
        fields = _stoichiometric_phase(phase, gibbs, symbols)
    else:
        fields = {"model_type": "sublattice", "raw_parameters": [_raw(p) for p in gibbs]}

    return Phase(
        name=phase.name,
        sublattice_sites=phase.sites,
        constituents=phase.constituents,
        magnetic_parameters=magnetic,
        type_definitions=type_definitions,
        **fields,
    )


def _element_order(
    database: TdbDatabase, elements: Sequence[str] | None, dependent_element: str | None, source: str
) -> tuple[str, str]:
    available = [symbol for symbol in database.elements if symbol not in NON_ELEMENTS]
    if len(available) != 2:
        raise ValueError(f"{source}: expected a binary database with 2 elements, got {available}")
    if elements is not None:
        order = [symbol.upper() for symbol in elements]
        if sorted(order) != sorted(available):
            raise ValueError(f"{source}: element order {list(elements)} does not match the database elements {available}")
        if dependent_element is not None and dependent_element.upper() != order[1]:
            raise ValueError(f"{source}: dependent element {dependent_element} is not the second of {list(elements)}")
        return order[0], order[1]
    if dependent_element is not None:
        dependent = dependent_element.upper()
        if dependent not in available:
            raise ValueError(f"{source}: dependent element {dependent_element} is not in the database ({available})")
        (base,) = [symbol for symbol in available if symbol != dependent]
        return base, dependent
    raise ValueError(
        f"{source}: the element order is not defined. Pass elements=[base, dependent] or dependent_element=, "
        "or give \"elements\" or \"dependent_element\" in the .meta.json file"
    )


def _display_symbol(symbol: str, preferred: Sequence[str]) -> str:
    """Symbol as spelled in `preferred` (e.g. "Ag"), else capitalised."""
    for spelling in preferred:
        if spelling.upper() == symbol:
            return spelling
    return symbol.capitalize()


def definition_from_database(
    database: TdbDatabase,
    *,
    elements: Sequence[str] | None = None,
    dependent_element: str | None = None,
    metadata: dict | None = None,
    source: str = "TDB database",
) -> SystemDefinition:
    """A SystemDefinition from a parsed database. Arguments take precedence
    over the corresponding `metadata` fields."""
    metadata = dict(metadata or {})
    if elements is None and dependent_element is None:
        elements = metadata.get("elements")
        dependent_element = metadata.get("dependent_element")
    base, dependent = _element_order(database, elements, dependent_element, source)

    spellings = [*(elements or []), *([dependent_element] if dependent_element else [])]
    symbols = {symbol: _display_symbol(symbol, spellings) for symbol in (base, dependent)}
    names = metadata.get("element_names", {})
    element_models = []
    for symbol in (base, dependent):
        tdb_element = database.elements[symbol]
        element_models.append(
            Element(
                symbol=symbols[symbol],
                name=names.get(symbols[symbol], symbols[symbol]),
                reference_state=tdb_element.reference_state.lower(),
                atomic_mass=tdb_element.mass if tdb_element.mass > 0 else None,
            )
        )

    return SystemDefinition(
        name=metadata.get("name", f"{symbols[base]}-{symbols[dependent]}"),
        source=metadata.get("_source"),
        status=metadata.get("_status"),
        status_reason=metadata.get("_status_reason"),
        elements=element_models,
        phases=[_phase(database, phase, base, dependent, symbols) for phase in database.phases.values()],
        t_range_k=metadata.get("t_range_k"),
        functions=database.functions,
    )


def load_tdb_system(
    path: str | Path, *, elements: Sequence[str] | None = None, dependent_element: str | None = None
) -> SystemDefinition:
    """Load a binary system from a TDB file and its optional
    <name>.meta.json (see docs/data_format.md)."""
    path = Path(path)
    meta = metadata_path(path)
    metadata = json.loads(meta.read_text()) if meta.exists() else {}
    return definition_from_database(
        read_tdb(path), elements=elements, dependent_element=dependent_element, metadata=metadata, source=str(path)
    )
