"""Convert a parsed TDB database into a binary SystemDefinition.

Every TDB phase becomes a Compound Energy Formalism phase (model_type
"sublattice"): its sublattices, site ratios, constituents and G/L
parameters are kept as written, per mole of formula units. TC and BMAGN
parameters, with the antiferromagnetic and structure factors of the
phase's MAGNETIC type definition, give its magnetic model. build_system
evaluates these (thermo/cef.py, thermo/sublattice.py).

Constituents must be the two elements of the system or the vacancy VA.
Other GES type definitions (e.g. a disordered part) are stored and make
build_system raise NotImplementedError.
"""
import json
from pathlib import Path
from typing import Sequence

from phase_diagram_explorer.models import (
    Element,
    MagneticSettings,
    Phase,
    RawParameter,
    SystemDefinition,
    metadata_path,
)
from phase_diagram_explorer.tdb.parser import (
    GIBBS_PARAMETERS,
    MAGNETIC_PARAMETERS,
    TdbDatabase,
    TdbError,
    TdbParameter,
    TdbPhase,
    read_tdb,
)
from phase_diagram_explorer.thermo.constants import DATABASE_GAS_CONSTANT

VACANCY = "VA"
WILDCARD = "*"
# ELEMENT entries that are not chemical elements.
NON_ELEMENTS = {"/-", VACANCY}


def _raw(parameter: TdbParameter) -> RawParameter:
    return RawParameter(
        kind=parameter.kind, constituents=parameter.constituents, order=parameter.order, function=parameter.function
    )


def _check_species(database: TdbDatabase, phase: str, names, allowed: set[str], line: int | None) -> None:
    where = f" (line {line})" if line else ""
    for name in names:
        if name in allowed:
            continue
        if name in database.species and name not in database.elements:
            raise NotImplementedError(
                f"phase {phase}{where}: species {name} ({database.species[name]}) is not an element; "
                "only element and VA constituents are supported"
            )
        raise TdbError(f"phase {phase}{where}: constituent {name} is not an element of the system")


def _phase(database: TdbDatabase, phase: TdbPhase, base: str, dependent: str) -> Phase:
    if phase.constituents is None:
        raise TdbError(f"phase {phase.name} (line {phase.line}) has no CONSTITUENT")
    allowed = {base, dependent, VACANCY}
    for species in phase.constituents:
        _check_species(database, phase.name, species, allowed, phase.line)

    parameters = [p for p in database.parameters if p.phase == phase.name]
    for parameter in parameters:
        if len(parameter.constituents) != len(phase.sites):
            raise TdbError(
                f"PARAMETER on line {parameter.line} gives {len(parameter.constituents)} sublattices; "
                f"phase {phase.name} has {len(phase.sites)}"
            )
        for k, species in enumerate(parameter.constituents):
            if species == [WILDCARD]:
                continue
            _check_species(database, phase.name, species, set(phase.constituents[k]), parameter.line)

    definitions = [database.type_definitions[code] for code in phase.types if code in database.type_definitions]
    magnetic = None
    for definition in definitions:
        if definition.is_magnetic:
            afm_factor, structure_factor = definition.magnetic_factors()
            magnetic = MagneticSettings(afm_factor=afm_factor, structure_factor=structure_factor)

    return Phase(
        name=phase.name,
        model_type="sublattice",
        sublattice_sites=phase.sites,
        constituents=phase.constituents,
        raw_parameters=[_raw(p) for p in parameters if p.kind in GIBBS_PARAMETERS],
        magnetic_parameters=[_raw(p) for p in parameters if p.kind in MAGNETIC_PARAMETERS],
        magnetic=magnetic,
        type_definitions=[f"TYPE_DEFINITION {d.code} {d.text}" for d in definitions if not d.is_sequential],
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
        phases=[_phase(database, phase, base, dependent) for phase in database.phases.values()],
        t_range_k=metadata.get("t_range_k"),
        functions=database.functions,
        gas_constant=metadata.get("gas_constant", DATABASE_GAS_CONSTANT),
        pressure_pa=metadata.get("pressure_pa", 101325.0),
        metadata=database.metadata,
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
