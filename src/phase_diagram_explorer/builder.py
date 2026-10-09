import logging

from phase_diagram_explorer.models import GibbsCoefficients, Phase, PiecewiseExpression, SystemDefinition
from phase_diagram_explorer.tdb.expression import FunctionTable, PiecewiseGibbs
from phase_diagram_explorer.thermo.cef import CEFModel, CEFParameter
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT
from phase_diagram_explorer.thermo.magnetic import MagneticModel
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase
from phase_diagram_explorer.thermo.sublattice import FIXED, SublatticePhase

logger = logging.getLogger(__name__)

ComputableSystem = dict[str, SublatticePhase]


def _gibbs(value: GibbsCoefficients | PiecewiseExpression, functions: FunctionTable, context: str):
    if isinstance(value, PiecewiseExpression):
        return PiecewiseGibbs(value, functions, context)
    return PureElementGibbs.from_coefficients(value)


def _interaction(value: float | PiecewiseExpression, functions: FunctionTable, context: str):
    if isinstance(value, PiecewiseExpression):
        return PiecewiseGibbs(value, functions, context)
    return value


def _solution_phase(phase: Phase, symbol_a: str, symbol_b: str, functions: FunctionTable) -> SolutionPhase:
    if not phase.end_members:
        raise ValueError(f"solution phase {phase.name!r} has no end_members Gibbs energy data")
    missing = {symbol_a, symbol_b} - set(phase.end_members)
    if missing:
        raise ValueError(f"solution phase {phase.name!r} is missing end members {sorted(missing)}")
    return SolutionPhase(
        _gibbs(phase.end_members[symbol_a], functions, f"phase {phase.name} end member {symbol_a}"),
        _gibbs(phase.end_members[symbol_b], functions, f"phase {phase.name} end member {symbol_b}"),
        L=[
            _interaction(L_v, functions, f"phase {phase.name} L{v}")
            for v, L_v in enumerate(phase.interaction_parameters)
        ],
        gas_constant=functions.gas_constant,
        name=phase.name,
    )


def _stoichiometric_phase(phase: Phase, symbol_a: str, symbol_b: str, functions: FunctionTable) -> StoichiometricPhase:
    if not phase.stoichiometry:
        raise ValueError(f"stoichiometric phase {phase.name!r} has no stoichiometry")
    missing = {symbol_a, symbol_b} - set(phase.stoichiometry)
    if missing:
        raise ValueError(f"stoichiometric phase {phase.name!r} is missing site counts for {sorted(missing)}")
    g_form = _gibbs(phase.formation, functions, f"phase {phase.name}") if phase.formation else PureElementGibbs()
    # Elements are referenced to their own stable solid (G = 0), so the
    # compound's Gibbs energy is its formation energy.
    return StoichiometricPhase(
        gibbs_a=PureElementGibbs(a=0.0),
        gibbs_b=PureElementGibbs(a=0.0),
        m=phase.stoichiometry[symbol_a],
        n=phase.stoichiometry[symbol_b],
        g_form=g_form,
        gas_constant=functions.gas_constant,
        name=phase.name,
    )


def _cef_parameters(parameters, functions: FunctionTable, phase: str) -> list[CEFParameter]:
    return [
        CEFParameter(
            tuple(tuple(species) for species in p.constituents),
            p.order,
            PiecewiseGibbs(p.function, functions, f"phase {phase} {p.kind}({':'.join(','.join(s) for s in p.constituents)};{p.order})"),
        )
        for p in parameters
    ]


def _sublattice_phase(phase: Phase, symbol_a: str, symbol_b: str, functions: FunctionTable) -> SublatticePhase:
    if not phase.sublattice_sites or not phase.constituents:
        raise ValueError(f"sublattice phase {phase.name!r} has no sublattices")
    curie = [p for p in phase.magnetic_parameters if p.kind == "TC"]
    moment = [p for p in phase.magnetic_parameters if p.kind == "BMAGN"]
    magnetic = None
    if phase.magnetic is not None:
        magnetic = MagneticModel(phase.magnetic.afm_factor, phase.magnetic.structure_factor)
    elif phase.magnetic_parameters:
        logger.warning("phase %s has TC/BMAGN parameters but no MAGNETIC type definition; they are ignored", phase.name)
    model = CEFModel(
        phase.sublattice_sites,
        phase.constituents,
        _cef_parameters(phase.raw_parameters, functions, phase.name),
        magnetic=magnetic,
        curie_temperature=_cef_parameters(curie, functions, phase.name),
        magnetic_moment=_cef_parameters(moment, functions, phase.name),
        gas_constant=functions.gas_constant,
        name=phase.name,
    )
    result = SublatticePhase(model, symbol_a.upper(), symbol_b.upper())
    if result.mode == FIXED:
        return StoichiometricPhase.from_model(model, symbol_a.upper(), symbol_b.upper())
    return result


def build_system(definition: SystemDefinition) -> ComputableSystem:
    """Construct evaluable thermodynamic phase objects from a system definition.

    Every Gibbs energy comes from the definition's own coefficient fields
    (end_members/interaction_parameters for solution phases,
    stoichiometry/formation for stoichiometric phases), looked up by element
    symbol. Composition is the mole fraction of `definition.dependent_element`,
    so end member A is `base_element` and end member B is `dependent_element`;
    the order of the phase list has no effect on any phase.

    "solution" and "stoichiometric" phases (JSON) and "sublattice" phases
    (TDB) are all Compound Energy Formalism phases (thermo/cef.py); the
    first two are its single-sublattice and fixed-composition special
    cases. The gas constant is the definition's, or 8.314462618 for JSON
    systems that do not set one.

    Raises ValueError if a phase lacks the data needed to evaluate it or an
    expression references an undefined FUNCTION, and NotImplementedError
    for models not implemented (e.g. a disordered-part type definition).
    """
    symbol_a = definition.base_element.symbol
    symbol_b = definition.dependent_element.symbol
    gas_constant = definition.gas_constant if definition.gas_constant is not None else CODATA_GAS_CONSTANT
    functions = FunctionTable(definition.functions, gas_constant=gas_constant, pressure=definition.pressure_pa)
    system: ComputableSystem = {}

    for phase in definition.phases:
        _check_supported(phase)
        if phase.model_type == "solution":
            system[phase.name] = _solution_phase(phase, symbol_a, symbol_b, functions)
        elif phase.model_type == "stoichiometric":
            system[phase.name] = _stoichiometric_phase(phase, symbol_a, symbol_b, functions)
        elif phase.model_type == "sublattice":
            system[phase.name] = _sublattice_phase(phase, symbol_a, symbol_b, functions)
        else:
            raise ValueError(f"unsupported model_type {phase.model_type!r} for phase {phase.name!r}")

    return system


def _check_supported(phase: Phase) -> None:
    unsupported = phase.unsupported_type_definitions
    if unsupported:
        raise NotImplementedError(f"phase {phase.name} uses a model this version cannot evaluate: {unsupported[0]}")


def is_computable(definition: SystemDefinition) -> bool:
    """True if build_system can evaluate every phase."""
    try:
        build_system(definition)
    except (ValueError, NotImplementedError):
        return False
    return True
