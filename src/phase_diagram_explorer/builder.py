from phase_diagram_explorer.models import GibbsCoefficients, Phase, PiecewiseExpression, SystemDefinition
from phase_diagram_explorer.tdb.expression import FunctionTable, PiecewiseGibbs
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

ComputableSystem = dict[str, SolutionPhase | StoichiometricPhase]
MAGNETIC_MODEL_MESSAGE = "magnetic model: added in 0.2.1"


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
    )


def build_system(definition: SystemDefinition) -> ComputableSystem:
    """Construct evaluable thermodynamic phase objects from a system definition.

    Every Gibbs energy comes from the definition's own coefficient fields
    (end_members/interaction_parameters for solution phases,
    stoichiometry/formation for stoichiometric phases), looked up by element
    symbol. Composition is the mole fraction of `definition.dependent_element`,
    so end member A is `base_element` and end member B is `dependent_element`;
    the order of the phase list has no effect on any phase.

    Raises ValueError if a phase lacks the data needed to evaluate it or an
    expression references an undefined FUNCTION, and NotImplementedError
    for phases that need a model not implemented yet (magnetic or
    multi-sublattice phases read from TDB files).
    """
    symbol_a = definition.base_element.symbol
    symbol_b = definition.dependent_element.symbol
    functions = FunctionTable(definition.functions)
    system: ComputableSystem = {}

    for phase in definition.phases:
        _check_supported(phase)
        if phase.model_type == "solution":
            system[phase.name] = _solution_phase(phase, symbol_a, symbol_b, functions)
        elif phase.model_type == "stoichiometric":
            system[phase.name] = _stoichiometric_phase(phase, symbol_a, symbol_b, functions)
        else:
            raise ValueError(f"unsupported model_type {phase.model_type!r} for phase {phase.name!r}")

    return system


def _check_supported(phase: Phase) -> None:
    if phase.is_magnetic:
        raise NotImplementedError(f"{MAGNETIC_MODEL_MESSAGE} (phase {phase.name} has a magnetic contribution)")
    other_models = [t for t in phase.type_definitions if "GES" in t.upper().split()]
    if other_models:
        raise NotImplementedError(f"phase {phase.name} uses a model this version cannot evaluate: {other_models[0]}")
    if phase.model_type == "sublattice":
        sublattices = [s for s in (phase.constituents or []) if s != ["VA"]]
        if len(sublattices) > 1:
            raise NotImplementedError(
                f"sublattice model: phase {phase.name} mixes species on {len(sublattices)} sublattices, "
                "which the substitutional and stoichiometric models cannot represent"
            )
        raise NotImplementedError(
            f"phase {phase.name} (constituents {phase.constituents}) is neither a binary substitutional solution "
            "nor a two-element compound"
        )


def is_computable(definition: SystemDefinition) -> bool:
    """True if build_system can evaluate every phase."""
    try:
        build_system(definition)
    except (ValueError, NotImplementedError):
        return False
    return True
