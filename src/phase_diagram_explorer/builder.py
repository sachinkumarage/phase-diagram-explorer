from phase_diagram_explorer.models import Phase, SystemDefinition
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.solution import SolutionPhase
from phase_diagram_explorer.thermo.stoichiometric import StoichiometricPhase

ComputableSystem = dict[str, SolutionPhase | StoichiometricPhase]


def _solution_phase(phase: Phase, symbol_a: str, symbol_b: str) -> SolutionPhase:
    if not phase.end_members:
        raise ValueError(f"solution phase {phase.name!r} has no end_members Gibbs energy data")
    missing = {symbol_a, symbol_b} - set(phase.end_members)
    if missing:
        raise ValueError(f"solution phase {phase.name!r} is missing end members {sorted(missing)}")
    return SolutionPhase(
        PureElementGibbs.from_coefficients(phase.end_members[symbol_a]),
        PureElementGibbs.from_coefficients(phase.end_members[symbol_b]),
        L=phase.interaction_parameters,
    )


def _stoichiometric_phase(phase: Phase, symbol_a: str, symbol_b: str) -> StoichiometricPhase:
    if not phase.stoichiometry:
        raise ValueError(f"stoichiometric phase {phase.name!r} has no stoichiometry")
    missing = {symbol_a, symbol_b} - set(phase.stoichiometry)
    if missing:
        raise ValueError(f"stoichiometric phase {phase.name!r} is missing site counts for {sorted(missing)}")
    g_form = PureElementGibbs.from_coefficients(phase.formation) if phase.formation else PureElementGibbs()
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

    Raises ValueError if a phase lacks the data needed to evaluate it.
    """
    symbol_a = definition.base_element.symbol
    symbol_b = definition.dependent_element.symbol
    system: ComputableSystem = {}

    for phase in definition.phases:
        if phase.model_type == "solution":
            system[phase.name] = _solution_phase(phase, symbol_a, symbol_b)
        elif phase.model_type == "stoichiometric":
            system[phase.name] = _stoichiometric_phase(phase, symbol_a, symbol_b)
        else:
            raise ValueError(f"unsupported model_type {phase.model_type!r} for phase {phase.name!r}")

    return system


def is_computable(definition: SystemDefinition) -> bool:
    """True if every phase has the Gibbs energy data build_system needs."""
    try:
        build_system(definition)
    except ValueError:
        return False
    return True
