import numpy as np

from phase_diagram_explorer.thermo.cef import CEFModel, CEFParameter
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.sublattice import SublatticePhase

# Default gas constant of the JSON models (see constants.py).
GAS_CONSTANT = CODATA_GAS_CONSTANT
_A, _B = "A", "B"


class SolutionPhase(SublatticePhase):
    """Binary substitutional solution phase using a Redlich-Kister excess term.

    G_m = x_A*G_A(T) + x_B*G_B(T)
          + R*T*(x_A*ln(x_A) + x_B*ln(x_B))
          + x_A*x_B * sum_v L_v * (x_A - x_B)^v

    This is the sublattice model (A,B)1 (see cef.py) with end members G_A,
    G_B and interaction parameters L_v. Each L_v is a number or, if it
    depends on temperature, an object with a G(T) method (like
    PureElementGibbs).
    """

    def __init__(
        self,
        gibbs_a: PureElementGibbs,
        gibbs_b: PureElementGibbs,
        L: list | None = None,
        gas_constant: float = GAS_CONSTANT,
        name: str = "solution",
    ):
        self.gibbs_a = gibbs_a
        self.gibbs_b = gibbs_b
        self.L = list(L) if L is not None else []
        parameters = [
            CEFParameter(((_A,),), 0, gibbs_a),
            CEFParameter(((_B,),), 0, gibbs_b),
        ] + [CEFParameter(((_A, _B),), v, L_v) for v, L_v in enumerate(self.L) if not _is_zero(L_v)]
        super().__init__(
            CEFModel([1.0], [[_A, _B]], parameters, gas_constant=gas_constant, name=name), _A, _B
        )


def _is_zero(value) -> bool:
    return not hasattr(value, "G") and np.all(np.asarray(value) == 0.0)
