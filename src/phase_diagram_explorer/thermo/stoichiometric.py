import numpy as np

from phase_diagram_explorer.thermo.cef import CEFModel, CEFParameter
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT
from phase_diagram_explorer.thermo.pure import PureElementGibbs
from phase_diagram_explorer.thermo.sublattice import FIXED, SublatticePhase


class _FormulaUnitGibbs:
    """(m + n) * (m*G_A + n*G_B + G_form): the per-atom energy of the
    two-element compound written per formula unit of m + n atoms."""

    def __init__(self, gibbs_a, gibbs_b, m: float, n: float, g_form):
        self.parts = (gibbs_a, gibbs_b, g_form)
        self.m, self.n = m, n

    def G(self, T):
        gibbs_a, gibbs_b, g_form = self.parts
        return (self.m + self.n) * (self.m * gibbs_a.G(T) + self.n * gibbs_b.G(T) + g_form.G(T))


class StoichiometricPhase(SublatticePhase):
    """Fixed-composition phase: a sublattice model in which every
    sublattice holds a single species (a compound, or a pure element).

    The two-element constructor keeps the JSON model, a compound A_m B_n with

        G(T) = m*G_A(T) + n*G_B(T) + G_form(T)

    per mole of atoms, where G_form(T) uses the same a + bT + cT*ln(T) +
    dT^2 + eT^3 + f/T polynomial form as a pure element Gibbs energy. It is
    the sublattice model (A)m(B)n with that energy per formula unit.
    """

    def __init__(
        self,
        gibbs_a: PureElementGibbs,
        gibbs_b: PureElementGibbs,
        m: float,
        n: float,
        g_form: PureElementGibbs | None = None,
        gas_constant: float = CODATA_GAS_CONSTANT,
        name: str = "compound",
    ):
        if m <= 0 or n <= 0:
            raise ValueError("stoichiometric coefficients m and n must be positive")
        self.gibbs_a = gibbs_a
        self.gibbs_b = gibbs_b
        self.m = m
        self.n = n
        self.g_form = g_form if g_form is not None else PureElementGibbs()
        energy = _FormulaUnitGibbs(gibbs_a, gibbs_b, m, n, self.g_form)
        model = CEFModel([m, n], [["A"], ["B"]], [CEFParameter((("A",), ("B",)), 0, energy)],
                         gas_constant=gas_constant, name=name)
        super().__init__(model, "A", "B")

    @classmethod
    def from_model(cls, model: CEFModel, base: str, dependent: str) -> "StoichiometricPhase":
        """A fixed-composition phase from a sublattice model."""
        phase = cls.__new__(cls)
        SublatticePhase.__init__(phase, model, base, dependent)
        if phase.mode != FIXED:
            raise ValueError(f"phase {model.name} does not have a fixed composition")
        return phase

    @property
    def composition(self) -> float:
        """Mole fraction of the dependent element."""
        return self.composition_range[0]

    def molar_gibbs(self, T, x=None):
        """G per mole of atoms (the composition argument is ignored)."""
        G = self.model.molar_gibbs(T, self._template(1))
        if np.ndim(T) == 0:
            return float(G[0])
        return np.asarray(G, dtype=float)
