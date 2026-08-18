import numpy as np

from phase_diagram_explorer.thermo.pure import PureElementGibbs


class StoichiometricPhase:
    """Fixed-composition compound A_m B_n.

    G(T) = m*G_A(T) + n*G_B(T) + G_form(T)

    where G_form(T) uses the same a + bT + cT*ln(T) + dT^2 + eT^3 + f/T
    polynomial form as a pure element Gibbs energy.
    """

    def __init__(
        self,
        gibbs_a: PureElementGibbs,
        gibbs_b: PureElementGibbs,
        m: float,
        n: float,
        g_form: PureElementGibbs | None = None,
    ):
        if m <= 0 or n <= 0:
            raise ValueError("stoichiometric coefficients m and n must be positive")
        self.gibbs_a = gibbs_a
        self.gibbs_b = gibbs_b
        self.m = m
        self.n = n
        self.g_form = g_form if g_form is not None else PureElementGibbs()

    @property
    def composition(self) -> float:
        """Mole fraction of B in the compound A_m B_n."""
        return self.n / (self.m + self.n)

    def molar_gibbs(self, T: float | np.ndarray) -> float | np.ndarray:
        return self.m * self.gibbs_a.G(T) + self.n * self.gibbs_b.G(T) + self.g_form.G(T)
