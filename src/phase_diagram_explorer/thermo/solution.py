import numpy as np

from phase_diagram_explorer.thermo.pure import PureElementGibbs

GAS_CONSTANT = 8.314462618


def _xlogx(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = x * np.log(x)
    return np.where(x > 0, result, 0.0)


class SolutionPhase:
    """Binary substitutional solution phase using a Redlich-Kister excess term.

    G_m = x_A*G_A(T) + x_B*G_B(T)
          + R*T*(x_A*ln(x_A) + x_B*ln(x_B))
          + x_A*x_B * sum_v L_v * (x_A - x_B)^v
    """

    def __init__(
        self,
        gibbs_a: PureElementGibbs,
        gibbs_b: PureElementGibbs,
        L: list[float] | None = None,
    ):
        self.gibbs_a = gibbs_a
        self.gibbs_b = gibbs_b
        self.L = list(L) if L is not None else []

    def _redlich_kister(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        diff = x_a - x_b
        excess = np.zeros_like(diff, dtype=float)
        for v, L_v in enumerate(self.L):
            excess = excess + L_v * diff**v
        return excess

    def molar_gibbs(self, T: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        x_b = np.asarray(x, dtype=float)
        x_a = 1.0 - x_b

        ideal = x_a * self.gibbs_a.G(T) + x_b * self.gibbs_b.G(T)
        entropy = GAS_CONSTANT * np.asarray(T, dtype=float) * (_xlogx(x_a) + _xlogx(x_b))
        excess = x_a * x_b * self._redlich_kister(x_a, x_b)

        result = ideal + entropy + excess

        if np.ndim(T) == 0 and np.ndim(x) == 0:
            return float(result)
        return result
