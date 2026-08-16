import json
from pathlib import Path

import numpy as np

from phase_diagram_explorer.models import GibbsCoefficients


class PureElementGibbs:
    """Gibbs energy of a pure element: G(T) = a + bT + cT*ln(T) + dT^2 + eT^3 + f/T"""

    def __init__(
        self,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0,
        d: float = 0.0,
        e: float = 0.0,
        f: float = 0.0,
        T_min: float | None = None,
        T_max: float | None = None,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.T_min = T_min
        self.T_max = T_max

    @classmethod
    def from_coefficients(cls, coefficients: GibbsCoefficients) -> "PureElementGibbs":
        return cls(**coefficients.model_dump())

    @classmethod
    def from_json(cls, path: str | Path) -> "PureElementGibbs":
        data = json.loads(Path(path).read_text())
        return cls.from_coefficients(GibbsCoefficients.model_validate(data))

    def _validate_range(self, T: np.ndarray) -> None:
        if self.T_min is not None and np.any(T < self.T_min):
            raise ValueError(f"temperature below valid range T_min={self.T_min}")
        if self.T_max is not None and np.any(T > self.T_max):
            raise ValueError(f"temperature above valid range T_max={self.T_max}")

    def G(self, T: float | np.ndarray) -> float | np.ndarray:
        T_arr = np.asarray(T, dtype=float)
        self._validate_range(T_arr)
        result = (
            self.a
            + self.b * T_arr
            + self.c * T_arr * np.log(T_arr)
            + self.d * T_arr**2
            + self.e * T_arr**3
            + self.f / T_arr
        )
        if np.ndim(T) == 0:
            return float(result)
        return result
