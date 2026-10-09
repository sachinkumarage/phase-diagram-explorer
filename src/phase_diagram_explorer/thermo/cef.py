"""Compound Energy Formalism (sublattice model).

A phase has sublattices s with site ratios a_s, each holding species i with
site fractions y_si (sum_i y_si = 1). Per mole of formula units

    G = sum_e  G_e(T) prod_s y_{s,e_s}                         (surface of reference)
      + R T sum_s a_s sum_i y_si ln y_si                        (ideal configurational entropy)
      + sum_L  prod_s prod_{i in L_s} y_si  L(T) (y_si - y_sj)^v  (excess, Redlich-Kister)
      + G_mag                                                   (Inden-Hillert-Jarl, magnetic.py)

where e runs over the end members (one species per sublattice) and L over
interaction parameters with two species i, j on one sublattice (order v);
interactions on several sublattices at once (reciprocal) or between three
species are accepted at order 0. A wildcard "*" on a sublattice means the
parameter does not depend on it (its site fractions sum to one).

The vacancy "VA" carries no atoms: the number of atoms per formula unit is
N = sum_s a_s (1 - y_s,VA), the Gibbs energy per mole of atoms is G / N and
the mole fraction of element k is sum_s a_s y_sk / N. TC and BMAGN of the
magnetic model are expanded over site fractions the same way as G.

Every quantity is evaluated vectorised over rows of a site-fraction matrix Y
(one column per sublattice-species pair), with analytic gradients.
"""
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from phase_diagram_explorer.thermo.constants import DATABASE_GAS_CONSTANT
from phase_diagram_explorer.thermo.magnetic import MagneticModel, magnetic_gibbs

VACANCY = "VA"
WILDCARD = "*"
VALUE_CACHE_SIZE = 256


@dataclass(frozen=True)
class CEFParameter:
    """A G or L parameter (or TC/BMAGN for the magnetic model): species per
    sublattice (interacting species together, or the wildcard "*"), the
    Redlich-Kister order, and a value given as a number or as an object
    with a G(T) method."""

    constituents: tuple[tuple[str, ...], ...]
    order: int
    value: object


def _value(value, T):
    return value.G(T) if hasattr(value, "G") else value


def _xlogx(y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = y * np.log(y)
    return np.where(y > 0.0, result, 0.0)


class _Term:
    """A compiled parameter: value * prod(Y[:, columns]) * (Y[:, i] - Y[:, j])^order."""

    __slots__ = ("value", "columns", "rk")

    def __init__(self, value, columns: tuple[int, ...], rk: tuple[int, int, int] | None):
        self.value = value
        self.columns = columns
        self.rk = rk

    def evaluate(self, T, Y: np.ndarray, gradient: np.ndarray | None, L=None):
        if L is None:
            L = _value(self.value, T)
        factors = [Y[..., c] for c in self.columns]
        product = np.ones(Y.shape[:-1]) if not factors else np.prod(factors, axis=0)
        if self.rk is None:
            rk = 1.0
        else:
            i, j, order = self.rk
            difference = Y[..., i] - Y[..., j]
            rk = difference**order
        term = L * product * rk
        if gradient is not None:
            for k, column in enumerate(self.columns):
                others = [f for m, f in enumerate(factors) if m != k]
                other_product = np.prod(others, axis=0) if others else 1.0
                gradient[..., column] += L * rk * other_product
            if self.rk is not None and self.rk[2] > 0:
                i, j, order = self.rk
                d_rk = L * product * order * difference ** (order - 1)
                gradient[..., i] += d_rk
                gradient[..., j] -= d_rk
        return term


class CEFModel:
    """Gibbs energy of a sublattice phase per mole of formula units as a
    function of site fractions (see module docstring)."""

    def __init__(
        self,
        sites: Sequence[float],
        constituents: Sequence[Sequence[str]],
        parameters: Sequence[CEFParameter],
        *,
        magnetic: MagneticModel | None = None,
        curie_temperature: Sequence[CEFParameter] = (),
        magnetic_moment: Sequence[CEFParameter] = (),
        gas_constant: float = DATABASE_GAS_CONSTANT,
        name: str = "phase",
    ):
        if len(sites) != len(constituents):
            raise ValueError(f"{name}: {len(sites)} site ratios for {len(constituents)} sublattices")
        self.name = name
        self.sites = tuple(float(a) for a in sites)
        self.constituents = tuple(tuple(species) for species in constituents)
        self.columns: list[tuple[int, str]] = [
            (s, species) for s, sublattice in enumerate(self.constituents) for species in sublattice
        ]
        self.column = {key: k for k, key in enumerate(self.columns)}
        self.gas_constant = float(gas_constant)
        self.magnetic = magnetic
        self.terms = [self._compile(p) for p in parameters]
        self.curie_terms = [self._compile(p) for p in curie_temperature]
        self.moment_terms = [self._compile(p) for p in magnetic_moment]
        # Atoms per formula unit contributed by each column (vacancies carry none).
        self.atom_vector = np.array([0.0 if species == VACANCY else self.sites[s] for s, species in self.columns])
        # Entropy only involves sublattices with more than one species.
        self._mixing = [
            (k, self.sites[s]) for k, (s, _) in enumerate(self.columns) if len(self.constituents[s]) > 1
        ]
        self._mixing_columns = np.array([k for k, _ in self._mixing], dtype=int)
        self._mixing_weights = np.array([w for _, w in self._mixing])
        self._atoms = [float(a) for a in self.atom_vector]
        self._values_cache: OrderedDict = OrderedDict()

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def sublattice_columns(self, s: int) -> list[int]:
        return [self.column[(s, species)] for species in self.constituents[s]]

    def species_vector(self, species: str) -> np.ndarray:
        """Moles of `species` per formula unit contributed by each column."""
        return np.array([self.sites[s] if name == species else 0.0 for s, name in self.columns])

    def _compile(self, parameter: CEFParameter) -> _Term:
        if len(parameter.constituents) != len(self.sites):
            raise ValueError(
                f"{self.name}: parameter {parameter.constituents} has {len(parameter.constituents)} sublattices, "
                f"the phase has {len(self.sites)}"
            )
        columns: list[int] = []
        mixing: list[tuple[int, ...]] = []
        for s, species in enumerate(parameter.constituents):
            if tuple(species) == (WILDCARD,):
                continue
            indices = []
            for name in species:
                if (s, name) not in self.column:
                    raise ValueError(f"{self.name}: parameter species {name} is not on sublattice {s + 1}")
                indices.append(self.column[(s, name)])
            if len(set(indices)) != len(indices):
                raise ValueError(f"{self.name}: parameter {parameter.constituents} repeats a species")
            columns += indices
            if len(indices) > 1:
                mixing.append(tuple(indices))

        rk = None
        if parameter.order != 0:
            if len(mixing) != 1 or len(mixing[0]) != 2:
                raise NotImplementedError(
                    f"{self.name}: order-{parameter.order} parameter {parameter.constituents} is not a binary "
                    "interaction on one sublattice (reciprocal and ternary parameters are supported at order 0)"
                )
            rk = (mixing[0][0], mixing[0][1], parameter.order)
        return _Term(parameter.value, tuple(columns), rk)

    def atoms(self, Y: np.ndarray) -> np.ndarray:
        """Atoms per formula unit, N = sum_s a_s (1 - y_s,VA)."""
        return Y @ self.atom_vector

    def _values(self, T):
        """Parameter values (G/L, TC, BMAGN terms) at T, cached for scalar T."""
        if np.ndim(T) != 0:
            return [[_value(t.value, T) for t in terms] for terms in (self.terms, self.curie_terms, self.moment_terms)]
        key = float(T)
        cached = self._values_cache.get(key)
        if cached is None:
            cached = [[float(_value(t.value, key)) for t in terms] for terms in (self.terms, self.curie_terms, self.moment_terms)]
            self._values_cache[key] = cached
            if len(self._values_cache) > VALUE_CACHE_SIZE:
                self._values_cache.popitem(last=False)
        return cached

    def _property(self, terms: list[_Term], values, T, Y: np.ndarray, gradient: bool):
        value = np.zeros(Y.shape[:-1])
        grad = np.zeros(Y.shape) if gradient else None
        for term, L in zip(terms, values):
            value = value + term.evaluate(T, Y, grad, L)
        return value, grad

    def formula_gibbs(self, T, Y: np.ndarray, gradient: bool = False):
        """G per mole of formula units at site fractions Y (..., n_columns),
        and its gradient with respect to Y if requested."""
        Y = np.asarray(Y, dtype=float)
        values, curie_values, moment_values = self._values(T)
        G, grad = self._property(self.terms, values, T, Y, gradient)

        if len(self._mixing_columns):
            RT = self.gas_constant * np.asarray(T, dtype=float)
            Y_mix = Y[..., self._mixing_columns]
            G = G + RT * (_xlogx(Y_mix) @ self._mixing_weights)
            if gradient:
                with np.errstate(divide="ignore"):
                    grad[..., self._mixing_columns] += (
                        (RT[..., None] if np.ndim(RT) else RT) * self._mixing_weights * (np.log(Y_mix) + 1.0)
                    )

        if self.magnetic is not None and self.curie_terms and self.moment_terms:
            TC, dTC = self._property(self.curie_terms, curie_values, T, Y, gradient)
            beta, dbeta = self._property(self.moment_terms, moment_values, T, Y, gradient)
            G_mag, dG_dTC, dG_dbeta = magnetic_gibbs(T, TC, beta, self.magnetic, self.gas_constant)
            G = G + G_mag
            if gradient:
                grad = grad + dG_dTC[..., None] * dTC + dG_dbeta[..., None] * dbeta
        return (G, grad) if gradient else G

    def molar_gibbs(self, T, Y: np.ndarray) -> np.ndarray:
        """G per mole of atoms at site fractions Y."""
        Y = np.asarray(Y, dtype=float)
        return self.formula_gibbs(T, Y) / self.atoms(Y)

    def molar_gibbs_gradient(self, T, Y: np.ndarray):
        """(G per mole of atoms, its gradient with respect to Y)."""
        Y = np.asarray(Y, dtype=float)
        G, grad = self.formula_gibbs(T, Y, gradient=True)
        N = self.atoms(Y)
        G_m = G / N
        return G_m, (grad - G_m[..., None] * self.atom_vector) / N[..., None]

    def curie_temperature(self, T, Y: np.ndarray) -> np.ndarray:
        return self._property(self.curie_terms, self._values(T)[1], T, np.asarray(Y, dtype=float), False)[0]

    def magnetic_moment(self, T, Y: np.ndarray) -> np.ndarray:
        return self._property(self.moment_terms, self._values(T)[2], T, np.asarray(Y, dtype=float), False)[0]

    # --- single points, in plain Python (the tangent solvers' hot path) -------------

    @staticmethod
    def _point_property(terms: list[_Term], values, y: list[float], grad: list[float] | None) -> float:
        total = 0.0
        for term, L in zip(terms, values):
            columns = term.columns
            product = 1.0
            for c in columns:
                product *= y[c]
            rk = 1.0
            if term.rk is not None:
                i, j, order = term.rk
                difference = y[i] - y[j]
                rk = difference**order
            total += L * product * rk
            if grad is not None:
                for k, c in enumerate(columns):
                    other = 1.0
                    for m, c2 in enumerate(columns):
                        if m != k:
                            other *= y[c2]
                    grad[c] += L * rk * other
                if term.rk is not None and term.rk[2] > 0:
                    d_rk = L * product * order * difference ** (order - 1)
                    grad[i] += d_rk
                    grad[j] -= d_rk
        return total

    def point_molar_gibbs(self, T: float, y: list[float], gradient: bool = False):
        """G per mole of atoms at one scalar T and one site-fraction row y,
        and its gradient with respect to y if requested."""
        values, curie_values, moment_values = self._values(T)
        grad = [0.0] * len(y) if gradient else None
        G = self._point_property(self.terms, values, y, grad)
        RT = self.gas_constant * T
        for k, weight in self._mixing:
            yk = y[k]
            if yk > 0.0:
                log_y = math.log(yk)
                G += RT * weight * yk * log_y
                if gradient:
                    grad[k] += RT * weight * (log_y + 1.0)
            elif gradient:
                grad[k] = -math.inf
        if self.magnetic is not None and self.curie_terms and self.moment_terms:
            d_TC = [0.0] * len(y) if gradient else None
            d_beta = [0.0] * len(y) if gradient else None
            TC = self._point_property(self.curie_terms, curie_values, y, d_TC)
            beta = self._point_property(self.moment_terms, moment_values, y, d_beta)
            G_mag, dG_dTC, dG_dbeta = (float(v) for v in magnetic_gibbs(T, TC, beta, self.magnetic, self.gas_constant))
            G += G_mag
            if gradient:
                grad = [g + dG_dTC * a + dG_dbeta * b for g, a, b in zip(grad, d_TC, d_beta)]
        N = 0.0
        for yk, atoms in zip(y, self._atoms):
            N += yk * atoms
        G_m = G / N
        if not gradient:
            return G_m
        return G_m, [(g - G_m * atoms) / N for g, atoms in zip(grad, self._atoms)]

    def site_fraction_matrix(self, fractions: Sequence[dict[str, float]]) -> np.ndarray:
        """One row of Y from site fractions given per sublattice as {species: y}."""
        row = np.zeros(self.n_columns)
        for s, sublattice in enumerate(fractions):
            for species, y in sublattice.items():
                row[self.column[(s, species)]] = y
        return row
