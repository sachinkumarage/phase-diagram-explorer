"""A sublattice (CEF) phase of a binary system, as a function of composition.

The composition x is the mole fraction of the dependent element per mole of
atoms, vacancies excluded:

    x = sum_s a_s y_s,dep / N,    N = sum_s a_s (1 - y_s,VA)

and every Gibbs energy is per mole of atoms (G / N). Depending on how many
site fractions are free, a phase is one of:

- fixed: every sublattice holds one species (a compound or a pure element),
  so x is fixed;
- direct: one sublattice mixes two species (a substitutional solution
  (A,B), or an interstitial (A)a(C,VA)c). x determines the site fractions,
  so G(x) is in closed form;
- minimise: more site fractions are free than composition fixes (internal
  degrees of freedom, e.g. ordering (A,B)2(A,B)1). G(x) is the minimum over
  the site fractions consistent with x, all within (1e-12, 1). With one
  internal degree of freedom this is a vectorised search over a sampled
  interval refined by root-finding on the derivative; with more it is SLSQP
  from several starting points. Results are cached per (T, x).

dG/dx is analytic in every mode: at a constrained minimum it is the
Lagrange multiplier of the composition constraint times N.
"""
import itertools
import math
import warnings
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from phase_diagram_explorer.thermo.cef import VACANCY, CEFModel
from phase_diagram_explorer.thermo.constants import MIN_SITE_FRACTION

FIXED = "fixed"
DIRECT = "direct"
MINIMISE = "minimise"

# Samples of the internal coordinate before refinement (one internal degree of freedom).
INTERNAL_SAMPLES = 48
INTERNAL_EDGE_SAMPLES = 8
REFINEMENT_ITERATIONS = 60
CACHE_SIZE = 50000
RANGE_TOLERANCE = 1e-12
# Compositions are kept this far inside the composition range when
# minimising, so that every site fraction can stay above MIN_SITE_FRACTION.
EDGE_MARGIN = 1e-9
# Refinement stops once the derivative along the internal coordinate is
# below this (J/mol per unit change of the coordinate).
SLOPE_TOLERANCE = 1e-9


class ConvergenceWarning(RuntimeWarning):
    pass


class SublatticePhase:
    """A binary phase described by a CEFModel; see module docstring."""

    def __init__(self, model: CEFModel, base: str, dependent: str):
        self.model = model
        self.base = base
        self.dependent = dependent
        for _, species in model.columns:
            if species not in (base, dependent, VACANCY):
                raise NotImplementedError(
                    f"phase {model.name}: species {species} is neither {base}, {dependent} nor {VACANCY}"
                )
        self._d = model.species_vector(dependent)
        self._n = model.atom_vector
        self.free_sublattices = [s for s, sp in enumerate(model.constituents) if len(sp) > 1]
        self._fixed_columns = [
            model.column[(s, sp[0])] for s, sp in enumerate(model.constituents) if len(sp) == 1
        ]
        self._free_columns = [c for s in self.free_sublattices for c in model.sublattice_columns(s)]

        compositions = []
        for choice in itertools.product(*model.constituents):
            row = np.zeros(model.n_columns)
            for s, species in enumerate(choice):
                row[model.column[(s, species)]] = 1.0
            atoms = row @ self._n
            if atoms <= 0.0:
                raise NotImplementedError(f"phase {model.name}: end member {':'.join(choice)} has no atoms")
            compositions.append(row @ self._d / atoms)
        self.composition_range = (float(min(compositions)), float(max(compositions)))

        n_free = sum(len(model.constituents[s]) - 1 for s in self.free_sublattices)
        if n_free == 0 or self.composition_range[0] == self.composition_range[1]:
            self.mode = FIXED
        elif n_free == 1:
            self.mode = DIRECT
            self._setup_direct()
        else:
            self.mode = MINIMISE
            self.internal_dof = n_free - 1
        self._cache: OrderedDict = OrderedDict()

    # --- common ------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def gas_constant(self) -> float:
        return self.model.gas_constant

    @property
    def has_internal_freedom(self) -> bool:
        return self.mode == MINIMISE

    def _template(self, n: int) -> np.ndarray:
        Y = np.zeros((n, self.model.n_columns))
        Y[:, self._fixed_columns] = 1.0
        return Y

    def composition_of(self, Y: np.ndarray) -> np.ndarray:
        return (Y @ self._d) / (Y @ self._n)

    def _in_range(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.composition_range
        return (x >= lo - RANGE_TOLERANCE) & (x <= hi + RANGE_TOLERANCE)

    def site_fractions(self, T: float, x) -> np.ndarray:
        """Equilibrium site fractions Y (n, n_columns) at compositions x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self.mode == FIXED:
            return self._template(len(x))
        if self.mode == DIRECT:
            return self._direct_Y(np.clip(self._direct_t(x), 0.0, 1.0))
        return self._minimised(T, x, with_site_fractions=True)[2]

    def molar_gibbs(self, T, x):
        """G per mole of atoms at composition x (inf outside the phase's
        composition range)."""
        if self.mode == DIRECT and np.ndim(T) == 0 and np.ndim(x) == 0:
            return self._direct_point(float(T), float(x), False)
        x_arr = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x_arr).ravel()
        if self.mode == DIRECT:
            G = self._direct_gibbs(T, flat)
        elif self.mode == MINIMISE:
            G = self._minimised(T, flat)[0]
        else:
            G = np.where(self._in_range(flat), self.model.molar_gibbs(T, self._template(len(flat))), np.inf)
        if np.ndim(T) == 0 and np.ndim(x) == 0:
            return float(G[0])
        return G.reshape(np.shape(x_arr)) if np.ndim(x_arr) else G

    def molar_gibbs_derivative(self, T, x):
        """dG/dx at fixed T (analytic)."""
        if self.mode == DIRECT and np.ndim(T) == 0 and np.ndim(x) == 0:
            return self._direct_point(float(T), float(x), True)
        x_arr = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x_arr).ravel()
        if self.mode == DIRECT:
            dG = self._direct_derivative(T, flat)
        elif self.mode == MINIMISE:
            dG = self._minimised(T, flat)[1]
        else:
            raise ValueError(f"phase {self.name} has a fixed composition")
        if np.ndim(T) == 0 and np.ndim(x) == 0:
            return float(dG[0])
        return dG.reshape(np.shape(x_arr))

    def hull_points(self, T: float, x_grid: np.ndarray):
        """(x, G) samples of the Gibbs curve for the convex hull: the grid
        points inside the composition range plus its exact end points."""
        lo, hi = self.composition_range
        x = x_grid[(x_grid >= lo) & (x_grid <= hi)]
        x = np.unique(np.concatenate([x, [lo, hi]])) if (lo > 0.0 or hi < 1.0) else x
        return x, np.asarray(self.molar_gibbs(T, x), dtype=float)

    # --- direct mode: one mixing sublattice with two species ---------------------------

    def _setup_direct(self):
        (s,) = self.free_sublattices
        p, q = self.model.sublattice_columns(s)
        self._p, self._q = p, q
        fixed = self._template(1)[0]
        self._D0, self._N0 = fixed @ self._d, fixed @ self._n
        self._dp, self._dq = self._d[p], self._d[q]
        self._np, self._nq = self._n[p], self._n[q]
        self._point_template = [float(v) for v in fixed]

    def _direct_point(self, T: float, x: float, derivative: bool) -> float:
        """G (or dG/dx) at one scalar (T, x), in plain Python."""
        lo, hi = self.composition_range
        if not lo - RANGE_TOLERANCE <= x <= hi + RANGE_TOLERANCE:
            return np.nan if derivative else np.inf
        t = (self._D0 + self._dp - x * (self._N0 + self._np)) / (x * (self._nq - self._np) - (self._dq - self._dp))
        t = min(max(t, 0.0), 1.0)
        y = list(self._point_template)
        y[self._p], y[self._q] = 1.0 - t, t
        if not derivative:
            return self.model.point_molar_gibbs(T, y)
        _, grad = self.model.point_molar_gibbs(T, y, gradient=True)
        dG_dt = grad[self._q] - grad[self._p]
        D = self._D0 + self._dp + (self._dq - self._dp) * t
        N = self._N0 + self._np + (self._nq - self._np) * t
        dx_dt = ((self._dq - self._dp) * N - D * (self._nq - self._np)) / N**2
        return dG_dt / dx_dt

    def _direct_t(self, x: np.ndarray) -> np.ndarray:
        """Site fraction of the second species of the mixing sublattice at x."""
        numerator = self._D0 + self._dp - x * (self._N0 + self._np)
        denominator = x * (self._nq - self._np) - (self._dq - self._dp)
        return numerator / denominator

    def _direct_Y(self, t: np.ndarray) -> np.ndarray:
        Y = self._template(len(t))
        Y[:, self._p] = 1.0 - t
        Y[:, self._q] = t
        return Y

    def _direct_gibbs(self, T, x: np.ndarray) -> np.ndarray:
        inside = self._in_range(x)
        t = np.clip(self._direct_t(x), 0.0, 1.0)
        G = self.model.molar_gibbs(T, self._direct_Y(t))
        return np.where(inside, G, np.inf)

    def _direct_derivative(self, T, x: np.ndarray) -> np.ndarray:
        inside = self._in_range(x)
        t = np.clip(self._direct_t(x), 0.0, 1.0)
        Y = self._direct_Y(t)
        _, grad = self.model.molar_gibbs_gradient(T, Y)
        dG_dt = grad[:, self._q] - grad[:, self._p]
        D = self._D0 + self._dp + (self._dq - self._dp) * t
        N = self._N0 + self._np + (self._nq - self._np) * t
        dx_dt = ((self._dq - self._dp) * N - D * (self._nq - self._np)) / N**2
        return np.where(inside, dG_dt / dx_dt, np.nan)

    # --- minimise mode: internal degrees of freedom ------------------------------

    def _constraints(self, x: float):
        """(A, b): linear equality constraints on the free columns at x:
        each free sublattice sums to one, and the composition is x."""
        columns = self._free_columns
        rows = []
        for s in self.free_sublattices:
            members = set(self.model.sublattice_columns(s))
            rows.append([1.0 if c in members else 0.0 for c in columns])
        fixed = self._template(1)[0]
        rows.append([self._d[c] - x * self._n[c] for c in columns])
        b = [1.0] * len(self.free_sublattices) + [x * (fixed @ self._n) - fixed @ self._d]
        return np.array(rows), np.array(b)

    def _full(self, z: np.ndarray) -> np.ndarray:
        Y = self._template(len(z))
        Y[:, self._free_columns] = z
        return Y

    def _constraint_batch(self, x: np.ndarray):
        """Stacked constraints (A (n, k+1, m), b (n, k+1)) for compositions x."""
        A1, b1 = self._constraints(0.0)
        A = np.repeat(A1[None, :, :], len(x), axis=0)
        b = np.repeat(b1[None, :], len(x), axis=0)
        n_free = np.array([self._n[c] for c in self._free_columns])
        fixed = self._template(1)[0]
        A[:, -1, :] = np.array([self._d[c] for c in self._free_columns])[None, :] - x[:, None] * n_free[None, :]
        b[:, -1] = x * (fixed @ self._n) - fixed @ self._d
        return A, b

    def _derivatives(self, T: float, x: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """dG/dx at constrained minima Z: lambda * N, with lambda the
        multiplier of the composition constraint (bound site fractions excluded)."""
        Y = self._full(Z)
        _, grad = self.model.molar_gibbs_gradient(T, Y)
        g = grad[:, self._free_columns]
        A, _ = self._constraint_batch(x)
        At = A.transpose(0, 2, 1).copy()
        bound = Z <= 10.0 * MIN_SITE_FRACTION
        At[bound] = 0.0
        g = np.where(bound, 0.0, g)
        multipliers = np.linalg.pinv(At) @ g[..., None]
        return multipliers[:, -1, 0] * self.model.atoms(Y)

    def _minimum(self, T: float, x: float):
        """(G, dG/dx, Y) at the constrained minimum for one composition."""
        G, dG, Y = self._minimised(T, np.array([float(x)]), with_site_fractions=True)
        return G[0], dG[0], Y[0]

    def _minimised(self, T, x: np.ndarray, with_site_fractions: bool = False):
        """(G, dG/dx[, Y]) at the constrained minima for compositions x,
        cached per (T, x)."""
        T = float(T)
        x = np.asarray(x, dtype=float)
        results = [self._cache.get((T, float(xi))) for xi in x]
        missing = [k for k, r in enumerate(results) if r is None]
        if missing:
            x_new = x[missing]
            G = np.full(len(x_new), np.inf)
            dG = np.full(len(x_new), np.nan)
            Y = np.full((len(x_new), self.model.n_columns), np.nan)
            inside = self._in_range(x_new)
            if inside.any():
                lo, hi = self.composition_range
                x_inner = np.clip(x_new[inside], lo + EDGE_MARGIN, hi - EDGE_MARGIN)
                if self.internal_dof == 1 and len(x_inner) == 1:
                    Z = self._minimise_point(T, float(x_inner[0]))[None, :]
                elif self.internal_dof == 1:
                    Z = self._minimise_one(T, x_inner)
                else:
                    Z = np.array([self._minimise_many(T, xi) for xi in x_inner])
                Y_in = self._full(Z)
                G[inside] = self.model.molar_gibbs(T, Y_in)
                dG[inside] = self._derivatives(T, x_inner, Z)
                Y[inside] = Y_in
            for k, index in enumerate(missing):
                result = (float(G[k]), float(dG[k]), Y[k])
                results[index] = result
                self._cache[(T, float(x[index]))] = result
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        G = np.array([r[0] for r in results])
        dG = np.array([r[1] for r in results])
        if with_site_fractions:
            return G, dG, np.array([r[2] for r in results])
        return G, dG

    def _minimise_one(self, T: float, x: np.ndarray) -> np.ndarray:
        """One internal degree of freedom, vectorised over compositions x:
        sample the feasible segment of each constraint line, then refine the
        best sample by regula falsi (Illinois) on the directional derivative.
        Returns the free site fractions Z (n, m)."""
        A, b = self._constraint_batch(x)
        z0 = (np.linalg.pinv(A) @ b[..., None])[..., 0]
        direction = np.linalg.svd(A)[2][:, -1, :]

        nonzero = np.abs(direction) > 1e-15
        safe = np.where(nonzero, direction, 1.0)
        bound_a = (MIN_SITE_FRACTION - z0) / safe
        bound_b = (1.0 - z0) / safe
        w_lo = np.max(np.where(nonzero, np.minimum(bound_a, bound_b), -np.inf), axis=1)
        w_hi = np.min(np.where(nonzero, np.maximum(bound_a, bound_b), np.inf), axis=1)
        if np.any(w_lo > w_hi):
            raise ValueError(f"phase {self.name}: no feasible site fractions at some of x = {x}")

        edge = np.geomspace(1e-10, 0.02, INTERNAL_EDGE_SAMPLES)
        u = np.unique(np.concatenate([np.linspace(0.0, 1.0, INTERNAL_SAMPLES), edge, 1.0 - edge]))
        W = w_lo[:, None] + u[None, :] * (w_hi - w_lo)[:, None]
        n, S = W.shape
        Z = z0[:, None, :] + W[..., None] * direction[:, None, :]
        G = self.model.molar_gibbs(T, self._full(Z.reshape(n * S, -1))).reshape(n, S)
        k = np.argmin(G, axis=1)
        rows = np.arange(n)

        def slope(w: np.ndarray) -> np.ndarray:
            Y = self._full(z0 + w[:, None] * direction)
            _, grad = self.model.molar_gibbs_gradient(T, Y)
            return np.einsum("ij,ij->i", grad[:, self._free_columns], direction)

        a = W[rows, np.maximum(k - 1, 0)]
        c = W[rows, np.minimum(k + 1, S - 1)]
        s_a, s_c = slope(a), slope(c)
        w_best = W[rows, k].copy()
        w_best = np.where((k == 0) & (s_a >= 0.0), a, w_best)
        w_best = np.where((k == S - 1) & (s_c <= 0.0), c, w_best)
        active = (s_a < 0.0) & (s_c > 0.0)
        best_slope = np.full(n, np.inf)
        for _ in range(REFINEMENT_ITERATIONS):
            if not active.any():
                break
            m = c - s_c * (c - a) / np.where(active, s_c - s_a, 1.0)
            m = np.where((m > a) & (m < c), m, (a + c) / 2.0)
            s_m = slope(m)
            better = active & (np.abs(s_m) < best_slope)
            w_best = np.where(better, m, w_best)
            best_slope = np.where(better, np.abs(s_m), best_slope)
            left = active & (s_m < 0.0)
            right = active & (s_m >= 0.0)
            a, s_a = np.where(left, m, a), np.where(left, s_m, s_a)
            s_c = np.where(left, s_c / 2.0, s_c)
            c, s_c = np.where(right, m, c), np.where(right, s_m, s_c)
            s_a = np.where(right, s_a / 2.0, s_a)
            active &= (c - a > 1e-14 * np.maximum(1.0, np.abs(m))) & (np.abs(s_m) > SLOPE_TOLERANCE)
        return z0 + w_best[:, None] * direction

    def _minimise_point(self, T: float, x: float) -> np.ndarray:
        """_minimise_one for a single composition, refining in plain Python."""
        A, b = self._constraint_batch(np.array([x]))
        z0 = (np.linalg.pinv(A[0]) @ b[0])
        direction = np.linalg.svd(A[0])[2][-1]
        nonzero = np.abs(direction) > 1e-15
        safe = np.where(nonzero, direction, 1.0)
        bound_a, bound_b = (MIN_SITE_FRACTION - z0) / safe, (1.0 - z0) / safe
        w_lo = float(np.max(np.where(nonzero, np.minimum(bound_a, bound_b), -np.inf)))
        w_hi = float(np.min(np.where(nonzero, np.maximum(bound_a, bound_b), np.inf)))
        if w_lo > w_hi:
            raise ValueError(f"phase {self.name}: no feasible site fractions at x = {x}")

        edge = np.geomspace(1e-10, 0.02, INTERNAL_EDGE_SAMPLES)
        u = np.unique(np.concatenate([np.linspace(0.0, 1.0, INTERNAL_SAMPLES), edge, 1.0 - edge]))
        W = w_lo + u * (w_hi - w_lo)
        G = self.model.molar_gibbs(T, self._full(z0[None, :] + W[:, None] * direction[None, :]))
        k = int(np.argmin(G))

        template = [float(v) for v in self._template(1)[0]]
        columns = self._free_columns
        z0_list, d_list = [float(v) for v in z0], [float(v) for v in direction]

        def slope(w: float) -> float:
            y = list(template)
            for c, z, d in zip(columns, z0_list, d_list):
                y[c] = z + w * d
            _, grad = self.model.point_molar_gibbs(T, y, gradient=True)
            return sum(grad[c] * d for c, d in zip(columns, d_list))

        a, c = float(W[max(k - 1, 0)]), float(W[min(k + 1, len(W) - 1)])
        s_a, s_c = slope(a), slope(c)
        w_best = float(W[k])
        if k == 0 and s_a >= 0.0:
            w_best = a
        elif k == len(W) - 1 and s_c <= 0.0:
            w_best = c
        elif s_a < 0.0 < s_c:
            best = math.inf
            for _ in range(REFINEMENT_ITERATIONS):
                m = c - s_c * (c - a) / (s_c - s_a)
                if not a < m < c:
                    m = (a + c) / 2.0
                s_m = slope(m)
                if abs(s_m) < best:
                    w_best, best = m, abs(s_m)
                if abs(s_m) <= SLOPE_TOLERANCE or c - a <= 1e-14 * max(1.0, abs(m)):
                    break
                if s_m < 0.0:
                    a, s_a, s_c = m, s_m, s_c / 2.0
                else:
                    c, s_c, s_a = m, s_m, s_a / 2.0
        return z0 + w_best * direction

    def _minimise_many(self, T: float, x: float) -> np.ndarray:
        """Several internal degrees of freedom: SLSQP from several starts."""
        A, b = self._constraints(x)
        n = len(self._free_columns)

        def objective(z):
            Y = self._full(z[None, :])
            G, grad = self.model.molar_gibbs_gradient(T, Y)
            return float(G[0]), grad[0, self._free_columns]

        rng = np.random.default_rng(0)
        starts = []
        z0, *_ = np.linalg.lstsq(A, b, rcond=None)
        for _ in range(64):
            if len(starts) >= 6:
                break
            z = rng.dirichlet(np.ones(n))
            z = z - A.T @ np.linalg.lstsq(A @ A.T, A @ z - b, rcond=None)[0]
            if np.all(z > MIN_SITE_FRACTION) and np.all(z < 1.0):
                starts.append(z)
        if not starts:
            starts.append(np.clip(z0, MIN_SITE_FRACTION, 1.0))

        best, converged = None, False
        for start in starts:
            result = minimize(
                objective, start, jac=True, method="SLSQP",
                bounds=[(MIN_SITE_FRACTION, 1.0)] * n,
                constraints=[{"type": "eq", "fun": lambda z: A @ z - b, "jac": lambda z: A}],
                options={"ftol": 1e-14, "maxiter": 500},
            )
            feasible = np.max(np.abs(A @ result.x - b)) < 1e-9
            if feasible and (best is None or result.fun < best.fun):
                best = result
            converged |= bool(result.success and feasible)
        if best is None or not converged:
            warnings.warn(
                f"phase {self.name}: site-fraction minimisation did not converge at T={T}, x={x}",
                ConvergenceWarning,
                stacklevel=3,
            )
            if best is None:
                return np.clip(z0, MIN_SITE_FRACTION, 1.0)
        return best.x
