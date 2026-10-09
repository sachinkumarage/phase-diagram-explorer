# Thermodynamic models

Every phase is a Compound Energy Formalism (sublattice) phase. A binary
substitutional solution, a stoichiometric compound and a pure-element phase
are special cases of it. Energies are in J/mol, temperatures in K, and R is
the system's gas constant (see [data_format.md](data_format.md)).

Implementation:
- `thermo/cef.py`: the Gibbs energy as a function of site fractions;
- `thermo/sublattice.py`: the binary phase as a function of composition;
- `thermo/magnetic.py`: the Inden-Hillert-Jarl magnetic term.

## Compound Energy Formalism

A phase has sublattices s with site ratios a_s. Sublattice s holds species
i, which can be the two elements or the vacancy VA, with site fractions
y_si, where Σ_i y_si = 1. The molar Gibbs energy per mole of formula units
is

```
G = G_ref + G_id + G_ex + G_mag
```

**Surface of reference.** End members e choose one species e_s on every
sublattice:

```
G_ref = Σ_e  G_e(T)  Π_s y_{s, e_s}
```

**Ideal configurational entropy**, weighted by the site ratios:

```
G_id = R T Σ_s a_s Σ_i y_si ln y_si
```

**Excess.** Interaction parameters have two species i, j on one sublattice
and one species on every other sublattice. Each is a Redlich-Kister series
of orders v = 0..n:

```
G_ex = Σ_L  Π_s Π_{k ∈ L_s} y_sk  ·  Σ_v  L_v(T) (y_si − y_sj)^v
```

The order of i and j is as written in the parameter: `L(FCC_A1,CU,AG:VA;1)`
multiplies (y_Cu − y_Ag). Parameters with interactions on several
sublattices (reciprocal) or between three species are supported at order
0. A wildcard `*` on a sublattice drops that sublattice from the product,
since its site fractions sum to one.

### Composition per mole of atoms

Vacancies carry no atoms. The number of atoms per formula unit and the mole
fraction of element k are

```
N = Σ_s a_s (1 − y_s,VA)
x_k = Σ_s a_s y_sk / N
```

The composition axis is x of the dependent element. Every Gibbs energy the
engine uses is per mole of atoms, G / N.

For example, in BCC_A2 (FE)1(C,VA)3 with y_C = 0.1, N = 1.3 and
x_C = 0.3 / 1.3 = 0.231. Its composition range is [0, 0.75], reached when
y_C = 1.

### Special cases

| Phase | Sublattices | G per mole of atoms |
|---|---|---|
| Substitutional solution | (A,B)1 | x_A G_A + x_B G_B + RT(x_A ln x_A + x_B ln x_B) + x_A x_B Σ_v L_v (x_A − x_B)^v |
| Same, with a vacancy sublattice | (A,B)1(VA)c | identical: the VA sublattice has no atoms and no entropy |
| Stoichiometric compound | (A)m(B)n | G_AmBn / (m + n), at x_B = n / (m + n) |
| Pure element | (C)1 | G_C, at x_C = 1 |

The JSON `"solution"` and `"stoichiometric"` models are built as these CEF
phases. A test checks that every system that existed in 0.2.0 still gives
the same equilibria to 1e-10.

### Equilibrium at fixed composition

A phase's G(x) is the minimum of G/N over all site fractions consistent
with x. That depends on how many site fractions are free.

- **Fixed:** every sublattice holds one species, so x is fixed.
- **Direct:** one sublattice mixes two species, p and q. Composition fixes
  y_q through

  ```
  x = (D0 + d_p (1 − y_q) + d_q y_q) / (N0 + n_p (1 − y_q) + n_q y_q)
  ```

  where D0 and N0 come from the single-species sublattices, and d and n are
  the dependent-element and atom contributions of p and q. This
  linear-fractional relation is inverted in closed form.
- **Minimise:** there are internal degrees of freedom. The constraints are
  linear in the free site fractions z:

  ```
  Σ_{i ∈ s} z_si = 1                               for each mixing sublattice s
  Σ_si a_s (δ_{i,dep} − x δ_{i≠VA}) z_si = const(x)  (composition)
  ```

  All site fractions are kept in (1e-12, 1).
  - **One internal degree of freedom:** z = z0(x) + w·d(x), where d spans
    the null space of the constraints. The feasible interval of w is
    sampled at about 60 points, including points near both ends where
    dilute ordered states lie. The best sample is then refined by regula
    falsi (Illinois) on dG/dw until the derivative is below 1e-9. This is
    vectorised over all compositions at once.
  - **Several internal degrees of freedom:** SLSQP with analytic gradients,
    run from several feasible starting points. If no start converges, a
    `ConvergenceWarning` is issued.

  Results are cached per (T, x).

### dG/dx

At a constrained minimum, the derivative is the multiplier λ of the
composition constraint:

```
∇_z (G/N) = λ ∇_z h + Σ_s μ_s e_s     ⇒     dG/dx = λ N
```

where h is the composition constraint. This is solved by least squares,
leaving out site fractions held at their lower bound. In direct mode the
same derivative is (dG/dy_q) / (dx/dy_q). Common tangents and invariant
reactions use these analytic derivatives.

## Inden-Hillert-Jarl magnetic model

Per mole of formula units:

```
G_mag = R T ln(β + 1) f(τ),      τ = T / T_C
```

where T_C (TC) and the mean magnetic moment β (BMAGN) are expanded over site
fractions in the same way as G_ref + G_ex, so they can depend on composition
through Redlich-Kister terms. With

```
A = 518/1125 + (11692/15975)(1/p − 1)
```

the function f is

```
τ < 1:   f(τ) = 1 − [ 79/(140 p τ) + (474/497)(1/p − 1)(τ³/6 + τ⁹/135 + τ¹⁵/600) ] / A
τ ≥ 1:   f(τ) = −[ τ⁻⁵/10 + τ⁻¹⁵/315 + τ⁻²⁵/1500 ] / A
```

The structure factor p is 0.28 for FCC and HCP, and 0.40 for BCC. f and
df/dτ are continuous at τ = 1, so G_mag and the magnetic entropy are
continuous at T_C; the heat capacity is not.

| τ | f, p = 0.28 | f, p = 0.40 |
|---|---|---|
| 0.5 | −0.742504 | −0.829738 |
| 1 | −0.044330 | −0.066638 |
| 2 | −0.001334 | −0.002005 |

**Antiferromagnetism (TDB convention).** The phase's TYPE_DEFINITION, for
example `GES A_P_D BCC_A2 MAGNETIC -1.0 0.4`, gives an antiferromagnetic
factor: −3 for FCC and HCP, −1 for BCC. A negative composition-weighted T_C
(a Néel temperature) or β is divided by this factor before use. When T_C = 0
or β = 0, the contribution is zero.

The gradient of G_mag with respect to site fractions,
(∂G/∂T_C) ∇T_C + (∂G/∂β) ∇β, enters dG/dx like every other term.

## References

- M. Hillert, L.-I. Staffansson, Acta Chem. Scand. 24 (1970) 3618
  (sublattice model).
- G. Inden, Proc. CALPHAD V (1976); M. Hillert, M. Jarl, CALPHAD 2 (1978)
  227 (magnetic model).
- H.L. Lukas, S.G. Fries, B. Sundman, *Computational Thermodynamics: The
  Calphad Method*, Cambridge University Press (2007).
