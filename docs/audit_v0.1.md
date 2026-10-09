# Ag-Cu physical-correctness audit (v0.1.0 → v0.1.1)

Scope: the Ag-Cu system as computed by the test suite and as shown in the
Streamlit app. This audit adds regression tests only; it changes no
production code.

## Summary

| Quantity | Reference | `tests/test_systems.py` (JSON model) | Streamlit app |
|---|---|---|---|
| Ag melting point (x=0) | 1234.93 K | 1234.93 K | 900 K |
| Cu melting point (x=1) | 1357.77 K | 1357.77 K | 900 K |
| Eutectic temperature | 1052 K (779 °C) | 1051.5 K (778.4 °C) | 650 K (377 °C) |
| Eutectic liquid x(Cu) | 0.399 | 0.388 | 0.452 |
| Phase at x=0, 600 K | FCC_AG | FCC_AG | **FCC_CU** |
| Phase at x=1, 600 K | FCC_CU | FCC_CU | **FCC_AG** |

The JSON model is close to the assessed diagram. The app is wrong because it
**never reads the Gibbs energy coefficients in `ag_cu.json`**. It shows a
placeholder diagram that only shares the phase names.

Reference values: Subramanian & Perepezko, J. Phase Equilib. 14 (1993) 62-75;
pure-element data from Dinsdale, CALPHAD 15 (1991) 317-425. The JSON's
`_source` field cites both.

## 1. Why the tests show ~779 °C and the app shows ~650 K (377 °C)

The two paths build **different thermodynamic models** from the same file.

- `tests/test_systems.py::_build_system` reads `end_members` and
  `interaction_parameters` from the JSON. In LIQUID, Ag has
  `G = 11090 − 8.9803·T` (zero at 1234.93 K) and Cu has `G = 13050 − 9.6113·T`
  (zero at 1357.77 K). FCC_AG has a +6400 J/mol Cu end member and FCC_CU a
  +4200 J/mol Ag end member.
- `app.py::_build_computable_system` ignores those fields. Its docstring says
  the "JSON schema records phase names and model types but not raw Gibbs
  energy coefficients". That stopped being true when the coefficients were
  added in 496704b. Instead it uses hard-coded constants:
  - Every phase whose name contains "liquid" gets
    `G = FUSION_ENTHALPY − (FUSION_ENTHALPY/MELTING_T)·T` with
    `FUSION_ENTHALPY = 9000` and `MELTING_T = 900.0` **for both end
    members**. So pure Ag and pure Cu both "melt" at 900 K. That is why
    LIQUID appears at x=0 and at x=1 at 900 K.
  - Solid solutions get endpoint offsets of 6000 and 9000 J/mol, chosen by
    list position.
  - With these placeholder curves, the three-phase reaction falls at about
    650 K.

**K/°C conversion is not the cause.** Every conversion uses 273.15:
650.0 K − 273.15 = 376.85 → "377 C", and 779 °C + 273.15 = 1052.15 K. The
app's units are inconsistent but correct:
- The temperature-range slider and the invariant table are in K.
- The diagram's y axis is labelled just "Temperature" (K, but no unit shown).
- The invariant markers are labelled in °C (`visualization.py`, "Eutectic 377 C").
- The Gibbs-curve title says `T=650.0` with no unit.

Mixing K and °C on the same figure makes the discrepancy look like a
conversion error when it is a modelling error.

**Which JSON the app loads.** The app loads the right file:
`SYSTEMS_DIR = parents[2]/data/systems` and `ag_cu.json` is the first option
alphabetically. The problem is what it does with the file, not which file it
opens. The values in the table above were reproduced by running both builders
on the same `ag_cu.json`, and by driving the app headlessly with
`streamlit.testing.v1.AppTest` (default sliders 400–900 K, 80×150 grid). That
run gives one eutectic, at 650.0 K, `FCC_CU+LIQUID+FCC_AG`, with a liquid
marker at x=0.452.

**Why the existing suite did not catch it.**
- `tests/test_systems.py` never imports or exercises `app.py`.
- Its eutectic tests use the window `T_range=(1000, 1150)`, which excludes both
  melting points, so melting is never checked.
- Its composition tolerance (±0.03) is wide enough that the JSON model's 0.388
  passes comfortably against 0.399.

## 2. Element → A/B mapping, JSON to plot

| Stage | Mapping |
|---|---|
| `ag_cu.json` `elements` | `[Ag, Cu]` |
| `SolutionPhase` | `x` is x_B; `gibbs_a` is at x=0, `gibbs_b` at x=1 |
| `_build_system` (tests) | `symbol_a, symbol_b = elements` → A=Ag, B=Cu ✔ |
| `_build_computable_system` (app) | **No element lookup.** A/B offsets alternate by `solution_index` |
| `compute_diagram` / `plot_diagram` | x axis = x_B, labelled "mole fraction B" without naming B |

**Why FCC_CU ends up at x=0.** `solution_index` is incremented for *every*
solution phase, LIQUID included. With the JSON order
`[LIQUID, FCC_AG, FCC_CU]`:

| Phase | `solution_index` | Branch | Endpoints (A, B) | Stable side |
|---|---|---|---|---|
| LIQUID | 0 | liquid branch | (9000−10T, 9000−10T) | — |
| FCC_AG | 1 (odd) | `gibbs_a=offset` | (6000, 0) | **x=1 (Cu side)** |
| FCC_CU | 2 (even) | `gibbs_b=offset` | (0, 9000) | **x=0 (Ag side)** |

So the terminal phases are swapped. The swap comes from where LIQUID sits in
the list, not from element ordering. Reordering phases in the JSON would
silently flip the diagram, and the phase names never affect which side a
solid lands on. The x axis also never says that B is Cu.

The same position-based approach affects stoichiometric phases. The app
always uses `m=1, n=1`, so in Al-Cu, AL2CU is drawn at x=0.5 instead of
x=1/3. Al-Cu is outside this audit's scope; this is noted for follow-up.

## 3. Does invariant detection depend on the temperature slider?

**Yes, on the range and grid sliders, not on the equilibrium-temperature slider.**

- `_cached_diagram` computes the diagram only over the "Temperature range (K)"
  slider. `detect_invariants` only compares adjacent rows of that grid, so an
  invariant outside the range is never found. With the JSON model and the
  default 400–900 K range, **no eutectic is reported at all**, because 1052 K
  is outside the window.
- The reported temperature is the midpoint of the two rows that straddle the
  reaction. Its error is up to ½·(T_max − T_min)/(n_T − 1). With the default
  80 points, that is ±3.2 K over 400–900 K and ±11.4 K over 200–2000 K. The
  same JSON model reports 1051.5 K (1000–1150, 151 pts), 1047.5 K (400–1500,
  80 pts) and 1054.4 K (200–2000, 80 pts). The phase compositions shift too,
  because they are refined at the row temperature, not the invariant
  temperature.
- A reaction that falls exactly on the first or last row has no neighbour on
  one side and is missed.
- The "Temperature (K) for equilibrium and Gibbs curves" and composition
  sliders only drive the single-point equilibrium readout and the
  Gibbs-curve plot. They do not affect detection.

## 4. Regression tests

`tests/test_physics_regression.py` drives the real app via
`streamlit.testing.v1.AppTest`. It reads the equilibrium readout and the
invariant markers on the diagram, so it tests what a user actually sees. The
four physics tests are `xfail(strict=True, raises=AssertionError)`. They fail
today for physical reasons, and setup or harness errors are not counted as
expected failures:

| Test | Assertion | Current failure |
|---|---|---|
| `test_ag_melts_at_reference_temperature` | x=0: no LIQUID at 1232.93 K, only LIQUID at 1236.93 K | LIQUID already stable below Tm |
| `test_cu_melts_at_reference_temperature` | x=1: same, around 1357.77 ± 2 K | LIQUID already stable below Tm |
| `test_eutectic_temperature_and_composition` | one eutectic in 1000–1100 K (300 pts) at 1052 ± 5 K, liquid x(Cu) = 0.399 ± 0.015 | no eutectic in range |
| `test_ag_rich_terminal_phase_is_on_the_ag_side` | 800 K: x=0 is FCC_AG, x=1 is FCC_CU | x=0 is FCC_CU |

There is also a non-xfail precondition, `test_ag_cu_json_lists_ag_then_cu`. It
pins `x = x(Cu)` for these assertions.

**Positive control.** A scratch copy of the app whose builder was replaced by
the JSON-reading `_build_system` passed all five tests. So the tests will
XPASS, which strict mode turns into a failure, as soon as the app uses the
JSON coefficients. At that point the markers must be removed.

Note: the JSON model's eutectic liquid composition (0.388) passes the
±0.015 band by only about 0.004. A refit of the FCC cross-endpoint energies,
or a real Redlich-Kister liquid interaction, would give more margin.

## Recommended fixes (not applied)

1. Make the app build its phases from `end_members`, `interaction_parameters`,
   `stoichiometry` and `formation`, keyed by element symbol, the same way as
   `tests/test_systems.py::_build_system`. Move that builder into the package
   so the app and the tests share one implementation. Then remove the xfail
   markers.
2. Label the composition axis and slider with the actual element, for
   example "x(Cu)".
3. Use one temperature unit across the axis, markers, tables and titles, and
   label it.
4. Find invariants on a dedicated grid (or by root-finding) that does not
   depend on the display range. Or at least warn when the expected
   invariants fall outside the selected range.
