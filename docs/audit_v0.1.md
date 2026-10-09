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

## Resolved in 0.1.2

Every Day 1 regression test in `tests/test_physics_regression.py` now passes
in the real app. Their xfail markers are removed, and the tolerances are
unchanged.

| Quantity | Reference | 0.1.1 app | 0.1.2 app |
|---|---|---|---|
| Ag melting point (x=0) | 1234.93 K | 900 K | 1234.93 K |
| Cu melting point (x=1) | 1357.77 K | 900 K | 1357.77 K |
| Eutectic temperature | 1052 K (779 °C) | 650 K | **1051.0 K (777.85 °C)** |
| Eutectic liquid x(Cu) | 0.399 ± 0.015 | 0.452 | **0.3878** (38.8 at%, 27.2 wt%) |
| Eutectic solid compositions x(Cu) | — | — | FCC_AG 0.2605, FCC_CU 0.5431 |
| Phase at x=0 / x=1, 800 K | FCC_AG / FCC_CU | FCC_CU / FCC_AG | FCC_AG / FCC_CU |

The eutectic temperature is resolved to ±1 K by the 2 K invariant grid. The
liquid composition still sits close to the lower edge of the tolerance band
(0.3878 against a minimum of 0.384). That gap is a limit of the model
parameters, not of the code.

### 1. One model builder (findings 1 and 2)

- `phase_diagram_explorer/builder.py::build_system(definition)` is the only
  place where Gibbs energy objects are built from a system definition. The
  app and `tests/test_systems.py` both use it.
- `_build_computable_system` is deleted, along with `MELTING_T`,
  `FUSION_ENTHALPY`, `BASE_OFFSET`, `OFFSET_STEP` and the placeholder
  stoichiometry `m=n=1`.
- A system without coefficient data, such as `example.json`, is now rejected
  with a `ValueError`. The app shows it as an error instead of drawing a
  made-up diagram.
- `tests/test_app.py` guards against regressions in two ways:
  - It checks the source of `app.py` and fails if the app constructs
    `PureElementGibbs`, `SolutionPhase` or `StoichiometricPhase`, imports the
    pure or solution thermo modules, or reintroduces the placeholder
    constants.
  - It runs the app with `build_system` wrapped so that LIQUID is dropped,
    then checks that the equilibrium readout reflects that change. This
    proves the app's system object comes from `build_system`.

### 2. Element-to-axis mapping (finding 2)

- `SystemDefinition.base_element` is the first listed element and sits at
  x=0. `SystemDefinition.dependent_element` is the second and sits at x=1.
  The composition axis is the mole fraction of the dependent element.
- `build_system` looks up end members and site counts by element symbol, so
  the order of the phase list has no effect.
- `tests/test_builder.py` reverses the phase list in both system JSON files
  and checks that the phase labels and phase fractions on the diagram are
  identical, and that the detected invariants are identical too.
- Stoichiometric compounds use their real site counts. AL2CU sits at
  x(Cu) = 1/3, and the tie lines on both sides of it end there exactly.
  The app draws it there too.

### 3. Invariant detection independent of the display (finding 3)

- System JSON has a new optional field, `t_range_k`. It is set to
  [500, 1450] K for Ag-Cu and Al-Cu.
- `invariants.detect_invariants_over_range` scans the system's full range on
  its own grid, with steps of at most 2 K and 201 compositions. It does not
  use the display grid.
- The app's range and grid sliders now only change what is drawn. The
  invariant table always lists every reaction in the full range. A reaction
  outside the displayed range is labelled "outside displayed temperature
  range" and gets no marker on the diagram.
- The detection grid has not been refined further. Root-finding is still
  deferred.

### 4. Units and labels (finding 1)

- `units.py` converts between K and °C, and between mole fraction and weight
  fraction. Weight fraction uses the `atomic_mass` now stored for each
  element in the system JSON.
- The app has toggles for K/°C and at%/wt%. Internal values stay in K and
  mole fraction; only the display is converted. The wt% option appears only
  when the system has atomic masses for both elements.
- Every axis, slider, marker and table uses the selected units and labels
  them. Examples: "Temperature (K)", "x(Cu) (at%)", "w(Cu) (wt%)" and
  "Molar Gibbs energy (J/mol)".
- The common tangent is drawn as a sampled curve, so it stays correct on the
  non-linear wt% axis.
- Tests cover conversion round trips and the Fe-C check: 0.76 wt% C works
  out to 3.44 at% C, compared with about 3.46 expected.

### 5. Default view

The Ag-Cu default display range is 500–1450 K. That covers the eutectic and
both melting points.

## Resolved in 0.1.3

The "Still open" items from 0.1.2 are resolved, or moved to the updated
list below.

### Stoichiometric compounds are exact hull points

- Each compound now enters the convex hull at its exact composition, taken
  from its site ratios. It is no longer snapped to the nearest point on the
  Gibbs-curve grid.
- If the overall composition lands exactly on a hull vertex, the result is
  that vertex's phase on its own.
- At x(Cu) = 1/3 in Al-Cu, the result is now single-phase AL2CU with
  fraction 1.0 for every grid size tested.

### Spurious Al-Cu invariants

Both 0.1.2 Al-Cu reactions were artifacts of the detector:
- the "eutectic" with liquid at 46 at% Cu,
- the FCC_AL + AL2CU + LIQUID "peritectic".

The cause was that two-phase fields were merged by phase set. The
LIQUID+AL2CU fields on both sides of the compound became one field.

The detector now works with contiguous two-phase segments in composition
order. It also requires the P1+P3 field to span the composition where P2
was stable.

With a composition grid fine enough to resolve it (n_x = 1001), the engine
finds the one real invariant in the provisional Al-Cu model:
L (x(Cu) ≈ 0.004) → FCC_AL + AL2CU at 931 K, just below the melting point
of Al.

`tests/test_systems.py` used to pass its Al-Cu invariant test only because
of the artifact. It now uses that finer grid. The thermodynamic parameters
are unchanged. The Ag-Cu result is unchanged too: 1051.0 K, with liquid at
x(Cu) = 0.3878.

### Miscibility gaps (engine capability)

- The hull now detects tie lines that join two tangent points on the same
  phase's Gibbs curve, where the bridged curve lies above the tie line.
  Those phases are reported as composition sets PHASE#1, PHASE#2, ...,
  numbered from left to right.
- Diagrams, invariant detection, tables and plots all handle these labels:
  - A single-phase region is drawn as its base phase, so ALPHA and ALPHA#1
    form one continuous field.
  - Invariants match phases by base name, so LIQUID → ALPHA#1 + ALPHA#2 is
    detected.
- These are validated against a synthetic symmetric regular solution,
  `tests/fixtures/regular_solution.json`, with L0 = 20000 J/mol:
  - the binodal at 800, 1000 and 1150 K matches the analytical root of
    ln(x/(1-x)) = L0(2x-1)/(RT) to within one grid step;
  - the gap is symmetric about x = 0.5;
  - the bisected critical temperature is within 0.02 K of
    L0/(2R) = 1202.72 K.
- A second fixture, `gap_eutectic.json`, checks a eutectic that decomposes
  into two composition sets.
- `ag_cu.json` still uses separate FCC_AG and FCC_CU phases. Converting it to
  one FCC_A1 phase waits for assessed parameters.

### Data status and system list

- `ag_cu.json` and `al_cu.json` are marked `"_status": "provisional"`, each
  with a `_status_reason`. The app shows a "Preview: provisional
  thermodynamic data, not yet validated." banner for them.
- `example.json`, which has no Gibbs energy data, moved to `tests/fixtures/`.
  The app's system selector now lists only systems that `build_system` can
  compute (`builder.is_computable`).

### Regression tests for the solvus

Two new strict-xfail tests in `tests/test_physics_regression.py` record the
assessed solid solubilities at the eutectic. They fail today against the
values below:

| Test | Assessed x(Cu) | Current x(Cu) |
|---|---|---|
| Ag-rich fcc phase | 0.141 ± 0.015 | 0.261 |
| Cu-rich fcc phase | 0.950 ± 0.010 | 0.543 |

### Tests and units

- There is a new `slow` pytest marker for the full-range invariant tests and
  the headless app tests:
  - a plain `pytest` run skips them and takes about 10 s;
  - `pytest -m "slow or not slow"` runs everything, which is what CI does.
- The Fe-C unit check is tightened to 3.44 ± 0.01 at% C for 0.76 wt% C.

## Resolved in 0.1.4

### Traced boundaries and vector rendering (grid-limited precision)

- `equilibrium.phase_assemblage(system, T)` returns every field across
  composition at one temperature. `compute_equilibrium` is built on it.
  - The sampled convex hull is used only to find which fields exist.
  - Each tie line is then refined to the exact common tangent
    (`equilibrium/tangent.py`, using an analytic dG/dx for solution phases).
  - Each tie line is checked against every other phase, and split if a phase
    lies below it. Ties whose tangents cross are merged.
  - Tie-line compositions therefore no longer depend on any composition grid
    or on the Gibbs-curve sampling.
- `tracing.trace_diagram` traces fields from these assemblages. It never
  uses grid cell labels, and it chooses its temperature levels adaptively:
  - every change of field structure is bracketed to 1e-3 K by bisection;
  - each invariant gets levels at ±1e-4 K;
  - an interval is subdivided, down to 0.05 K, where a boundary moves more
    than 0.01 in x or bends more than 0.001.
- The ~0.005-wide Al-rich LIQUID + FCC_AL field is traced, even though it is
  narrower than one step of the app's grid.
- The app no longer draws a heatmap.
  - Phase regions are filled polygons bounded by the traced lines.
  - Invariants are horizontal lines across their three compositions.
  - Fields are labelled at their centroids with plain phase names, for
    example "LIQUID + FCC_AG". (From 0.2.0, a miscibility gap is labelled
    "ALPHA#1 + ALPHA#2".)
  - Hovering a field shows its phases; hovering a boundary shows T and
    composition.
- The composition-points slider is gone, because drawing no longer uses a
  composition grid. "Temperature points" now sets the number of starting
  levels for the tracer.

### Invariants by root-finding

- Candidate reactions come from changes in the phase assemblage between
  temperature levels.
- Each candidate is refined with `brentq` on the three-phase condition: the
  reacting phase just touches the common tangent of the other two. The
  temperature tolerance is 1e-6 K.
- All three phase compositions are reported.
- A reaction with the same phases within 0.5 K of another is merged with it.
- Results are identical for n_x = 101 and n_x = 1001, and for scan steps
  from 2 K to 25 K.

Ag-Cu eutectic (root-found, provisional data):

| Quantity | 0.1.3 (2 K grid) | 0.1.4 (root-found) |
|---|---|---|
| T | 1051.0 K | **1051.739 K (778.589 °C)** |
| x(Cu) FCC_AG | 0.2605 | **0.261164** |
| x(Cu) LIQUID | 0.3878 | **0.387868** |
| x(Cu) FCC_CU | 0.5431 | **0.542955** |

The tolerances of the existing Ag-Cu tests are unchanged.

### Al-Cu tests that encoded non-physical results

- The `tests/test_systems.py` test that needed n_x = 1001 is removed. That
  test asserted the provisional model's eutectic (liquid at ~0.4 at% Cu) as
  if it were a physical result.
- In its place, `test_engine_finds_single_al_cu_reaction` checks only that
  the engine finds exactly one LIQUID + FCC_AL + AL2CU reaction for the
  current data, on the default grid. The app's invariant table now lists
  this reaction: 930.306 K, with FCC_AL at 0.0009, LIQUID at 0.0056 and
  AL2CU at 0.3333.
- Three new strict-xfail tests record the assessed Al-Cu eutectic:
  821 ± 3 K, liquid x(Cu) = 0.173 ± 0.010, and maximum Cu solubility in
  FCC_AL x(Cu) = 0.0248 ± 0.003.

### Figure export

- "Export figure (SVG)" and "Export figure (PDF)" render the traced diagram
  with matplotlib, in a white publication style. The figure has:
  - axis labels with units;
  - each invariant temperature written beside its line;
  - a footnote naming the data source and its provisional status.
- Exported files contain only lines, polygons and text, so they have no
  raster images. Tests check for `<image` in SVG and `/Subtype /Image` in
  PDF.

### Performance

Ag-Cu at the app's default settings (500–1450 K, invariants over the full
system range):

| | Diagram | Invariants | Total |
|---|---|---|---|
| 0.1.3 (grid labels + 2 K invariant grid) | 0.48 s | 3.59 s | **4.06 s** |
| 0.1.4 (traced + root-found) | 0.98 s | 1.09 s | **2.07 s** |

These are medians of three local runs. The 0.1.4 diagram time includes
building the Plotly figure.

## Resolved in 0.2.0

### Engine cross-validated against pycalphad

`tests/validation/` runs both engines on the same TDB files: the provisional
Ag-Cu and Al-Cu systems exported with `write_tdb`, and the two synthetic
gap fixtures. It compares them at 30 (T, x) points per system:

- the stable phase sets agree at every point;
- phase fractions and phase compositions agree to within 2e-5, against a
  tolerance of 1e-3;
- the Ag-Cu eutectic is 1051.739 K from both engines;
- the regular-solution binodal agrees to within 1e-5.

Further checks:

- On the gap eutectic, the engine matches the analytical temperature,
  643.192 K. pycalphad's value is 0.07 K lower.
- pycalphad's Pb-Sn literature test database (Ngai and Chang 1981) gives
  matching Gibbs energies, equilibria and the eutectic (454.56 K).

So the engine solves the models correctly. The open Ag-Cu and Al-Cu issues
below are in the data, not in the engine. The full report is
`docs/validation/engine_vs_pycalphad.md`.

### TDB import and export

- Assessed databases can now be loaded directly, which is the planned route
  to fixing the Ag-Cu and Al-Cu data.
- Magnetic and multi-sublattice phases are read but rejected by
  `build_system` with `NotImplementedError`, so their contributions are
  never silently dropped.

### Review fixes

- Gap fields are labelled with their composition sets.
- Tables and labels round to 0.1 K and 0.1 at%.
- The CLI `plot` command works and is smoke-tested.

## Still open

- **Ag-Cu solvus.** The simplified FCC_AG and FCC_CU parameters put the
  eutectic solid solubilities at x(Cu) = 0.2612 and 0.5430, against assessed
  values of 0.141 and 0.950. Two strict-xfail tests record this. It will be
  fixed by assessed database parameters, together with converting the
  system to a single FCC_A1 phase, which the engine supports.
- **Ag-Cu eutectic liquid composition.** It is 0.3879, which clears the
  0.399 ± 0.015 band by only about 0.004.
- **Al-Cu data (provisional).** The model has no Cu-rich solid phases, so
  LIQUID is stable at the Cu-rich end at every temperature. The eutectic is
  at 930.3 K with liquid x(Cu) = 0.0056 and FCC_AL x(Cu) = 0.0009. The
  assessed values are 821 K, 0.173 and 0.0248. Three strict-xfail tests
  record this.
- **Detection limits.** A field that appears and disappears again between
  two scan levels (5 K apart for invariants; the tracer's starting levels
  for drawing) is not found. A miscibility gap is detected from the sampled
  Gibbs curves, so a gap narrower than the Gibbs-curve sampling step is
  missed. In practice this only matters within about 0.02 K of a critical
  temperature.
- **Models not yet evaluated.** Magnetic contributions (TC, BMAGN) and
  multi-sublattice phases from TDB files raise `NotImplementedError`. The
  magnetic model is planned for 0.2.1.
- **Boundary drawing.** Between adaptive levels, boundary lines are
  straight segments. Points on the lines are exact equilibrium compositions,
  but between levels a line can deviate from the true curve by up to the
  0.001 bend tolerance.
