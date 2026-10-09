# Changelog

## 0.2.0 - 2026-10-10

### Added
- TDB import (`tdb/` package, and `load_system` on `.tdb` files).
  - Reads the binary subset: ELEMENT, FUNCTION, PHASE, CONSTITUENT,
    PARAMETER (G, and L of Redlich-Kister orders 0..n), TYPE_DEFINITION,
    DEFINE_SYSTEM_DEFAULT and DEFAULT_COMMAND. Handles comments,
    multi-line commands and Thermo-Calc keyword abbreviations.
  - Unsupported keywords and parameter types raise `NotImplementedError`
    naming them and their line number.
  - Substitutional phases (including ones with a VA-only sublattice) and
    stoichiometric compounds are converted to the existing models, per mole
    of atoms.
  - TC/BMAGN parameters, magnetic type definitions and multi-sublattice
    phases are stored. `build_system` raises `NotImplementedError` for them
    instead of ignoring them.
  - The element order and the dependent element come from
    `<name>.meta.json` or from `load_system(..., elements=...)` /
    `dependent_element=`. The order is never guessed.
- Safe TDB expression evaluation (`tdb/expression.py`).
  - A tokenizer plus a recursive-descent parser, evaluated with numpy; no
    `eval`/`exec`. Supports `+ - * / **`, `T`, `R`, `LN`, `LOG`, `EXP`, and
    nested FUNCTION references, which are checked for undefined names and
    cycles.
  - Piecewise functions with Y/N ranges. Evaluating outside every range
    raises `ValueError`.
- TDB export: `tdb.writer.write_tdb` and `write_metadata`, and the CLI
  `export-tdb` command. JSON → TDB → JSON round trips give identical Gibbs
  energies, with relative difference < 1e-10 at 20 (T, x) points per phase.
- TDB test fixtures:
  - `tests/fixtures/ag_cu_provisional.tdb` and `al_cu_provisional.tdb`,
    labelled as provisional and not literature data;
  - synthetic `regular_solution.tdb` and `gap_eutectic.tdb`.
- Cross-validation against pycalphad.
  - New `validation` extra and pytest marker, a separate CI job,
    `tests/validation/`, and `python -m phase_diagram_explorer.validation`
    for the report (`docs/validation/engine_vs_pycalphad.md`, with overlay
    plots in `docs/img/`).
  - Same stable phases at all 30 points of every system. Phase fractions and
    compositions agree to within 2e-5.
  - Ag-Cu eutectic: 1051.739 K from both engines.
  - Regular-solution binodal agrees to within 1e-5.
  - Also checked against pycalphad's Pb-Sn literature test database:
    Gibbs energies, equilibria, and the eutectic at 454.56 K.
- CLI: `phase-diagram-explorer plot --system ag_cu` writes an SVG or PDF
  figure; there are also `export-tdb` and `list` commands. Installed as a
  console script.
- The app downloads the invariant reactions as a full-precision CSV
  (`export.invariants_csv`).
- `docs/data_format.md` documents the JSON format, the supported TDB subset
  and the `.meta.json` fields.

### Changed
- Miscibility-gap fields are labelled "ALPHA#1 + ALPHA#2", not
  "ALPHA + ALPHA".
- Display precision: T to 0.1 K (or 0.1 °C), compositions to 0.1 at% or
  wt%, and phase fractions to 0.001, in the app's tables, hover text and
  labels. The API and the CSV export keep full precision.
- `SolutionPhase` accepts temperature-dependent interaction parameters
  (anything with a `G(T)` method).
- The app and CLI list `.json`, `.yaml` and `.tdb` systems, but not
  `*.meta.json` files (`models.system_files`).
- `is_computable` returns False for phases that raise
  `NotImplementedError`.
- Expensive diagram computations are shared through session-scoped
  fixtures (`tests/conftest.py`).
  - The tests that existed in 0.1.4 run in 14.1 s by default, down from
    19.5 s.
  - The whole default run, with 210 tests instead of 123, takes 16.7 s.
- The default pytest run deselects `slow` and `validation` tests. Run
  `pytest -m ""` for everything.

### Fixed
- The CLI only printed its name. It now has working commands, and the
  README examples match it.

## 0.1.4 - 2026-10-09

### Added
- Phase boundary tracing (`tracing.trace_diagram`). Boundaries are traced
  from exact tie-line endpoint compositions, independent of any composition
  grid. Temperature levels are adaptive: field changes are bisected,
  invariants get levels at ±1e-4 K, and intervals are subdivided where a
  boundary moves far or bends sharply.
- Vector rendering. Phase regions are polygons built from the traced
  boundaries and invariant lines, labelled at their centroids with plain
  phase names. Invariant reactions are horizontal lines across their three
  compositions. Hover shows the phases of a field, or T and composition on a
  boundary.
- "Export figure" as SVG or PDF (`export.py`): a white publication style,
  axis labels with units, invariant temperatures annotated, the data source
  as a footnote, and no raster content.
- `equilibrium.phase_assemblage` and exact common-tangent refinement
  (`equilibrium/tangent.py`), with an analytic `SolutionPhase.molar_gibbs_derivative`.
- `_source` is read into `SystemDefinition.source`.
- Strict-xfail Al-Cu physics tests for the assessed eutectic (821 K, liquid
  x(Cu) = 0.173, FCC_AL x(Cu) = 0.0248).
- Tests for tracing (no NaN gaps, boundaries meeting invariant lines,
  non-overlapping polygons covering the diagram, the traced binodal against
  the analytical solution), grid-independent invariants and vector export.

### Changed
- Invariant reactions are refined by root-finding on the three-phase
  condition (to 1e-6 K), report all three phase compositions, and are
  deduplicated (same phases within 0.5 K). Results no longer depend on the
  composition grid or the scan step.
- `compute_equilibrium` returns exact common-tangent compositions instead of
  Gibbs-curve grid points.
- `plot_diagram` takes a `TracedDiagram`. The heatmap of grid labels is
  gone, and so is the app's composition-points slider.
- The Al-Cu test that needed n_x = 1001 and asserted the provisional model's
  eutectic is replaced by `test_engine_finds_single_al_cu_reaction`.
- Ag-Cu eutectic, now root-found: 1051.739 K; x(Cu) = 0.261164 (FCC_AG),
  0.387868 (LIQUID), 0.542955 (FCC_CU).
- Ag-Cu diagram plus invariants at the default settings: about 2.1 s, down
  from about 4.1 s.

## 0.1.3 - 2026-10-09

### Added
- Miscibility gap support: the equilibrium solver reports two tangent points
  on one phase's Gibbs curve as composition sets (`PHASE#1`, `PHASE#2`).
  Diagrams, invariant detection, tables and plots handle these labels.
  Validated against a synthetic symmetric regular solution
  (`tests/fixtures/regular_solution.json`): the analytical binodal at three
  temperatures, symmetry about x = 0.5, and Tc = L0/(2R) to within 1 K.
- `"_status"` / `"_status_reason"` fields in system JSON. Ag-Cu and Al-Cu are
  marked provisional, and the app shows a preview banner for them.
- `builder.is_computable`. The app lists only systems with complete Gibbs
  energy data.
- Strict-xfail Ag-Cu solvus regression tests: assessed eutectic solid
  solubilities x(Cu) = 0.141 and 0.950.
- `slow` pytest marker. The default local run skips slow tests; CI runs the
  full suite.

### Fixed
- Stoichiometric compounds enter the convex hull at their exact composition,
  not the nearest grid point. At x(Cu) = 1/3 in Al-Cu the equilibrium is
  single-phase AL2CU.
- Invariant detection no longer merges separate two-phase fields that share
  the same phase pair. This removes the spurious Al-Cu "eutectic" (liquid
  at 46 at% Cu) and "peritectic" reported in 0.1.2.

### Changed
- `data/systems/example.json` moved to `tests/fixtures/`.
- The Al-Cu invariant test in `tests/test_systems.py` uses a composition grid
  fine enough to resolve the ~0.005-wide Al-rich liquid field.
- Tightened the Fe-C conversion check to 3.44 ± 0.01 at% C.

## 0.1.2 - 2026-10-09

### Fixed
- The Streamlit app now builds its thermodynamic models from the system JSON
  coefficients through `builder.build_system`, shared with the tests. The
  hard-coded placeholder curves are gone (`_build_computable_system`,
  `MELTING_T`, and the other placeholders). Ag-Cu now shows Ag melting at
  1234.93 K, Cu at 1357.77 K, and a eutectic at 1051.0 K with liquid
  x(Cu) = 0.388.
- Terminal phases are placed by thermodynamics, never by phase-list order.
  FCC_AG is now on the Ag side (x = 0).
- Stoichiometric compounds are placed at their site-ratio composition
  (AL2CU at x(Cu) = 1/3).
- Invariant reactions are detected over the system's full temperature range
  on a dedicated grid, independent of the displayed range and grid.

### Added
- `SystemDefinition.base_element` / `dependent_element`. The composition
  axis is the mole fraction of the second listed element.
- Optional `t_range_k` (analysis and default display range) and per-element
  `atomic_mass` fields in system JSON. Both are set for Ag-Cu and Al-Cu, and
  the Ag-Cu default range of 500-1450 K covers both melting points.
- `units.py`: K/°C and mole/weight fraction conversion.
- App toggles for K/°C and at%/wt%, with unit-labelled axes, sliders, markers
  and tables. The invariant table lists reactions outside the displayed range
  with a note.
- `invariants.detect_invariants_over_range`.
- Tests for the builder, units and app wiring.

### Changed
- Removed the xfail markers from `tests/test_physics_regression.py`; all of
  its tests now pass.
- `docs/audit_v0.1.md`: added a "Resolved in 0.1.2" section.

## 0.1.1 - 2026-10-09

### Added
- `tests/test_physics_regression.py`: strict-xfail regression tests that drive
  the Streamlit app for the Ag-Cu system (pure Ag/Cu melting points, eutectic
  temperature and composition, Ag-rich terminal phase on the x(Cu)=0 side).
- `docs/audit_v0.1.md`: physical-correctness audit of the Ag-Cu system. The app
  ignores the Gibbs coefficients in the system JSON and assigns terminal solid
  phases by list position, so it shows a placeholder diagram (eutectic ~650 K,
  both elements melting at 900 K, FCC_CU at x=0).

### Changed
- Ignore `.DS_Store` files.

## 0.1.0

- Initial release.
