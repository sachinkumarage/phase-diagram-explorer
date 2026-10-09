# Changelog

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
