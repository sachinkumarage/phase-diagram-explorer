# Changelog

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
