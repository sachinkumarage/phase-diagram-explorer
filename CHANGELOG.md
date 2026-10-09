# Changelog

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
