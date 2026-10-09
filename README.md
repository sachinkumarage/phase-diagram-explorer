# Phase Diagram Explorer

Phase Diagram Explorer is a scientific Python toolkit for computing and visualizing the thermodynamic phase diagrams of materials and chemical systems, using CALPHAD-style thermodynamic models to map regions of pressure, temperature, and composition space to the equilibrium phases (solid, liquid, gas, and their polymorphs) that are stable under those conditions, and rendering the resulting phase boundaries, triple points, and critical points as interactive plots for materials engineers, metallurgists, and researchers.

## Running the web app

An interactive Streamlit app is included for exploring computed phase diagrams: selecting a system, adjusting the temperature range and composition, and viewing stable phases, phase fractions, phase compositions, and Gibbs energy curves with the common tangent construction. Run it with:

```
streamlit run src/phase_diagram_explorer/app.py
```

Phase boundaries are traced from exact equilibrium tie lines and invariant reactions are located by root-finding, so neither depends on a composition grid. The diagram can be exported as a vector figure (SVG or PDF) with the "Export figure" buttons.

Only systems in `data/systems/` whose every phase can be evaluated are listed: JSON files, or TDB databases with a `<name>.meta.json` giving the element order. Systems whose data are marked `"_status": "provisional"` are shown with a preview banner. The invariant reactions can be downloaded as CSV at full precision.

## Command line

```
phase-diagram-explorer plot --system ag_cu                    # writes ag_cu_phase_diagram.svg
phase-diagram-explorer plot --system ag_cu --output ag_cu.pdf --unit C
phase-diagram-explorer export-tdb --system ag_cu --output ag_cu.tdb   # also writes ag_cu.meta.json
phase-diagram-explorer list
```

`--system` is a name in `data/systems/` (or `$PHASE_DIAGRAM_SYSTEMS_DIR`) or a path to a `.json` or `.tdb` file. `python -m phase_diagram_explorer` works the same way.

## Data formats

Systems are JSON files or TDB databases (binary subset: substitutional solutions and stoichiometric compounds; magnetic and multi-sublattice phases are read but not yet evaluated). See [docs/data_format.md](docs/data_format.md).

```python
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.tdb.writer import write_tdb

definition = load_system("tests/fixtures/ag_cu_provisional.tdb")   # element order from ag_cu_provisional.meta.json
definition = load_system("other.tdb", elements=["Pb", "Sn"])       # or given explicitly
system = build_system(definition)
write_tdb(load_system("data/systems/ag_cu.json"), "ag_cu.tdb")
```

## Validation against pycalphad

The equilibrium engine is cross-validated against [pycalphad](https://pycalphad.org) on the same TDB files: stable phases, phase fractions and compositions at 30 points per system, eutectic temperatures and a miscibility-gap binodal. Results are in [docs/validation/engine_vs_pycalphad.md](docs/validation/engine_vs_pycalphad.md). pycalphad is an optional dependency:

```
pip install -e ".[validation]"
pytest -m validation
python -m phase_diagram_explorer.validation     # regenerate the report and plots
```

## Running the tests

Slow tests (full-range tracing, export and headless Streamlit app tests) are marked `slow`, and the pycalphad cross-validation tests are marked `validation`. Both are skipped by default, so a plain local run is fast:

```
pytest                         # same as: pytest -m "not slow and not validation"
```

Run everything (validation tests skip with a reason if pycalphad is not installed) with:

```
pytest -m ""
```

CI runs the full suite on Python 3.11 and 3.12, and the validation tests in a separate job with `.[validation]` installed.
