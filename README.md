# Phase Diagram Explorer

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://phase-diagram-explorer.streamlit.app/)
[![Tests](https://github.com/sachinkumarage/phase-diagram-explorer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/sachinkumarage/phase-diagram-explorer/actions/workflows/tests.yml)

Phase Diagram Explorer is a scientific Python toolkit for computing and visualizing the thermodynamic phase diagrams of materials and chemical systems, using CALPHAD-style thermodynamic models to map regions of pressure, temperature, and composition space to the equilibrium phases (solid, liquid, gas, and their polymorphs) that are stable under those conditions, and rendering the resulting phase boundaries, triple points, and critical points as interactive plots for materials engineers, metallurgists, and researchers.

## Running the web app

An interactive Streamlit app is included for exploring computed phase diagrams. You can select a system, adjust the temperature range and composition, and view stable phases, phase fractions, phase compositions, and Gibbs energy curves with the common tangent construction.

Phase boundaries are traced from exact equilibrium tie lines, and invariant reactions are located by root-finding, so neither depends on a composition grid. The diagram can be exported as a vector figure (SVG or PDF) with the "Export figure" buttons, and the invariant reactions can be downloaded as CSV at full precision.

Phase fields are labelled with textbook names such as L, (Ag) and "L + (Ag)"; the legend, tables and hover text also give the CALPHAD phase names. Phase fractions are mole fractions in at% and mass fractions in wt%, and "Show lever-rule calculation" shows the lever rule in the current unit. The Gibbs energy curves are always plotted against mole fraction, where the common tangent construction is valid, and show only the phases stable somewhere in the diagram unless "Show metastable phases" is ticked. The app opens in °C and at% unless a system sets its own units.

Only systems in `data/systems/` whose every phase can be evaluated are listed: JSON files, or TDB databases with a `<name>.meta.json` that gives the element order. By default the app lists only validated systems (Ag-Cu, Fe-C and Pb-Sn) and opens Ag-Cu. Ticking "Show preview systems (provisional data)" in the sidebar adds systems marked `"_status": "provisional"`, such as Al-Cu, which are shown with the banner "Preview: provisional thermodynamic data, not yet validated."

### Run online

The app is deployed on Streamlit Community Cloud: https://phase-diagram-explorer.streamlit.app/

### Run locally

```
pip install -r requirements.txt     # the package plus its pinned runtime dependencies
streamlit run streamlit_app.py
```

`streamlit_app.py` at the repository root is the entry point used by Streamlit Community Cloud. `streamlit run src/phase_diagram_explorer/app.py` also works with the package installed. Settings are in `.streamlit/config.toml`. The theme follows the viewer's light or dark mode. No secrets are needed, and `.streamlit/secrets.toml` is git-ignored.

At the default settings, the diagram and invariant reactions are loaded from `data/precomputed/` (one file per diagram mode for Fe-C), so the app starts quickly on a small machine. If a system's database cannot be downloaded (for example if NIST is unreachable), the page still works from the precomputed data. That covers the diagram, invariant table, citations, unit toggles, and the equilibrium at the selected temperature and composition, which is read from a precomputed equilibrium table to within 1e-4. Only the Gibbs energy curves and the total Gibbs energy need the database, and that section says so. Other temperature ranges and resolutions are computed live and cached. "High resolution" shows an estimated time before it runs. The equilibrium at the selected temperature and composition is always computed live.

### Regenerating precomputed data

Regenerate the precomputed diagrams whenever system data or the engine change (every engine change bumps the package version, which makes the files stale):

```
python scripts/precompute.py            # writes data/precomputed/<system>.npz
python scripts/precompute.py --check    # exit status 1 if any file is missing or out of date
```

Each file records the SHA256 of its source data, the package version, the gas constant, the temperature range, the settings and the computation time. The app only uses a file whose metadata matches; otherwise it computes live and says so. A test fails if any app system's file is missing or stale.

Runtime dependencies go into both `pyproject.toml` and `requirements.txt` (with `~=` pins). Development and validation tools (`pytest`, `pycalphad`) never go into `requirements.txt`.

## Data sources

**Ag-Cu** is the assessed description from the NIST solder database (its Ag-Cu-Va subsystem):

- Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST Standard Reference Database 139, National Institute of Standards and Technology, https://doi.org/10.18434/T4759N (Accessed 12 October 2026).
- Ag-Cu assessment: F.H. Hayes, H.L. Lukas, G. Effenberg, G. Petzow, Z. Metallkd. 77 (1986) 749-754.
- Unary data: A.T. Dinsdale, CALPHAD 15 (1991) 317-425.

License: https://www.nist.gov/open/license. The database file is **downloaded from NIST** (https://www.metallurgy.nist.gov/phase/solder/NIST-solder.tdb) and verified by SHA256. The same file provides Ag-Cu and Pb-Sn. It is not redistributed in this repository: `data/tdb/` is git-ignored. Fetch it with:

```
python scripts/fetch_tdb.py          # downloads to data/tdb/ and checks size and SHA256
```

The app and `load_system` also download it on first use. A failed download, or a file that doesn't match, is an error; there is no fallback to other data. Sources are registered in `data/sources.json`. The app shows each system's citations below the diagram, and `docs/validation/ag_cu.md` compares the computed Ag-Cu diagram with NIST's calculation, experiment and pycalphad.

**Pb-Sn** is the assessed description from the same NIST solder database (its Pb-Sn-Va subsystem). Its composition axis is x(Sn), and the app opens it in °C and wt% Sn:

- Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST Standard Reference Database 139, National Institute of Standards and Technology, https://doi.org/10.18434/T4759N (Accessed 12 October 2026).
- Pb-Sn assessment: H.J. Fecht, M.-X. Zhang, Y.A. Chang, J.H. Perepezko, Metall. Trans. A 20A (1989) 795-803.
- Unary data: A.T. Dinsdale, CALPHAD 15 (1991) 317-425.
- NIST Pb-Sn page: https://www.metallurgy.nist.gov/phase/solder/pbsn.html

For Pb-Sn the app adds a short note on 63Sn-37Pb solder (near-eutectic), with the eutectic and the melting (mushy) range of the selected alloy, i.e. liquidus minus solidus. `docs/validation/pb_sn.md` compares the computed diagram with NIST's calculation, experiment and pycalphad. pycalphad's own Pb-Sn test database (Ngai and Chang 1981) is used only as a cross-validation fixture.

**Fe-C** is the assessed description in pycalphad's test database `cfe_broshe.tdb` (release 0.11.2). The app shows two diagrams, chosen with "Diagram": the metastable Fe-Fe3C diagram of steel practice (graphite suspended, the default) and the stable Fe-graphite diagram. Both run from 0 to 6.69 wt% C (Fe3C, x(C) = 0.25) and 400 to 1600 °C, in °C and wt% by default. Fields carry textbook names (alpha and delta ferrite by temperature region, gamma austenite, Fe3C cementite, graphite). The A1, A3 and Acm lines are labelled, and the steel / cast iron division at 2.14 wt% C is a dashed line. Invariant labels show this database's values (eutectoid 726.5 °C, eutectic 1148.2 °C, peritectic 1494.6 °C; stable eutectoid 738.0 °C, eutectic 1153.4 °C), which may differ slightly from older textbook figures:

- B. Hallstedt, D. Djurovic, J. von Appen, R. Dronskowski, A. Dick, F. Koermann, T. Hickel, J. Neugebauer, Calphad 34 (2010) 129-133.
- A.T. Dinsdale, CALPHAD 15 (1991) 317-425.
- Brosh et al., CALPHAD 31 (2007) 173-185 (equation of state; not used at 1 atm beyond its contribution at that pressure).
- Database file: pycalphad 0.11.2 test database cfe_broshe.tdb.

The file is **fetched from the pycalphad repository** at the 0.11.2 release tag (https://raw.githubusercontent.com/pycalphad/pycalphad/0.11.2/pycalphad/tests/databases/cfe_broshe.tdb) and verified by SHA256, like the NIST file; it is not redistributed here. It carries no license text of its own; the pycalphad repository is MIT-licensed.

Two phases of the file are left out, as the app also notes:
- **M7C3_D101** (Fe7C3): its parameters are attributed only to an unidentified "This work" in the file. Below x(C) = 0.25 it is never stable, so leaving it out changes nothing in the displayed diagram (a test checks this to 1e-8).
- **DIAMOND_A4**: stable only at high pressure. With graphite suspended, ferrite + diamond would replace ferrite + Fe3C below 589 °C, so it must stay out of the metastable diagram.

`docs/validation/fe_c.md` compares both diagrams with pycalphad on the same file, this database's reference values and literature values.

**Al-Cu** is still a provisional model, shown with a preview banner. The former provisional Ag-Cu model is kept as a test fixture, `tests/fixtures/ag_cu_provisional.json`.

## Command line

```
phase-diagram-explorer plot --system ag_cu                    # writes ag_cu_phase_diagram.svg
phase-diagram-explorer plot --system ag_cu --output ag_cu.pdf --unit C
phase-diagram-explorer export-tdb --system ag_cu --output ag_cu.tdb   # also writes ag_cu.meta.json
phase-diagram-explorer list
```

`--system` is a name in `data/systems/` (or `$PHASE_DIAGRAM_SYSTEMS_DIR`) or a path to a `.json` or `.tdb` file. `python -m phase_diagram_explorer` works the same way.

## Data formats

Systems are JSON files or binary TDB databases. Phases use the Compound Energy Formalism: substitutional and interstitial solutions, ordered phases with internal degrees of freedom, compounds and pure-element phases. Magnetic phases use the Inden-Hillert-Jarl model. Literature databases such as Fe-C and Cu-Mg load directly, with their references available in `system.metadata["references"]`. See [docs/data_format.md](docs/data_format.md) for the formats and [docs/theory.md](docs/theory.md) for the models.

```python
from phase_diagram_explorer.models import load_system
from phase_diagram_explorer.builder import build_system
from phase_diagram_explorer.tdb.writer import write_tdb

definition = load_system("data/systems/ag_cu.meta.json")          # assessed Ag-Cu (downloads the NIST database if missing)
definition = load_system("data/tdb/nist_solder.tdb", elements=["AG", "CU"])  # binary subsystem of a multicomponent database
definition = load_system("tests/fixtures/ag_cu_provisional.tdb")   # element order from ag_cu_provisional.meta.json
definition = load_system("fe_c.tdb", elements=["Fe", "C"])          # or given explicitly (x = mole fraction of C)
definition = load_system("fe_c.tdb", elements=["Fe", "C"], gas_constant=8.3145)  # R defaults to 8.31451 for TDB
system = build_system(definition)
write_tdb(load_system("data/systems/al_cu.json"), "al_cu.tdb")
```

## Validation against pycalphad

The equilibrium engine is cross-validated against [pycalphad](https://pycalphad.org) on the same TDB files. The comparison covers synthetic fixtures (interstitial, magnetic, miscibility gap) and literature databases (Pb-Sn, Cu-Mg, Fe-C). For each system it checks Gibbs energies at the same site fractions, stable phases, fractions and compositions at 30 points, every invariant temperature, and a miscibility-gap binodal. Results are in [docs/validation/engine_vs_pycalphad.md](docs/validation/engine_vs_pycalphad.md). pycalphad is an optional dependency:

```
pip install -e ".[dev,validation]"
pytest -m validation
python -m phase_diagram_explorer.validation     # regenerate the report and plots
```

## Running the tests

Install the test runner with `pip install -e ".[dev]"`. Three markers are skipped by default, so a plain local run is fast:

- `slow`: full-range tracing, export and headless Streamlit app tests;
- `validation`: the pycalphad cross-validation tests;
- `network`: real downloads from the official database URLs, run with `pytest -m network`.

```
pytest                         # same as: pytest -m "not slow and not validation and not network"
```

Run everything (validation tests skip with a reason if pycalphad is not installed) with:

```
pytest -m ""
```

CI does the following:
- runs the full suite, including the network tests, on Python 3.11 and 3.12, after fetching the databases and running `python scripts/precompute.py --check`;
- runs the validation tests in a separate job with `.[validation]` installed;
- runs a cloud smoke test in a fresh virtual environment with only `pip install -r requirements.txt`:
  - first with no database and downloads disabled (`python scripts/smoke_test.py --offline`);
  - then after fetching the databases (`python scripts/smoke_test.py`).

  Each run starts the app with Streamlit's AppTest and checks the public defaults and both preview states. It loads every system from precomputed data, toggles units, and records the cold-start time in the job summary.
