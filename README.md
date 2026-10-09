# Phase Diagram Explorer

Phase Diagram Explorer is a scientific Python toolkit for computing and visualizing the thermodynamic phase diagrams of materials and chemical systems, using CALPHAD-style thermodynamic models to map regions of pressure, temperature, and composition space to the equilibrium phases (solid, liquid, gas, and their polymorphs) that are stable under those conditions, and rendering the resulting phase boundaries, triple points, and critical points as interactive plots for materials engineers, metallurgists, and researchers.

## Running the web app

An interactive Streamlit app is included for exploring computed phase diagrams: selecting a system, adjusting the temperature range and composition, and viewing stable phases, phase fractions, phase compositions, and Gibbs energy curves with the common tangent construction. Run it with:

```
streamlit run src/phase_diagram_explorer/app.py
```

Only systems in `data/systems/` with complete Gibbs energy data are listed. Systems whose data are marked `"_status": "provisional"` are shown with a preview banner.

## Running the tests

Slow tests (full-range invariant detection and headless Streamlit app tests) are marked `slow` and skipped by default, so a plain local run is fast:

```
pytest                         # same as: pytest -m "not slow"
```

Run the full suite, as CI does, with:

```
pytest -m "slow or not slow"
```
