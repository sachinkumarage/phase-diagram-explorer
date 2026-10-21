# Changelog

## 0.4.0 - 2026-10-22

### Added
- **Assessed Fe-C system** (`data/systems/fe_c.meta.json`), listed by default.
  - Data: pycalphad's test database `cfe_broshe.tdb` (release 0.11.2): Hallstedt et al., Calphad 34 (2010)
    129-133, with Dinsdale (1991) and the equation of state of Brosh et al. (2007).
  - Fetched from the pycalphad repository at the 0.11.2 tag and verified by SHA256
    (`data/sources.json`, `pycalphad_cfe_broshe`); never committed. The file carries no license text of its
    own; the pycalphad repository is MIT-licensed.
  - Gustafson (1985) is not cited: the abstract of Hallstedt et al. (2010) does not say that its solution
    phases come from it, and the full text could not be checked.
- **Diagram modes**: "Metastable Fe-Fe3C (steel practice)" (default, GRAPHITE suspended) and "Stable
  Fe-graphite". Each mode has its own precomputed file (`fe_c.metastable.npz`, `fe_c.stable.npz`), covered by
  the stale-data check, and works without the database (diagram, invariants, point equilibrium).
- **Suspended phases**: M7C3_D101 (parameters attributed only to an unidentified "This work") and
  DIAMOND_A4, with the reasons shown in the app and the docs.
  - A test checks that results with and without M7C3_D101 are identical within 1e-8 below x(C) = 0.25.
  - DIAMOND_A4 must stay out of the metastable diagram: with graphite suspended, ferrite + diamond would
    replace ferrite + Fe3C below 589 °C.
- **Fe-C display**:
  - 0 to 6.69 wt% C (x = 0.25, Fe3C) and 400 to 1600 °C, in °C and wt% by default; compositions in the
    tables to 0.001 wt%.
  - Textbook names: L, alpha (ferrite) and delta (ferrite) by temperature region, gamma (austenite),
    Fe3C (cementite), graphite.
  - A1, A3 and Acm labelled; the 2.14 wt% C steel / cast iron division as a dashed line with "Steels" and
    "Cast irons".
  - Invariant labels with this database's values, and a note that they may differ slightly from older
    textbook figures.
- **Validation** (`tests/test_fe_c.py`, `tests/validation/test_fe_c_vs_pycalphad.py`,
  `docs/validation/fe_c.md` with an overlay plot):
  - against pycalphad on the same file, both modes: invariant temperatures within 0.01 K, compositions
    within 1e-4 in x;
  - against this database's reference values: 0.05 K and 0.0005 wt%;
  - against literature values, with tolerances from the spread between published assessments;
  - exactly three invariant reactions in each mode over 400-1600 °C (eutectoid, eutectic, peritectic).
- Metadata fields `suspended_phases`, `modes`, `x_range`, `diagram_note` and `diagram_marks`
  (docs/data_format.md); `load_system(..., include_suspended=True)`.

### Changed
- Field labels are placed in the part of each field inside the displayed composition range, and a field
  traced as several strips is labelled once.
- The smoke test loads Fe-C in both modes, offline and online, and selects unit radios by label.

## 0.3.3 - 2026-10-13

Presentation fixes and textbook-style labels. No thermodynamic parameters
and no engine code changed; the precomputed data were regenerated for the
new version and the stored display names.

### Fixed
- **Phase fractions now match the composition unit.**
  - In at% the app shows mole fractions of the phases. In wt% it shows mass
    fractions, w_i = n_i M_i / Σ n_j M_j, with M_i the molar mass of phase i
    from its composition.
  - The column header says which: "phase fraction (mol)" or "phase fraction
    (mass)".
  - Previously, mole fractions were shown in wt% mode as well, so they did
    not satisfy the lever rule in the displayed wt% compositions.
  - The precomputed (no-database) path gives the same fractions as the live
    path.
- **The Gibbs energy chart always uses mole fraction.** Its x axis is x (at%)
  even in wt% mode, where it shows "Gibbs energy curves are shown against
  mole fraction, where the common tangent construction is valid." The
  common tangent is now a straight line in the plotted coordinates.
- **No tracebacks for visitors.** `showErrorDetails = "none"` is set in
  `.streamlit/config.toml`. Unexpected errors show "Something went wrong
  while loading this system. Please try another system or reload the page."
  and are logged in full.

### Changed
- **Textbook phase-field labels.**
  - Systems can give display names in their `.meta.json`
    (`"display_names"`):
    - Ag-Cu: L, (Ag) and (Cu), the two sides of the FCC_A1 gap.
    - Pb-Sn: L, (Pb) for FCC_A1 and (Sn) for BCT_A5.
  - A composition set is named by its side of the gap, not by #1/#2.
  - Field labels, the equilibrium tables and the invariant table use these
    names, e.g. "L + (Ag)", "(Ag) + (Cu)", "L -> (Ag) + (Cu)".
  - The legend, the tables and the hover text also show the CALPHAD names,
    e.g. "(Ag) FCC_A1".
- **Label placement.**
  - A field label goes inside its field where it fits. Otherwise it goes
    outside, with a short leader line.
  - The invariant temperature label sits above its line, clear of boundary
    lines.
  - No label is clipped by the plot area. The SVG/PDF export (app and
    `phase-diagram-explorer plot`) uses the same names and layout.
- **Gibbs energy chart.**
  - By default, only phases that are stable somewhere in the system's
    default diagram are shown. "Show metastable phases" adds the others.
  - The system info line lists stable phases first, then "Metastable in this
    database: ...".
- **Public default temperature unit is °C** (composition stays at%), unless
  a system sets its own.
- **Molar Gibbs energy** is shown as an integer in J/mol. "Download
  equilibrium (CSV, full precision)" gives it at full precision, together
  with the mole and mass fractions and the phase compositions.

### Added
- "Show lever-rule calculation" below the phase fractions in two-phase
  fields: the formula and the numbers in the current composition unit, on
  the same basis (mol or mass) as the table.

## 0.3.2 - 2026-10-13

### Fixed
- **Database download refused with HTTP 403** (the live app's Ag-Cu
  failure).
  - `sources.fetch`, used by `scripts/fetch_tdb.py` and the app, sent
    Python's default User-Agent (`Python-urllib/3.x`), which NIST's server
    refuses from any network.
  - Downloads now send `phase-diagram-explorer/<version>
    (+https://github.com/sachinkumarage/phase-diagram-explorer)`. SHA256 and
    size verification and the 3 retries are unchanged, and no mirror is used.
  - **CI was red** for `67697b4` (Ag-Cu) and `bdf4aad` (Pb-Sn): the fetch
    step got HTTP 403 and those commits' tests never ran in CI.
  - **Local runs did not catch it** because the database file had been
    downloaded beforehand with curl (so it was only verified, never
    fetched), and the fetch tests used `file://` URLs.
  - See `docs/audit_v0.1.md`.
- **Trivial common tangents for miscibility gaps.** When the first
  refinement failed, a fallback start inside the gap could converge to two
  nearby points near the spinodal: a 2e-6 wide FCC_A1 gap tie line at
  504.25 K in Ag-Cu, instead of 0.0018 to 0.9998. Such solutions are now
  rejected, and a regression test covers this case. Every equilibrium
  recorded with 0.2.0 is unchanged.

### Added
- **The app works without the database.** If a system's database cannot be
  downloaded, it renders everything except the Gibbs energy curves from
  precomputed data, with no database access:
  - the diagram, invariant table, citations and unit toggles;
  - the equilibrium at the selected (T, x), from the new
    `lookup.equilibrium_from_trace`;
  - Pb-Sn's melting range.

  Only the Gibbs energy section says "Gibbs energy curves need the NIST
  database, which could not be downloaded right now." The total Gibbs
  energy shows "Needs the NIST database." A failed download is retried at
  most every 5 minutes, and `PHASE_DIAGRAM_DISABLE_DOWNLOADS=1` disables
  downloads.
- **Precomputed files now also hold:**
  - an equilibrium table: the diagram traced 0.25 K apart, with extra levels
    around the start and end of each two-phase field. It matches
    `compute_equilibrium` to 4.9e-7 (Ag-Cu) and 9.3e-6 (Pb-Sn) over 1,120
    points each.
  - a summary of the system definition (names, elements, phase names,
    citations, display settings; no thermodynamic parameters).

  The staleness check no longer needs the database. File sizes: Ag-Cu
  162 KiB, Al-Cu 168 KiB, Pb-Sn 72 KiB.
- **Public defaults:** only validated systems are listed, and Ag-Cu is
  selected. The sidebar checkbox "Show preview systems (provisional data)",
  off by default, adds Al-Cu with its banner.
- **Tests:**
  - `tests/test_lookup.py`: 20 points each for Ag-Cu and Pb-Sn, within
    1e-4 of `compute_equilibrium`, covering the gap, the eutectic and the
    pure ends; no stored gap tie line is degenerate.
  - `tests/test_network.py` (marker `network`, deselected by default): a real
    download from the official URL through the fetch code path, with SHA256
    verification. CI runs it.
  - User-Agent and disabled-download tests.
  - AppTests for a failed download, the public defaults and Pb-Sn without
    the database.
- `scripts/smoke_test.py --offline`, and a `cloud-smoke` CI step that runs
  it with downloads disabled and no database before the normal run. Both
  runs check the public defaults and both preview states.
- `trace_diagram(..., extra_levels=...)`.
- README: "Live demo" and CI status badges, and the live app URL
  https://phase-diagram-explorer.streamlit.app/.

### Changed
- `tests/fixtures/nist_database_info.tdb` is replaced by the synthetic
  `tests/fixtures/database_info_line_breaks.tdb`, which keeps the
  apostrophe line-break test. The expression tests use a synthetic unary
  function, and no tracked file contains text from the NIST database file.
- `melting_range` accepts an `equilibrium` callable, so it can run on the
  precomputed table.

## 0.3.1 - 2026-10-13

### Added
- **Assessed Pb-Sn system** (`data/systems/pb_sn.meta.json`): the Pb-Sn-Va
  subsystem of the NIST solder database (same file and SHA256 as Ag-Cu).
  - The composition axis is x(Sn), and the default view is 300–650 K in °C
    and wt% Sn.
  - It has no preview banner. The app shows the citations: Kattner (2017),
    NIST SRD 139; H.J. Fecht, M.-X. Zhang, Y.A. Chang, J.H. Perepezko,
    Metall. Trans. A 20A (1989) 795-803; A.T. Dinsdale, CALPHAD 15 (1991)
    317-425; and the NIST Pb-Sn page.
  - Eutectic at 455.391 K (182.24 °C): liquid x(Sn) = 0.74119, (Pb) 0.28929,
    (Sn) 0.98021. NIST: 182.2 °C, 62.13, 18.91 and 96.59 wt% Sn.
- Industrial context box for Pb-Sn: 63Sn-37Pb solder is near-eutectic. The
  box shows the eutectic temperature and the solidus, liquidus and melting
  (mushy) range of the selected alloy, for example 0.9 K at 63 wt% Sn.
  - Systems define the note with an `industry_note` field.
  - The solidus and liquidus come from the new `solidification.melting_range`.
- Per-system default display units (`"display"` in the `.meta.json`). The
  unit selectors reset to each system's defaults when you switch systems.
- `tests/test_pb_sn_assessed.py`:
  - (a) NIST's eutectic, 455.35 K ± 0.2 K, and compositions ± 0.0005, with
    the mass % to mole fraction conversion in the docstring;
  - (b) experiment: 183 ± 2 °C, 61.9 ± 1.5 wt% Sn, about 19 ± 2 wt% Sn in
    (Pb), about 2.5 ± 1.5 wt% Pb in (Sn), and Pb and Sn melting at
    600.61 K and 505.08 K ± 0.1 K;
  - (d) exactly one invariant, a eutectic;
  - (e) the app uses the NIST data, and pycalphad's Pb-Sn test database
    stays a test fixture only.
- (c) pycalphad cross-validation of Pb-Sn on the same file (case
  "Pb-Sn (NIST)").
- `docs/validation/pb_sn.md`: engine vs NIST vs experiment vs pycalphad,
  with an overlay plot. `python -m phase_diagram_explorer.validation
  --systems` writes both per-system pages.
- The smoke test checks that Pb-Sn loads with citations, the info box and
  no banner.

### Fixed
- **Miscibility gaps with both composition sets near the pure ends.** The
  tie line stayed at grid points (for Ag-Cu at 525 K: 0.002004 and 1.0,
  instead of 0.0024831 and 0.9996607).
  - The common-tangent refinement now also tries each composition set close
    to its own end.
  - This was the cause of the largest Ag-Cu differences from pycalphad in
    0.3.0 (4.4e-4); the largest is now 1.8e-6.
  - New regression tests cover the regular-solution binodal at 300–450 K
    (analytical) and the Ag-Cu low-temperature solvus (common-tangent
    condition).
  - Every equilibrium recorded with 0.2.0 is unchanged.

### Changed
- Precomputed data regenerated for all systems. Pb-Sn is new, at 14.9 KiB.

## 0.3.0 - 2026-10-12

### Added
- **Assessed Ag-Cu system** (`data/systems/ag_cu.meta.json`): the Ag-Cu-Va
  subsystem of the NIST solder database. It has a single FCC_A1 phase with
  a miscibility gap, no provisional status and no preview banner. The app
  shows its three citations, with links, below the diagram:
  - Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics -
    Solder Systems, NIST Standard Reference Database 139, National
    Institute of Standards and Technology, https://doi.org/10.18434/T4759N
    (Accessed 12 October 2026);
  - F.H. Hayes, H.L. Lukas, G. Effenberg, G. Petzow, Z. Metallkd. 77
    (1986) 749-754;
  - A.T. Dinsdale, CALPHAD 15 (1991) 317-425.
- Downloaded databases, which are never committed:
  - `data/sources.json` registers each database's official URL, SHA256,
    size, citation, license mode and license URL
    (https://www.nist.gov/open/license).
  - `scripts/fetch_tdb.py` and `sources.py` download into `data/tdb/` (3
    attempts with backoff) and accept a file only if its size and SHA256
    match. Errors are clear and there is no silent fallback.
  - `load_system` and the app download a missing database on first use. The
    app shows a spinner, and a clear error if the download fails.
  - `data/tdb/` is git-ignored.
- Binary subsystems of multicomponent TDB files
  (`load_system(path, elements=["AG", "CU"])`, `tdb.convert.extract_subsystem`):
  - the requested elements plus VA and /- are kept;
  - phases left with an empty sublattice and parameters involving other
    elements (ternary ones included) are dropped;
  - what was dropped is reported in `system.metadata["subsystem"]`.
- A system can be a `.meta.json` naming a registered `source`, with its
  `citations`.
- Tests:
  - `tests/test_ag_cu_assessed.py`:
    - (a) NIST's calculated eutectic, 1052.25 K ± 0.2 K, with liquid, (Ag)
      and (Cu) at x(Cu) = 0.40250, 0.13273, 0.95460 ± 0.0005;
    - (b) experiment: 1052 ± 3 K, 0.399 ± 0.010, 0.141 ± 0.015,
      0.950 ± 0.010, and Ag and Cu melting at 1234.93 K and 1357.77 K
      ± 0.1 K;
    - (d) exactly one invariant reaction in 500–1450 K;
    - melting detected next to each pure end, and the pure-element Gibbs
      crossing.
  - (c) pycalphad cross-validation on the same file
    (`tests/validation/`, case "Ag-Cu (NIST)").
  - `tests/test_subsystem.py`: a synthetic A-B-C fixture.
  - `tests/test_sources.py`: fetch, checksum and retry behaviour.
  - App tests for citations, download on first use and a failed download.
  - The NIST DATABASE_INFO block as a parser fixture.
- `docs/validation/ag_cu.md`: engine vs NIST vs experiment vs pycalphad,
  with an overlay plot.

### Changed
- `data/systems/ag_cu.json` (provisional, two FCC phases) moved to
  `tests/fixtures/ag_cu_provisional.json`. The engine tests that used it now
  use the fixture.
- The Ag-Cu solvus tests in `tests/test_physics_regression.py` lost their
  strict-xfail markers and pass with unchanged tolerances. The only
  remaining xfails are the three Al-Cu tests.
- Precomputed data are regenerated. For a downloaded database, the
  staleness check uses the registered SHA256 of the TDB file.
- The smoke test checks that Ag-Cu shows its citations and no banner, and
  that Al-Cu still shows the banner.
- GitHub Actions moved to `actions/checkout@v7`, `actions/setup-python@v7`
  and `actions/cache@v6` (Node.js 24). Every job restores `data/tdb/` from
  a cache keyed by `data/sources.json`, which holds the SHA256, then fetches
  and verifies. A failed download with no cache fails the job.

### Fixed
- TDB parser: in `DATABASE_INFO`, `'` is Thermo-Calc's line-break marker,
  not a quote. Single quotes now quote text only in `LIST_OF_REFERENCES` and
  `ADD_REFERENCES`. The NIST solder database failed to load because of this.
- Tie-line refinement next to a pure element. When the L + FCC field is
  narrower than one Gibbs-curve grid cell (within about 1.7 K of Cu's
  melting point), refinement failed and kept the hull's grid points. It now
  retries from points inside the tie line and close to the pure end.
  - Melting detection agrees with the pure-element Gibbs crossing to within
    0.002 K (Ag and Cu).
  - All equilibria recorded with 0.2.0 are unchanged.
  - The traced Al-Cu boundary within 0.3 K of the Al melting point is now
    resolved.

## 0.2.2 - 2026-10-10

### Added
- Streamlit Community Cloud deployment:
  - `streamlit_app.py` at the repository root runs the app from the
    package. It works with the package installed, or straight from `src/`.
  - `requirements.txt` holds the package (`-e .`) plus its runtime
    dependencies with `~=` pins.
  - `.streamlit/config.toml` sets a headless server and an accent colour
    only, so light and dark themes both work. `.streamlit/secrets.toml` is
    git-ignored.
- Precomputed diagrams (`precomputed.py`, `scripts/precompute.py`,
  `data/precomputed/<system>.npz`).
  - Each file holds the traced diagram, the invariant reactions and the
    default equilibrium at the app's default settings.
  - Its metadata records the source SHA256, package version, gas constant,
    temperature range, settings and computation time.
  - The app uses a file only while its metadata matches; otherwise it
    computes live and shows a notice.
  - Sizes: ag_cu 15.1 KiB, al_cu 12.6 KiB. Computation time: 1.2 s and
    1.3 s.
- Tests:
  - `tests/test_precomputed.py`: every app system's file is present and up
    to date, under 1 MB, round-trips, detects changed data or version, and
    (slow) equals a live computation;
  - `tests/test_requirements.py`: every runtime dependency in
    `pyproject.toml` is in `requirements.txt`, with `~=` pins, and no
    dev/validation tool is;
  - AppTest checks of the entry point, loading each system from precomputed
    data, switching systems and units, the live-computation notice for
    missing or stale data, the High resolution warning, and deferred
    exports.
- `scripts/smoke_test.py` and a `cloud-smoke` CI job: a fresh virtual
  environment with only `requirements.txt`, then the AppTest smoke test,
  with the cold-start time written to the job summary. The test job also
  runs `scripts/precompute.py --check`.
- "High resolution" option (240 temperature levels, 2000 Gibbs-curve
  points). It shows an estimated time, scaled from the precomputed run,
  before it runs.

### Changed
- The app's UI is a `main()` function. Systems are loaded with
  `st.cache_resource`. Live computations use `st.cache_data`, keyed by
  system, data hash, temperature range, settings and package version, with
  a spinner.
- Figure exports are rendered only when a download button is clicked, and
  matplotlib is imported on first use. Neither is in the start-up path any
  more.
- Text on the plot area is dark in both light and dark themes.
- `pytest` moved from the runtime dependencies to a `dev` extra. CI
  installs `.[dev]`.

### Performance (AppTest, local machine)

| | 0.2.1 | 0.2.2 |
|---|---|---|
| Cold start (first run, Ag-Cu) | 4.0 s | 1.5–1.7 s |
| Switch to Al-Cu | 1.6 s | 0.05 s |

In a fresh virtual environment installed only from `requirements.txt`, the
very first start took 43.8 s, spent loading the newly installed packages
for the first time. The next start took 1.49 s.

## 0.2.1 - 2026-10-10

### Added
- Compound Energy Formalism (sublattice model), in `thermo/cef.py` and
  `thermo/sublattice.py`. Phases may have any number of sublattices, site
  ratios and vacancies, for example FCC_A1 (FE)1(C,VA)1 and BCC_A2
  (FE)1(C,VA)3. The model includes:
  - a surface of reference;
  - ideal configurational entropy weighted by site ratios;
  - Redlich-Kister excess on any sublattice (reciprocal and ternary
    parameters at order 0, and wildcard `*` parameters);
  - analytic gradients.
- Compositions and Gibbs energies of sublattice phases are per mole of
  atoms, with vacancies excluded (documented in `docs/theory.md` and
  `docs/data_format.md`).
- Phases with internal degrees of freedom are minimised over their site
  fractions at fixed composition:
  - site fractions stay within (1e-12, 1);
  - one internal degree of freedom uses a vectorised search refined to
    |dG/dw| < 1e-9;
  - more than one uses SLSQP from several starts, with a convergence check.

  Results are cached per (T, x). dG/dx comes from the Lagrange multiplier.
- Inden-Hillert-Jarl magnetic model (`thermo/magnetic.py`):
  - p = 0.28 for FCC/HCP and 0.40 for BCC;
  - TC and BMAGN may depend on composition;
  - the TDB antiferromagnetic convention (negative TC/BMAGN divided by -3
    or -1) is handled;
  - the TDB parameter alias `BM` is accepted.
- More of the TDB format is read:
  - `SPECIES` and `TEMPERATURE_LIMITS`;
  - metadata keywords (`DATABASE_INFO`, `VERSION_DATE`, `ASSESSED_SYSTEMS`,
    `REFERENCE_FILE`, `LIST_OF_REFERENCES`, `ADD_REFERENCES`). These never
    raise; references are in `system.metadata["references"]`;
  - other metadata-only keywords are stored with a logged warning;
  - `P` in expressions;
  - quoted reference texts that contain `!`.

  Keywords that could affect the thermodynamics still raise
  `NotImplementedError` with the keyword and line number.
- Per-system gas constant (`gas_constant`) and pressure (`pressure_pa`),
  including `load_system(..., gas_constant=...)`. TDB systems default to
  R = 8.31451.
- `write_tdb` also writes sublattice and magnetic phases, and references.
- Synthetic TDB fixtures: `interstitial.tdb` ((A)1(C,VA)1 + (A)1(C,VA)3 +
  graphite-like C + liquid) and `magnetic.tdb` (ferro- and
  antiferromagnetic BCC/FCC with composition-dependent TC).
- Cross-validation against pycalphad now covers these fixtures and two more
  literature databases from pycalphad's test set:
  - Cu-Mg (Liang et al. 1998: Laves phase with internal ordering, wildcard
    parameters, pure-Mg HCP);
  - Fe-C (Hallstedt et al. 2010 with the Brosh equation of state: magnetic
    and interstitial phases, carbides, graphite).

  Results across 9 systems:
  - stable phases agree at 270/270 points;
  - fractions and compositions agree within 2e-5;
  - all 13 invariants agree within 0.005 K;
  - Gibbs energies at the same R agree within 2e-7 J/mol.

  Each comparison reports the gas constant it uses.
- `docs/theory.md`: equations for the sublattice and magnetic models.
- `tests/test_identical_to_0_2_0.py` checks every 0.2.0 system against
  equilibria recorded with 0.2.0. All agree within 1e-10 (observed within
  4e-15), and invariants agree to the root tolerance.

### Changed
- `SolutionPhase` and `StoichiometricPhase` are now special cases of the
  sublattice model, and every TDB phase is a sublattice phase
  (`model_type` "sublattice"). `build_system` no longer raises for magnetic
  or multi-sublattice phases. Only non-magnetic GES type definitions (e.g.
  `DIS_PART`) still raise `NotImplementedError`.
- JSON systems keep R = 8.314462618 unless they set `gas_constant`, so their
  results are unchanged. The exported TDB fixtures record this R in their
  `.meta.json`, and `write_metadata` writes the system's R.
- Phases may have a restricted composition range (for example
  (FE)1(C,VA)3 is limited to x <= 0.75), and pure-element phases are
  fixed-composition end points. The convex hull and the common-tangent
  solver work within each phase's range.
- Faster evaluation: parameters are evaluated once per temperature, and
  single-point evaluations use plain Python. Ag-Cu diagram plus invariants
  went from 1.92 s to 1.20 s (median of three runs).

  | System | Invariants | Trace | Total |
  |---|---|---|---|
  | Ag-Cu | 0.64 s | 0.56 s | 1.20 s |
  | Al-Cu | 0.76 s | 0.46 s | 1.22 s |
  | Interstitial (synthetic) | 0.94 s | 0.44 s | 1.38 s |
  | Magnetic (synthetic) | 1.86 s | 2.03 s | 3.89 s |
  | Cu-Mg (Laves with internal ordering) | 12.6 s | 9.8 s | 22.5 s |
  | Fe-C (Brosh EOS functions) | 7.6 s | 3.3 s | 10.9 s |

### Fixed
- A compound lying above a solution phase's curve was taken as evidence of
  a miscibility gap in that phase. Only the phase's own points now count,
  so a congruently melting compound no longer produces a false
  "LIQUID#1 + compound + LIQUID#2" reaction.

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
