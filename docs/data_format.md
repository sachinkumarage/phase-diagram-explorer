# System data formats

A binary system is loaded with `models.load_system(path)` from one of:

- a JSON (or YAML) system file, which holds everything in one file;
- a TDB database plus an optional `<name>.meta.json` metadata file.

The app and the CLI list every `.json`, `.yaml`, `.yml` and `.tdb` file in
the systems directory (`data/systems/`, or `$PHASE_DIAGRAM_SYSTEMS_DIR`),
except `*.meta.json`. They offer only systems whose every phase can be
evaluated (`builder.is_computable`).

Conventions for every format:

- Temperatures are in K. Energies are in J/mol of atoms.
- The composition axis is the mole fraction x of the **dependent element**,
  the second element. x = 0 is the pure **base element**, the first element.

## JSON

```json
{
  "name": "Ag-Cu",
  "_source": "Free-text description of where the data come from.",
  "_status": "provisional",
  "_status_reason": "Why the data are not yet validated.",
  "elements": [
    {"symbol": "Ag", "name": "Silver", "reference_state": "fcc_a1", "atomic_mass": 107.8682},
    {"symbol": "Cu", "name": "Copper", "reference_state": "fcc_a1", "atomic_mass": 63.546}
  ],
  "t_range_k": [500.0, 1450.0],
  "phases": [
    {
      "name": "LIQUID",
      "model_type": "solution",
      "end_members": {
        "Ag": {"a": 11090.0, "b": -8.980266087956402},
        "Cu": {"a": 13050.0, "b": -9.611348019178505}
      },
      "interaction_parameters": [-1000.0, 500.0]
    },
    {
      "name": "AL2CU",
      "model_type": "stoichiometric",
      "stoichiometry": {"Al": 2, "Cu": 1},
      "formation": {"a": -12000.0}
    }
  ]
}
```

(The AL2CU phase is shown only to illustrate the format. A real system lists
phases of its own two elements.)

| Field | Meaning |
|---|---|
| `name` | Display name. |
| `elements` | Exactly two elements: base first, then dependent. `atomic_mass` (g/mol) is optional and enables wt% display. |
| `t_range_k` | Optional `[T_min, T_max]`. Invariants are always found over this range, and it is the default display range. |
| `_source`, `_status`, `_status_reason` | Optional provenance. `"_status": "provisional"` shows a preview banner. |

### Gibbs energies

A Gibbs energy is either a coefficient object or a piecewise expression.

- **Coefficient object:** G(T) = a + bT + cT ln T + dT² + eT³ + f/T. Every
  coefficient defaults to 0. Optional `T_min` and `T_max` make evaluation
  outside that range an error.
- **Piecewise expression:** `{"intervals": [{"T_min": ..., "T_max": ...,
  "expression": "..."}]}`. The expressions use the TDB syntax described
  below. TDB-loaded systems use this form.

### Phase models

**`"solution"`** is a substitutional solution:

    G = x_A G_A + x_B G_B + RT (x_A ln x_A + x_B ln x_B) + x_A x_B Σ_v L_v (x_A − x_B)^v

- `end_members` maps each element symbol to that element's Gibbs energy in
  this phase.
- `interaction_parameters` lists L_0, L_1, ... Each is a number or a
  piecewise expression.
- A is the base element and B the dependent element, so odd orders change
  sign if the element order is swapped.

**`"stoichiometric"`** is a compound with a fixed composition.

- `stoichiometry` gives the moles of each element per formula unit. The
  compound sits at x = n_B / (n_A + n_B).
- `formation` is its Gibbs energy **per mole of atoms**, on the same
  reference as the end members. The JSON example systems take each element's
  own stable solid as G = 0, so this is the energy of formation.

**`"sublattice"`** is a Compound Energy Formalism phase. Every phase read
from a TDB file has this type (see [theory.md](theory.md)).
- `sublattice_sites` gives the site ratio of each sublattice, and
  `constituents` gives the species on each one. Species are element symbols
  as written in the TDB file, plus `VA`, the vacancy.
- `raw_parameters` holds the G and L parameters per mole of formula units.
- `magnetic_parameters` holds TC and BMAGN. `magnetic`, which gives the
  `afm_factor` and `structure_factor`, comes from the phase's MAGNETIC type
  definition.
- `type_definitions` keeps the GES TYPE_DEFINITION commands the phase uses,
  as read.

"solution" and "stoichiometric" are special cases of the same model:
- a solution is the single sublattice (A,B)1;
- a compound is (A)m(B)n.

They give the same results as before 0.2.1.

### Gas constant

`gas_constant` (J/(mol K)) sets R for ideal mixing, for the magnetic term
and for `R` in expressions.
- JSON systems that do not set it use 8.314462618, as every system did
  before 0.2.1, so their results are unchanged.
- TDB systems default to 8.31451, the usual database convention, unless
  their `.meta.json` sets `gas_constant`.
- `load_system(path, gas_constant=...)` overrides either default.
- `pressure_pa` (default 101325) is the value of `P` in TDB expressions.

### Composition of sublattice phases

The composition axis is a mole fraction **per mole of atoms**. Vacancies are
not atoms, so for site ratios a_s and site fractions y_si:

    x_dep = sum_s a_s y_s,dep / N,    N = sum_s a_s (1 - y_s,VA)

and every Gibbs energy is per mole of atoms, G_formula / N. For example,
BCC_A2 (FE)1(C,VA)3 with y_C = 0.1 has x_C = 0.3 / 1.3 = 0.231, and its
composition range is 0 to 0.75.

## TDB

`load_system("x.tdb")` reads binary TDB databases.
`tdb.writer.write_tdb(system, path)` writes them. The CLI command
`phase-diagram-explorer export-tdb` writes a TDB file together with its
`.meta.json`.

### Syntax

- `$` starts a comment that runs to the end of the line.
- A command runs over as many lines as it needs, until its terminating `!`.
- In metadata commands, single-quoted text (reference texts) may contain
  `!` and `$`. Elsewhere `'` is an ordinary character, for example a
  TYPE_DEFINITION code.
- Keywords may be abbreviated word by word, as in Thermo-Calc, for example
  `PARA`, `FUNCT`, `TYPE_DEF`, `DEF_SYS_DEF` or `LIST_OF_REF`. `TEMP-LIM` and
  `DATABASE_INFORMATION` are also accepted.

**Thermodynamic keywords**

| Keyword | Handling |
|---|---|
| `ELEMENT` | Symbol, reference state and mass. `/-` and `VA` are not chemical elements. |
| `SPECIES` | Parsed and stored (`name`, formula). As a constituent, only element species and `VA` are supported: another species raises `NotImplementedError` naming it. |
| `FUNCTION` | A named piecewise function. It can be referenced from other functions and from parameters, with or without a trailing `#`. |
| `TYPE_DEFINITION` | `MAGNETIC` definitions give the phase's antiferromagnetic and structure factors. Plain `SEQ` definitions are ignored. Other GES definitions (e.g. `DIS_PART`) are stored, and `build_system` raises `NotImplementedError` for them. |
| `PHASE` | Name (a `:L`-style suffix is dropped), type codes, and site ratios. |
| `CONSTITUENT` | Species on each sublattice. `%` markers are dropped. |
| `PARAMETER` | `G` and `L`, Redlich-Kister orders 0..n on any sublattice; `TC` and `BMAGN` (alias `BM`) for the magnetic model. `*` is a wildcard sublattice. |
| `TEMPERATURE_LIMITS` | Default lower and upper limits for later functions and parameters. |
| `DEFINE_SYSTEM_DEFAULT`, `DEFAULT_COMMAND` | Read and ignored. |

**Metadata keywords** are stored and never raise:

| Keyword | Stored as |
|---|---|
| `LIST_OF_REFERENCES`, `ADD_REFERENCES` | `system.metadata["references"]`, as `{id: text}` |
| `DATABASE_INFO` | `system.metadata["database_info"]` |
| `VERSION_DATE`, `VERSION_DATA`, `ASSESSED_SYSTEMS`, `REFERENCE_FILE` | `system.metadata["version_date"]` etc. |

Some other keywords carry no Gibbs energy data:
`ZERO_VOLUME_SPECIES`, `DIFFUSION`, `DATABASE_TITLE`, `DATABASE_VERSION`,
and any name containing `REFERENCE` or `INFO`. These are stored under
`metadata["ignored"]`, and a warning is logged.

Every other keyword might affect the thermodynamics, for example
`ADD_CONSTITUENT`, `TABLE` or `FTP_FILE`. So might any other parameter type,
for example `MQ` or `NT`. These raise `NotImplementedError` naming the
keyword or type and its line number.

### Expressions

- Expressions use `+ - * / **`, parentheses, `T`, `P` (the system's pressure),
  `R` (the system's gas constant), numbers (`1.5E+03` and Fortran `1.5D+03`),
  `LN()`, `LOG()` (also natural log, as in Thermo-Calc), `EXP()`, and
  FUNCTION names. Nothing else is accepted.
- They are parsed by a small recursive-descent parser and evaluated with
  numpy. `eval()` and `exec()` are never used.
- Function references are checked when the system is built. An undefined
  name or a circular reference is an error.
- For a single temperature, each FUNCTION is evaluated once and cached.

### Piecewise functions

    298.15  <expr>;  1234.93  Y  <expr>;  3000  N  <reference> !

- Each interval runs from its lower limit to the next limit.
- `Y` continues with another interval and `N` ends the function. The
  reference after `N` is dropped.
- As in Thermo-Calc:
  - an omitted lower limit is 298.15 K;
  - an omitted upper limit, or `,,`, is 6000 K (both are changed by
    `TEMPERATURE_LIMITS`);
  - a final limit without `Y` or `N` ends the function.
- Evaluating outside every interval raises `ValueError`.

### How TDB phases are evaluated

Every phase is a sublattice phase, and its constituents must be the two
elements or `VA`. How it is solved depends on how many site fractions are
free (see [theory.md](theory.md)):
- **fixed:** every sublattice holds one species, as in a compound or a
  pure-element phase such as graphite;
- **direct:** one sublattice mixes two species, as in a substitutional
  solution or an interstitial (FE)1(C,VA)3;
- **minimise:** there are internal degrees of freedom, as in ordering
  (CU,MG)2(CU,MG)1. G(x) is the minimum over the site fractions.

Magnetic phases add the Inden-Hillert-Jarl term. A phase with TC or BMAGN
parameters but no MAGNETIC type definition gets no magnetic contribution,
as in pycalphad, and a warning is logged.

### Writing

`write_tdb` supports `"solution"`, `"stoichiometric"` and `"sublattice"`
phases. It writes:

- `ELEMENT` lines for `/-`, `VA` and both elements, with their masses;
- the functions;
- a `% SEQ` type definition;
- one `PHASE` / `CONSTITUENT` / `PARAMETER` block per phase, with a
  MAGNETIC type definition for magnetic phases;
- the references, as `LIST_OF_REFERENCES`.

How energies are written:
- Coefficient energies become a single expression, valid over 1–10000 K
  unless they set `T_min` or `T_max`.
- A compound A_m B_n gets sublattices of m and n sites. Its G parameter is
  (m + n) times the per-atom energy.
- Numbers keep full precision, so reading the file back gives identical
  Gibbs energies.
- The gas constant is not part of the TDB format. `write_metadata` stores it
  in the `.meta.json`.

## `<name>.meta.json`

The TDB format does not hold the element order or display information. They
are read from a metadata file next to the TDB file: `ag_cu.meta.json` for
`ag_cu.tdb`.

```json
{
  "name": "Ag-Cu",
  "elements": ["Ag", "Cu"],
  "element_names": {"Ag": "Silver", "Cu": "Copper"},
  "t_range_k": [500.0, 1450.0],
  "gas_constant": 8.314462618,
  "_source": "...",
  "_status": "provisional",
  "_status_reason": "..."
}
```

| Field | Meaning |
|---|---|
| `elements` | `[base, dependent]`. This sets the composition axis, and its spelling ("Ag") is used for display. |
| `dependent_element` | Alternative to `elements`: the dependent element alone. |
| `gas_constant` | R for this system. Optional; the TDB default is 8.31451. |
| `pressure_pa` | Pressure for `P` in expressions. Optional; the default is 101325. |
| `name`, `element_names`, `t_range_k`, `_source`, `_status`, `_status_reason` | As in JSON. Optional. `name` defaults to "Base-Dependent". |

Arguments to `load_system(path, elements=[...])` or
`load_system(path, dependent_element="Cu")` take precedence over the
metadata. If neither the arguments nor the metadata give the element order,
`load_system` raises `ValueError` rather than guessing.

## Test fixtures

`tests/fixtures/*_provisional.tdb` are the provisional example systems,
exported with `write_tdb`, for testing the TDB reader and for
cross-validation against pycalphad. They are **not literature data**.
`regular_solution.tdb` and `gap_eutectic.tdb` are synthetic, and so are:
- `interstitial.tdb`: (A)1(C,VA)1, (A)1(C,VA)3, graphite-like C and a
  liquid;
- `magnetic.tdb`: ferro- and antiferromagnetic BCC/FCC with a
  composition-dependent TC.

`reference_equilibria_0.2.0.json` holds equilibria computed with 0.2.0. A
test checks that 0.2.1 reproduces them.
