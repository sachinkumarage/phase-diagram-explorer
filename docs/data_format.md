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

**`"sublattice"`** is a TDB phase that no model here evaluates yet (see
below). Its `sublattice_sites`, `constituents` and `raw_parameters` are kept
as read.

Phases read from a TDB file also carry:
- `sublattice_sites` and `constituents`;
- `magnetic_parameters`, holding the TC and BMAGN parameters;
- `type_definitions`, holding the GES TYPE_DEFINITION commands the phase
  uses.

## TDB (supported subset)

`load_system("x.tdb")` reads the binary subset of the TDB format.
`tdb.writer.write_tdb(system, path)` writes it. The CLI command
`phase-diagram-explorer export-tdb` writes a TDB file together with its
`.meta.json`.

### Syntax

- `$` starts a comment that runs to the end of the line.
- A command runs over as many lines as it needs, until its terminating `!`.
- Keywords may be abbreviated word by word, as in Thermo-Calc, for example
  `PARA`, `FUNCT`, `TYPE_DEF` or `DEF_SYS_DEF`.

| Keyword | Handling |
|---|---|
| `ELEMENT` | Symbol, reference state and mass. `/-` and `VA` are not chemical elements. |
| `FUNCTION` | A named piecewise function. It can be referenced from other functions and from parameters, with or without a trailing `#`. |
| `TYPE_DEFINITION` | Stored. Plain `SEQ` definitions are ignored. GES definitions are attached to the phases that use their code. |
| `PHASE` | Name (a `:L`-style suffix is dropped), type codes, and site counts per sublattice. |
| `CONSTITUENT` | Species on each sublattice. `%` markers are dropped. |
| `PARAMETER` | `G` and `L`, Redlich-Kister orders 0..n, are evaluated. `TC` and `BMAGN` are stored. |
| `DEFINE_SYSTEM_DEFAULT`, `DEFAULT_COMMAND` | Read and ignored. |

Any other keyword, such as `SPECIES`, `DATABASE_INFO` or
`LIST_OF_REFERENCES`, and any other parameter type, such as `MQ` or `NT`,
raises `NotImplementedError` naming it and its line number.

### Expressions

- Expressions use `+ - * / **`, parentheses, `T`, `R` (8.314462618), numbers
  (`1.5E+03` and Fortran `1.5D+03`), `LN()`, `LOG()` (also natural log, as
  in Thermo-Calc), `EXP()`, and FUNCTION names. Nothing else is accepted.
- They are parsed by a small recursive-descent parser and evaluated with
  numpy. `eval()` and `exec()` are never used.
- Function references are checked when the system is built. An undefined
  name or a circular reference is an error.

### Piecewise functions

    298.15  <expr>;  1234.93  Y  <expr>;  3000  N  <reference> !

- Each interval runs from its lower limit to the next limit.
- `Y` continues with another interval and `N` ends the function. The
  reference after `N` is dropped.
- As in Thermo-Calc:
  - an omitted lower limit is 298.15 K;
  - an omitted upper limit, or `,,`, is 6000 K;
  - a final limit without `Y` or `N` ends the function.
- Evaluating outside every interval raises `ValueError`.

### How phases map to models

- **One mixing sublattice holding both elements.** Any other sublattices
  hold only `VA`. This becomes `"solution"`.
  - G and L parameters are divided by the site count, which gives energies
    per mole of atoms.
  - An interaction listed as (dependent, base), for example
    `L(FCC_A1,CU,AG:VA;1)` with Ag as the base element, has its odd orders
    negated.
- **Every non-`VA` sublattice holds one element, and both elements occur.**
  This becomes `"stoichiometric"`. The single G parameter is divided by the
  total number of sites.
- **Anything else** becomes `"sublattice"`: mixing on several sublattices,
  or a phase of only one element. `build_system` raises
  `NotImplementedError` until the sublattice model exists.
- **Magnetic phases:** phases with TC or BMAGN parameters, or with a
  `MAGNETIC` type definition such as AFCC or ABCC. `build_system` raises
  `NotImplementedError("magnetic model: added in 0.2.1 ...")` rather than
  silently ignoring the magnetic contribution.

### Writing

`write_tdb` supports `"solution"` and `"stoichiometric"` phases. It writes
the following:

- `ELEMENT` lines for `/-`, `VA` and both elements, with their masses;
- the functions;
- a `% SEQ` type definition;
- one `PHASE` / `CONSTITUENT` / `PARAMETER` block per phase.

How energies are written:
- Coefficient energies become a single expression, valid over 1–10000 K
  unless they set `T_min` or `T_max`.
- A compound A_m B_n gets sublattices of m and n sites. Its G parameter is
  (m + n) times the per-atom energy.
- Numbers keep full precision, so reading the file back gives identical
  Gibbs energies.

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
  "_source": "...",
  "_status": "provisional",
  "_status_reason": "..."
}
```

| Field | Meaning |
|---|---|
| `elements` | `[base, dependent]`. This sets the composition axis, and its spelling ("Ag") is used for display. |
| `dependent_element` | Alternative to `elements`: the dependent element alone. |
| `name`, `element_names`, `t_range_k`, `_source`, `_status`, `_status_reason` | As in JSON. Optional. `name` defaults to "Base-Dependent". |

Arguments to `load_system(path, elements=[...])` or
`load_system(path, dependent_element="Cu")` take precedence over the
metadata. If neither the arguments nor the metadata give the element order,
`load_system` raises `ValueError` rather than guessing.

## Test fixtures

`tests/fixtures/*_provisional.tdb` are the provisional example systems,
exported with `write_tdb`, for testing the TDB reader and for
cross-validation against pycalphad. They are **not literature data**.
`regular_solution.tdb` and `gap_eutectic.tdb` are synthetic.
