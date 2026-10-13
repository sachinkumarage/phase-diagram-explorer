"""Precomputed phase diagrams for the app.

Tracing a diagram and finding its invariants takes seconds (Fe-C about 11 s,
Cu-Mg about 22 s locally), too long for an app start on a small cloud
machine. scripts/precompute.py stores, for each app system,

- the traced diagram over the system's analysis range at the app's default
  settings (DEFAULT_LEVELS temperature levels, DEFAULT_POINTS Gibbs-curve
  points),
- the invariant reactions over that range,
- the equilibrium at the app's default (T, x),
- an equilibrium table: the same diagram traced on levels at most
  TABLE_STEP_K apart, from which lookup.equilibrium_from_trace reads the
  equilibrium at any (T, x) to within 1e-4 without the database,
- a summary of the system definition (names, elements and molar masses,
  phase names, citations, display settings; no thermodynamic parameters),
  so that the app can show a system whose database cannot be downloaded,

in data/precomputed/<system>.npz, with metadata: the SHA256 of the source
data file (and of its .meta.json for TDB systems), the package version, the
gas constant, the temperature range, the settings and the computation time.

A file is used only while its metadata matches the current data and package
version (status()); otherwise the app computes live. Regenerate with

    python scripts/precompute.py

whenever system data or the engine change (the version is bumped with every
engine change, which makes every file stale).
"""
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_diagram_explorer import __version__
from phase_diagram_explorer.builder import build_system, is_computable
from phase_diagram_explorer.equilibrium.equilibrium import EquilibriumResult, compute_equilibrium
from phase_diagram_explorer.invariants import InvariantReaction, detect_invariants_over_range
from phase_diagram_explorer.models import (
    Phase,
    SystemDefinition,
    is_sourced_system,
    load_system,
    metadata_path,
    system_files,
    system_name,
)
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT, DATABASE_GAS_CONSTANT
from phase_diagram_explorer.tracing import DEFAULT_LEVELS, TracedDiagram, TracedField, trace_diagram

SUFFIX = ".npz"
DEFAULT_POINTS = 500
# Default composition of the app's equilibrium readout (mole fraction).
DEFAULT_X = 0.5
# Analysis range (K) for systems without t_range_k (the app's slider limits).
FALLBACK_T_RANGE_K = (200.0, 2000.0)
# Largest spacing (K) of the equilibrium table's temperature levels: linear
# interpolation between them reproduces compute_equilibrium to about 1e-5
# in compositions and phase fractions (measured: Ag-Cu, Pb-Sn).
TABLE_STEP_K = 0.25
# Offsets (K) of extra table levels around the start and end of each
# two-phase field (see equilibrium_table).
TABLE_END_OFFSETS_K = (0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3)
# Metadata that must match for a file to be used.
CHECKED_FIELDS = ("source_sha256", "package_version", "gas_constant", "t_range_k", "n_levels", "n_points")


def default_precomputed_dir() -> Path:
    """$PHASE_DIAGRAM_PRECOMPUTED_DIR, or data/precomputed in the source tree."""
    return Path(
        os.environ.get("PHASE_DIAGRAM_PRECOMPUTED_DIR", Path(__file__).resolve().parents[2] / "data" / "precomputed")
    )


def analysis_range(definition: SystemDefinition) -> tuple[float, float]:
    return tuple(float(T) for T in definition.t_range_k) if definition.t_range_k else FALLBACK_T_RANGE_K


def default_temperature(definition: SystemDefinition) -> float:
    """The app's default temperature for the equilibrium readout: the middle
    of the analysis range."""
    T_min, T_max = analysis_range(definition)
    return (T_min + T_max) / 2.0


def gas_constant(definition: SystemDefinition) -> float:
    return definition.gas_constant if definition.gas_constant is not None else CODATA_GAS_CONSTANT


def source_hash(system_path: str | Path) -> str:
    """SHA256 of the system's data: the system file, followed by its
    .meta.json for TDB files. For a system whose database is downloaded
    (a .meta.json naming a data source) it combines the .meta.json with the
    registered SHA256 of the database file, so that a different database
    makes the precomputed data stale without the file being present."""
    path = Path(system_path)
    digest = hashlib.sha256(path.read_bytes())
    if is_sourced_system(path):
        from phase_diagram_explorer.sources import source

        digest.update(source(json.loads(path.read_text())["source"]).sha256.encode())
        return digest.hexdigest()
    meta = metadata_path(path)
    if path.suffix.lower() == ".tdb" and meta.exists():
        digest.update(meta.read_bytes())
    return digest.hexdigest()


def precomputed_path(system_path: str | Path, directory: str | Path | None = None) -> Path:
    return Path(directory or default_precomputed_dir()) / f"{system_name(system_path)}{SUFFIX}"


def app_systems(systems_dir: str | Path) -> list[Path]:
    """The systems the app offers: every system file whose phases can all be evaluated."""
    return [path for path in system_files(systems_dir) if is_computable(load_system(path))]


def _definition_without_database(system_path: str | Path) -> SystemDefinition | dict:
    """What expected_metadata needs, without loading (or downloading) a
    sourced system's database: its .meta.json has the range and gas constant."""
    if is_sourced_system(system_path):
        meta = json.loads(Path(system_path).read_text())
        return {"t_range_k": meta.get("t_range_k"), "gas_constant": meta.get("gas_constant", DATABASE_GAS_CONSTANT)}
    return load_system(system_path)


def expected_metadata(system_path: str | Path, definition: SystemDefinition | None = None) -> dict:
    """The metadata a precomputed file must have to be used. Never needs a
    sourced system's database."""
    definition = definition or _definition_without_database(system_path)
    if isinstance(definition, dict):
        T_range = list(definition["t_range_k"]) if definition["t_range_k"] else list(FALLBACK_T_RANGE_K)
        R = definition["gas_constant"]
    else:
        T_range, R = list(analysis_range(definition)), gas_constant(definition)
    return {
        "system": Path(system_path).name,
        "source_sha256": source_hash(system_path),
        "package_version": __version__,
        "gas_constant": R,
        "t_range_k": [float(T) for T in T_range],
        "n_levels": DEFAULT_LEVELS,
        "n_points": DEFAULT_POINTS,
    }


def definition_summary(definition: SystemDefinition) -> dict:
    """Everything the app shows about a system except its thermodynamic
    parameters (no Gibbs energy expressions are stored)."""
    return {
        "name": definition.name,
        "_source": definition.source,
        "_status": definition.status,
        "_status_reason": definition.status_reason,
        "elements": [element.model_dump() for element in definition.elements],
        "phases": [phase.name for phase in definition.phases],
        "t_range_k": list(definition.t_range_k) if definition.t_range_k else None,
        "gas_constant": definition.gas_constant,
        "data_source": definition.data_source,
        "citations": [citation.model_dump() for citation in definition.citations],
        "display": definition.display,
        "industry_note": definition.industry_note,
    }


def summary_definition(summary: dict) -> SystemDefinition:
    """A SystemDefinition rebuilt from definition_summary: names, elements,
    phases (without parameters), citations and display settings."""
    data = dict(summary)
    data["phases"] = [Phase(name=name, model_type="sublattice") for name in summary["phases"]]
    return SystemDefinition.model_validate(data)


@dataclass
class Precomputed:
    metadata: dict
    traced: TracedDiagram
    invariants: list[InvariantReaction]
    default_equilibrium: EquilibriumResult
    table: TracedDiagram | None = None
    summary: dict | None = None

    @property
    def definition(self) -> SystemDefinition | None:
        return summary_definition(self.summary) if self.summary else None

    def stale_fields(self, expected: dict) -> list[str]:
        return [key for key in CHECKED_FIELDS if self.metadata.get(key) != expected.get(key)]


def compute(system_path: str | Path) -> Precomputed:
    """Everything the app shows at its default settings, timed."""
    definition = load_system(system_path)
    start = time.perf_counter()
    system = build_system(definition)
    T_range = analysis_range(definition)
    invariants = detect_invariants_over_range(system, T_range, n_points=DEFAULT_POINTS)
    traced = trace_diagram(system, T_range, invariants=invariants, n_levels=DEFAULT_LEVELS, n_points=DEFAULT_POINTS)
    equilibrium = compute_equilibrium(system, default_temperature(definition), DEFAULT_X, n_points=DEFAULT_POINTS)
    metadata = expected_metadata(system_path, definition)
    metadata["computation_time_s"] = round(time.perf_counter() - start, 3)
    table = equilibrium_table(system, T_range, invariants)
    metadata["table_step_k"] = TABLE_STEP_K
    return Precomputed(metadata, traced, invariants, equilibrium, table, definition_summary(definition))


def equilibrium_table(system: dict, T_range: tuple[float, float], invariants) -> TracedDiagram:
    """The traced diagram behind lookup.equilibrium_from_trace: levels at
    most TABLE_STEP_K apart, plus levels at TABLE_END_OFFSETS_K on both sides
    of every temperature where a two-phase field starts or ends (e.g. a
    pure-element melting point). Near those temperatures a field is narrow,
    so the lever rule amplifies interpolation errors; the extra levels keep
    phase fractions within 1e-4 of compute_equilibrium there too."""
    n_levels = int(np.ceil((T_range[1] - T_range[0]) / TABLE_STEP_K)) + 1
    coarse = trace_diagram(system, T_range, invariants=invariants, n_levels=n_levels, n_points=DEFAULT_POINTS)
    ends = {T for field in coarse.fields if field.is_two_phase for T in (field.T[0], field.T[-1])}
    ends -= {float(T_range[0]), float(T_range[1])}
    extra = [T + sign * offset for T in sorted(ends) for offset in TABLE_END_OFFSETS_K for sign in (-1.0, 1.0)]
    return trace_diagram(
        system, T_range, invariants=invariants, n_levels=n_levels, n_points=DEFAULT_POINTS, extra_levels=extra
    )


def _reaction_to_dict(reaction: InvariantReaction) -> dict:
    return {
        "temperature": reaction.temperature,
        "type": reaction.type,
        "phases": list(reaction.phases),
        "composition": reaction.composition,
    }


def _reaction_from_dict(data: dict) -> InvariantReaction:
    return InvariantReaction(data["temperature"], data["type"], tuple(data["phases"]), dict(data["composition"]))


def _text(value) -> np.ndarray:
    return np.array(json.dumps(value))


def _traced_arrays(traced: TracedDiagram, prefix: str) -> dict:
    arrays = {
        f"{prefix}T_range": np.array(traced.T_range, dtype=float),
        f"{prefix}levels": np.asarray(traced.levels, dtype=float),
        f"{prefix}field_phases": _text([list(field.phases) for field in traced.fields]),
        f"{prefix}trace_invariants": _text([_reaction_to_dict(r) for r in traced.invariants]),
    }
    for k, field in enumerate(traced.fields):
        arrays[f"{prefix}field_{k}"] = np.array([field.T, field.x_min, field.x_max], dtype=float)
    return arrays


def _traced_from(data, prefix: str) -> TracedDiagram:
    fields = []
    for k, phases in enumerate(json.loads(str(data[f"{prefix}field_phases"]))):
        T, x_min, x_max = data[f"{prefix}field_{k}"]
        fields.append(TracedField(tuple(phases), list(T), list(x_min), list(x_max)))
    return TracedDiagram(
        T_range=tuple(float(T) for T in data[f"{prefix}T_range"]),
        fields=fields,
        invariants=[_reaction_from_dict(r) for r in json.loads(str(data[f"{prefix}trace_invariants"]))],
        levels=np.array(data[f"{prefix}levels"]),
    )


TABLE_PREFIX = "table_"


def to_bytes(precomputed: Precomputed) -> bytes:
    equilibrium = precomputed.default_equilibrium
    arrays = {
        "metadata": _text(precomputed.metadata),
        "invariants": _text([_reaction_to_dict(r) for r in precomputed.invariants]),
        "default_equilibrium": _text(
            {
                "T": equilibrium.T,
                "x_overall": equilibrium.x_overall,
                "stable_phases": equilibrium.stable_phases,
                "phase_compositions": equilibrium.phase_compositions,
                "phase_fractions": equilibrium.phase_fractions,
                "total_gibbs": equilibrium.total_gibbs,
            }
        ),
        **_traced_arrays(precomputed.traced, ""),
    }
    if precomputed.table is not None:
        arrays.update(_traced_arrays(precomputed.table, TABLE_PREFIX))
    if precomputed.summary is not None:
        arrays["summary"] = _text(precomputed.summary)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def save(precomputed: Precomputed, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(precomputed))
    return path


def load(path: str | Path) -> Precomputed:
    with np.load(path, allow_pickle=False) as data:
        def text(key):
            return json.loads(str(data[key]))

        return Precomputed(
            metadata=text("metadata"),
            traced=_traced_from(data, ""),
            invariants=[_reaction_from_dict(r) for r in text("invariants")],
            default_equilibrium=EquilibriumResult(**text("default_equilibrium")),
            table=_traced_from(data, TABLE_PREFIX) if f"{TABLE_PREFIX}levels" in data.files else None,
            summary=text("summary") if "summary" in data.files else None,
        )


def status(system_path: str | Path, directory: str | Path | None = None) -> tuple[Precomputed | None, str | None]:
    """(precomputed result, None) if an up-to-date file exists, else
    (None, reason)."""
    path = precomputed_path(system_path, directory)
    if not path.exists():
        return None, f"no precomputed file {path.name}"
    try:
        precomputed = load(path)
    except Exception as error:  # a damaged or foreign file is treated as missing
        return None, f"{path.name} could not be read ({type(error).__name__})"
    stale = precomputed.stale_fields(expected_metadata(system_path))
    if stale:
        return None, f"{path.name} is out of date ({', '.join(stale)} changed)"
    return precomputed, None
