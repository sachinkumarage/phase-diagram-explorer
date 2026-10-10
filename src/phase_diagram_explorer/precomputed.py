"""Precomputed phase diagrams for the app.

Tracing a diagram and finding its invariants takes seconds (Fe-C about 11 s,
Cu-Mg about 22 s locally), too long for an app start on a small cloud
machine. scripts/precompute.py stores, for each app system,

- the traced diagram over the system's analysis range at the app's default
  settings (DEFAULT_LEVELS temperature levels, DEFAULT_POINTS Gibbs-curve
  points),
- the invariant reactions over that range,
- the equilibrium at the app's default (T, x),

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
from phase_diagram_explorer.models import SystemDefinition, load_system, metadata_path, system_files
from phase_diagram_explorer.thermo.constants import CODATA_GAS_CONSTANT
from phase_diagram_explorer.tracing import DEFAULT_LEVELS, TracedDiagram, TracedField, trace_diagram

SUFFIX = ".npz"
DEFAULT_POINTS = 500
# Default composition of the app's equilibrium readout (mole fraction).
DEFAULT_X = 0.5
# Analysis range (K) for systems without t_range_k (the app's slider limits).
FALLBACK_T_RANGE_K = (200.0, 2000.0)
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
    """SHA256 of the system file, followed by its .meta.json for TDB files."""
    path = Path(system_path)
    digest = hashlib.sha256(path.read_bytes())
    meta = metadata_path(path)
    if path.suffix.lower() == ".tdb" and meta.exists():
        digest.update(meta.read_bytes())
    return digest.hexdigest()


def precomputed_path(system_path: str | Path, directory: str | Path | None = None) -> Path:
    return Path(directory or default_precomputed_dir()) / f"{Path(system_path).stem}{SUFFIX}"


def app_systems(systems_dir: str | Path) -> list[Path]:
    """The systems the app offers: every system file whose phases can all be evaluated."""
    return [path for path in system_files(systems_dir) if is_computable(load_system(path))]


def expected_metadata(system_path: str | Path, definition: SystemDefinition | None = None) -> dict:
    definition = definition or load_system(system_path)
    return {
        "system": Path(system_path).name,
        "source_sha256": source_hash(system_path),
        "package_version": __version__,
        "gas_constant": gas_constant(definition),
        "t_range_k": list(analysis_range(definition)),
        "n_levels": DEFAULT_LEVELS,
        "n_points": DEFAULT_POINTS,
    }


@dataclass
class Precomputed:
    metadata: dict
    traced: TracedDiagram
    invariants: list[InvariantReaction]
    default_equilibrium: EquilibriumResult

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
    return Precomputed(metadata, traced, invariants, equilibrium)


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


def to_bytes(precomputed: Precomputed) -> bytes:
    traced = precomputed.traced
    equilibrium = precomputed.default_equilibrium
    arrays = {
        "metadata": _text(precomputed.metadata),
        "T_range": np.array(traced.T_range, dtype=float),
        "levels": np.asarray(traced.levels, dtype=float),
        "field_phases": _text([list(field.phases) for field in traced.fields]),
        "trace_invariants": _text([_reaction_to_dict(r) for r in traced.invariants]),
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
    }
    for k, field in enumerate(traced.fields):
        arrays[f"field_{k}"] = np.array([field.T, field.x_min, field.x_max], dtype=float)
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

        fields = []
        for k, phases in enumerate(text("field_phases")):
            T, x_min, x_max = data[f"field_{k}"]
            fields.append(TracedField(tuple(phases), list(T), list(x_min), list(x_max)))
        traced = TracedDiagram(
            T_range=tuple(float(T) for T in data["T_range"]),
            fields=fields,
            invariants=[_reaction_from_dict(r) for r in text("trace_invariants")],
            levels=np.array(data["levels"]),
        )
        equilibrium = EquilibriumResult(**text("default_equilibrium"))
        return Precomputed(
            metadata=text("metadata"),
            traced=traced,
            invariants=[_reaction_from_dict(r) for r in text("invariants")],
            default_equilibrium=equilibrium,
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
