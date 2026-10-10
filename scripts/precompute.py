"""Precompute the app's phase diagrams (see phase_diagram_explorer.precomputed).

    python scripts/precompute.py            # regenerate every app system
    python scripts/precompute.py --check    # exit 1 if any file is missing or stale

Regenerate whenever system data or the engine change.
"""
import argparse
import sys
from pathlib import Path

from phase_diagram_explorer.models import default_systems_dir
from phase_diagram_explorer.precomputed import (
    app_systems,
    compute,
    default_precomputed_dir,
    precomputed_path,
    save,
    status,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--systems-dir", type=Path, default=default_systems_dir())
    parser.add_argument("--output-dir", type=Path, default=default_precomputed_dir())
    parser.add_argument("--check", action="store_true", help="only check that every file is up to date")
    arguments = parser.parse_args(argv)

    problems = 0
    for system_path in app_systems(arguments.systems_dir):
        if arguments.check:
            _, reason = status(system_path, arguments.output_dir)
            print(f"{system_path.name}: {reason or 'up to date'}")
            problems += reason is not None
            continue
        precomputed = compute(system_path)
        path = save(precomputed, precomputed_path(system_path, arguments.output_dir))
        print(
            f"{system_path.name}: {path} ({path.stat().st_size / 1024:.1f} KiB, "
            f"computed in {precomputed.metadata['computation_time_s']:.2f} s)"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
