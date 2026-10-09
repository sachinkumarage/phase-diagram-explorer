"""Command line interface.

    phase-diagram-explorer plot --system ag_cu [--output ag_cu.pdf] [--unit C]
    phase-diagram-explorer export-tdb --system ag_cu --output ag_cu.tdb
    phase-diagram-explorer list

A --system value is a path to a system file, or the name of one (without
extension) in the systems directory ($PHASE_DIAGRAM_SYSTEMS_DIR, or
data/systems).
"""
import argparse
import sys
from pathlib import Path

from phase_diagram_explorer import __version__
from phase_diagram_explorer.models import default_systems_dir, load_system, system_files
from phase_diagram_explorer.units import CELSIUS, KELVIN

# Temperature range (K) for systems without "t_range_k".
DEFAULT_T_RANGE_K = (200.0, 2000.0)
FIGURE_FORMATS = ("svg", "pdf")


def resolve_system(value: str, systems_dir: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    matches = [candidate for candidate in system_files(systems_dir) if candidate.stem == value]
    if not matches:
        available = ", ".join(sorted({candidate.stem for candidate in system_files(systems_dir)})) or "none"
        raise SystemExit(f"error: no system {value!r} in {systems_dir} (available: {available})")
    return matches[0]


def _plot(arguments) -> None:
    from phase_diagram_explorer.builder import build_system
    from phase_diagram_explorer.export import export_figure
    from phase_diagram_explorer.tracing import trace_diagram

    path = resolve_system(arguments.system, arguments.systems_dir)
    definition = load_system(path)
    output = Path(arguments.output or f"{path.stem}_phase_diagram.{arguments.format or 'svg'}")
    fmt = arguments.format or output.suffix.lstrip(".").lower()
    if fmt not in FIGURE_FORMATS:
        raise SystemExit(f"error: unsupported figure format {fmt!r}; use one of {', '.join(FIGURE_FORMATS)}")

    T_range = tuple(arguments.t_range) if arguments.t_range else (definition.t_range_k or DEFAULT_T_RANGE_K)
    traced = trace_diagram(build_system(definition), T_range)
    output.write_bytes(
        export_figure(
            traced, definition.name, fmt,
            dependent_symbol=definition.dependent_element.symbol,
            temperature_unit=CELSIUS if arguments.unit == "C" else KELVIN,
            footnote=f"Data source: {path.name} ({definition.name}).",
        )
    )
    print(output)


def _export_tdb(arguments) -> None:
    from phase_diagram_explorer.tdb.writer import write_metadata, write_tdb

    path = resolve_system(arguments.system, arguments.systems_dir)
    definition = load_system(path)
    output = Path(arguments.output or f"{path.stem}.tdb")
    write_tdb(definition, output)
    print(output)
    print(write_metadata(definition, output))


def _list(arguments) -> None:
    for path in system_files(arguments.systems_dir):
        print(path.stem if path.suffix == ".json" else path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase-diagram-explorer",
        description="Explore and visualize thermodynamic phase diagrams",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--systems-dir", type=Path, default=default_systems_dir(), help="directory of system files")
    commands = parser.add_subparsers(dest="command")

    plot = commands.add_parser("plot", help="trace a phase diagram and write it as SVG or PDF")
    plot.add_argument("--system", required=True, help="system name in the systems directory, or a file path")
    plot.add_argument("--output", help="figure file (default: <system>_phase_diagram.svg)")
    plot.add_argument("--format", choices=FIGURE_FORMATS, help="figure format (default: from --output, else svg)")
    plot.add_argument("--unit", choices=("K", "C"), default="K", help="temperature unit")
    plot.add_argument("--t-range", nargs=2, type=float, metavar=("T_MIN", "T_MAX"), help="temperature range in K")
    plot.set_defaults(handler=_plot)

    export = commands.add_parser("export-tdb", help="write a system as a TDB file plus its .meta.json")
    export.add_argument("--system", required=True, help="system name in the systems directory, or a file path")
    export.add_argument("--output", help="TDB file (default: <system>.tdb)")
    export.set_defaults(handler=_export_tdb)

    listing = commands.add_parser("list", help="list the systems in the systems directory")
    listing.set_defaults(handler=_list)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
