"""Download every registered database (data/sources.json) from its official
URL into data/tdb/ and verify its size and SHA256.

    python scripts/fetch_tdb.py           # fetch missing files, verify all
    python scripts/fetch_tdb.py --check   # verify only, never download

Exits with status 1, and a message naming the file and the URL, if a
download fails or a file does not match.
"""
import argparse
import sys
from pathlib import Path

from phase_diagram_explorer.sources import SourceError, ensure, load_registry, tdb_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--directory", type=Path, default=None, help=f"target directory (default {tdb_dir()})")
    parser.add_argument("--check", action="store_true", help="verify existing files only")
    arguments = parser.parse_args(argv)

    failed = False
    for name, entry in load_registry().items():
        try:
            path = ensure(name, arguments.directory, download=not arguments.check)
            print(f"{name}: {path} ({entry.size} bytes, SHA256 verified)")
        except SourceError as error:
            print(f"ERROR {name}: {error}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
