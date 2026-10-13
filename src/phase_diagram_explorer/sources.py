"""Thermodynamic databases that are downloaded, not redistributed.

data/sources.json registers each database: its official URL, SHA256 and
size, citation and license. fetch() downloads a database into data/tdb/
(3 attempts with backoff), and accepts it only if both the size and the
SHA256 match; ensure() returns the local file, fetching it if it is
missing. A mismatch or a failed download raises SourceError; nothing falls
back silently to other data. Requests carry a descriptive User-Agent
(user_agent()): NIST's server refuses Python's default one with HTTP 403.
Setting PHASE_DIAGRAM_DISABLE_DOWNLOADS=1 refuses all downloads.

    python scripts/fetch_tdb.py      # fetch and verify every registered database
"""
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETRIES = 3
BACKOFF_S = 2.0
TIMEOUT_S = 60.0
PROJECT_URL = "https://github.com/sachinkumarage/phase-diagram-explorer"
# Set to "1" to refuse all downloads (only files already in data/tdb/ are used).
DISABLE_DOWNLOADS_ENV = "PHASE_DIAGRAM_DISABLE_DOWNLOADS"


def user_agent() -> str:
    """Descriptive User-Agent for downloads. Some servers (NIST's among
    them) refuse Python's default "Python-urllib/3.x" with HTTP 403."""
    from phase_diagram_explorer import __version__

    return f"phase-diagram-explorer/{__version__} (+{PROJECT_URL})"


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    name: str
    file: str
    url: str
    sha256: str
    size: int
    citation: str
    license_mode: str
    license_url: str
    title: str = ""
    short_name: str = ""  # e.g. "NIST", for messages
    doi: str = ""
    page: str = ""
    note: str = ""


def registry_path() -> Path:
    """$PHASE_DIAGRAM_SOURCES, or data/sources.json in the source tree."""
    return Path(os.environ.get("PHASE_DIAGRAM_SOURCES", ROOT / "data" / "sources.json"))


def tdb_dir() -> Path:
    """$PHASE_DIAGRAM_TDB_DIR, or data/tdb in the source tree."""
    return Path(os.environ.get("PHASE_DIAGRAM_TDB_DIR", ROOT / "data" / "tdb"))


def load_registry(path: str | Path | None = None) -> dict[str, Source]:
    data = json.loads(Path(path or registry_path()).read_text())
    return {name: Source(name=name, **entry) for name, entry in data.items()}


def source(name: str) -> Source:
    registry = load_registry()
    if name not in registry:
        raise SourceError(f"unknown data source {name!r}; registered: {', '.join(registry) or 'none'}")
    return registry[name]


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path: str | Path, entry: Source) -> None:
    """Raise SourceError unless `path` has the registered size and SHA256."""
    path = Path(path)
    size = path.stat().st_size
    if size != entry.size:
        raise SourceError(f"{path}: size {size} bytes, expected {entry.size} for {entry.name} ({entry.url})")
    digest = sha256_of(path)
    if digest != entry.sha256:
        raise SourceError(f"{path}: SHA256 {digest}, expected {entry.sha256} for {entry.name} ({entry.url})")


def fetch(
    entry: Source, directory: str | Path | None = None, retries: int = RETRIES, backoff: float | None = None
) -> Path:
    """Download `entry` into `directory` (default tdb_dir()) and verify it.
    The file is written only once verified."""
    backoff = BACKOFF_S if backoff is None else backoff
    directory = Path(directory or tdb_dir())
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entry.file
    errors = []
    if os.environ.get(DISABLE_DOWNLOADS_ENV) == "1":
        raise SourceError(f"downloads are disabled ({DISABLE_DOWNLOADS_ENV}=1); {entry.name} not fetched from {entry.url}")
    request = urllib.request.Request(entry.url, headers={"User-Agent": user_agent()})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                content = response.read()
        except OSError as error:
            errors.append(f"attempt {attempt}: {error}")
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
            continue
        with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".part") as handle:
            handle.write(content)
        try:
            verify(handle.name, entry)
        except SourceError:
            Path(handle.name).unlink()
            raise
        os.replace(handle.name, target)
        return target
    raise SourceError(
        f"could not download {entry.name} from {entry.url} after {retries} attempts "
        f"({'; '.join(errors)}). Download it manually to {target} and check its SHA256 is {entry.sha256}."
    )


def ensure(name: str, directory: str | Path | None = None, download: bool = True) -> Path:
    """The verified local file of a registered database, fetched if missing."""
    entry = source(name)
    path = Path(directory or tdb_dir()) / entry.file
    if not path.exists():
        if not download:
            raise SourceError(f"{path} is missing; run: python scripts/fetch_tdb.py")
        return fetch(entry, path.parent)
    verify(path, entry)
    return path
