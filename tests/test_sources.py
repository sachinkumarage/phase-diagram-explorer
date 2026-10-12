"""Downloaded databases (data/sources.json, sources.py, scripts/fetch_tdb.py).
Downloads are tested with local file:// URLs; the real NIST file is fetched
by scripts/fetch_tdb.py (CI runs it before the tests)."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from phase_diagram_explorer import sources
from phase_diagram_explorer.sources import SourceError, ensure, fetch, load_registry, verify

ROOT = Path(__file__).resolve().parents[1]
NIST_CITATION = (
    "Kattner, U.R. (2017), Phase Diagrams & Computational Thermodynamics - Solder Systems, NIST Standard "
    "Reference Database 139, National Institute of Standards and Technology, https://doi.org/10.18434/T4759N "
    "(Accessed 12 October 2026)."
)


def _local_source(tmp_path: Path, content: bytes = b"ELEMENT AG FCC_A1 107.87 0 0 !\n") -> sources.Source:
    origin = tmp_path / "origin" / "remote.tdb"
    origin.parent.mkdir()
    origin.write_bytes(content)
    return sources.Source(
        name="test_db", file="test_db.tdb", url=origin.as_uri(), sha256=hashlib.sha256(content).hexdigest(),
        size=len(content), citation="Test citation.", license_mode="fetch", license_url="https://example.org/license",
    )


def test_nist_solder_is_registered():
    entry = load_registry()["nist_solder"]
    assert entry.url == "https://www.metallurgy.nist.gov/phase/solder/NIST-solder.tdb"
    assert entry.sha256 == "56a0b25021f5dd4d2efbb9ed89ad8ad06929796e963c40ede857fcc9f72df8f3"
    assert entry.size == 19312
    assert entry.file == "nist_solder.tdb"
    assert entry.license_mode == "fetch"
    assert entry.license_url == "https://www.nist.gov/open/license"
    assert entry.citation == NIST_CITATION


def test_registered_database_is_present_and_verified():
    """Run "python scripts/fetch_tdb.py" (CI does) if this fails."""
    path = ensure("nist_solder", download=False)
    assert path == ROOT / "data" / "tdb" / "nist_solder.tdb"


def test_database_files_are_never_committed():
    assert "data/tdb/" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(["git", "ls-files", "data/tdb"], cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    assert tracked.stdout.strip() == ""


def test_fetch_downloads_and_verifies(tmp_path):
    entry = _local_source(tmp_path)
    path = fetch(entry, tmp_path / "tdb")
    assert path == tmp_path / "tdb" / "test_db.tdb"
    verify(path, entry)
    assert list((tmp_path / "tdb").iterdir()) == [path]  # no partial files left


def test_fetch_rejects_a_wrong_checksum_without_writing(tmp_path):
    entry = replace(_local_source(tmp_path), sha256="0" * 64)
    with pytest.raises(SourceError, match="SHA256"):
        fetch(entry, tmp_path / "tdb")
    assert list((tmp_path / "tdb").iterdir()) == []


def test_fetch_rejects_a_wrong_size(tmp_path):
    entry = replace(_local_source(tmp_path), size=1)
    with pytest.raises(SourceError, match="size"):
        fetch(entry, tmp_path / "tdb")


def test_fetch_retries_three_times_with_backoff(tmp_path, monkeypatch):
    attempts, sleeps = [], []

    def failing_urlopen(url, timeout):
        attempts.append(url)
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(sources.time, "sleep", sleeps.append)
    with pytest.raises(SourceError, match="after 3 attempts") as error:
        fetch(_local_source(tmp_path), tmp_path / "tdb")
    assert len(attempts) == 3 and sleeps == [2.0, 4.0]
    assert "Download it manually" in str(error.value)


def test_fetch_succeeds_after_a_transient_failure(tmp_path, monkeypatch):
    real_urlopen = urllib.request.urlopen
    calls = []

    def flaky_urlopen(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("temporary failure")
        return real_urlopen(url, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    assert fetch(_local_source(tmp_path), tmp_path / "tdb").exists()
    assert len(calls) == 2


def test_ensure_never_uses_a_modified_file(tmp_path, monkeypatch):
    entry = _local_source(tmp_path)
    registry = tmp_path / "sources.json"
    fields = {k: v for k, v in entry.__dict__.items() if k != "name"}
    registry.write_text(json.dumps({"test_db": fields}))
    monkeypatch.setenv("PHASE_DIAGRAM_SOURCES", str(registry))
    monkeypatch.setenv("PHASE_DIAGRAM_TDB_DIR", str(tmp_path / "tdb"))

    with pytest.raises(SourceError, match="missing"):
        ensure("test_db", download=False)
    path = ensure("test_db")  # fetched on first use
    assert path.read_bytes() == Path(entry.url.removeprefix("file://")).read_bytes()
    path.write_bytes(b"modified")
    with pytest.raises(SourceError, match="size"):
        ensure("test_db")
    with pytest.raises(SourceError, match="unknown data source"):
        ensure("nope")


@pytest.mark.slow
def test_fetch_script_fails_clearly(tmp_path):
    entry = replace(_local_source(tmp_path), url=(tmp_path / "missing.tdb").as_uri())
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps({"test_db": {k: v for k, v in entry.__dict__.items() if k != "name"}}))
    env = {"PATH": str(Path(sys.executable).parent), "PHASE_DIAGRAM_SOURCES": str(registry)}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_tdb.py"), "--directory", str(tmp_path / "tdb")],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 1
    assert "ERROR test_db" in result.stderr and "after 3 attempts" in result.stderr
