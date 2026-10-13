"""Real downloads from the official URLs (marker "network"; run with
pytest -m network). These go through the same code path as
scripts/fetch_tdb.py and the app, into a temporary directory."""
import pytest

from phase_diagram_explorer.sources import fetch, load_registry, sha256_of, user_agent

pytestmark = pytest.mark.network


@pytest.mark.parametrize("name", list(load_registry()))
def test_official_download_verifies(tmp_path, name):
    entry = load_registry()[name]
    path = fetch(entry, tmp_path)
    assert path.parent == tmp_path
    assert path.stat().st_size == entry.size
    assert sha256_of(path) == entry.sha256


def test_user_agent_is_descriptive():
    from phase_diagram_explorer import __version__

    assert user_agent() == f"phase-diagram-explorer/{__version__} (+https://github.com/sachinkumarage/phase-diagram-explorer)"
