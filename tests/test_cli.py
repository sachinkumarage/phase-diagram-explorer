"""Command line interface smoke tests."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from phase_diagram_explorer import cli
from phase_diagram_explorer.models import load_system

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = ROOT / "data" / "systems"
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def test_plot_writes_a_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--systems-dir", str(SYSTEMS_DIR), "plot", "--system", "ag_cu"]) == 0
    figure = tmp_path / "ag_cu_phase_diagram.svg"
    assert figure.read_bytes().lstrip().startswith(b"<?xml")


def test_plot_pdf_from_tdb_file(tmp_path):
    output = tmp_path / "gap.pdf"
    cli.main(["plot", "--system", str(FIXTURES_DIR / "gap_eutectic.tdb"), "--output", str(output), "--unit", "C"])
    assert output.read_bytes().startswith(b"%PDF")


def test_unknown_system_is_reported(capsys):
    with pytest.raises(SystemExit, match="no system 'nope'"):
        cli.main(["--systems-dir", str(SYSTEMS_DIR), "plot", "--system", "nope"])


def test_export_tdb_and_list(tmp_path, capsys):
    shutil.copy(SYSTEMS_DIR / "ag_cu.json", tmp_path)
    output = tmp_path / "ag_cu.tdb"
    cli.main(["--systems-dir", str(tmp_path), "export-tdb", "--system", "ag_cu", "--output", str(output)])
    assert (tmp_path / "ag_cu.meta.json").exists()
    assert load_system(output).name == "Ag-Cu"

    cli.main(["--systems-dir", str(tmp_path), "list"])
    assert capsys.readouterr().out.splitlines()[-2:] == ["ag_cu", "ag_cu.tdb"]


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "plot" in capsys.readouterr().out


@pytest.mark.slow
def test_console_command_writes_a_figure(tmp_path):
    """The documented command, run as installed: phase-diagram-explorer plot --system ag_cu."""
    command = shutil.which("phase-diagram-explorer", path=str(Path(sys.executable).parent))
    argv = [command] if command else [sys.executable, "-m", "phase_diagram_explorer"]
    completed = subprocess.run(
        [*argv, "plot", "--system", "ag_cu"], cwd=tmp_path, capture_output=True, text=True, timeout=300,
        env={"PATH": str(Path(sys.executable).parent), "PHASE_DIAGRAM_SYSTEMS_DIR": str(SYSTEMS_DIR)},
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "ag_cu_phase_diagram.svg").stat().st_size > 0
