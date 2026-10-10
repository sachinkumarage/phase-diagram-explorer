"""requirements.txt (the cloud runtime) matches pyproject.toml."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
DEV_ONLY = {"pytest", "pycalphad"}


def _name(requirement: str) -> str:
    """Normalised distribution name (PEP 503) of a requirement string."""
    name = re.split(r"[\s\[<>=!~;]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements() -> list[str]:
    lines = [line.split("#", 1)[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    return [line for line in lines if line]


def _pinned() -> dict[str, str]:
    return {_name(line): line for line in _requirements() if not line.startswith("-")}


def test_installs_the_package_itself():
    assert "-e ." in _requirements()


def test_every_runtime_dependency_is_in_requirements():
    runtime = {_name(dependency) for dependency in PYPROJECT["project"]["dependencies"]}
    assert runtime, "pyproject.toml lists no runtime dependencies"
    assert runtime <= set(_pinned()), runtime - set(_pinned())


def test_requirements_are_runtime_dependencies_with_compatible_release_pins():
    runtime = {_name(dependency) for dependency in PYPROJECT["project"]["dependencies"]}
    for name, line in _pinned().items():
        assert name in runtime, f"{line} is not a runtime dependency in pyproject.toml"
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+~=\d+(\.\d+)+", line), f"{line} is not a ~= pin"


def test_no_validation_or_dev_dependency_in_requirements():
    optional = {
        _name(dependency) for group in PYPROJECT["project"]["optional-dependencies"].values() for dependency in group
    }
    assert DEV_ONLY <= optional
    assert not (optional | DEV_ONLY) & set(_pinned())
    assert not (optional | DEV_ONLY) & {_name(d) for d in PYPROJECT["project"]["dependencies"]}
