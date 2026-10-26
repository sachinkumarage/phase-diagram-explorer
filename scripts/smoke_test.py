"""App smoke test for the cloud runtime, using only requirements.txt
(no pytest): start the app with Streamlit's AppTest, check the public
defaults (the validated systems Ag-Cu, Fe-C and Pb-Sn, Ag-Cu selected, °C
and at%), that every system loads from precomputed data with its citations
and no warning, switch systems, diagram modes (Fe-C metastable and stable)
and unit toggles, and report the cold-start and per-system load times.

    python scripts/smoke_test.py             # database available (or downloadable)
    python scripts/smoke_test.py --offline   # database unavailable: page must still work

With --offline, downloads are disabled and data/tdb is not used; the test
checks that each sourced system shows the Gibbs energy notice and still
renders its diagram, invariant table and point equilibrium.

Exits with status 1 on any failure.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = ROOT / "streamlit_app.py"
PRECOMPUTED_NOTE = "Diagram and invariant reactions from precomputed data"
VALIDATED = ["ag_cu", "fe_c", "pb_sn"]
DEFAULT_UNITS = ["°C", "at%"]
# "Gibbs energy curves need the <database> database, which could not be downloaded right now."
GIBBS_NOTICE = ("Gibbs energy curves need the ", " database, which could not be downloaded right now.")
SOURCED = {"ag_cu", "fe_c", "pb_sn"}
# Citation that each system must show
CITATIONS = {
    "ag_cu": "https://doi.org/10.18434/T4759N",
    "pb_sn": "https://www.metallurgy.nist.gov/phase/solder/pbsn.html",
    "fe_c": "https://doi.org/10.1016/j.calphad.2010.01.004",
}
# Systems with diagram modes: every mode must load from precomputed data.
MODES = {"fe_c": ["metastable", "stable"]}
# Systems with an industrial-context info box, and how it starts.
INFO_BOXES = {"pb_sn": "63Sn-37Pb solder (63 wt% Sn) is near-eutectic"}


def radio(at: AppTest, label: str):
    (match,) = [element for element in at.radio if element.label == label]
    return match


def check(at: AppTest, context: str, offline: bool = False, system: str | None = None) -> None:
    if at.exception:
        raise AssertionError(f"{context}: {at.exception[0].value}")
    if at.error:
        raise AssertionError(f"{context}: error shown: {at.error[0].value}")
    if not any(caption.value.startswith(PRECOMPUTED_NOTE) for caption in at.caption):
        raise AssertionError(f"{context}: not loaded from precomputed data")
    markdown = [m.value for m in at.markdown]
    if "**Stable phases**" not in markdown or not at.table:
        raise AssertionError(f"{context}: point equilibrium or invariant table missing")
    gibbs_notice = any(i.value.startswith(GIBBS_NOTICE[0]) and i.value.endswith(GIBBS_NOTICE[1]) for i in at.info)
    if offline and system in SOURCED and not gibbs_notice:
        raise AssertionError(f"{context}: Gibbs energy notice missing without the database")
    if not offline and gibbs_notice:
        raise AssertionError(f"{context}: Gibbs energy notice shown although the database is available")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="App smoke test")
    parser.add_argument("--offline", action="store_true", help="simulate an unreachable database")
    offline = parser.parse_args(argv).offline
    if offline:
        os.environ["PHASE_DIAGRAM_DISABLE_DOWNLOADS"] = "1"
        os.environ["PHASE_DIAGRAM_TDB_DIR"] = tempfile.mkdtemp(prefix="no_tdb_")
    print(f"mode: {'offline (database unavailable)' if offline else 'online'}")

    start = time.perf_counter()
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=300).run()
    cold_start = time.perf_counter() - start
    check(at, "cold start", offline, at.selectbox[0].value)
    print(f"cold start ({at.selectbox[0].value}): {cold_start:.2f} s")

    # public defaults: validated systems only, Ag-Cu selected, in °C and at%
    if list(at.selectbox[0].options) != VALIDATED or at.selectbox[0].value != "ag_cu":
        raise AssertionError(f"public defaults: {list(at.selectbox[0].options)}, selected {at.selectbox[0].value}")
    units = [radio(at, "Temperature unit").value, radio(at, "Composition unit").value]
    if units != DEFAULT_UNITS:
        raise AssertionError(f"public default units: {units}")
    print(f"systems: {list(at.selectbox[0].options)}")
    run_systems(at, VALIDATED, offline)
    print(f"smoke test passed ({'offline' if offline else 'online'})")
    return 0


def run_systems(at: AppTest, systems: list[str], offline: bool) -> None:
    for system in systems:
        start = time.perf_counter()
        at.selectbox[0].set_value(system)
        at.run()
        elapsed = time.perf_counter() - start
        check(at, system, offline, system)
        if at.warning:
            raise AssertionError(f"{system}: unexpected warning: {at.warning[0].value}")
        markdown = [m.value for m in at.markdown]
        cited = "**Data sources**" in markdown
        citation = CITATIONS.get(system)
        if citation and not (cited and citation in markdown[markdown.index("**Data sources**") + 1]):
            raise AssertionError(f"{system}: citation {citation} not shown")
        info = [i.value for i in at.info if system in INFO_BOXES and i.value.startswith(INFO_BOXES[system])]
        if system in INFO_BOXES and len(info) != 1:
            raise AssertionError(f"{system}: info box not shown")
        notes = ["citations" if cited else "no citations"]
        if info:
            notes.append("info box")
        print(f"load {system}: {elapsed:.2f} s ({', '.join(notes)})")
        for mode in MODES.get(system, [None]):
            if mode is not None:
                start = time.perf_counter()
                radio(at, "Diagram").set_value(mode)
                at.run()
                check(at, f"{system} ({mode})", offline, system)
                print(f"  mode {mode}: {time.perf_counter() - start:.2f} s")
            for temperature_unit, composition_unit in [("°C", "wt%"), ("K", "at%")]:
                radio(at, "Temperature unit").set_value(temperature_unit)
                radio(at, "Composition unit").set_value(composition_unit)
                at.run()
                check(at, f"{system} {mode or ''} in {temperature_unit}, {composition_unit}", offline, system)
        if system in MODES:
            radio(at, "Diagram").set_value(MODES[system][0])
            at.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"smoke test FAILED: {error}")
        sys.exit(1)
