"""App smoke test for the cloud runtime, using only requirements.txt
(no pytest): start the app with Streamlit's AppTest, check that every
system loads from precomputed data, that assessed systems show their
citations and no preview banner while provisional ones show the banner,
switch systems and unit toggles, and report the cold-start and per-system
load times.

    python scripts/smoke_test.py

Exits with status 1 on any failure.
"""
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = ROOT / "streamlit_app.py"
PRECOMPUTED_NOTE = "Diagram and invariant reactions from precomputed data"
PROVISIONAL_BANNER = "Preview: provisional thermodynamic data, not yet validated."
# Expected per system: (preview banner shown, citation that must be shown)
EXPECTED = {
    "ag_cu": (False, "https://doi.org/10.18434/T4759N"),
    "al_cu": (True, None),
    "pb_sn": (False, "https://www.metallurgy.nist.gov/phase/solder/pbsn.html"),
}
# Systems with an industrial-context info box, and how it starts.
INFO_BOXES = {"pb_sn": "63Sn-37Pb solder (63 wt% Sn) is near-eutectic"}


def check(at: AppTest, context: str) -> None:
    if at.exception:
        raise AssertionError(f"{context}: {at.exception[0].value}")
    if not any(caption.value.startswith(PRECOMPUTED_NOTE) for caption in at.caption):
        raise AssertionError(f"{context}: not loaded from precomputed data")


def main() -> int:
    start = time.perf_counter()
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=300).run()
    cold_start = time.perf_counter() - start
    check(at, "cold start")
    systems = list(at.selectbox[0].options)
    print(f"cold start ({at.selectbox[0].value}): {cold_start:.2f} s")

    for system in systems:
        start = time.perf_counter()
        at.selectbox[0].set_value(system)
        at.run()
        elapsed = time.perf_counter() - start
        check(at, system)
        banner = PROVISIONAL_BANNER in [w.value for w in at.warning]
        markdown = [m.value for m in at.markdown]
        cited = "**Data sources**" in markdown
        expected_banner, citation = EXPECTED.get(system, (None, None))
        if expected_banner is not None and banner != expected_banner:
            raise AssertionError(f"{system}: preview banner {'shown' if banner else 'missing'}")
        if citation and not (cited and citation in markdown[markdown.index("**Data sources**") + 1]):
            raise AssertionError(f"{system}: citation {citation} not shown")
        info = [i.value for i in at.info if system in INFO_BOXES and i.value.startswith(INFO_BOXES[system])]
        if system in INFO_BOXES and len(info) != 1:
            raise AssertionError(f"{system}: info box not shown")
        notes = ["preview banner" if banner else "no banner", "citations" if cited else "no citations"]
        if info:
            notes.append("info box")
        print(f"load {system}: {elapsed:.2f} s ({', '.join(notes)})")
        for temperature_unit, composition_unit in [("°C", "wt%"), ("K", "at%")]:
            at.radio[0].set_value(temperature_unit)
            at.radio[1].set_value(composition_unit)
            at.run()
            check(at, f"{system} in {temperature_unit}, {composition_unit}")
    print(f"smoke test passed for {len(systems)} systems")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"smoke test FAILED: {error}")
        sys.exit(1)
