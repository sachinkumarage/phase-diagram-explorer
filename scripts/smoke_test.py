"""App smoke test for the cloud runtime, using only requirements.txt
(no pytest): start the app with Streamlit's AppTest, check that every
system loads from precomputed data, switch systems and unit toggles, and
report the cold-start and per-system load times.

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
        banner = [w.value for w in at.warning]
        print(f"load {system}: {elapsed:.2f} s{' (preview banner shown)' if PROVISIONAL_BANNER in banner else ''}")
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
