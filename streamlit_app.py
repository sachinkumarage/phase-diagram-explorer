"""Streamlit entry point (Streamlit Community Cloud and local runs):

    streamlit run streamlit_app.py

Runs the app from the phase_diagram_explorer package (src layout). If the
package is not installed, it is imported from src/ directly. System data and
precomputed diagrams are read from data/ next to this file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import phase_diagram_explorer  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("PHASE_DIAGRAM_SYSTEMS_DIR", str(ROOT / "data" / "systems"))
os.environ.setdefault("PHASE_DIAGRAM_PRECOMPUTED_DIR", str(ROOT / "data" / "precomputed"))

from phase_diagram_explorer.app import main  # noqa: E402

main()
