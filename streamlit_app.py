"""Entry point for Streamlit Community Cloud (main file: streamlit_app.py). The demo lives in
demo/app.py; this shim exists so the platform's conventions (root-level main file and
requirements.txt) are met without duplicating the app."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "demo" / "app.py"), run_name="__main__")
