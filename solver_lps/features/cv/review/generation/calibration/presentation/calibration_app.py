import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("streamlit_app.py")
    runpy.run_path(str(target), run_name="__main__")
