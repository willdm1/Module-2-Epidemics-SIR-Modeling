from pathlib import Path
import os
import runpy

REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "Code"

def _run_from_code(script_name: str):
    old = Path.cwd()
    try:
        os.chdir(CODE_DIR)
        runpy.run_path(str(CODE_DIR / script_name), run_name="__main__")
    finally:
        os.chdir(old)

def exploratory_analysis_day1():
    _run_from_code("exploratory_analysis_day1.py")

def exploratory_analysis_day2():
    _run_from_code("exploratory_analysis_day2.py")