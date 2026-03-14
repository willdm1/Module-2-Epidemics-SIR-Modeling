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

def exploratory_analysis_day1_2b():
    _run_from_code("exploratory_analysis_day1_2b.py")

def exploratory_analysis_day2_2b():
    _run_from_code("exploratory_analysis_day2_2b.py")

def exploratory_analysis_day3_2c():
    _run_from_code("exploratory_analysis_day3_2c.py")

def exploratory_analysis_day3_2d():
    _run_from_code("exploratory_analysis_day3_2d.py")

def exploratory_analysis_day3_2e():
    _run_from_code("exploratory_analysis_day3_2e.py")

def exploratory_analysis_day3_2f():
    _run_from_code("exploratory_analysis_day3_2f.py")

def exploratory_analysis_day4_2g():
    _run_from_code("exploratory_analysis_day4_2g.py")

def exploratory_analysis_day4_2h():
    _run_from_code("exploratory_analysis_day4_2h.py")