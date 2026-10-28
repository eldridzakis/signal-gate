"""Populates pre-rendered execution outputs for notebooks/05_frontier_llm_benchmark.ipynb."""

import json
import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / 'notebooks' / '05_frontier_llm_benchmark.ipynb'
FIGURES_DIR = REPO_ROOT / 'notebooks' / 'figures'

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Global execution context
exec_globals = {}

def capture_exec(code_str):
    """Executes code string and captures stdout."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code_str, exec_globals)
    finally:
        sys.stdout = old_stdout
    return buffer.getvalue()

exec_counter = 1

# --- Cell 1: Setup ---
c1_code = "".join(nb['cells'][1]['source'])
c1_stdout = capture_exec(c1_code)
nb['cells'][1]['execution_count'] = exec_counter
nb['cells'][1]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in c1_stdout.strip().split("\n")]
}]
exec_counter += 1

# --- Cell 3: Data Ingestion ---
c3_code = "".join(nb['cells'][3]['source'])
c3_stdout = capture_exec(c3_code)
nb['cells'][3]['execution_count'] = exec_counter
nb['cells'][3]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in c3_stdout.strip().split("\n")]
}]
exec_counter += 1

# --- Cell 5: Table Display ---
c5_code = "".join(nb['cells'][5]['source'])
table_res = eval(c5_code.strip(), exec_globals)
nb['cells'][5]['execution_count'] = exec_counter
html_lines = [l + "\n" for l in table_res.to_html(classes="table table-striped table-hover", index=False).split("\n")]
text_lines = [l + "\n" for l in table_res.to_string(index=False).split("\n")]
nb['cells'][5]['outputs'] = [{
    "data": {
        "text/html": html_lines,
        "text/plain": text_lines
    },
    "execution_count": exec_counter,
    "metadata": {},
    "output_type": "execute_result"
}]
exec_counter += 1

# --- Cell 7: Fig 1 Amnesia ---
c7_code = "".join(nb['cells'][7]['source'])
capture_exec(c7_code)
fig1_path = FIGURES_DIR / 'fig1_search_space_amnesia.png'
with open(fig1_path, 'rb') as f:
    fig1_b64 = base64.b64encode(f.read()).decode('ascii')
nb['cells'][7]['execution_count'] = exec_counter
nb['cells'][7]['outputs'] = [{
    "data": {
        "image/png": fig1_b64,
        "text/plain": ["<Figure size 1500x825 with 1 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]
exec_counter += 1

# --- Cell 9: Fig 2 Null Honeypot ---
c9_code = "".join(nb['cells'][9]['source'])
capture_exec(c9_code)
fig2_path = FIGURES_DIR / 'fig2_null_honeypot_fdr.png'
with open(fig2_path, 'rb') as f:
    fig2_b64 = base64.b64encode(f.read()).decode('ascii')
nb['cells'][9]['execution_count'] = exec_counter
nb['cells'][9]['outputs'] = [{
    "data": {
        "image/png": fig2_b64,
        "text/plain": ["<Figure size 1350x750 with 1 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]
exec_counter += 1

# --- Cell 11: Fig 3 Collinear Power ---
c11_code = "".join(nb['cells'][11]['source'])
capture_exec(c11_code)
fig3_path = FIGURES_DIR / 'fig3_collinear_power_gap.png'
with open(fig3_path, 'rb') as f:
    fig3_b64 = base64.b64encode(f.read()).decode('ascii')
nb['cells'][11]['execution_count'] = exec_counter
nb['cells'][11]['outputs'] = [{
    "data": {
        "image/png": fig3_b64,
        "text/plain": ["<Figure size 1350x750 with 1 Axes>"]
    },
    "metadata": {},
    "output_type": "display_data"
}]
exec_counter += 1

# --- Cell 13: Qualitative Quotes ---
c13_code = "".join(nb['cells'][13]['source'])
c13_stdout = capture_exec(c13_code)
nb['cells'][13]['execution_count'] = exec_counter
nb['cells'][13]['outputs'] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in c13_stdout.split("\n")]
}]

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully populated all outputs for {NOTEBOOK_PATH}!")
