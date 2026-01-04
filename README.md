# Bloom Filter Project (Python)

Implementation and experimental evaluation of the Bloom filter, including:
- Classic Bloom filter (k independent hashes via `mmh3` seeds)
- Variant using **double hashing** (Kirsch–Mitzenmacher style)

## Project structure
- `src/` — implementation + experiments
- `requirements.txt` — dependencies

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run experiments
From the project root:
```powershell
python -m src.experiments
```

Outputs:
- prints a summary table to the console
- saves CSV + plots into a `results/` folder (auto-created)

