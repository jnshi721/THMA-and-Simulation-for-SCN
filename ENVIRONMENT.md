# Reproducible environment (Windows)

This folder is intended to be runnable and reproducible on Windows.

## Python version

- Python 3.10.x is recommended (the original environment used 3.10.11).

## Dependency files

- `requirements.txt`: a minimal, human-maintained dependency list (recommended for most users).
- `requirements.lock.txt`: an exact, pinned dependency list exported via `pip freeze` (recommended for strict reproducibility).

## Create / rebuild a virtual environment

1) Create a venv:

```
python -m venv .venv
```

2) Install dependencies (choose one):

- Minimal dependencies:

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- Fully pinned dependencies:

```
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

## Run scripts

From `CompartmentModel/`:

```
.\.venv\Scripts\python.exe .\Appendix_RDI_plot.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"
.\.venv\Scripts\python.exe .\toy_RPI_baseline.py --targets 8,9,10 --trials 1000000 --seed 20260127 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv"
.\.venv\Scripts\python.exe .\toy_RPI_compare.py --budget 1000 --reps 30 --seed 1 --targets 8,9,10 --base_levels 2,5,7 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv" --baseline_csv "result/toy_RPI_baseline_targets_8-9-10_trials1000000_seed20260127.csv"
```

## Export / update the lock file

After you have a working environment in `.venv`, you can update the lock file with:

```
.\.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```
