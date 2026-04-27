# CompartmentModel

Code for our paper: "Heterogeneous Mean-Field Approximation and Simulation Models for Dynamic Risk Propagation in Complex Supply Chain Networks", including:

- THMA model
- Exact model
- Simulation model
- RDI(t) indicator based on the maximum eigenvalue of the symmetric part of the Risk Propagation Information Matrix
- RPI(eta) indicator based on rare-event probability estimation via importance splitting (IS)

The project keeps code, data, and outputs separated:

```
CompartmentModel/
  *.py                 # scripts (code)
  data/                # input CSV files
  result/              # generated PNG/CSV outputs
```

## Environment

See `ENVIRONMENT.md` for reproducible setup instructions (Windows + venv).

## Quick start (Windows / PowerShell)

Create a virtual environment and install dependencies:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Common commands

All scripts below default to reading input CSVs from `data/` and writing outputs to `result/`.

### 1) RDI(t) curve

- Purpose: plot the maximum eigenvalue of the symmetric part of the Risk Propagation Information Matrix P(t) as RDI(t)
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv`
- Outputs: `result/RDI_T{t_end}_dt{dt}_params_{params}.png`
- Command:

```
.\.venv\Scripts\python.exe .\plot_RDI.py --t_end 5 --dt 0.1
```

### 2) Toy experiment 1: THMA p(t)

- Purpose: plot THMA p(t) for the 3 preset infection scenarios
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter m.csv`
- Outputs: `result/toy_experiment_1_T{t_end}_dt{dt}_params_{params}_cases.png`
- Command:

```
.\.venv\Scripts\python.exe .\toy_experiment_1.py --t_end 0.4 --dt 0.01
```

### 3) Toy experiment 2: THMA vs Exact p(t)

- Purpose: compare THMA (solid) vs Exact (dashed) p(t) under the same preset infection scenarios
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv`
- Outputs: `result/toy_experiment_2_T{t_end}_dt{dt}_params_{params}_thma_vs_exact.png`
- Command:

```
.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 0.4 --dt 0.01
```

### 4) RPI(eta) baseline (high-trial MC)

- Purpose: generate a high-trial Monte Carlo baseline for target eta values (used as a reference in comparisons)
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`
- Outputs: `result/toy_RPI_baseline_targets_{targets}_trials{trials}_seed{seed}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\toy_RPI_baseline.py --targets 8,9,10 --trials 1000000 --seed 20260127
```

### 5) RPI(eta) single-run estimators (MC / IS)

- Purpose: run a single MC estimate and/or a single IS estimate (useful for quick checks)
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`
- Outputs: console output only
- Command:

```
.\.venv\Scripts\python.exe .\toy_RPI_MC.py --trials 1000000 --target_healthy 8 --seed 20260127
.\.venv\Scripts\python.exe .\toy_RPI_IS.py --levels 2,4,6,8 --seed 1
```

### 6) RPI(eta) comparison: MC vs IS (serial)

- Purpose: compare MC vs IS across multiple independent replications (`--reps`)
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`, plus the baseline CSV in `result/`
- Outputs:
  - `result/toy_RPI_compare_box_targets_{targets}_budget{budget}_reps{reps}_seed{seed}.png`
  - `result/toy_RPI_compare_results_{targets}_budget{budget}_reps{reps}_seed{seed}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\toy_RPI_compare.py --budget 1000 --reps 100 --seed 1
```

### 7) Listed medical: network plot

- Purpose: visualize the listed medical supply-chain network, colored by industry code
- Default inputs: `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Company Information.csv`
- Outputs: `result/medical_plot_medical_supply_chain.png`
- Command:

```
.\.venv\Scripts\python.exe .\medical_plot.py --seed 20260127
```

### 8) Listed medical: find important enterprise (simulation heatmap + ranking)

- Purpose: simulate p(t) for each enterprise as the initial infection source, then rank enterprises (AUC-based)
- Default inputs: `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Industry parameter.csv`
- Outputs:
  - `result/medical_find_important_enterprise_heatmap_T{t_end}_dt{dt}_all.png`
  - `result/medical_find_important_enterprise_summary_T{t_end}_dt{dt}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\medical_find_important_enterprise.py --t_end 1.0 --dt 0.1 --times 1000 --seed 1
```

## Paper-to-code mapping (draft)

This section is a **draft mapping** inferred from existing `result/` filenames and the current scripts in this folder.

Replace the `TBD` cells with your final paper figure/table numbers (e.g., "Figure 3", "Table 2") once you decide them.

| Paper item (TBD) | Script | Purpose | Default inputs | Default outputs (`result/`) | Example command |
|---|---|---|---|---|---|
| TBD | `plot_RDI.py` | RDI(t) curve | `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv` | `RDI_T{t_end}_dt{dt}_params_{params}.png` | `.\.venv\Scripts\python.exe .\plot_RDI.py --t_end 5 --dt 0.1` |
| TBD | `toy_experiment_1.py` | THMA p(t) for preset infections | `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter m.csv` | `toy_experiment_1_T{t_end}_dt{dt}_params_{params}_cases.png` | `.\.venv\Scripts\python.exe .\toy_experiment_1.py --t_end 0.4 --dt 0.01` |
| TBD | `toy_experiment_2.py` | THMA (solid) vs Exact (dashed) p(t) | `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv` | `toy_experiment_2_T{t_end}_dt{dt}_params_{params}_thma_vs_exact.png` | `.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 0.4 --dt 0.01` |
| TBD | `toy_RPI_baseline.py` | RPI(eta) baseline (high-trial MC) | `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv` | `toy_RPI_baseline_targets_{targets}_trials{trials}_seed{seed}.csv` | `.\.venv\Scripts\python.exe .\toy_RPI_baseline.py --targets 8,9,10 --trials 1000000 --seed 20260127` |
| TBD | `toy_RPI_compare.py` | RPI(eta) MC vs IS (boxplot + table) | `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`, plus baseline CSV | `toy_RPI_compare_box_targets_{targets}_budget{budget}_reps{reps}_seed{seed}.png` and `toy_RPI_compare_results_{targets}_budget{budget}_reps{reps}_seed{seed}.csv` | `.\.venv\Scripts\python.exe .\toy_RPI_compare.py --budget 1000 --reps 100 --seed 1` |
| TBD | `medical_plot.py` | Network plot (colored by industry code) | `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Company Information.csv` | `medical_plot_medical_supply_chain.png` | `.\.venv\Scripts\python.exe .\medical_plot.py --seed 20260127` |
| TBD | `medical_find_important_enterprise.py` | Ranking + simulation heatmap | `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Industry parameter.csv` | `medical_find_important_enterprise_summary_T{t_end}_dt{dt}.csv` and `medical_find_important_enterprise_heatmap_T{t_end}_dt{dt}_all.png` | `.\.venv\Scripts\python.exe .\medical_find_important_enterprise.py --t_end 1.0 --dt 0.1 --times 1000 --seed 1` |

## Notes

- Input CSV files are expected under `data/` by default. You can override paths via `--edge_csv` / `--params_csv` / `--info_csv`.
- Generated files are saved under `result/` by default (unless you pass absolute paths).

