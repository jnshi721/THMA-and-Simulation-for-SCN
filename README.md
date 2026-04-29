# THMA and Simulation for SCN

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

## Data files and formats

Most scripts take one or more CSV inputs. The three common types are:

- **Edge CSV (`--edge_csv`)**: supply chain network edges
  - Required columns: `enterprise_i`, `enterprise_j`
  - Each row is an undirected edge between two enterprises.
- **Params CSV (`--params_csv`)**: node-level model parameters
  - Required columns: `enterprise`, `beta_a`, `beta_b`, `beta_c`, `beta_i`, `delta_a`, `delta_b`, `delta_c`, `delta_i`
  - Exactly one row per enterprise; enterprises must cover all nodes in the edge CSV.
  - Values are numeric (int/float). `beta_*` and `delta_*` are parameters of the propagation rate function and the recovery rate function, respectively.
- **Company info CSV (`--info_csv`)**: listed medical dataset for visualization
  - Required columns: `enterprise`, `industry code`
  - Other columns are allowed (and kept), e.g. `short name`, `full name`, `industry name`.

### Included datasets (`data/`)

- `data/Kodak Digital Camera Supply Chain.csv`
  - Edge CSV for the Kodak toy network (node IDs like `A`, `B`, ...).
- `data/Kodak parameter s.csv`, `data/Kodak parameter m.csv`, `data/Kodak parameter l.csv`
  - Params CSVs for the Kodak network with different parameter settings.
- `data/Kodak IS parameter.csv`
  - Params CSV for the Kodak network used in estimation of RPI(eta) via IS.
- `data/Listed Medical Industry Supply Chain Network.csv`
  - Edge CSV for the listed medical supply chain network.
- `data/Listed Medical Industry parameter.csv`
  - Params CSV for the listed medical network.
- `data/Listed Medical Company Information.csv`
  - Company info CSV for the listed medical network.

To run on your own dataset, create CSV files with the same column names and pass them via the corresponding CLI flags.

## Common commands

All scripts below default to reading input CSVs from `data/` and writing outputs to `result/`.


### 1) Toy experiment 1: THMA p(t)

- Purpose: plot THMA p(t) for the three distinct infection scenarios
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter m.csv`
- Outputs: `result/toy_experiment_1_T{t_end}_dt{dt}_params_{params}_cases.png`
- Command:

```
.\.venv\Scripts\python.exe .\toy_experiment_1.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter m.csv"
```

### 2) Toy experiment 2: THMA vs Exact p(t)

- Purpose: compare THMA (solid) vs Exact (dashed) p(t) under the three distinct infection scenarios
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv`
- Outputs: `result/toy_experiment_2_T{t_end}_dt{dt}_params_{params}_thma_vs_exact.png`
- Command:

```
.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"
```

### 3) RPI(eta) baseline (large-scale MC estimation)

- Purpose: generate a baseline using large-scale Monte Carlo simulation for target eta values
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`
- Outputs: `result/toy_RPI_baseline_targets_{targets}_trials{trials}_seed{seed}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\toy_RPI_baseline.py --targets 8,9,10 --trials 1000000 --seed 20260127 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv"
```

### 4) RPI(eta) comparison: MC vs IS

- Purpose: compare MC vs IS across multiple independent replications
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak IS parameter.csv`, plus the baseline CSV in `result/`
- Outputs:
  - `result/toy_RPI_compare_box_targets_{targets}_budget{budget}_reps{reps}_seed{seed}.png`
  - `result/toy_RPI_compare_results_{targets}_budget{budget}_reps{reps}_seed{seed}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\toy_RPI_compare.py --budget 1000 --reps 100 --seed 1 --targets 8,9,10 --base_levels 2,5,7 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv" --baseline_csv "result/toy_RPI_baseline_targets_8-9-10_trials1000000_seed20260127.csv"
```

### 5) Listed medical: network plot

- Purpose: visualize the listed medical supply-chain network, colored by industry code
- Default inputs: `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Company Information.csv`
- Outputs: `result/medical_plot_medical_supply_chain.png`
- Command:

```
.\.venv\Scripts\python.exe .\medical_SCN_plot.py --seed 20260127 --edge_csv "data/Listed Medical Industry Supply Chain Network.csv" --info_csv "data/Listed Medical Company Information.csv"
```

### 6) Listed medical: find important enterprise

- Purpose: simulate p(t) for each enterprise as the initial infection source, then rank enterprises (AUC-based)
- Default inputs: `data/Listed Medical Industry Supply Chain Network.csv`, `data/Listed Medical Industry parameter.csv`
- Outputs:
  - `result/medical_find_important_enterprise_heatmap_T{t_end}_dt{dt}_all.png`
  - `result/medical_find_important_enterprise_summary_T{t_end}_dt{dt}.csv`
- Command:

```
.\.venv\Scripts\python.exe .\medical_find_important_enterprise.py --t_end 1.0 --dt 0.1 --times 1000 --seed 1 --edge_csv "data/Listed Medical Industry Supply Chain Network.csv" --params_csv "data/Listed Medical Industry parameter.csv"
```

### 7) RDI(t) curve

- Purpose: plot the maximum eigenvalue of the symmetric part of the Risk Propagation Information Matrix P(t) as RDI(t)
- Default inputs: `data/Kodak Digital Camera Supply Chain.csv`, `data/Kodak parameter s.csv`
- Outputs: `result/RDI_T{t_end}_dt{dt}_params_{params}.png`
- Command:

```
.\.venv\Scripts\python.exe .\Appendix_RDI_plot.py --t_end 5 --dt 0.1 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"
```

## Paper-to-code

This section maps each paper figure/table to the exact script and command (including CSV inputs) used to generate it.

| Paper figure/table | Script | CSV inputs | Python command |
|---|---|---|---|
| Figure 4 | `toy_experiment_1.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter s.csv | `.\.venv\Scripts\python.exe .\toy_experiment_1.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"` |
| Figure 5 | `toy_experiment_1.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter m.csv | `.\.venv\Scripts\python.exe .\toy_experiment_1.py --t_end 3 --dt 0.1 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter m.csv"` |
| Figure 6 | `toy_experiment_2.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter m.csv | `.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 2 --dt 0.1 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter m.csv"` |
| Figure 7 (a) | `toy_experiment_2.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter s.csv | `.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"` |
| Figure 7 (b) | `toy_experiment_2.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter l.csv | `.\.venv\Scripts\python.exe .\toy_experiment_2.py --t_end 0.1 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter l.csv"` |
| Table 2, Figure 8 | `toy_RPI_compare.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak IS parameter.csv<br>toy_RPI_baseline_targets_8-9-10_trials1000000_seed20260127.csv | `.\.venv\Scripts\python.exe .\toy_RPI_baseline.py --targets 8,9,10 --trials 1000000 --seed 20260127 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv"`<br>`.\.venv\Scripts\python.exe .\toy_RPI_compare.py --budget 1000 --reps 100 --seed 1 --targets 8,9,10 --base_levels 2,5,7 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak IS parameter.csv" --baseline_csv "result/toy_RPI_baseline_targets_8-9-10_trials1000000_seed20260127.csv"` |
| Figure 9 | `medical_SCN_plot.py` | Listed Medical Industry Supply Chain Network.csv<br>Listed Medical Company Information.csv | `.\.venv\Scripts\python.exe .\medical_SCN_plot.py --seed 20260127 --edge_csv "data/Listed Medical Industry Supply Chain Network.csv" --info_csv "data/Listed Medical Company Information.csv"` |
| Figure 10 | `medical_find_important_enterprise.py` | Listed Medical Industry Supply Chain Network.csv<br>Listed Medical Industry parameter.csv | `.\.venv\Scripts\python.exe .\medical_find_important_enterprise.py --t_end 1.0 --dt 0.1 --times 1000 --seed 1 --edge_csv "data/Listed Medical Industry Supply Chain Network.csv" --params_csv "data/Listed Medical Industry parameter.csv"` |
| Figure D.1 | `Appendix_RDI_plot.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter s.csv | `.\.venv\Scripts\python.exe .\Appendix_RDI_plot.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter s.csv"` |
| Figure D.2 | `Appendix_RDI_plot.py` | Kodak Digital Camera Supply Chain.csv<br>Kodak parameter m.csv | `.\.venv\Scripts\python.exe .\Appendix_RDI_plot.py --t_end 0.4 --dt 0.01 --edge_csv "data/Kodak Digital Camera Supply Chain.csv" --params_csv "data/Kodak parameter m.csv"` |

## Notes

- Input CSV files are expected under `data/` by default. You can override paths via `--edge_csv` / `--params_csv` / `--info_csv`.
- Generated files are saved under `result/` by default (unless you pass absolute paths).
