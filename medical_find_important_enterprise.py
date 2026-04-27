from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import data_loader
import model_sim as sim


STANDARD_PARAM_ORDER = [
    "enterprise",
    "beta_a",
    "beta_b",
    "beta_c",
    "beta_i",
    "delta_a",
    "delta_b",
    "delta_c",
    "delta_i",
]


def load_medical_data(
    edge_csv: str | Path,
    params_csv: str | Path,
) -> tuple[list[str], nx.Graph, list[list[int]], pd.DataFrame]:
    edges = pd.read_csv(edge_csv)
    required_edges = {"enterprise_i", "enterprise_j"}
    if not required_edges.issubset(edges.columns):
        raise KeyError(
            f"Edge CSV must contain columns {sorted(required_edges)}; got: {list(edges.columns)}"
        )

    edges = edges.copy()
    edges["enterprise_i"] = edges["enterprise_i"].astype(str).str.strip()
    edges["enterprise_j"] = edges["enterprise_j"].astype(str).str.strip()

    g_raw = nx.from_pandas_edgelist(
        edges, "enterprise_i", "enterprise_j", create_using=nx.Graph()
    )
    def _node_sort_key(node: str) -> tuple[int, int, str]:
        s = str(node).strip()
        try:
            return (0, int(s), "")
        except ValueError:
            return (1, 0, s)

    nodelist = sorted(g_raw.nodes(), key=_node_sort_key)

    params_raw = pd.read_csv(params_csv)
    params_raw.columns = [str(c).strip() for c in params_raw.columns]
    missing = [c for c in STANDARD_PARAM_ORDER if c not in params_raw.columns]
    if missing:
        raise KeyError(f"Params CSV missing columns: {missing}")

    # Reorder columns to match the standard parameter layout.
    params_raw = params_raw.loc[:, STANDARD_PARAM_ORDER].copy()
    params_raw["enterprise"] = params_raw["enterprise"].astype(str).str.strip()

    params = params_raw.set_index("enterprise").reindex(nodelist)
    missing_nodes = [n for n in nodelist if n not in set(params_raw["enterprise"])]
    if missing_nodes:
        raise ValueError(f"Missing parameter rows for nodes: {missing_nodes}")
    if params.isna().any(axis=1).any():
        bad = [str(n) for n, ok in params.isna().any(axis=1).items() if bool(ok)]
        raise ValueError(f"Parameter row has missing values for nodes: {bad}")

    g = nx.Graph()
    g.add_nodes_from(nodelist)
    g.add_edges_from(g_raw.edges())
    neighbors = data_loader.build_neighbors(nodelist, g)
    return nodelist, g, neighbors, params


def simulation_curve(
    time_steps: np.ndarray,
    *,
    nodelist: list[str],
    neighbors: list[list[int]],
    params: pd.DataFrame,
    initially_infected: list[str],
    times: int,
    seed: int | None,
) -> np.ndarray:
    n = len(nodelist)
    z0 = data_loader.bitmask_from_infected(nodelist, initially_infected)

    rng = np.random.default_rng(seed)
    acc = np.zeros(len(time_steps), dtype=float)
    for _ in range(int(times)):
        sim_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
        ti, Zi = sim.RiskPropagationSimulation(
            z0,
            float(time_steps[-1]),
            2,
            nodelist,
            neighbors,
            params,
            False,
            seed=sim_seed,
        )
        for k, t in enumerate(time_steps):
            acc[k] += sim.infection_fraction_from_path(ti, Zi, float(t), n)
    return acc / float(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_csv",
        type=str,
        default=None,
        help="Edge CSV (default: Listed Medical Industry Supply Chain Network.csv)",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default=None,
        help="Params CSV (default: Listed Medical Industry parameter.csv)",
    )
    parser.add_argument(
        "--infected",
        action="append",
        default=None,
        help="Initial infected enterprise (repeatable). Default: all enterprises, one by one.",
    )
    parser.add_argument("--t_end", type=float, default=1.0, help="End time (start is 0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step dt")
    parser.add_argument("--times", type=int, default=1000, help="Simulation runs")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (optional)")
    parser.add_argument("--out", type=str, default=None, help="Output png filename (optional)")
    parser.add_argument("--out_csv", type=str, default=None, help="Output summary CSV filename (optional)")
    args = parser.parse_args()

    t_end = float(args.t_end)
    dt = float(args.dt)
    if t_end <= 0 or dt <= 0:
        raise ValueError("--t_end and --dt must be > 0")
    if int(args.times) <= 0:
        raise ValueError("--times must be > 0")

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    result_dir = here / "result"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Listed Medical Industry Supply Chain Network.csv")
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Listed Medical Industry parameter.csv")

    nodelist, g, neighbors, params = load_medical_data(edge_csv, params_csv)
    a = nx.to_numpy_array(g, nodelist=nodelist, dtype=float)

    time_steps = np.arange(0.0, t_end + 1e-12, dt, dtype=float)

    if args.infected:
        targets = [str(x).strip() for x in args.infected if str(x).strip()]
    else:
        targets = list(nodelist)
    for node in targets:
        if node not in nodelist:
            raise ValueError(f"Initial infected node {node!r} not in nodes: {nodelist}")

    rng = np.random.default_rng(int(args.seed)) if args.seed is not None else None
    sim_curves: dict[str, np.ndarray] = {}
    for node in targets:
        sim_seed = int(rng.integers(0, 2**32 - 1)) if rng is not None else None
        sim_curves[node] = simulation_curve(
            time_steps,
            nodelist=nodelist,
            neighbors=neighbors,
            params=params,
            initially_infected=[node],
            times=int(args.times),
            seed=sim_seed,
        )

    def _trapz(y: np.ndarray, x: np.ndarray) -> float:
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(np.trapz(y, x))

    summary_rows = []
    for node in targets:
        auc_sim = _trapz(sim_curves[node], time_steps)
        summary_rows.append(
            {
                "enterprise": node,
                "p_end_sim": float(sim_curves[node][-1]),
                "auc_sim": auc_sim,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("auc_sim", ascending=False)

    out_csv = args.out_csv or f"medical_find_important_enterprise_summary_T{t_end}_dt{dt}.csv"
    out_csv_path = Path(out_csv)
    if not out_csv_path.is_absolute():
        out_csv_path = result_dir / out_csv_path
    summary.to_csv(out_csv_path, index=False)
    print(f"Saved summary CSV: {out_csv_path}")

    sim_matrix = np.vstack([sim_curves[node] for node in targets])

    # Make the heatmap readable even when there are many enterprises.
    base_font = float(plt.rcParams.get("font.size", 10))
    font_size = base_font * 2.25  # +50% on top of the previous 1.5x scaling
    fig_height = max(10.0, (0.45 * len(targets)) * (font_size / (base_font * 1.5)))
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height), constrained_layout=True)
    vmin, vmax = 0.0, 1.0
    cmap = "viridis"

    im = ax.imshow(
        sim_matrix,
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_title("Simulation p(t)", fontsize=font_size)

    yticks = np.arange(len(targets))
    ax.set_yticks(yticks)
    ax.set_yticklabels(targets, fontsize=font_size)
    ax.set_ylabel("enterprise", fontsize=font_size)

    # Map x-ticks to time values.
    # Keep a sparse set of ticks, but always include 0.3/0.6/0.9 when they are in-range.
    xticks = set(np.linspace(0, len(time_steps) - 1, num=min(8, len(time_steps)), dtype=int).tolist())
    for t_req in (0.3, 0.6, 0.9):
        if float(time_steps[0]) - 1e-12 <= t_req <= float(time_steps[-1]) + 1e-12:
            idx = int(np.argmin(np.abs(time_steps - float(t_req))))
            xticks.add(idx)
    xticks = sorted(xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{time_steps[i]:.2g}" for i in xticks], fontsize=font_size)
    ax.set_xlabel("t", fontsize=font_size)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("p(t)", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    out = args.out or f"medical_find_important_enterprise_heatmap_T{t_end}_dt{dt}_all.png"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


