from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp

import data_loader
import model_thma as thma


def thma_infection_fraction_curve(
    time_steps: np.ndarray,
    *,
    a: np.ndarray,
    params,
    initially_infected: list[str],
    nodelist: list[str],
) -> np.ndarray:
    infected_set = set(initially_infected)
    v0 = np.array([1.0 if n in infected_set else 0.0 for n in nodelist], dtype=float)
    sol = solve_ivp(
        lambda t, v: thma.ODEs(a, params, float(t), v),
        (0.0, float(time_steps[-1])),
        v0,
        t_eval=time_steps,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    return np.mean(sol.y, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_csv",
        type=str,
        default=None,
        help="Edge CSV (columns enterprise_i, enterprise_j); default: Kodak Digital Camera Supply Chain.csv",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default=None,
        help="Params CSV (enterprise,beta_a,beta_b,beta_c,beta_i,delta_a,delta_b,delta_c,delta_i); default: Kodak parameter s.csv",
    )
    parser.add_argument("--t_end", type=float, default=3, help="End time (start is 0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step dt")
    parser.add_argument("--out", type=str, default=None, help="Output png filename (optional)")
    args = parser.parse_args()

    t_end = float(args.t_end)
    dt = float(args.dt)
    if t_end <= 0 or dt <= 0:
        raise ValueError("--t_end and --dt must be > 0")

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    result_dir = here / "result"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Kodak Digital Camera Supply Chain.csv")
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Kodak parameter m.csv")

    nodelist, g, params = data_loader.load_graph_and_params(edge_csv, params_csv)
    a = nx.to_numpy_array(g, nodelist=nodelist, dtype=float)

    scenarios = [
        ("Preset Infection", ["A", "C", "D", "F", "I"]),
        ("Hub Infection", ["E", "J"]),
        ("Full Infection", list(nodelist)),
    ]
    for name, infected_nodes in scenarios:
        for node in infected_nodes:
            if node not in nodelist:
                raise ValueError(f"{name}: node {node!r} not in nodes: {nodelist}")

    time_steps = np.arange(0.0, t_end + 1e-12, dt, dtype=float)
    y_curves: dict[str, np.ndarray] = {}
    for name, infected_nodes in scenarios:
        y_curves[name] = thma_infection_fraction_curve(
            time_steps,
            a=a,
            params=params,
            initially_infected=infected_nodes,
            nodelist=nodelist,
        )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    markevery = max(1, int(round(0.1 / dt)))
    style = [
        ("#1f77b4", "o"),
        ("#2ca02c", "^"),
        ("#d62728", "s"),
    ]
    l_inf = []
    for i, (name, _) in enumerate(scenarios):
        color, marker = style[i % len(style)]
        l_inf += ax1.plot(
            time_steps,
            y_curves[name],
            color=color,
            marker=marker,
            markevery=markevery,
            label=name,
        )

    font_size = plt.rcParams.get("font.size", 10) * 1.5
    ax1.set_xlabel("t", fontsize=font_size)
    ax1.set_ylabel("p(t)", fontsize=font_size)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax1.legend(l_inf, [ln.get_label() for ln in l_inf], loc="best", fontsize=font_size)
    ax1.tick_params(axis="both", labelsize=font_size)
    #ax1.set_title(f"Toy experiment 1 (T={t_end}, dt={dt}, params={params_csv.name})")
    fig.tight_layout()

    out = args.out or f"toy_experiment_1_T{t_end}_dt{dt}_params_{params_csv.stem}_cases.png"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
