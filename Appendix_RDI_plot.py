from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import data_loader
import rate_function as rates


def ps_eigs_at_time(a: np.ndarray, params, t: float) -> float:
    beta_t = rates.beta_rate(
        float(t),
        params["beta_a"].to_numpy(dtype=float),
        params["beta_b"].to_numpy(dtype=float),
        params["beta_c"].to_numpy(dtype=float),
        params["beta_i"].to_numpy(dtype=float),
    )
    delta_t = rates.delta_rate(
        float(t),
        params["delta_a"].to_numpy(dtype=float),
        params["delta_b"].to_numpy(dtype=float),
        params["delta_c"].to_numpy(dtype=float),
        params["delta_i"].to_numpy(dtype=float),
    )

    # P(t) = A B(t) - D(t), where B(t)=diag(beta_t), D(t)=diag(delta_t)
    # Since B(t) is diagonal, A@B(t) is just scaling each column j by beta_t[j].
    p = (a * beta_t) - np.diag(delta_t)

    # Symmetric part: P^S(t) = (P(t)^T + P(t))/2
    ps = 0.5 * (p + p.T)

    # P^S is symmetric => use eigvalsh (real eigenvalues, faster/more stable).
    eigs = np.linalg.eigvalsh(ps)
    return float(eigs[-1])


def ps_eigs_curve(time_steps: np.ndarray, a: np.ndarray, params) -> np.ndarray:
    lam_max = np.zeros(len(time_steps), dtype=float)
    for k, t in enumerate(time_steps):
        lam_max[k] = ps_eigs_at_time(a, params, float(t))
    return lam_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_end", type=float, default=0.4, help="End time (start is 0)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step Δt")
    parser.add_argument("--edge_csv", type=str, default=None, help="Edge CSV (optional)")
    parser.add_argument("--params_csv", type=str, default=None, help="Params CSV (optional)")
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
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Kodak parameter s.csv")

    nodelist, g, params = data_loader.load_graph_and_params(edge_csv, params_csv)
    a = nx.to_numpy_array(g, nodelist=nodelist, dtype=float)

    time_steps = np.arange(0.0, t_end + 1e-12, dt, dtype=float)
    lam_max = ps_eigs_curve(time_steps, a, params)

    fig, ax = plt.subplots(figsize=(10, 6))
    markevery = max(1, int(round(0.1 / dt)))
    ax.plot(time_steps, lam_max, color="#9467bd", marker="D", markevery=markevery)

    font_size = plt.rcParams.get("font.size", 10) * 1.5
    ax.set_xlabel("t", fontsize=font_size)
    ax.set_ylabel("RDI(t)", fontsize=font_size)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=font_size)
    fig.tight_layout()

    out = args.out or f"RDI_T{t_end}_dt{dt}_params_{params_csv.stem}.png"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
