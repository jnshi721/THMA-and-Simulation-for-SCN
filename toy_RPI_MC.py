from __future__ import annotations

from pathlib import Path
import argparse

import toy_RPI_core as RPI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", type=str, default=None, help="Edge CSV (default: Kodak Digital Camera Supply Chain.csv)")
    parser.add_argument("--params_csv", type=str, default=None, help="Params CSV (default: Kodak IS parameter.csv)")
    parser.add_argument(
        "--origin",
        type=str,
        default="all_infected",
        choices=["all_infected"],
        help="Initial state; currently only supports all_infected (RPI).",
    )
    parser.add_argument(
        "--target_healthy",
        type=int,
        default=None,
        help="Target healthy nodes count (default: N, i.e. all healthy).",
    )
    parser.add_argument(
        "--target_eta",
        type=int,
        default=None,
        help="Alias of --target_healthy (eta = target healthy nodes count).",
    )
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Kodak Digital Camera Supply Chain.csv")
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Kodak IS parameter.csv")

    inp = RPI.load_inputs(edge_csv, params_csv)
    n = len(inp.nodelist)
    origin_z = RPI.bitmask_all_infected(n)
    if args.target_healthy is not None and args.target_eta is not None:
        if int(args.target_healthy) != int(args.target_eta):
            raise ValueError("--target_healthy and --target_eta both provided but differ.")
    target_healthy = (
        int(args.target_healthy)
        if args.target_healthy is not None
        else int(args.target_eta)
        if args.target_eta is not None
        else int(n)
    )

    p_hat = RPI.estimate_mc(inp, origin_z=origin_z, target_healthy=target_healthy, trials=args.trials, seed=args.seed)
    print(
        f"MC estimate: p={p_hat} (trials={args.trials}, origin=all_infected, target_healthy={target_healthy}, N={n})"
    )
