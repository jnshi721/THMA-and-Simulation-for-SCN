from __future__ import annotations

from pathlib import Path
import argparse
import csv

import toy_RPI_core as RPI


def _parse_int_list(s: str) -> list[int]:
    items: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_csv",
        type=str,
        default=None,
        help="Edge CSV (default: Kodak Digital Camera Supply Chain.csv)",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default=None,
        help="Params CSV (default: Kodak IS parameter.csv)",
    )
    parser.add_argument("--targets", type=str, default="8,9,10", help="Comma-separated target healthy counts (eta1,eta2,eta3)")
    parser.add_argument("--trials", type=int, default=1_000_000, help="Monte Carlo trials per target eta")
    parser.add_argument("--seed", type=int, default=20260127, help="Random seed")
    parser.add_argument("--out_csv", type=str, default=None, help="Output baseline CSV filename (optional)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    result_dir = here / "result"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Kodak Digital Camera Supply Chain.csv")
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Kodak IS parameter.csv")

    inp = RPI.load_inputs(edge_csv, params_csv)
    n = len(inp.nodelist)
    origin_z = RPI.bitmask_all_infected(n)

    targets = _parse_int_list(args.targets)
    if any(eta <= 0 or eta > n for eta in targets):
        raise ValueError(f"--targets must be in [1, {n}]")

    rows: list[tuple[int, float]] = []
    for eta in targets:
        p_hat = RPI.estimate_mc(
            inp,
            origin_z=origin_z,
            target_healthy=int(eta),
            trials=int(args.trials),
            seed=int(args.seed),
        )
        rows.append((int(eta), float(p_hat)))
        print(f"eta={eta}: baseline MC p={p_hat} (trials={args.trials}, seed={args.seed})")

    out = args.out_csv or f"toy_RPI_baseline_targets_{'-'.join(map(str, targets))}_trials{int(args.trials)}_seed{int(args.seed)}.csv"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_eta", "baseline_mc_p", "trials", "seed", "edge_csv", "params_csv"])
        for eta, p in rows:
            w.writerow([eta, p, int(args.trials), int(args.seed), str(edge_csv.name), str(params_csv.name)])

    print(f"Saved baseline CSV: {out_path}")



