from __future__ import annotations

from pathlib import Path
import argparse

import toy_RPI_core as RPI


def _parse_int_list(s: str) -> list[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", type=str, default=None, help="Edge CSV (default: Kodak Digital Camera Supply Chain.csv)")
    parser.add_argument("--params_csv", type=str, default=None, help="Params CSV (default: Kodak IS parameter.csv)")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated healthy-count thresholds, increasing (e.g. 2,4,6,8,10).",
    )
    parser.add_argument(
        "--sims",
        type=str,
        default=None,
        help="Comma-separated simulations per stage, same length as --levels (e.g. 200,200,200,200,200).",
    )
    parser.add_argument("--max_particles", type=int, default=2000, help="Cap number of success particles per stage")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--out", type=str, default=None, help="Optional: save a small txt summary")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    result_dir = here / "result"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Kodak Digital Camera Supply Chain.csv")
    params_csv = Path(args.params_csv) if args.params_csv else (data_dir / "Kodak IS parameter.csv")

    inp = RPI.load_inputs(edge_csv, params_csv)
    n = len(inp.nodelist)
    origin_z = RPI.bitmask_all_infected(n)

    if args.levels is None:
        # Default: ~5 stages from low -> high healthy counts, ending at N.
        levels = sorted({max(1, int(round(k * n / 5))) for k in range(1, 5)} | {n})
    else:
        levels = _parse_int_list(args.levels)
    if args.sims is None:
        # Default budget: 2000 trajectories total, split evenly across stages.
        per_stage = max(50, int(round(2000 / max(1, len(levels)))))
        sims = [per_stage] * len(levels)
    else:
        sims = _parse_int_list(args.sims)

    p_hat = RPI.estimate_importance_splitting(
        inp,
        origin_z=origin_z,
        levels_healthy=levels,
        sims_per_stage=sims,
        max_particles=args.max_particles,
        seed=args.seed,
    )
    msg = (
        f"IS estimate: p={p_hat} (origin=all_infected, levels_healthy={levels}, sims_per_stage={sims}, N={n})"
    )
    print(msg)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = result_dir / out_path
        out_path.write_text(msg + "\n", encoding="utf-8")
