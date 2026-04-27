from __future__ import annotations

from pathlib import Path
import argparse
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

import toy_RPI_core as RPI


def _parse_int_list(s: str) -> list[int]:
    items: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _allocate_budget(total: int, stages: int) -> list[int]:
    if total <= 0 or stages <= 0:
        raise ValueError("total and stages must be > 0")
    q, r = divmod(int(total), int(stages))
    sims = [q] * stages
    for i in range(r):
        sims[i] += 1
    return sims


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _sample_var(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.var(x, ddof=1))


def _fmt_sig_no_sci(x: float, sig: int = 3) -> str:
    """Format to significant digits without scientific notation."""
    if not np.isfinite(x):
        return str(x)
    if x == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    scale = int(sig - 1 - exp)
    xr = round(float(x), scale)
    decimals = max(scale, 0)
    return f"{xr:.{decimals}f}"


def _fmt_sci_sig(x: float, sig: int = 3) -> str:
    if not np.isfinite(x):
        return str(x)
    decimals = max(0, int(sig) - 1)
    return f"{float(x):.{decimals}e}"


def _task_seed(base_seed: int | None, eta: int, rep: int, stream: int) -> int | None:
    """
    Derive per-(eta,rep,stream) seeds deterministically.
    Matches toy_RPI_Compare_parallel.py so results can align.
    """
    if base_seed is None:
        return None
    ss = np.random.SeedSequence([int(base_seed), int(eta), int(rep), int(stream)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def run_replicates(
    *,
    inp: RPI.RPIInputs,
    origin_z: int,
    target_healthy: int,
    reps: int,
    mc_trials: int,
    levels: list[int],
    is_budget: int,
    max_particles: int | None,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    sims = _allocate_budget(int(is_budget), len(levels))
    p_mc = np.zeros(int(reps), dtype=float)
    p_is = np.zeros(int(reps), dtype=float)

    for r in range(int(reps)):
        s_mc = _task_seed(seed, int(target_healthy), int(r), 0)
        s_is = _task_seed(seed, int(target_healthy), int(r), 1)

        p_mc[r] = RPI.estimate_mc(
            inp,
            origin_z=origin_z,
            target_healthy=int(target_healthy),
            trials=int(mc_trials),
            seed=s_mc,
        )
        p_is[r] = RPI.estimate_importance_splitting(
            inp,
            origin_z=origin_z,
            levels_healthy=levels,
            sims_per_stage=sims,
            max_particles=max_particles,
            seed=s_is,
        )

    return p_mc, p_is


def _load_baseline_csv(path: Path) -> dict[int, float]:
    baseline: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            eta = int(row["target_eta"])
            p = float(row["baseline_mc_p"])
            baseline[eta] = p
    return baseline


def _load_levels_csv(path: Path) -> dict[int, list[int]]:
    levels_map: dict[int, list[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            eta = int(row["target_eta"])
            if row.get("levels"):
                levels = _parse_int_list(row["levels"])
            elif row.get("base_levels"):
                levels = _parse_int_list(row["base_levels"])
                if levels and levels[-1] != eta:
                    levels = levels + [eta]
            else:
                raise ValueError(f"Levels CSV missing 'levels' or 'base_levels' for target_eta={eta}")
            if not levels or levels[-1] != eta:
                levels = levels + [eta]
            levels_map[eta] = levels
    return levels_map


if __name__ == "__main__":
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_csv", type=str, default=None, help="Edge CSV (default: Kodak Digital Camera Supply Chain.csv)")
    parser.add_argument("--params_csv", type=str, default=None, help="Params CSV (default: Kodak IS parameter.csv)")

    parser.add_argument("--targets", type=str, default="8,9,10", help="Comma-separated target healthy counts (eta1,eta2,eta3)")
    parser.add_argument(
        "--base_levels",
        type=str,
        default=None,
        help="Comma-separated IS intermediate healthy levels (required unless --levels_csv is provided).",
    )
    parser.add_argument("--levels_csv", type=str, default=None, help="CSV with per-target levels (from toy_RPI_IS_auto_levels.py)")

    parser.add_argument("--budget", type=int, default=10000, help="Total trajectory budget per estimator per replication")
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=None,
        help="Baseline CSV produced by toy_RPI_baseline.py (required unless you edit code).",
    )
    parser.add_argument("--reps", type=int, default=30, help="Independent replications (boxplot sample size)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed base (optional)")
    parser.add_argument("--max_particles", type=int, default=2000, help="Cap support set size used for IS resampling")

    parser.add_argument("--out", type=str, default=None, help="Output png filename (optional)")
    parser.add_argument("--out_results_csv", type=str, default=None, help="Output results CSV filename (optional)")
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
    base_levels = _parse_int_list(args.base_levels) if args.base_levels else None
    levels_csv = Path(args.levels_csv) if args.levels_csv else None
    if any(eta <= 0 or eta > n for eta in targets):
        raise ValueError(f"--targets must be in [1, {n}]")
    if base_levels is not None and levels_csv is not None:
        raise ValueError("Use either --base_levels or --levels_csv, not both.")
    if base_levels is None and levels_csv is None:
        # Default: try to reuse the CSV produced by toy_RPI_IS_auto_levels.py.
        pattern = f"toy_RPI_auto_levels_targets_{'-'.join(map(str, targets))}_*.csv"
        matches = sorted(result_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            levels_csv = matches[0]
            print(f"Using auto levels CSV: {levels_csv}")
        else:
            raise ValueError(
                "You must provide --base_levels or --levels_csv. "
                f"(No auto levels CSV found matching {pattern!r} in {result_dir})"
            )
    if base_levels is not None:
        if any(l <= 0 or l > n for l in base_levels):
            raise ValueError(f"--base_levels must be in [1, {n}]")
        if any(base_levels[i] <= base_levels[i - 1] for i in range(1, len(base_levels))):
            raise ValueError("--base_levels must be strictly increasing")
    if int(args.budget) <= 0 or int(args.reps) <= 0:
        raise ValueError("--budget and --reps must be > 0")

    # Same trajectory-count budget for MC and IS.
    mc_trials = int(args.budget)
    is_budget = int(args.budget)

    results: dict[int, dict[str, np.ndarray]] = {}
    baseline: dict[int, float] = {}
    baseline_csv = Path(args.baseline_csv) if args.baseline_csv else None
    if baseline_csv is None:
        # We intentionally do NOT auto-run 1e6 trials here (too slow); require the baseline script output.
        default_name = f"toy_RPI_baseline_targets_{'-'.join(map(str, targets))}_trials1000000_seed20260127.csv"
        candidate = result_dir / default_name
        if candidate.exists():
            baseline_csv = candidate
        else:
            raise FileNotFoundError(
                "Missing baseline CSV. Run toy_RPI_baseline.py first, or pass --baseline_csv.\n"
                f"Expected (default): {candidate}"
            )

    baseline = _load_baseline_csv(baseline_csv)
    missing = [int(eta) for eta in targets if int(eta) not in baseline]
    if missing:
        raise ValueError(f"Baseline CSV does not contain targets: {missing} (file: {baseline_csv})")

    levels_map = _load_levels_csv(levels_csv) if levels_csv is not None else {}
    missing_levels = [int(eta) for eta in targets if levels_csv is not None and int(eta) not in levels_map]
    if missing_levels:
        raise ValueError(f"Levels CSV does not contain targets: {missing_levels} (file: {levels_csv})")

    for eta in targets:
        levels = levels_map[int(eta)] if levels_csv is not None else list(base_levels) + [int(eta)]
        p_mc, p_is = run_replicates(
            inp=inp,
            origin_z=origin_z,
            target_healthy=int(eta),
            reps=int(args.reps),
            mc_trials=mc_trials,
            levels=levels,
            is_budget=is_budget,
            max_particles=int(args.max_particles) if args.max_particles is not None else None,
            # Use the same base seed scheme as toy_RPI_Compare_parallel.py so results can match exactly.
            seed=int(args.seed) if args.seed is not None else None,
        )
        results[int(eta)] = {"mc": p_mc, "is": p_is}

    # Plot: only MC and IS boxplots (no IS-MC box).
    fig, ax = plt.subplots(figsize=(12, 6))
    data = []
    tick_labels = []
    for eta in targets:
        data.append(results[int(eta)]["mc"])
        tick_labels.append(f"MC η={eta}")
        data.append(results[int(eta)]["is"])
        tick_labels.append(f"IS η={eta}")

    ax.boxplot(data, tick_labels=tick_labels, showmeans=False)
    ax.grid(True, alpha=0.3)
    font_size = plt.rcParams.get("font.size", 10) * 1.5
    ax.set_ylabel("RPI(η)", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    #ax.set_title(f"Toy RPI: MC vs IS (budget={args.budget} per replication, reps={args.reps}, base_levels={base_levels}, params={params_csv.name})")

    # Overlay baseline as red solid lines: only in its own eta region.
    for j, eta in enumerate(targets):
        y = baseline[int(eta)]
        x0 = 2 * j + 1 - 0.35
        x1 = 2 * j + 2 + 0.35
        ax.hlines(y, x0, x1, colors="red", linestyles="-", linewidth=1.6)
    fig.tight_layout()

    seed_tag = f"seed{args.seed}" if args.seed is not None else "seedNone"
    out = (
        args.out
        or f"toy_RPI_compare_box_targets_{'-'.join(map(str, targets))}_budget{args.budget}_reps{args.reps}_{seed_tag}.png"
    )
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Table output: mean/variance for MC and IS.
    print("")
    print(
        f"{'target_eta':>10}  {'MC mean':>12}  {'MC Var':>12}  "
        f"{'IS mean':>12}  {'IS Var':>12}"
    )
    table_rows: list[dict[str, str]] = []
    for eta in targets:
        mc_mu = _mean(results[int(eta)]["mc"])
        is_mu = _mean(results[int(eta)]["is"])
        mc_var = _sample_var(results[int(eta)]["mc"])
        is_var = _sample_var(results[int(eta)]["is"])
        row = {
            "target_eta": str(int(eta)),
            "mc_mean": _fmt_sci_sig(mc_mu),
            "mc_var": _fmt_sci_sig(mc_var),
            "is_mean": _fmt_sci_sig(is_mu),
            "is_var": _fmt_sci_sig(is_var),
            "levels_healthy": ",".join(str(x) for x in (levels_map[int(eta)] if levels_csv is not None else list(base_levels) + [int(eta)])),
        }
        table_rows.append(row)
        print(
            f"{eta:>10d}  {row['mc_mean']:>12}  {row['mc_var']:>12}  "
            f"{row['is_mean']:>12}  {row['is_var']:>12}"
        )

    t_end = time.perf_counter()
    print("")
    elapsed = float(t_end - t_start)
    print(f"Elapsed time (seconds): {_fmt_sig_no_sci(elapsed)}")

    # Save the same table to CSV (keep naming consistent with the plot).
    out_results_csv = args.out_results_csv
    if out_results_csv is None:
        out_results_csv = (
            f"toy_RPI_compare_results_{'-'.join(map(str, targets))}"
            f"_budget{args.budget}_reps{args.reps}_{seed_tag}.csv"
        )
    out_results_path = Path(out_results_csv)
    if not out_results_path.is_absolute():
        out_results_path = result_dir / out_results_path
    for row in table_rows:
        row["elapsed_seconds"] = _fmt_sci_sig(elapsed)
    with out_results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "target_eta",
                "mc_mean",
                "mc_var",
                "is_mean",
                "is_var",
                "levels_healthy",
                "elapsed_seconds",
            ],
        )
        w.writeheader()
        w.writerows(table_rows)
    print(f"Saved: {out_results_path}")



