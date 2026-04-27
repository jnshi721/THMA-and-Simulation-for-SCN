from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import networkx as nx

import rates
import data_loader


def FindInfectedNodes(Z: int, N: int) -> list[int]:
    return [i for i in range(N) if (Z & (1 << i)) != 0]


def load_graph_params_neighbors(
    supply_chain_csv: str | Path, params_csv: str | Path
) -> tuple[list[str], list[list[int]], pd.DataFrame]:
    nodelist, g, params = data_loader.load_graph_and_params(supply_chain_csv, params_csv)
    neighbors = data_loader.build_neighbors(nodelist, g)
    return nodelist, neighbors, params


def _rates_at_time(t: float, params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    beta_t = rates.beta_rate(
        t,
        params["beta_a"].to_numpy(),
        params["beta_b"].to_numpy(),
        params["beta_c"].to_numpy(),
        params["beta_i"].to_numpy(),
    )
    delta_t = rates.delta_rate(
        t,
        params["delta_a"].to_numpy(),
        params["delta_b"].to_numpy(),
        params["delta_c"].to_numpy(),
        params["delta_i"].to_numpy(),
    )
    return beta_t, delta_t


def _infection_weights(
    infected: set[int], susceptible: list[int], neighbors: list[list[int]], beta_t: np.ndarray
) -> np.ndarray:
    w = np.zeros(len(susceptible), dtype=float)
    for k, node in enumerate(susceptible):
        rate = 0.0
        for nb in neighbors[node]:
            if nb in infected:
                rate += float(beta_t[nb])
        w[k] = rate
    return w


def RiskPropagationSimulation(
    Z0: int,
    T: float,
    state_node: int,
    nodelist: list[str],
    neighbors: list[list[int]],
    beta_delta_param: pd.DataFrame,
    flag: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(nodelist)
    if state_node != 2:
        raise ValueError("This implementation currently supports state_node=2 only.")

    rng = np.random.default_rng(seed)

    t = 0.0
    Z = int(Z0)
    ti = [t]
    Zi = [Z]

    while (t <= T or flag) and Z != 0:
        beta_t, delta_t = _rates_at_time(t, beta_delta_param)

        infected_nodes = FindInfectedNodes(Z, N)
        infected_set = set(infected_nodes)
        susceptible_nodes = [i for i in range(N) if i not in infected_set]

        sum_delta = float(delta_t[infected_nodes].sum()) if infected_nodes else 0.0
        beta_w = _infection_weights(infected_set, susceptible_nodes, neighbors, beta_t) if susceptible_nodes else np.array([])
        sum_beta = float(beta_w.sum()) if beta_w.size else 0.0

        event_rate = sum_beta + sum_delta
        if event_rate <= 0.0:
            break

        while True:
            lambda_max = event_rate
            dt = float(rng.exponential(1.0 / lambda_max))
            t_candidate = t + dt
            if t_candidate > T:
                return np.array(ti, dtype=float), np.array(Zi, dtype=int)

            beta_cand, delta_cand = _rates_at_time(t_candidate, beta_delta_param)
            sum_delta_c = float(delta_cand[infected_nodes].sum()) if infected_nodes else 0.0
            beta_w_c = (
                _infection_weights(infected_set, susceptible_nodes, neighbors, beta_cand)
                if susceptible_nodes
                else np.array([])
            )
            sum_beta_c = float(beta_w_c.sum()) if beta_w_c.size else 0.0
            event_rate_c = sum_beta_c + sum_delta_c

            if float(rng.random()) <= (event_rate_c / lambda_max if lambda_max > 0 else 0.0):
                t = t_candidate
                beta_t = beta_cand
                delta_t = delta_cand
                sum_delta = sum_delta_c
                sum_beta = sum_beta_c
                beta_w = beta_w_c
                event_rate = event_rate_c
                break
            else:
                t = t_candidate

        if t > T:
            break

        if sum_beta <= 0.0:
            # 只能恢复
            if sum_delta <= 0.0 or not infected_nodes:
                break
            probs = delta_t[infected_nodes] / sum_delta
            chosen = int(rng.choice(infected_nodes, p=probs))
            Z &= ~(1 << chosen)
        else:
            u = float(rng.random())
            if u < (sum_delta / event_rate if event_rate > 0 else 0.0):
                if sum_delta <= 0.0 or not infected_nodes:
                    break
                probs = delta_t[infected_nodes] / sum_delta
                chosen = int(rng.choice(infected_nodes, p=probs))
                Z &= ~(1 << chosen)
            else:
                if not susceptible_nodes:
                    break
                probs = beta_w / sum_beta
                chosen = int(rng.choice(susceptible_nodes, p=probs))
                Z |= 1 << chosen

        ti.append(t)
        Zi.append(Z)

    return np.array(ti, dtype=float), np.array(Zi, dtype=int)


def infection_fraction_from_path(ti: np.ndarray, Zi: np.ndarray, t: float, N: int) -> float:
    if t <= float(ti[0]):
        state = int(Zi[0])
    else:
        idx = int(np.searchsorted(ti, t, side="right") - 1)
        idx = max(0, min(idx, len(Zi) - 1))
        state = int(Zi[idx])
    return float(state.bit_count() / N)


def mean_infection_fraction_at_time(
    t: float,
    *,
    supply_chain_csv: str | Path,
    params_csv: str | Path,
    infected: list[str] | None,
    times: int,
    seed: int | None = None,
) -> float:
    nodelist, neighbors, params = load_graph_params_neighbors(supply_chain_csv, params_csv)
    N = len(nodelist)
    Z0 = data_loader.bitmask_from_infected(nodelist, infected)

    rng = np.random.default_rng(seed)
    fracs = []
    for _ in range(int(times)):
        sim_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
        ti, Zi = RiskPropagationSimulation(Z0, t, 2, nodelist, neighbors, params, False, seed=sim_seed)
        fracs.append(infection_fraction_from_path(ti, Zi, t, N))
    return float(np.mean(fracs))


if __name__ == "__main__":
    #   python model_sim.py --t 1.5 --infected A --times 10000
    #   python model_sim.py --t 1.5 --infected A --infected C --times 2000
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1.0, help="Time point t")
    parser.add_argument(
        "--infected",
        action="append",
        default=None,
        help="Initial infected enterprise (repeatable, e.g. --infected A --infected C)",
    )
    parser.add_argument("--times", type=int, default=1000, help="Number of simulation runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    supply_chain_csv = data_dir / "Kodak Digital Camera Supply Chain.csv"
    params_csv = data_dir / "Kodak parameter m.csv"

    mean_frac = mean_infection_fraction_at_time(
        args.t,
        supply_chain_csv=supply_chain_csv,
        params_csv=params_csv,
        infected=args.infected,
        times=args.times,
        seed=args.seed,
    )

    infected_str = ",".join(args.infected) if args.infected else "<default>"
    print(f"t={args.t} infected={infected_str} times={args.times} mean_infection_fraction={mean_frac}")

