from __future__ import annotations

from pathlib import Path
import argparse
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import solve_ivp

import rates
import data_loader


def ODEs(A: np.ndarray, beta_delta_param: pd.DataFrame, t: float, v: np.ndarray) -> np.ndarray:
    beta_t = rates.beta_rate(
        t,
        beta_delta_param["beta_a"].to_numpy(),
        beta_delta_param["beta_b"].to_numpy(),
        beta_delta_param["beta_c"].to_numpy(),
        beta_delta_param["beta_i"].to_numpy(),
    )
    delta_t = rates.delta_rate(
        t,
        beta_delta_param["delta_a"].to_numpy(),
        beta_delta_param["delta_b"].to_numpy(),
        beta_delta_param["delta_c"].to_numpy(),
        beta_delta_param["delta_i"].to_numpy(),
    )

    infection_pressure = A @ (beta_t * v)
    return (1.0 - v) * infection_pressure - delta_t * v


def NumericalSolution(
    ODEs_func,
    T: float,
    num: int,
    v0: np.ndarray,
    A: np.ndarray,
    beta_delta_param: pd.DataFrame,
):
    t_span = (0.0, float(T))
    t_eval = np.linspace(0.0, float(T), int(num))
    return solve_ivp(
        lambda t, v: ODEs_func(A, beta_delta_param, t, v),
        t_span,
        np.asarray(v0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

def Z0_to_vector(Z0, N):


    binary_representation = bin(Z0)[2:].zfill(N)  # 使用 zfill 保证二进制长度为 N

    vector = [int(bit) for bit in binary_representation]
    # 反转向量
    vector = vector[::-1]

    return vector


def infection_fraction_at_time(
    t: float,
    *,
    supply_chain_csv: str | Path,
    params_csv: str | Path,
    initially_infected: Iterable[str] | None = None,
) -> float:
    nodelist, g, params = data_loader.load_graph_and_params(supply_chain_csv, params_csv)
    a = nx.to_numpy_array(g, nodelist=nodelist, dtype=float)

    if initially_infected is None:
        initially_infected = [nodelist[0]]
    infected = set(str(x).strip() for x in initially_infected)
    v0 = np.array([1.0 if n in infected else 0.0 for n in nodelist], dtype=float)

    sol = solve_ivp(
        lambda tt, vv: ODEs(a, params, tt, vv),
        (0.0, float(t)),
        v0,
        t_eval=[float(t)],
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    return float(np.mean(sol.y[:, -1]))


if __name__ == "__main__":
    #   python model_thma.py --t 1.5 --infected A
    #   python model_thma.py --t 1.5 --infected A --infected C
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1.0, help="Time point t")
    parser.add_argument(
        "--infected",
        action="append",
        default=None,
        help="Initial infected enterprise (repeatable, e.g. --infected A --infected C)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    supply_chain_csv = data_dir / "Kodak Digital Camera Supply Chain.csv"
    params_csv = data_dir / "Kodak parameter m.csv"

    frac = infection_fraction_at_time(
        args.t,
        supply_chain_csv=supply_chain_csv,
        params_csv=params_csv,
        initially_infected=args.infected,
    )
    infected_str = ",".join(args.infected) if args.infected else "<default>"
    print(f"t={args.t} infected={infected_str} infection_fraction={frac}")

