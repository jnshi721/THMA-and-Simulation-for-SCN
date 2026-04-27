from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.linalg import expm
import networkx as nx

import rates
import data_loader

def get_set_bits(n, N):
    bits = []
    for i in range(N):
        if (n & (1 << i)) != 0:
            bits.append(i)
    return bits

def InfinitesimalGenerator(beta, delta, state_node, N, A):
    state_network = state_node ** N
    Q = np.zeros((state_network, state_network))

    # 填写Q矩阵的非对角元素q_ij
    for j in range(state_network):
        for i in range(state_network):
            if i == j:
                continue
            else:
                d = i ^ j
                Xm = int(d).bit_length() - 1 # 状态发生改变的节点
                if d & (d-1) == 0:
                    if i > j:
                        Q[i][j] = delta[Xm]
                    elif i < j:
                        bits = []
                        for k in range(N):
                            if A[Xm, k] != 0 and (i & (1 << k)) != 0:
                                bits.append(k)
                        for k in bits:
                            Q[i][j] = Q[i][j] + beta[k]

    # 填写Q矩阵的对角元素q_ii
    row_sum = np.sum(Q, axis=1)
    for i in range(state_network):
        Q[i][i] = -row_sum[i]

    return Q

def ExactModel(T1, T2, beta_delta_param, G, state_node, S0, *, nodelist: list[str] | None = None):

    if nodelist is None:
        nodelist = list(G.nodes())
    N = len(nodelist)

    beta_a = beta_delta_param["beta_a"].to_numpy()
    beta_b = beta_delta_param["beta_b"].to_numpy()
    beta_c = beta_delta_param["beta_c"].to_numpy()
    beta_i = beta_delta_param["beta_i"].to_numpy()
    delta_a = beta_delta_param["delta_a"].to_numpy()
    delta_b = beta_delta_param["delta_b"].to_numpy()
    delta_c = beta_delta_param["delta_c"].to_numpy()
    delta_i = beta_delta_param["delta_i"].to_numpy()

    beta, delta = rates.integrate_beta_delta(
        float(T1),
        float(T2),
        beta_a=beta_a,
        beta_b=beta_b,
        beta_c=beta_c,
        beta_i=beta_i,
        delta_a=delta_a,
        delta_b=delta_b,
        delta_c=delta_c,
        delta_i=delta_i,
    )

    # 邻接矩阵
    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)

    # 计算无穷小算子Q
    Q = InfinitesimalGenerator(beta, delta, state_node, N, A)

    e_Qt = expm(Q)  # 计算矩阵指数

    ST = np.dot(S0, e_Qt)

    return ST

def PrevalenceRate(ST, state_node, N):

    pre_num = 0
    Y = state_node ** N

    for i in range(Y):
        binary = bin(i)[2:]
        num = binary.count('1')
        pi = ST[0][i]
        pre_num = pre_num + pi * num

    return pre_num/N


def infection_fraction_at_time(
    t: float,
    *,
    supply_chain_csv: str | Path,
    params_csv: str | Path,
    initially_infected: list[str] | None = None,
    state_node: int = 2,
) -> float:
    nodelist, g, params = data_loader.load_graph_and_params(supply_chain_csv, params_csv)

    N = len(nodelist)
    if N > 15:
        raise ValueError(f"Exact model state space is 2^N; N={N} is too large for this script.")

    if initially_infected is None:
        initially_infected = [nodelist[0]]
    infected_set = set(str(x).strip() for x in initially_infected)
    index = {n: i for i, n in enumerate(nodelist)}
    Z0 = 0
    for node in infected_set:
        if node not in index:
            raise ValueError(f"Initial infected node {node!r} not in graph nodes: {nodelist}")
        Z0 += state_node ** index[node]

    Y = state_node ** N
    S0 = np.zeros((1, Y))
    S0[:, Z0] = 1

    ST = ExactModel(0.0, float(t), params, g, state_node, S0, nodelist=nodelist)
    return float(PrevalenceRate(ST, state_node, N))


if __name__ == "__main__":
    #   python model_exact.py --t 1.5 --infected A
    #   python model_exact.py --t 1.5 --infected A --infected C
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
        state_node=2,
    )
    infected_str = ",".join(args.infected) if args.infected else "<default>"
    print(f"t={args.t} infected={infected_str} infection_fraction={frac}")

