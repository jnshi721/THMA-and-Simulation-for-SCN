from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import data_loader
import rates


@dataclass(frozen=True)
class RPIInputs:
    nodelist: list[str]
    neighbors: list[list[int]]  # adjacency list in nodelist index space
    beta_a: np.ndarray
    beta_b: np.ndarray
    beta_c: np.ndarray
    beta_i: np.ndarray
    delta_a: np.ndarray
    delta_b: np.ndarray
    delta_c: np.ndarray
    delta_i: np.ndarray


def load_inputs(edge_csv: str | Path, params_csv: str | Path) -> RPIInputs:
    # Support either:
    # - params file with an explicit node id column ("enterprise"), or
    # - params file without node ids (assume row order matches sorted nodelist).
    params_head = pd.read_csv(params_csv, nrows=1)
    node_col = "enterprise" if "enterprise" in params_head.columns else None

    nodelist, g, params = data_loader.load_graph_and_params(edge_csv, params_csv, node_col=node_col)
    neighbors = data_loader.build_neighbors(nodelist, g)

    return RPIInputs(
        nodelist=nodelist,
        neighbors=neighbors,
        beta_a=params["beta_a"].to_numpy(dtype=float),
        beta_b=params["beta_b"].to_numpy(dtype=float),
        beta_c=params["beta_c"].to_numpy(dtype=float),
        beta_i=params["beta_i"].to_numpy(dtype=float),
        delta_a=params["delta_a"].to_numpy(dtype=float),
        delta_b=params["delta_b"].to_numpy(dtype=float),
        delta_c=params["delta_c"].to_numpy(dtype=float),
        delta_i=params["delta_i"].to_numpy(dtype=float),
    )


def bitmask_all_infected(n: int) -> int:
    return (1 << n) - 1


def bitmask_from_infected(nodelist: list[str], infected: Iterable[str] | None) -> int:
    # Same mapping used in other scripts: bit i corresponds to nodelist[i].
    return data_loader.bitmask_from_infected(nodelist, infected, default_first=False)


def infected_count(z: int) -> int:
    return int(z.bit_count())


def healthy_count(z: int, n: int) -> int:
    return int(n - z.bit_count())


def _rates_at_time(inp: RPIInputs, t: float) -> tuple[np.ndarray, np.ndarray]:
    beta_t = rates.beta_rate(t, inp.beta_a, inp.beta_b, inp.beta_c, inp.beta_i)
    delta_t = rates.delta_rate(t, inp.delta_a, inp.delta_b, inp.delta_c, inp.delta_i)
    return beta_t, delta_t


def _infected_nodes(z: int, n: int) -> list[int]:
    return [i for i in range(n) if (z & (1 << i)) != 0]


def _infection_weights(
    *,
    infected_set: set[int],
    susceptible_nodes: list[int],
    neighbors: list[list[int]],
    beta_t: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    # Susceptible node i gets infected at rate sum_{j in infected neighbors} beta_j(t).
    nodes = []
    w = []
    for node in susceptible_nodes:
        rate = 0.0
        for nb in neighbors[node]:
            if nb in infected_set:
                rate += float(beta_t[nb])
        if rate > 0.0:
            nodes.append(node)
            w.append(rate)
    if not w:
        return [], np.array([], dtype=float)
    return nodes, np.asarray(w, dtype=float)


def simulate_until_hit_or_return(
    inp: RPIInputs,
    *,
    t0: float,
    z0: int,
    origin_z: int,
    target_healthy: int,
    rng: np.random.Generator,
) -> tuple[float, int, bool]:
    """
    Runs a single CTMC trajectory (non-homogeneous rates) until it either:
      - hits healthy_count >= target_healthy  (success=True), or
      - returns to origin_z after at least one event (success=False).

    Returns: (t_end, z_end, success)
    """
    n = len(inp.nodelist)
    if target_healthy < 0 or target_healthy > n:
        raise ValueError(f"target_healthy must be in [0, {n}]")

    t = float(t0)
    z = int(z0)
    steps = 0

    while True:
        if healthy_count(z, n) >= target_healthy:
            return t, z, True
        if steps > 0 and z == origin_z:
            return t, z, False

        infected_nodes = _infected_nodes(z, n)
        if not infected_nodes:
            # All healthy => would have returned above as success.
            return t, z, True

        infected_set = set(infected_nodes)
        susceptible_nodes = [i for i in range(n) if i not in infected_set]

        beta_t, delta_t = _rates_at_time(inp, t)

        sum_delta = float(delta_t[infected_nodes].sum()) if infected_nodes else 0.0
        inf_nodes, beta_w = _infection_weights(
            infected_set=infected_set,
            susceptible_nodes=susceptible_nodes,
            neighbors=inp.neighbors,
            beta_t=beta_t,
        )
        sum_beta = float(beta_w.sum()) if beta_w.size else 0.0

        event_rate = sum_beta + sum_delta
        if event_rate <= 0.0:
            # Degenerate: no events possible.
            return t, z, False

        # Thinning step: assumes event_rate(t) for the fixed state is an upper bound
        # over the waiting period (true when rates are non-increasing in t).
        while True:
            lambda_max = event_rate
            dt = float(rng.exponential(1.0 / lambda_max))
            t_candidate = t + dt

            beta_c, delta_c = _rates_at_time(inp, t_candidate)
            sum_delta_c = float(delta_c[infected_nodes].sum()) if infected_nodes else 0.0
            _, beta_w_c = _infection_weights(
                infected_set=infected_set,
                susceptible_nodes=susceptible_nodes,
                neighbors=inp.neighbors,
                beta_t=beta_c,
            )
            sum_beta_c = float(beta_w_c.sum()) if beta_w_c.size else 0.0
            event_rate_c = sum_beta_c + sum_delta_c

            accept_prob = (event_rate_c / lambda_max) if lambda_max > 0 else 0.0
            if float(rng.random()) <= accept_prob:
                t = t_candidate
                beta_t = beta_c
                delta_t = delta_c
                sum_delta = sum_delta_c
                sum_beta = sum_beta_c
                beta_w = beta_w_c
                inf_nodes = _infection_weights(
                    infected_set=infected_set,
                    susceptible_nodes=susceptible_nodes,
                    neighbors=inp.neighbors,
                    beta_t=beta_t,
                )[0]
                event_rate = event_rate_c
                break
            else:
                t = t_candidate

        # Choose event type and node.
        if sum_beta <= 0.0:
            # Only recovery possible.
            probs = delta_t[infected_nodes] / sum_delta if sum_delta > 0 else None
            chosen = int(rng.choice(infected_nodes, p=probs))
            z &= ~(1 << chosen)
        elif sum_delta <= 0.0:
            # Only infection possible.
            if not inf_nodes or beta_w.size == 0:
                return t, z, False
            probs = beta_w / sum_beta
            chosen = int(rng.choice(inf_nodes, p=probs))
            z |= 1 << chosen
        else:
            if float(rng.random()) < (sum_delta / event_rate):
                probs = delta_t[infected_nodes] / sum_delta
                chosen = int(rng.choice(infected_nodes, p=probs))
                z &= ~(1 << chosen)
            else:
                if not inf_nodes or beta_w.size == 0:
                    return t, z, False
                probs = beta_w / sum_beta
                chosen = int(rng.choice(inf_nodes, p=probs))
                z |= 1 << chosen

        steps += 1


def estimate_mc(
    inp: RPIInputs,
    *,
    origin_z: int,
    target_healthy: int,
    trials: int,
    seed: int | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(int(trials)):
        _, _, ok = simulate_until_hit_or_return(
            inp,
            t0=0.0,
            z0=origin_z,
            origin_z=origin_z,
            target_healthy=target_healthy,
            rng=rng,
        )
        successes += int(ok)
    return float(successes / float(trials))


def estimate_importance_splitting(
    inp: RPIInputs,
    *,
    origin_z: int,
    levels_healthy: list[int],
    sims_per_stage: list[int],
    max_particles: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Importance splitting estimator for:
      P(hit level_m before returning to origin_z),
    where each stage r estimates:
      P(hit level_r before returning to origin_z | started from hitting distribution of level_{r-1}).
    """
    n = len(inp.nodelist)
    if not levels_healthy:
        raise ValueError("levels_healthy must not be empty.")
    if len(levels_healthy) != len(sims_per_stage):
        raise ValueError("levels_healthy and sims_per_stage must have the same length.")

    levels = [int(x) for x in levels_healthy]
    sims = [int(x) for x in sims_per_stage]
    if any(x <= 0 for x in sims):
        raise ValueError("All sims_per_stage values must be > 0.")
    if any(l < 0 or l > n for l in levels):
        raise ValueError(f"All levels_healthy must be in [0, {n}].")
    if any(levels[i] <= levels[i - 1] for i in range(1, len(levels))):
        raise ValueError("levels_healthy must be strictly increasing.")

    rng = np.random.default_rng(seed)

    # Standard splitting with resampling:
    # - Stage i uses sims_per_stage[i] particles.
    # - Each particle is simulated once until it either hits level_i or returns to origin_z.
    # - The conditional probability is estimated by the success fraction.
    # - Success end states are resampled (with replacement) to seed the next stage.
    p_hat = 1.0

    m0 = sims[0]
    particles: list[tuple[float, int]] = [(0.0, int(origin_z)) for _ in range(int(m0))]

    for i, level in enumerate(levels):
        successes: list[tuple[float, int]] = []
        for (t0, z0) in particles:
            t_end, z_end, ok = simulate_until_hit_or_return(
                inp,
                t0=t0,
                z0=z0,
                origin_z=origin_z,
                target_healthy=level,
                rng=rng,
            )
            if ok:
                successes.append((t_end, z_end))

        p_stage = (len(successes) / float(len(particles))) if particles else 0.0
        p_hat *= p_stage
        if not successes:
            return 0.0

        if i == len(levels) - 1:
            return float(p_hat)

        # Optional cap on the support set used for resampling (performance only).
        if max_particles is not None and len(successes) > int(max_particles):
            idx = rng.choice(len(successes), size=int(max_particles), replace=False)
            successes = [successes[int(j)] for j in idx]

        m_next = sims[i + 1]
        idx = rng.integers(0, len(successes), size=int(m_next))
        particles = [successes[int(j)] for j in idx]

    return float(p_hat)


