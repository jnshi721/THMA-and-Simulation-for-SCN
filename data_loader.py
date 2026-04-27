from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import networkx as nx


REQUIRED_PARAM_COLUMNS: tuple[str, ...] = (
    "beta_a",
    "beta_b",
    "beta_c",
    "beta_i",
    "delta_a",
    "delta_b",
    "delta_c",
    "delta_i",
)


def load_graph_and_params(
    edge_csv: str | Path,
    params_csv: str | Path,
    *,
    edge_source_col: str = "enterprise_i",
    edge_target_col: str = "enterprise_j",
    node_col: str | None = "enterprise",
    required_param_cols: Iterable[str] = REQUIRED_PARAM_COLUMNS,
    coerce_node_ids_to_str: bool = True,
) -> tuple[list[str], nx.Graph, pd.DataFrame]:
    edges = pd.read_csv(edge_csv)
    required_edges = {edge_source_col, edge_target_col}
    if not required_edges.issubset(edges.columns):
        raise KeyError(
            f"Edge CSV must contain columns {sorted(required_edges)}; got: {list(edges.columns)}"
        )

    required_param_cols = tuple(required_param_cols)

    params_raw = pd.read_csv(params_csv)
    required_params = set(required_param_cols) if node_col is None else {node_col, *required_param_cols}
    if not required_params.issubset(params_raw.columns):
        raise KeyError(
            f"Params CSV must contain columns {sorted(required_params)}; got: {list(params_raw.columns)}"
        )

    if coerce_node_ids_to_str and node_col is not None:
        edges = edges.copy()
        edges[edge_source_col] = edges[edge_source_col].astype(str).str.strip()
        edges[edge_target_col] = edges[edge_target_col].astype(str).str.strip()

        params_raw = params_raw.copy()
        params_raw[node_col] = params_raw[node_col].astype(str).str.strip()

    g_raw = nx.from_pandas_edgelist(edges, edge_source_col, edge_target_col, create_using=nx.Graph())
    nodelist = sorted(g_raw.nodes(), key=str)

    if node_col is None:
        if len(params_raw) != len(nodelist):
            raise ValueError(
                f"Params CSV has no node column; expected exactly {len(nodelist)} rows to match graph nodes, got {len(params_raw)}."
            )
        params = params_raw.copy()
        params.index = nodelist
    else:
        params = params_raw.set_index(node_col).reindex(nodelist)

        missing_nodes = [n for n in nodelist if n not in set(params_raw[node_col])]
        if missing_nodes:
            raise ValueError(f"Missing parameter rows for nodes: {missing_nodes}")
    missing_values = params.loc[:, list(required_param_cols)].isna().any(axis=1)
    if bool(missing_values.any()):
        bad = [str(n) for n, is_bad in missing_values.items() if bool(is_bad)]
        raise ValueError(f"Parameter row has missing values for nodes: {bad}")

    g = nx.Graph()
    g.add_nodes_from(nodelist)
    g.add_edges_from(g_raw.edges())

    return nodelist, g, params


def build_neighbors(nodelist: list[str], g: nx.Graph) -> list[list[int]]:
    idx = {n: i for i, n in enumerate(nodelist)}
    neighbors: list[list[int]] = [[] for _ in range(len(nodelist))]
    for u, v in g.edges():
        iu, iv = idx[u], idx[v]
        neighbors[iu].append(iv)
        neighbors[iv].append(iu)
    return neighbors


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def bitmask_from_infected(
    nodelist: list[str],
    infected: Iterable[str] | None,
    *,
    default_first: bool = True,
) -> int:
    if not nodelist:
        raise ValueError("Empty node list.")
    if infected is None:
        infected_nodes: list[str] = []
    else:
        infected_nodes = dedupe_preserve_order(infected)
    if default_first and len(infected_nodes) == 0:
        infected_nodes = [nodelist[0]]

    idx = {n: i for i, n in enumerate(nodelist)}
    z0 = 0
    for node in infected_nodes:
        if node not in idx:
            raise ValueError(f"Initial infected node {node!r} not in nodes: {nodelist}")
        z0 |= 1 << idx[node]
    return z0


