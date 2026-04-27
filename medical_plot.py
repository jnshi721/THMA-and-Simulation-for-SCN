from __future__ import annotations

from pathlib import Path
import argparse
import random

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _build_graph(edge_csv: Path) -> nx.Graph:
    edges = pd.read_csv(edge_csv)
    required = {"enterprise_i", "enterprise_j"}
    if not required.issubset(edges.columns):
        raise KeyError(f"Edge CSV must contain columns {sorted(required)}; got: {list(edges.columns)}")

    edges = edges.copy()
    edges["enterprise_i"] = edges["enterprise_i"].astype(str).str.strip()
    edges["enterprise_j"] = edges["enterprise_j"].astype(str).str.strip()
    g = nx.from_pandas_edgelist(edges, "enterprise_i", "enterprise_j", create_using=nx.Graph())

    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        sizes = sorted((len(c) for c in components), reverse=True)
        raise ValueError(f"Input network is not connected. Components={len(components)}, sizes={sizes}")
    return g


def _load_industry_info(info_csv: Path) -> dict[str, str]:
    info = pd.read_csv(info_csv)
    required = {"enterprise", "industry code"}
    if not required.issubset(info.columns):
        raise KeyError(f"Company info CSV must contain columns {sorted(required)}; got: {list(info.columns)}")

    info = info.copy()
    info["enterprise"] = info["enterprise"].astype(str).str.strip()
    info["industry code"] = info["industry code"].astype(str).str.strip()
    return dict(zip(info["enterprise"], info["industry code"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_csv",
        type=str,
        default=None,
        help="Edge CSV (default: Listed Medical Industry Supply Chain Network.csv)",
    )
    parser.add_argument(
        "--info_csv",
        type=str,
        default=None,
        help="Company info CSV (default: Listed Medical Company Information.csv)",
    )
    parser.add_argument("--seed", type=int, default=20260127, help="Random seed for reproducible layout")
    parser.add_argument("--out", type=str, default=None, help="Output png filename (optional)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    result_dir = here / "result"
    edge_csv = Path(args.edge_csv) if args.edge_csv else (data_dir / "Listed Medical Industry Supply Chain Network.csv")
    info_csv = Path(args.info_csv) if args.info_csv else (data_dir / "Listed Medical Company Information.csv")

    g = _build_graph(edge_csv)
    industry_map = _load_industry_info(info_csv)

    # Assign industry code as layer for layout and coloring.
    present_industries = sorted(
        {industry_map.get(node) for node in g.nodes() if industry_map.get(node)}
    )
    if not present_industries:
        raise ValueError("No industry code found for nodes in the graph.")

    has_unknown = False
    for node in g.nodes():
        industry = industry_map.get(node)
        if industry:
            g.nodes[node]["layer"] = industry
        else:
            g.nodes[node]["layer"] = "Unknown"
            has_unknown = True

    base_colors = list(mcolors.TABLEAU_COLORS.values())
    if len(base_colors) < len(present_industries) + 1:
        base_colors += list(mcolors.CSS4_COLORS.values())
    industry_color = {ind: base_colors[i % len(base_colors)] for i, ind in enumerate(present_industries)}
    if has_unknown:
        industry_color["Unknown"] = "#999999"

    node_colors = [industry_color[g.nodes[n]["layer"]] for n in g.nodes()]

    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal", scale=0.8)
    pos = {n: (xy[0] * 0.7, xy[1]) for n, xy in pos.items()}

    plt.figure(figsize=(8, 6))
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_color=node_colors,
        edgecolors="black",
        node_size=600,
        font_size=9,
        font_color="white",
        edge_color="black",
        width=1.2,
        style="solid",
        connectionstyle="arc3,rad=0.2",
        arrows=True,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=ind,
            markerfacecolor=industry_color[ind],
            markersize=12,
        )
        for ind in sorted(industry_color.keys())
    ]
    plt.legend(handles=legend_handles, handletextpad=0.4, title="Industry", loc="upper left", fontsize=14, title_fontsize=14)

    out = args.out or "medical_plot_medical_supply_chain.png"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = result_dir / out_path
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


