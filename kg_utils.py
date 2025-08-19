import re
import textwrap
import json
import zlib
import uuid
import colorsys
from collections import defaultdict

import pandas as pd
import networkx as nx
from networkx.algorithms import community


def clean_and_wrap(text, width=50):
    """Remove HTML tags and wrap text to the specified width."""
    no_html = re.sub(r'<.*?>', '', text)
    normalized = re.sub(r'\s+', ' ', no_html).strip()
    return textwrap.fill(
        normalized, width=width, break_long_words=False, break_on_hyphens=False
    )


def node_kind(n: dict):
    """Unified accessor for a node's category.

    Prefer explicit 'type', then fall back to 'group', else 'default'.
    """
    return n.get("type") or n.get("group") or "default"


def prepare_kgs_data(subgraph_nodes, subgraph_edges):
    """
    Prepare data in the .kgs format to be compatible with 02-trim_nodes.py.

    The format is a zlib-compressed JSON object representing an IsolationState.
    Original node/edge properties are preserved, with only vis.js-specific
    fields added or overwritten.
    """
    node_degrees = defaultdict(int)
    for edge in subgraph_edges:
        node_degrees[edge.get("source")] += 1
        node_degrees[edge.get("target")] += 1

    processed_nodes = []
    for node in subgraph_nodes:
        new_node = dict(node)
        new_node["value"] = node_degrees[new_node.get("id")]
        new_node["shape"] = "dot"

        if "summary" not in new_node:
            new_node["summary"] = ""
        if "type" not in new_node:
            new_node["type"] = new_node.get("group", "default")

        new_node["title"] = clean_and_wrap(new_node.get("summary", ""))

        if "color" not in new_node:
            group = new_node.get("group") or new_node.get("type") or "default"
            new_node["_group_for_color"] = group

        if "font" not in new_node:
            new_node["font"] = {"color": "white", "strokeWidth": 2, "strokeColor": "black"}

        processed_nodes.append(new_node)

    groups_for_coloring = defaultdict(list)
    for node in processed_nodes:
        if "_group_for_color" in node:
            groups_for_coloring[node["_group_for_color"]].append(node)
            del node["_group_for_color"]

    group_names = sorted(groups_for_coloring)
    total_groups = len(group_names)
    for idx, group in enumerate(group_names):
        h = idx / max(1, total_groups)
        s, v = 0.6, 0.85
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_hex = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        for node in groups_for_coloring[group]:
            node["color"] = {"background": color_hex}

    processed_edges = []
    for edge in subgraph_edges:
        new_edge = dict(edge)
        new_edge["from"] = new_edge.pop("source")
        new_edge["to"] = new_edge.pop("target")

        if "id" not in new_edge:
            new_edge["id"] = str(uuid.uuid4())
        if "label" not in new_edge:
            new_edge["label"] = new_edge.get("type", "")
        if "arrows" not in new_edge:
            new_edge["arrows"] = "to"
        if "font" not in new_edge:
            new_edge["font"] = {"color": "white", "strokeWidth": 2, "strokeColor": "black"}

        processed_edges.append(new_edge)

    state = {
        "type": "IsolationState",
        "version": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "isolateStack": [
            {"nodes": processed_nodes, "edges": processed_edges, "deleted": []}
        ],
    }

    json_string = json.dumps(state, indent=2)
    compressed_data = zlib.compress(json_string.encode("utf-8"))
    return compressed_data


def get_subgraph(graph_obj, result_node_ids, node_lookup, valid_edges):
    """Return the subgraph containing result nodes and their immediate neighbors."""
    neighbors = {
        n
        for node_id in result_node_ids
        if node_id in graph_obj
        for n in list(graph_obj.predecessors(node_id)) + list(graph_obj.successors(node_id))
    }
    subgraph_node_ids = set(result_node_ids) | neighbors

    subgraph_nodes = [node_lookup[nid] for nid in subgraph_node_ids if nid in node_lookup]
    subgraph_edges = [
        edge
        for edge in valid_edges
        if edge.get("source") in subgraph_node_ids and edge.get("target") in subgraph_node_ids
    ]

    return subgraph_nodes, subgraph_edges


def calculate_graph_metrics(graph, node_lookup, warn=None):
    """
    Calculate centrality and community metrics for a graph.

    Parameters
    ----------
    graph : nx.Graph
        Graph to analyze.
    node_lookup : dict
        Mapping of node IDs to node information.
    warn : callable, optional
        Function used to emit warning messages.
    """
    if warn is None:
        warn = lambda *args, **kwargs: None  # noqa: E731

    if not graph.nodes():
        return None

    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1.0e-6)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {n: 0.0 for n in graph.nodes()}
        warn("Eigenvector centrality did not converge.")

    pagerank = nx.pagerank(graph)

    try:
        communities_generator = community.greedy_modularity_communities(graph.to_undirected())
        communities = [list(c) for c in communities_generator]
        node_community = {node_id: i for i, com in enumerate(communities) for node_id in com}
    except Exception as e:  # pragma: no cover - community detection is best effort
        communities = []
        node_community = {}
        warn(f"Could not perform community detection: {e}")

    metrics_df = pd.DataFrame(
        {
            "Degree": degree_centrality,
            "Betweenness": betweenness_centrality,
            "Eigenvector": eigenvector_centrality,
            "PageRank": pagerank,
        }
    )
    metrics_df["Community"] = pd.Series(node_community)
    metrics_df["Label"] = [node_lookup[nid].get("label", nid) for nid in metrics_df.index]

    metrics_df = metrics_df.reset_index().rename(columns={"index": "ID"})
    column_order = [
        "ID",
        "Label",
        "Degree",
        "Betweenness",
        "Eigenvector",
        "PageRank",
        "Community",
    ]
    metrics_df = metrics_df[[col for col in column_order if col in metrics_df.columns]]
    return metrics_df

