import json
import os
import re
import textwrap
import uuid
import zlib

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from networkx.algorithms import community
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kg_compressor import KGCompressor


def clean_and_wrap(text, width=50):
    # Remove all HTML tags (anything between < and >)
    no_html = re.sub(r"<.*?>", "", text)
    # Replace multiple whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", no_html).strip()
    # Wrap text to the desired width, breaking only at word boundaries
    wrapped = textwrap.fill(
        normalized,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    # Ensure explicit "\n" (textwrap.fill already uses \n at end of lines)
    return wrapped


st.set_page_config(layout="wide")
st.title("Knowledge Graph Semantic Explorer")
# Width (in pixels) for embedded graph iframes
GRAPH_HTML_WIDTH = 1200

# ---- Session State Management ----
if st.session_state.get("_hard_reset_flag", False):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def hard_reset():
    st.session_state._hard_reset_flag = True


# ---- Model Configuration ----
MODEL_OPTIONS = {
    "Large (bge-large-en-v1.5)": "BAAI/bge-large-en-v1.5",
    "Small (bge-micro-v2)": "TaylorAI/bge-micro-v2",
    "BioBERT (biomedical)": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
}
MODEL_DISPLAY_NAMES = list(MODEL_OPTIONS.keys())


# ---- Helper Functions ----
def decompress_file(file_content, file_name):
    try:
        data = KGCompressor.decompress_data(file_content)
        if "nodes" not in data or "edges" not in data:
            st.error("Invalid file: 'nodes' or 'edges' key missing.")
            return None
        if file_name.endswith(".kge"):
            if "embeddings" not in data:
                st.warning(
                    "Warning: .kge file is missing 'embeddings' data. Will regenerate.",
                )
            else:
                data["embeddings"] = np.array(data["embeddings"])
        return data
    except Exception as e:
        st.error(f"Failed to decompress file. It may be corrupted. Details: {e}")
        return None


def compress_kge_data(nodes, edges, embeddings, model_name):
    try:
        embeddings_list = embeddings.tolist()
        combined_data = {
            "nodes": nodes,
            "edges": edges,
            "embeddings": embeddings_list,
            "model_name": model_name,
        }
        return KGCompressor.compress_data(combined_data)
    except Exception as e:
        st.error(f"Failed to compress data for saving: {e}")
        return None


def prepare_kgs_data(subgraph_nodes, subgraph_edges):
    """
    Prepares data in the .kgs format (zlib compressed JSON).

    Ensures edges have the correct 'from', 'to', and 'id' keys.
    Ensure all nodes have a unique ID, default to label if not present
    """
    for node in subgraph_nodes:
        if "id" not in node:
            node["id"] = node.get("label", str(uuid.uuid4()))

    # Correctly format edges
    formatted_edges = []
    for edge in subgraph_edges:
        formatted_edge = {
            "from": edge.get("source"),
            "to": edge.get("target"),
            "id": str(uuid.uuid4()),  # Add a unique ID
            "label": edge.get("label", ""),
            "arrows": edge.get("arrows", "to"),
        }
        # Carry over font styling if it exists
        if "font" in edge:
            formatted_edge["font"] = edge["font"]
        formatted_edges.append(formatted_edge)

    state = {
        "type": "IsolationState",
        "version": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "isolateStack": [
            {
                "nodes": subgraph_nodes,
                "edges": formatted_edges,
                "deleted": [],
            },
        ],
    }
    json_string = json.dumps(state, indent=2)
    compressed_data = zlib.compress(json_string.encode("utf-8"))
    return compressed_data


def display_subgraph(
    graph_obj,
    result_node_ids,
    node_lookup,
    valid_edges,
    vis_options,
    source_node_id=None,
):
    """
    Refactored function to generate and display a subgraph visualization.

    Returns the nodes and edges of the displayed subgraph.
    """
    neighbors = set(
        n
        for node_id in result_node_ids
        if node_id in graph_obj
        for n in list(graph_obj.predecessors(node_id))
        + list(graph_obj.successors(node_id))
    )
    subgraph_node_ids = set(result_node_ids) | neighbors

    sub_net = Network(
        height="100vh",
        width="100%",
        notebook=False,
        directed=True,
        cdn_resources="in_line",
    )
    sub_net.set_options(json.dumps(vis_options))

    subgraph_nodes = []
    for node_id in subgraph_node_ids:
        if node_id in node_lookup:
            node_info = node_lookup[node_id]
            if node_id == source_node_id:
                color = "red"
            elif node_id in result_node_ids:
                color = "#00ff00"
            else:
                color = "#97c2fc"
            sub_net.add_node(
                node_id,
                label=node_info["label"],
                title=clean_and_wrap(node_info.get("summary", "")),
                color=color,
            )
            subgraph_nodes.append(node_info)

    subgraph_edges = []
    for edge in valid_edges:
        if edge["source"] in subgraph_node_ids and edge["target"] in subgraph_node_ids:
            sub_net.add_edge(
                edge["source"],
                edge["target"],
                title=edge.get("label", ""),
                label=edge.get("label", ""),
            )
            subgraph_edges.append(edge)

    sub_net.generate_html()
    st.components.v1.html(
        sub_net.html,
        height=650,
        width=GRAPH_HTML_WIDTH,
        scrolling=True,
    )

    return subgraph_nodes, subgraph_edges


# ---- Sidebar Setup & File Upload ----
st.sidebar.title("File Operations")
uploaded_file = st.sidebar.file_uploader(
    "Load Graph",
    type=["kgc", "kge"],
    help="Load a `.kgc` (graph only) or `.kge` (graph + embeddings) file.",
)

if uploaded_file is None:
    st.info("Please upload a .kgc or .kge file to begin.")
    st.stop()

# ---- File Processing and State Initialization ----
if (
    "graph_data" not in st.session_state
    or st.session_state.get("uploaded_file_name") != uploaded_file.name
):
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.status_messages = []
    file_content = uploaded_file.read()
    st.session_state.graph_data = decompress_file(file_content, uploaded_file.name)

    model_index = 0  # Default to the first model
    if st.session_state.graph_data and uploaded_file.name.endswith(".kge"):
        loaded_model_name = st.session_state.graph_data.get("model_name")
        if loaded_model_name:
            if loaded_model_name in MODEL_OPTIONS.values():
                # Find the index of the loaded model
                model_index = list(MODEL_OPTIONS.values()).index(loaded_model_name)
            else:
                st.session_state.status_messages.append(
                    {
                        "type": "warning",
                        "msg": f"Model '{loaded_model_name}' from file is not available. Defaulting to '{MODEL_DISPLAY_NAMES[0]}'.",
                    },
                )
        else:
            st.session_state.status_messages.append(
                {
                    "type": "info",
                    "msg": f"No model specified in file. Defaulting to '{MODEL_DISPLAY_NAMES[0]}'.",
                },
            )
    st.session_state.model_index = model_index
    st.rerun()

# ---- Sidebar UI ----
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Select Embedding Model",
    MODEL_DISPLAY_NAMES,
    index=st.session_state.get("model_index", 0),
    help="Select the model for semantic search. 'Large' is accurate, 'Small' is fast, 'BioBERT' is for biomedical text.",
)
MODEL_NAME = MODEL_OPTIONS[model_choice]
st.sidebar.button(
    "Reset App State",
    on_click=hard_reset,
    help="Click to clear all loaded data and start over.",
)

# ---- Status Message Area in Sidebar ----
if "status_messages" in st.session_state and st.session_state.status_messages:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Graph Status")
    for msg_info in st.session_state.status_messages:
        if msg_info["type"] == "success":
            st.sidebar.success(msg_info["msg"])
        elif msg_info["type"] == "warning":
            st.sidebar.warning(msg_info["msg"])
        elif msg_info["type"] == "info":
            st.sidebar.info(msg_info["msg"])

# ---- Embedding Generation and Loading ----
if (
    "embeddings" not in st.session_state
    or st.session_state.get("current_model_name") != MODEL_NAME
):
    st.session_state.current_model_name = MODEL_NAME
    with st.spinner(f"Initializing embedding model ({MODEL_NAME})..."):
        st.session_state.model = SentenceTransformer(MODEL_NAME)

    if st.session_state.graph_data:
        nodes = st.session_state.graph_data["nodes"]
        edges = st.session_state.graph_data["edges"]

        regenerate_embeddings = True
        if (
            uploaded_file.name.endswith(".kge")
            and "embeddings" in st.session_state.graph_data
        ):
            loaded_model_name = st.session_state.graph_data.get("model_name")
            if loaded_model_name == MODEL_NAME:
                st.session_state.embeddings = st.session_state.graph_data["embeddings"]
                if not any(
                    d.get("msg", "").startswith("Loaded embeddings")
                    for d in st.session_state.status_messages
                ):
                    st.session_state.status_messages.append(
                        {
                            "type": "success",
                            "msg": f"Loaded embeddings from .kge file (Model: {loaded_model_name}).",
                        },
                    )
                regenerate_embeddings = False

        if regenerate_embeddings:
            with st.spinner(f"Generating embeddings with {MODEL_NAME}..."):
                node_texts = [n.get("summary") or n["label"] for n in nodes]
                st.session_state.embeddings = st.session_state.model.encode(
                    node_texts,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                )
            st.session_state.status_messages.append(
                {"type": "success", "msg": f"Embeddings generated with {MODEL_NAME}."},
            )

        kge_data = compress_kge_data(
            nodes,
            edges,
            st.session_state.embeddings,
            MODEL_NAME,
        )
        if kge_data:
            st.session_state.kge_data = kge_data
    st.rerun()

if "graph_data" not in st.session_state or not st.session_state.graph_data:
    st.error("Failed to load or process the graph file.")
    st.stop()

# ---- Main Application Logic ----
nodes = st.session_state.graph_data["nodes"]
edges = st.session_state.graph_data["edges"]
embeddings = st.session_state.embeddings

node_ids = [n["id"] for n in nodes]
node_lookup = {node["id"]: node for node in nodes}
node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

st.sidebar.download_button(
    label="Save Enhanced Graph (.kge)",
    data=st.session_state.get("kge_data", b""),
    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_embeddings.kge",
    mime="application/octet-stream",
    help="Save the graph and its embeddings to a compressed .kge file. Disabled until processing is complete.",
    disabled="kge_data" not in st.session_state,
)

valid_node_ids = set(node["id"] for node in nodes)
valid_edges = [
    edge
    for edge in edges
    if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
]
skipped_edges = [edge for edge in edges if edge not in valid_edges]

G = nx.DiGraph()
G.add_nodes_from((node["id"], node) for node in nodes)
G.add_edges_from([(e["source"], e["target"], e) for e in valid_edges])

vis_options = {
    "interaction": {"zoomSpeed": 0.2},
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -100,
            "centralGravity": 0.01,
            "springLength": 200,
            "springConstant": 0.08,
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
    },
}

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Full Graph",
        "Find Similar Nodes",
        "Semantic Search",
        "Diagnostics",
        "Graph Anatomy",
    ],
)

with tab1:
    st.header("Interactive Full Graph")
    net = Network(
        height="100vh",
        width="100%",
        notebook=False,
        directed=True,
        cdn_resources="in_line",
    )
    net.set_options(json.dumps(vis_options))
    for node in nodes:
        label = node["label"]
        tooltip = clean_and_wrap(node.get("summary", ""))
        net.add_node(node["id"], label=label, title=tooltip)
    for edge in valid_edges:
        net.add_edge(
            edge["source"],
            edge["target"],
            title=edge.get("label", ""),
            label=edge.get("label", ""),
        )

    net.generate_html()
    st.components.v1.html(net.html, height=800, width=GRAPH_HTML_WIDTH, scrolling=True)

with tab2:
    st.header("Find Similar Nodes")
    selected_sim_node = st.selectbox(
        "Select a node to find similar ones",
        sorted(node_ids),
        key="sim_node",
    )
    top_k = st.slider("Number of similar nodes to find", 1, 15, 5, key="sim_k")

    if selected_sim_node:
        idx = node_id_to_index[selected_sim_node]
        sims = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings)[0]
        top_indices = np.argsort(sims)[::-1][1 : top_k + 1]

        st.write(f"Top {top_k} nodes similar to **{selected_sim_node}**:")
        result_node_ids = [selected_sim_node]
        for i in top_indices:
            st.write(f"- {node_ids[i]} (Similarity: {sims[i]:.2f})")
            result_node_ids.append(node_ids[i])

        subgraph_nodes, subgraph_edges = display_subgraph(
            G,
            result_node_ids,
            node_lookup,
            valid_edges,
            vis_options,
            source_node_id=selected_sim_node,
        )

        if subgraph_nodes:
            kgs_data = prepare_kgs_data(subgraph_nodes, subgraph_edges)
            st.download_button(
                label="Export Subgraph (.kgs)",
                data=kgs_data,
                file_name=f"subgraph_{selected_sim_node}.kgs",
                mime="application/octet-stream",
                help="Save the displayed subgraph for use in an interactive HTML graph.",
            )
            st.info(
                f"Subgraph contains {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges.",
            )

with tab3:
    st.header("Semantic Search")
    query = st.text_input("Search for nodes based on meaning:")
    search_top_k = st.slider("Number of search results", 1, 15, 5, key="search_k")

    if query:
        query_vec = st.session_state.model.encode(
            [query],
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        sims = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:search_top_k]

        st.write("Top matching nodes:")
        result_node_ids = [node_ids[i] for i in top_indices]
        for i in top_indices:
            st.write(f"- {node_ids[i]} (Similarity: {sims[i]:.2f})")

        subgraph_nodes, subgraph_edges = display_subgraph(
            G,
            result_node_ids,
            node_lookup,
            valid_edges,
            vis_options,
        )

        if subgraph_nodes:
            kgs_data = prepare_kgs_data(subgraph_nodes, subgraph_edges)
            st.download_button(
                label="Export Subgraph (.kgs)",
                data=kgs_data,
                file_name="subgraph_search.kgs",
                mime="application/octet-stream",
                help="Save the displayed subgraph for use in an interactive HTML graph.",
            )
            st.info(
                f"Subgraph contains {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges.",
            )

with tab4:
    st.header("Graph Diagnostics")
    st.write(f"Total Nodes: {len(nodes)}")
    st.write(f"Total Valid Edges: {len(valid_edges)}")

    if skipped_edges:
        st.warning(f"{len(skipped_edges)} edge(s) were skipped due to missing nodes.")
        with st.expander("Show Skipped Edges"):
            st.json(skipped_edges)

    orphans = [n for n in G.nodes if G.degree(n) == 0]
    if orphans:
        st.info(f"{len(orphans)} orphan node(s) (no edges).")
        with st.expander("Show Orphan Nodes"):
            st.code("\n".join(sorted(orphans)), language="text")

    st.subheader("Node Inspector")
    selected_node_id = st.selectbox(
        "Choose a node to inspect:",
        sorted(list(valid_node_ids)),
    )
    if selected_node_id:
        st.json(node_lookup[selected_node_id])

with tab5:
    st.header("Graph Anatomy")

    @st.cache_data
    def calculate_graph_metrics(_G):
        if not _G.nodes():
            return None

        # Centrality Measures
        degree_centrality = nx.degree_centrality(_G)
        betweenness_centrality = nx.betweenness_centrality(_G)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(
                _G,
                max_iter=1000,
                tol=1.0e-6,
            )
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = dict.fromkeys(_G.nodes(), 0.0)
            st.warning("Eigenvector centrality did not converge.")

        pagerank = nx.pagerank(_G)

        # Community Detection
        try:
            communities_generator = community.greedy_modularity_communities(
                _G.to_undirected(),
            )
            communities = [list(c) for c in communities_generator]
            node_community = {
                node_id: i for i, com in enumerate(communities) for node_id in com
            }
        except Exception as e:
            communities = []
            node_community = {}
            st.warning(f"Could not perform community detection: {e}")

        # Combine into a DataFrame, ensuring proper alignment
        metrics_df = pd.DataFrame(
            {
                "Degree": degree_centrality,
                "Betweenness": betweenness_centrality,
                "Eigenvector": eigenvector_centrality,
                "PageRank": pagerank,
            },
        )
        metrics_df["Community"] = pd.Series(node_community)
        # Map the node IDs (index) to their labels
        metrics_df["Label"] = [
            node_lookup[nid].get("label", nid) for nid in metrics_df.index
        ]

        # Reorder columns for display and add ID
        metrics_df = metrics_df.reset_index().rename(columns={"index": "ID"})

        # Ensure desired column order
        column_order = [
            "ID",
            "Label",
            "Degree",
            "Betweenness",
            "Eigenvector",
            "PageRank",
            "Community",
        ]
        metrics_df = metrics_df[
            [col for col in column_order if col in metrics_df.columns]
        ]

        return metrics_df

    metrics_df = calculate_graph_metrics(G)

    if metrics_df is not None:
        st.subheader("Graph Metrics Data")
        st.dataframe(metrics_df.sort_values(by="Degree", ascending=False))

        st.subheader("Graph Anatomy Map")

        # Visualization settings
        hub_threshold = metrics_df["Degree"].quantile(0.9)
        bridge_threshold = metrics_df["Betweenness"].quantile(0.9)

        net_anatomy = Network(
            height="100vh",
            width="100%",
            notebook=False,
            directed=True,
            cdn_resources="in_line",
        )
        net_anatomy.set_options(json.dumps(vis_options))

        # Use metrics_df as a lookup, but iterate over G.nodes() for consistency
        metrics_lookup = metrics_df.set_index("ID")

        # Add nodes with visual encoding
        for node_id in G.nodes():
            # Look up metrics for the node
            try:
                row = metrics_lookup.loc[node_id]
                label = row["Label"]
                size = 15 + row["Degree"] * 50  # Scale size by degree

                # Color by community
                community_id = row["Community"]
                if pd.notna(community_id):
                    hue = int((community_id * 137.5) % 360)
                    color = f"hsl({hue}, 70%, 50%)"
                else:
                    color = "#808080"  # Grey for no community

                border_color = "black"
                if row["Betweenness"] > bridge_threshold:
                    border_color = "red"  # Highlight bridges

                title = f"""
                <b>{label}</b><br>
                Degree: {row["Degree"]:.3f}<br>
                Betweenness: {row["Betweenness"]:.3f}<br>
                Eigenvector: {row["Eigenvector"]:.3f}<br>
                PageRank: {row["PageRank"]:.3f}<br>
                Community: {row["Community"]}
                """

                net_anatomy.add_node(
                    node_id,
                    label=label,
                    title=title,
                    size=size,
                    color=color,
                    borderWidth=3 if border_color == "red" else 1,
                    borderColor=border_color,
                )
            except KeyError:
                # This node was in the graph but not in the metrics, add it with default styling
                node_info = node_lookup.get(node_id, {})
                label = node_info.get("label", node_id)
                title = f"<b>{label}</b><br>(Metrics not available)"
                net_anatomy.add_node(node_id, label=label, title=title, color="#CCCCCC")

        # Add edges
        for edge in valid_edges:
            # Ensure both source and target nodes exist before adding an edge
            if edge["source"] in G.nodes() and edge["target"] in G.nodes():
                net_anatomy.add_edge(
                    edge["source"],
                    edge["target"],
                    title=edge.get("label", ""),
                    label=edge.get("label", ""),
                )

        # Save and display the graph
        net_anatomy.generate_html()
        st.components.v1.html(
            net_anatomy.html,
            height=800,
            width=GRAPH_HTML_WIDTH,
            scrolling=True,
        )
    else:
        st.info("Graph is empty. No metrics to display.")
