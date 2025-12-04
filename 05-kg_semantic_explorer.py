import streamlit as st
st.set_page_config(layout="wide")
st.markdown(
    "<style>iframe[title='st.components.v1.html']{max-width:100% !important;width:100% !important;}</style>",
    unsafe_allow_html=True
)
import json
import re
import tempfile
from pyvis.network import Network
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from UTL_kg_compressor import KGCompressor
import pandas as pd
from networkx.algorithms import community
import colorsys
import plotly.graph_objects as go

from UTL_kg_utils import (
    clean_and_wrap,
    node_kind,
    prepare_kgs_data,
    get_subgraph,
    calculate_graph_metrics,
)


# st.markdown("""
    # <style>
    # /* Use full-width content area */
    # .block-container {max-width: 100% !important; padding-left: 1rem; padding-right: 1rem;}
    # /* Make the components iframe stretch */
    # iframe[title="st.components.v1.html"] {width: 100% !important;}
    # /* Ensure tab panels take full width */
    # [data-testid="stTabs"] div[role="tabpanel"] {width: 100%;}
    # </style>
    # """, unsafe_allow_html=True)

st.title("Knowledge Graph Semantic Explorer")

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
    "BioBERT (biomedical)": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
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
                st.warning("Warning: .kge file is missing 'embeddings' data. Will regenerate.")
            else:
                data["embeddings"] = np.array(data["embeddings"])
        return data
    except Exception as e:
        st.error(f"Failed to decompress file. It may be corrupted. Details: {e}")
        return None

def compress_kge_data(nodes, edges, embeddings, model_name):
    try:
        embeddings_list = embeddings.tolist()
        combined_data = {"nodes": nodes, "edges": edges, "embeddings": embeddings_list, "model_name": model_name}
        return KGCompressor.compress_data(combined_data)
    except Exception as e:
        st.error(f"Failed to compress data for saving: {e}")
        return None


def generate_edge_color_map(edges):
    edge_types = sorted({e.get("type", "default") for e in edges})
    total = len(edge_types)
    color_map = {}
    for idx, etype in enumerate(edge_types):
        h = idx / max(1, total)
        s, v = 0.6, 0.85
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_map[etype] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return color_map


def display_subgraph(graph_obj, result_node_ids, node_lookup, valid_edges, vis_options, source_node_id=None, edge_colors=None):
    """
    Generates and displays a subgraph visualization without returning data.
    """
    subgraph_nodes, subgraph_edges = get_subgraph(graph_obj, result_node_ids, node_lookup, valid_edges)
    
    sub_net = Network(height="600px", width="100%", notebook=False, directed=True)
    sub_net.set_options(json.dumps(vis_options))

    for node_info in subgraph_nodes:
        node_id = node_info['id']
        if node_id == source_node_id:
            color = "red"
        elif node_id in result_node_ids:
            color = "#00ff00"
        else:
            color = "#97c2fc"
        sub_net.add_node(node_id, label=node_info["label"], title=clean_and_wrap(node_info.get("summary", "")), color=color)

    for edge in subgraph_edges:
        edge_label = edge.get("label") or edge.get("type", "")
        e_type = edge.get("type", "")
        e_color = edge_colors.get(e_type) if edge_colors else None
        width = edge.get("weight", 1)
        sub_net.add_edge(
            edge["source"],
            edge["target"],
            title=edge_label,
            label=edge_label,
            color=e_color,
            width=width,
        )
        
    html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    try:
        sub_net.write_html(html_file.name)
        with open(html_file.name, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=800, width=1600, scrolling=True)
    finally:
        html_file.close()
        os.remove(html_file.name)


# ---- Sidebar Setup & File Upload ----
class LocalFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.path = path
    def read(self):
        with open(self.path, 'rb') as f:
            return f.read()

st.sidebar.title("File Operations")
uploaded_file_widget = st.sidebar.file_uploader(
    "Load Graph",
    type=["kgc", "kge"],
    help="Load a `.kgc` (graph only) or `.kge` (graph + embeddings) file."
)

if st.sidebar.button("Load Demo Graph"):
    st.session_state.use_demo = True

uploaded_file = None
if uploaded_file_widget is not None:
    uploaded_file = uploaded_file_widget
    st.session_state.use_demo = False
elif st.session_state.get("use_demo", False):
    if os.path.exists("demo.kge"):
        uploaded_file = LocalFile("demo.kge")
    else:
        st.sidebar.error("Demo file 'demo.kge' not found.")

if uploaded_file is None:
    st.info("Please upload a .kgc or .kge file to begin, or click 'Load Demo Graph'.")
    st.stop()

# ---- File Processing and State Initialization ----
if "graph_data" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
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
                st.session_state.status_messages.append({
                    "type": "warning",
                    "msg": f"Model '{loaded_model_name}' from file is not available. Defaulting to '{MODEL_DISPLAY_NAMES[0]}'."
                })
        else:
            st.session_state.status_messages.append({
                "type": "info",
                "msg": f"No model specified in file. Defaulting to '{MODEL_DISPLAY_NAMES[0]}'."
            })
    st.session_state.model_index = model_index
    st.rerun()

# ---- Sidebar UI ----
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Select Embedding Model",
    MODEL_DISPLAY_NAMES,
    index=st.session_state.get('model_index', 0),
    help="Select the model for semantic search. 'Large' is accurate, 'Small' is fast, 'BioBERT' is for biomedical text."
)
MODEL_NAME = MODEL_OPTIONS[model_choice]
st.sidebar.button("Reset App State", on_click=hard_reset, help="Click to clear all loaded data and start over.")

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
if "embeddings" not in st.session_state or st.session_state.get("current_model_name") != MODEL_NAME:
    st.session_state.current_model_name = MODEL_NAME
    with st.spinner(f"Initializing embedding model ({MODEL_NAME})..."):
        st.session_state.model = SentenceTransformer(MODEL_NAME)

    if st.session_state.graph_data:
        nodes = st.session_state.graph_data["nodes"]
        edges = st.session_state.graph_data["edges"]
        
        regenerate_embeddings = True
        if uploaded_file.name.endswith(".kge") and "embeddings" in st.session_state.graph_data:
            loaded_model_name = st.session_state.graph_data.get("model_name")
            if loaded_model_name == MODEL_NAME:
                st.session_state.embeddings = st.session_state.graph_data["embeddings"]
                if not any(d.get('msg', '').startswith('Loaded embeddings') for d in st.session_state.status_messages):
                    st.session_state.status_messages.append({"type": "success", "msg": f"Loaded embeddings from .kge file (Model: {loaded_model_name})."})
                regenerate_embeddings = False

        if regenerate_embeddings:
            with st.spinner(f"Generating embeddings with {MODEL_NAME}..."):
                node_texts = [n.get("summary") or n["label"] for n in nodes]
                st.session_state.embeddings = st.session_state.model.encode(node_texts, convert_to_tensor=False, normalize_embeddings=True)
            st.session_state.status_messages.append({"type": "success", "msg": f"Embeddings generated with {MODEL_NAME}."})

        kge_data = compress_kge_data(nodes, edges, st.session_state.embeddings, MODEL_NAME)
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
    data=st.session_state.get('kge_data', b''),
    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_embeddings.kge",
    mime="application/octet-stream",
    help="Save the graph and its embeddings to a compressed .kge file. Disabled until processing is complete.",
    disabled='kge_data' not in st.session_state
)

valid_node_ids = set(node["id"] for node in nodes)
valid_edges = [edge for edge in edges if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids]
skipped_edges = [edge for edge in edges if edge not in valid_edges]

edge_color_map = generate_edge_color_map(valid_edges)

G = nx.DiGraph()
G.add_nodes_from((node["id"], node) for node in nodes)
G.add_edges_from([(e["source"], e["target"], e) for e in valid_edges])

# Compute community assignments once for optional coloring
communities = community.greedy_modularity_communities(G.to_undirected())
community_map = {n: idx for idx, comm in enumerate(communities) for n in comm}
community_colors = {}
community_names = {}

total_comms = len(communities)
for idx, comm in enumerate(communities):
    # Color generation
    h = idx / max(1, total_comms)
    s, v = 0.6, 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    community_colors[idx] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    # Name generation: Find node with highest degree in this community
    if comm:
        central_node_id = max(comm, key=lambda n: G.degree[n])
        central_label = node_lookup[central_node_id].get("label", central_node_id)
        # Remove square brackets and all text between them
        central_label = re.sub(r"[.*?]", "", central_label)
        # Ensure only one space before "Cluster" by stripping leading/trailing whitespace
        community_names[idx] = f"{central_label.strip()} Cluster"
    else:
        community_names[idx] = f"Community {idx}"

vis_options = {
    "interaction": { "zoomSpeed": 0.2 },
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -100,
            "centralGravity": 0.01,
            "springLength": 200,
            "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
    }
}

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Full Graph", "Find Similar Nodes", "Semantic Search", "Diagnostics", "Graph Anatomy"])

with tab1:
    st.header("Interactive Full Graph")
    color_by_community = st.checkbox("Colour nodes by community", value=True)
    layout_choice = st.selectbox("Layout Style", ["Force-directed", "Hierarchical", "Radial"])
    renderer_choice = st.radio("Renderer", ["2D", "3D"], horizontal=True)

    node_types = sorted({node_kind(node) for node in nodes})
    selected_types = st.multiselect("Node types to display", node_types, default=node_types)
    filtered_nodes = [node for node in nodes if node_kind(node) in selected_types]
    filtered_node_ids = {n["id"] for n in filtered_nodes}
    filtered_edges = [
        edge for edge in valid_edges if edge["source"] in filtered_node_ids and edge["target"] in filtered_node_ids
    ]

    if renderer_choice == "2D":
        net = Network(height="750px", width="100%", notebook=False, directed=True)
        if layout_choice == "Force-directed":
            net.set_options(json.dumps(vis_options))
        elif layout_choice == "Hierarchical":
            hier_options = {
                "layout": {"hierarchical": {"enabled": True, "direction": "UD", "sortMethod": "directed"}},
                "physics": {"enabled": False},
            }
            net.set_options(json.dumps(hier_options))
        elif layout_choice == "Radial":
            net.set_options(json.dumps({"physics": {"enabled": False}}))
            pos = nx.circular_layout(G.subgraph(filtered_node_ids))
        else:
            pos = None

        for node in filtered_nodes:
            label = node["label"]
            tooltip = clean_and_wrap(node.get("summary", ""))
            kwargs = {}
            if layout_choice == "Radial":
                x, y = pos.get(node["id"], (0, 0))
                kwargs.update({"x": x * 1000, "y": y * 1000, "fixed": True})
            if color_by_community:
                comm_idx = community_map.get(node["id"])
                kwargs["color"] = community_colors.get(comm_idx, "#97c2fc")
                comm_name = community_names.get(comm_idx, "Unknown")
                tooltip = f"Community: {comm_name}\n\n{tooltip}"
                
            net.add_node(node["id"], label=label, title=tooltip, **kwargs)

        for edge in filtered_edges:
            edge_label = edge.get("label") or edge.get("type", "")
            e_type = edge.get("type", "")
            e_color = edge_color_map.get(e_type)
            width = edge.get("weight", 1)
            net.add_edge(
                edge["source"],
                edge["target"],
                title=edge_label,
                label=edge_label,
                color=e_color,
                width=width,
            )

        html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        try:
            net.write_html(html_file.name)
            with open(html_file.name, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=800, width=1600, scrolling=True)
        finally:
            html_file.close()
            os.remove(html_file.name)
    else:
        subG = G.subgraph(filtered_node_ids)
        pos3d = nx.spring_layout(subG, dim=3)
        edge_traces = []
        for edge in filtered_edges:
            x0, y0, z0 = pos3d[edge["source"]]
            x1, y1, z1 = pos3d[edge["target"]]
            e_type = edge.get("type", "")
            e_color = edge_color_map.get(e_type)
            width = edge.get("weight", 1)
            edge_traces.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color=e_color, width=width),
                    hoverinfo="none",
                )
            )

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_colors = []
        for node in filtered_nodes:
            x, y, z = pos3d[node["id"]]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node["label"])
            if color_by_community:
                comm_idx = community_map.get(node["id"])
                node_colors.append(community_colors.get(comm_idx, "#97c2fc"))
            else:
                node_colors.append("#97c2fc")

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=6, color=node_colors),
            text=node_text,
            hoverinfo="text",
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Find Similar Nodes")
    selected_sim_node = st.selectbox("Select a node to find similar ones", sorted(node_ids), key="sim_node")
    top_k = st.slider("Number of similar nodes to find", 1, 15, 5, key="sim_k")

    if selected_sim_node:
        idx = node_id_to_index[selected_sim_node]
        sims = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings)[0]
        top_indices = np.argsort(sims)[::-1][1:top_k+1]

        st.write(f"Top {top_k} nodes similar to **{selected_sim_node}**:")
        result_node_ids = [selected_sim_node]
        for i in top_indices:
            st.write(f"- {node_ids[i]} (Similarity: {sims[i]:.2f})")
            result_node_ids.append(node_ids[i])

        # Display the subgraph
        display_subgraph(
            G,
            result_node_ids,
            node_lookup,
            valid_edges,
            vis_options,
            source_node_id=selected_sim_node,
            edge_colors=edge_color_map,
        )

        # Get the same subgraph data for exporting
        subgraph_nodes, subgraph_edges = get_subgraph(G, result_node_ids, node_lookup, valid_edges)

        with st.expander("DEBUG: Data prepared for export (Find Similar Nodes)"):
            st.write("Node IDs:", sorted([n['id'] for n in subgraph_nodes]))
            st.write(f"Edge Count: {len(subgraph_edges)}")

        if subgraph_nodes:
            kgs_data = prepare_kgs_data(subgraph_nodes, subgraph_edges)
            st.download_button(
                label="Export State (.kgs)",
                data=kgs_data,
                file_name=f"subgraph_{selected_sim_node}.kgs",
                mime="application/octet-stream",
                help="Save the displayed subgraph for use in an interactive HTML graph."
            )
            st.info(f"Subgraph contains {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges.")

with tab3:
    st.header("Semantic Search")
    query = st.text_input("Search for nodes based on meaning:")
    search_top_k = st.slider("Number of search results", 1, 15, 5, key="search_k")

    if query:
        query_vec = st.session_state.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        sims = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:search_top_k]

        st.write("Top matching nodes:")
        result_node_ids = [node_ids[i] for i in top_indices]
        for i in top_indices:
            st.write(f"- {node_ids[i]} (Similarity: {sims[i]:.2f})")

        # Display the subgraph
        display_subgraph(
            G,
            result_node_ids,
            node_lookup,
            valid_edges,
            vis_options,
            edge_colors=edge_color_map,
        )

        # Get the same subgraph data for exporting
        subgraph_nodes, subgraph_edges = get_subgraph(G, result_node_ids, node_lookup, valid_edges)

        with st.expander("DEBUG: Data prepared for export (Semantic Search)"):
            st.write("Node IDs:", sorted([n['id'] for n in subgraph_nodes]))
            st.write(f"Edge Count: {len(subgraph_edges)}")

        if subgraph_nodes:
            kgs_data = prepare_kgs_data(subgraph_nodes, subgraph_edges)
            st.download_button(
                label="Export State (.kgs)",
                data=kgs_data,
                file_name=f"subgraph_search.kgs",
                mime="application/octet-stream",
                help="Save the displayed subgraph for use in an interactive HTML graph."
            )
            st.info(f"Subgraph contains {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges.")

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
    selected_node_id = st.selectbox("Choose a node to inspect:", sorted(list(valid_node_ids)))
    if selected_node_id:
        st.json(node_lookup[selected_node_id])

with tab5:
    st.header("Graph Anatomy")

    @st.cache_data
    def cached_graph_metrics(_G):
        return calculate_graph_metrics(_G, node_lookup, warn=st.warning)

    metrics_df = cached_graph_metrics(G)

    if metrics_df is not None:
        st.subheader("Graph Metrics Data")
        
        # Create a display dataframe with named communities
        display_df = metrics_df.copy()
        display_df["Community"] = display_df["Community"].map(lambda x: community_names.get(int(x), str(x)) if pd.notna(x) else "None")
        
        st.dataframe(display_df.sort_values(by="Degree", ascending=False))

        st.subheader("Graph Anatomy Map")
        
        # Visualization settings
        hub_threshold = metrics_df["Degree"].quantile(0.9)
        bridge_threshold = metrics_df["Betweenness"].quantile(0.9)

        net_anatomy = Network(height="750px", width="100%", notebook=False, directed=True)
        net_anatomy.set_options(json.dumps(vis_options))

        # Use metrics_df as a lookup, but iterate over G.nodes() for consistency
        metrics_lookup = metrics_df.set_index('ID')

        # Add nodes with visual encoding
        for node_id in G.nodes():
            # Look up metrics for the node
            try:
                row = metrics_lookup.loc[node_id]
                label = row["Label"]
                size = 15 + row["Degree"] * 50  # Scale size by degree
                
                # Color by community using precomputed palette
                community_id = row["Community"]
                if pd.notna(community_id):
                    comm_idx = int(community_id)
                    color = community_colors.get(comm_idx, "#808080")
                    comm_name = community_names.get(comm_idx, str(comm_idx))
                else:
                    color = "#808080"  # Grey for no community
                    comm_name = "None"

                border_color = "black"
                if row["Betweenness"] > bridge_threshold:
                    border_color = "red" # Highlight bridges

                title = f"""
                <b>{label}</b><br>
                Degree: {row['Degree']:.3f}<br>
                Betweenness: {row['Betweenness']:.3f}<br>
                Eigenvector: {row['Eigenvector']:.3f}<br>
                PageRank: {row['PageRank']:.3f}<br>
                Community: {comm_name}
                """
                
                net_anatomy.add_node(node_id, label=label, title=title, size=size, color=color, borderWidth=3 if border_color == "red" else 1, borderColor=border_color)
            except KeyError:
                # This node was in the graph but not in the metrics, add it with default styling
                node_info = node_lookup.get(node_id, {})
                label = node_info.get('label', node_id)
                title = f"<b>{label}</b><br>(Metrics not available)"
                net_anatomy.add_node(node_id, label=label, title=title, color="#CCCCCC")


        # Add edges
        for edge in valid_edges:
            # Ensure both source and target nodes exist before adding an edge
            if edge["source"] in G.nodes() and edge["target"] in G.nodes():
                edge_label = edge.get("label") or edge.get("type", "")
                e_type = edge.get("type", "")
                e_color = edge_color_map.get(e_type)
                width = edge.get("weight", 1)
                net_anatomy.add_edge(
                    edge["source"],
                    edge["target"],
                    title=edge_label,
                    label=edge_label,
                    color=e_color,
                    width=width,
                )

        # Save and display the graph
        html_file_anatomy = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        try:
            net_anatomy.write_html(html_file_anatomy.name)
            with open(html_file_anatomy.name, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=800, width=1600, scrolling=True)
        finally:
            html_file_anatomy.close()
            os.remove(html_file_anatomy.name)
    else:
        st.info("Graph is empty. No metrics to display.")
