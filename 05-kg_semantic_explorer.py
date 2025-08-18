import streamlit as st
st.set_page_config(layout="wide")
st.markdown(
    "<style>iframe[title='st.components.v1.html']{max-width:100% !important;width:100% !important;}</style>",
    unsafe_allow_html=True
)
import json
import tempfile
from pyvis.network import Network
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
from kg_compressor import KGCompressor
import re
import textwrap
import pandas as pd
from networkx.algorithms import community
import zlib
import uuid
import colorsys
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

def clean_and_wrap(text, width=50):
    # Remove all HTML tags (anything between < and >)
    no_html = re.sub(r'<.*?>', '', text)
    # Replace multiple whitespace (including newlines) with a single space
    normalized = re.sub(r'\s+', ' ', no_html).strip()
    # Wrap text to the desired width, breaking only at word boundaries
    wrapped = textwrap.fill(normalized, width=width, break_long_words=False, break_on_hyphens=False)
    # Ensure explicit "\n" (textwrap.fill already uses \n at end of lines)
    return wrapped


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


def node_kind(n: dict):
    """Unified accessor for a node's category.
    Prefer explicit 'type', then fall back to 'group', else 'default'.
    """
    return n.get('type') or n.get('group') or 'default'

def prepare_kgs_data(subgraph_nodes, subgraph_edges):
    """
    Prepares data in the .kgs format to be compatible with 02-trim_nodes.py.
    This format is a zlib-compressed JSON object representing an "IsolationState".
    It now aims to preserve all original node/edge properties from the subgraph,
    while adding/overwriting only necessary vis.js-specific fields.
    """
    
    # Calculate node degrees for the 'value' property.
    node_degrees = defaultdict(int)
    for edge in subgraph_edges:
        node_degrees[edge.get("source")] += 1
        node_degrees[edge.get("target")] += 1

    # Process Nodes: Preserve all original properties, then add/override vis.js specific ones.
    processed_nodes = []
    for node in subgraph_nodes:
        new_node = dict(node) # Start with a shallow copy of the original node

        # Add 'value' property for sizing (always calculated)
        new_node['value'] = node_degrees[new_node.get("id")]

        # Explicitly set shape to 'dot' as requested
        new_node['shape'] = 'dot'

        # Ensure 'summary' and 'type' are present (add defaults only if truly missing)
        if 'summary' not in new_node:
            new_node['summary'] = ''
        if 'type' not in new_node:
            new_node['type'] = new_node.get('group', 'default')

        # Set the 'title' property for popover text
        new_node['title'] = clean_and_wrap(new_node.get("summary", ""))

        # Assign distinct colors based on group/type, but only if 'color' is not already defined
        if 'color' not in new_node:
            group = new_node.get("group") or new_node.get("type") or "default"
            # Temporarily store nodes by group to assign colors
            # This part needs to be done after all nodes are processed to ensure consistent color assignment
            # So, we'll collect nodes and then assign colors in a separate loop or function.
            # For now, let's just mark them for color assignment.
            new_node['_group_for_color'] = group # Use a temporary key

        # Apply default font styling, but only if 'font' is not already defined
        if 'font' not in new_node:
            new_node['font'] = {"color": "white", "strokeWidth": 2, "strokeColor": "black"}
        
        processed_nodes.append(new_node)

    # Second pass for color assignment (after all nodes are in processed_nodes)
    # This ensures consistent color mapping across all nodes based on their groups.
    groups_for_coloring = defaultdict(list)
    for node in processed_nodes:
        if '_group_for_color' in node:
            groups_for_coloring[node['_group_for_color']].append(node)
            del node['_group_for_color'] # Clean up temporary key

    group_names = sorted(groups_for_coloring)
    total_groups = len(group_names)

    for idx, group in enumerate(group_names):
        h = idx / max(1, total_groups)
        s, v = 0.6, 0.85
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color_hex = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        for node in groups_for_coloring[group]:
            node['color'] = {"background": color_hex}


    # Process Edges: Preserve all original properties, then add/override vis.js specific ones.
    processed_edges = []
    for edge in subgraph_edges:
        new_edge = dict(edge) # Start with a shallow copy of the original edge

        # Transform 'source'/'target' to 'from'/'to' (vis.js format)
        new_edge['from'] = new_edge.pop('source')
        new_edge['to'] = new_edge.pop('target')

        # Ensure 'id' is present
        if 'id' not in new_edge:
            new_edge['id'] = str(uuid.uuid4())

        # Ensure 'label' is present (use 'type' if 'label' is missing)
        if 'label' not in new_edge:
            new_edge['label'] = new_edge.get('type', '')

        # Ensure 'arrows' is present
        if 'arrows' not in new_edge:
            new_edge['arrows'] = 'to' # Default direction

        # Apply default font styling, but only if 'font' is not already defined
        if 'font' not in new_edge:
            new_edge['font'] = {"color": "white", "strokeWidth": 2, "strokeColor": "black"}
        
        # Ensure 'color' is present (if not, vis.js will use default)
        # No explicit default color assignment here, vis.js handles it.

        processed_edges.append(new_edge)

    # Build the final state object matching the structure from 02-trim_nodes.py.
    state = {
        "type": "IsolationState",
        "version": 1,
        "timestamp": pd.Timestamp.now().isoformat(),
        "isolateStack": [{
            "nodes": processed_nodes,
            "edges": processed_edges,
            "deleted": []
        }]
    }

    json_string = json.dumps(state, indent=2)
    compressed_data = zlib.compress(json_string.encode('utf-8'))
    return compressed_data

def get_subgraph(graph_obj, result_node_ids, node_lookup, valid_edges):
    """
    Calculates the subgraph containing the result nodes and their immediate neighbors.
    """
    neighbors = set(n for node_id in result_node_ids if node_id in graph_obj for n in list(graph_obj.predecessors(node_id)) + list(graph_obj.successors(node_id)))
    subgraph_node_ids = set(result_node_ids) | neighbors

    # Return the full node object from the lookup
    subgraph_nodes = [node_lookup[nid] for nid in subgraph_node_ids if nid in node_lookup]
    
    # Return the full edge object
    subgraph_edges = [edge for edge in valid_edges if edge.get('source') in subgraph_node_ids and edge.get('target') in subgraph_node_ids]
    
    return subgraph_nodes, subgraph_edges

def display_subgraph(graph_obj, result_node_ids, node_lookup, valid_edges, vis_options, source_node_id=None):
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
        sub_net.add_edge(edge["source"], edge["target"], title=edge_label, label=edge_label)
        
    html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    try:
        sub_net.write_html(html_file.name)
        with open(html_file.name, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=800, width=1600, scrolling=True)
    finally:
        html_file.close()
        os.remove(html_file.name)


# ---- Sidebar Setup & File Upload ----
st.sidebar.title("File Operations")
uploaded_file = st.sidebar.file_uploader(
    "Load Graph",
    type=["kgc", "kge"],
    help="Load a `.kgc` (graph only) or `.kge` (graph + embeddings) file."
)

if uploaded_file is None:
    st.info("Please upload a .kgc or .kge file to begin.")
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

# Precompute 2D projection of embeddings
if "embeddings_2d" not in st.session_state or st.session_state.embeddings_2d.shape[0] != len(embeddings):
    pca = PCA(n_components=2)
    st.session_state.embeddings_2d = pca.fit_transform(embeddings)
projection_2d = st.session_state.embeddings_2d

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

G = nx.DiGraph()
G.add_nodes_from((node["id"], node) for node in nodes)
G.add_edges_from([(e["source"], e["target"], e) for e in valid_edges])

# Compute community assignments once for optional coloring
communities = community.greedy_modularity_communities(G.to_undirected())
community_map = {n: idx for idx, comm in enumerate(communities) for n in comm}
community_colors = {}
total_comms = len(communities)
for idx in range(total_comms):
    h = idx / max(1, total_comms)
    s, v = 0.6, 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    community_colors[idx] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Full Graph", "Find Similar Nodes", "Semantic Search", "Diagnostics", "Graph Anatomy", "Embedding Map"])

with tab1:
    st.header("Interactive Full Graph")
    color_by_community = st.checkbox("Color nodes by community")
    node_types = sorted({node_kind(node) for node in nodes})
    selected_types = st.multiselect("Node types to display", node_types, default=node_types)
    filtered_nodes = [node for node in nodes if node_kind(node) in selected_types]
    filtered_node_ids = {n["id"] for n in filtered_nodes}
    filtered_edges = [edge for edge in valid_edges if edge["source"] in filtered_node_ids and edge["target"] in filtered_node_ids]

    net = Network(height="750px", width="100%", notebook=False, directed=True)
    net.set_options(json.dumps(vis_options))
    for node in filtered_nodes:
        label = node["label"]
        tooltip = clean_and_wrap(node.get("summary", ""))
        if color_by_community:
            comm_idx = community_map.get(node["id"])
            color = community_colors.get(comm_idx, "#97c2fc")
            net.add_node(node["id"], label=label, title=tooltip, color=color)
        else:
            net.add_node(node["id"], label=label, title=tooltip)
    for edge in filtered_edges:
        edge_label = edge.get("label") or edge.get("type", "")
        net.add_edge(edge["source"], edge["target"], title=edge_label, label=edge_label)

    html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    try:
        net.write_html(html_file.name)
        with open(html_file.name, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=800, width=1600, scrolling=True)
    finally:
        html_file.close()
        os.remove(html_file.name)

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
        display_subgraph(G, result_node_ids, node_lookup, valid_edges, vis_options, source_node_id=selected_sim_node)

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
        display_subgraph(G, result_node_ids, node_lookup, valid_edges, vis_options)

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
    def calculate_graph_metrics(_G):
        if not _G.nodes():
            return None

        # Centrality Measures
        degree_centrality = nx.degree_centrality(_G)
        betweenness_centrality = nx.betweenness_centrality(_G)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(_G, max_iter=1000, tol=1.0e-6)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {n: 0.0 for n in _G.nodes()}
            st.warning("Eigenvector centrality did not converge.")
        
        pagerank = nx.pagerank(_G)

        # Community Detection
        try:
            communities_generator = community.greedy_modularity_communities(_G.to_undirected())
            communities = [list(c) for c in communities_generator]
            node_community = {node_id: i for i, com in enumerate(communities) for node_id in com}
        except Exception as e:
            communities = []
            node_community = {}
            st.warning(f"Could not perform community detection: {e}")

        # Combine into a DataFrame, ensuring proper alignment
        metrics_df = pd.DataFrame({
            "Degree": degree_centrality,
            "Betweenness": betweenness_centrality,
            "Eigenvector": eigenvector_centrality,
            "PageRank": pagerank
        })
        metrics_df['Community'] = pd.Series(node_community)
        # Map the node IDs (index) to their labels
        metrics_df['Label'] = [node_lookup[nid].get('label', nid) for nid in metrics_df.index]
        
        # Reorder columns for display and add ID
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'ID'})
        
        # Ensure desired column order
        column_order = ['ID', 'Label', 'Degree', 'Betweenness', 'Eigenvector', 'PageRank', 'Community']
        metrics_df = metrics_df[[col for col in column_order if col in metrics_df.columns]]

        return metrics_df

    metrics_df = calculate_graph_metrics(G)

    if metrics_df is not None:
        st.subheader("Graph Metrics Data")
        st.dataframe(metrics_df.sort_values(by="Degree", ascending=False))

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
                
                # Color by community
                community_id = row["Community"]
                if pd.notna(community_id):
                    hue = int((community_id * 137.5) % 360)
                    color = f"hsl({hue}, 70%, 50%)"
                else:
                    color = "#808080" # Grey for no community

                border_color = "black"
                if row["Betweenness"] > bridge_threshold:
                    border_color = "red" # Highlight bridges

                title = f"""
                <b>{label}</b><br>
                Degree: {row['Degree']:.3f}<br>
                Betweenness: {row['Betweenness']:.3f}<br>
                Eigenvector: {row['Eigenvector']:.3f}<br>
                PageRank: {row['PageRank']:.3f}<br>
                Community: {row['Community']}
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
                net_anatomy.add_edge(edge["source"], edge["target"], title=edge_label, label=edge_label)    

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

with tab6:
    st.header("Embedding Map")
    if embeddings is not None and len(embeddings):
        coords = projection_2d
        labels = [node["label"] for node in nodes]
        df_map = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels,
            "id": node_ids,
        })

        fig = px.scatter(df_map, x="x", y="y", text="label", hover_name="label")
        fig.update_traces(textposition="top center")

        selected_idx = st.session_state.get("selected_map_idx")
        if selected_idx is not None:
            fig.update_traces(marker=dict(color="LightGray", size=8))
            fig.add_trace(go.Scatter(
                x=[df_map.iloc[selected_idx]["x"]],
                y=[df_map.iloc[selected_idx]["y"]],
                text=[df_map.iloc[selected_idx]["label"]],
                mode="markers+text",
                marker=dict(color="red", size=12),
            ))

        selected_points = plotly_events(fig, click_event=True, hover_event=False, key="embedding_map")
        if selected_points:
            st.session_state.selected_map_idx = selected_points[0]["pointIndex"]
            st.rerun()

        if selected_idx is not None:
            node = nodes[selected_idx]
            st.subheader(node.get("label", node.get("id")))
            st.json(node)
    else:
        st.info("No embeddings available to visualize.")