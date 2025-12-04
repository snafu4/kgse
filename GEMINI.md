# Project Overview

This project is a Python-based web application called "Knowledge Graph Semantic Explorer". It's designed to visualize and explore knowledge graphs, allowing users to interact with and analyze complex network data. The application is built primarily with Streamlit for the user interface, and it leverages several key libraries for its functionality:

- **Graph Visualization & Analysis:** `pyvis` and `networkx` are used to render and analyze the graph structures.
- **Semantic Search:** `sentence-transformers` and `scikit-learn` provide the foundation for semantic search capabilities, enabling users to find nodes based on meaning rather than just keywords.
- **3D Visualization:** `plotly` is used to create interactive 3D representations of the graphs.

The application loads graph data from custom compressed file formats (`.kgc` and `.kge` from an application created by the same developer) and provides a rich, interactive experience for exploring the relationships and properties of the graph's nodes and edges.

## Features

The application is organized into several tabs, each providing a different set of tools for exploring the knowledge graph:

- **Full Graph:** An interactive 2D and 3D visualization of the entire knowledge graph. Users can pan, zoom, and inspect nodes and edges.
- **Find Similar Nodes:** Allows users to select a node and find other nodes that are semantically similar to it, based on their text embeddings.
- **Semantic Search:** Provides a search bar where users can enter a query to find the most relevant nodes in the graph based on semantic meaning.
- **Diagnostics:** Displays diagnostic information about the graph, such as the number of nodes and edges, and lists any orphan nodes or skipped edges.
- **Graph Anatomy:** A tab for more advanced analysis, showing graph metrics like degree, betweenness centrality, and community detection.

# Key Files

- **`05-kg_semantic_explorer.py`**: The main entry point of the application. This file contains the Streamlit UI code and orchestrates the different components of the application.
- **`UTL_kg_utils.py`**: A utility module that provides a wide range of helper functions for graph manipulation, including functions for calculating graph metrics, preparing data for visualization, and cleaning text.
- **`UTL_kg_compressor.py`**: A simple utility for compressing and decompressing graph data into the custom `.kgc` and `.kge` file formats.
- **`requirements.txt`**: The list of Python dependencies required to run the project.
- **`demo.html`**: A standalone webpage that displays an interactive knowledge graph. When you open it, it shows a large, explorable network of ideas or entities connected by relationships. You can zoom, pan, and click around to see how everything links together. All the data for the graph is already built into the file, so it works on its own with no setup. It’s basically a packaged, portable viewer for exploring a structured map of information.

# Data Formats

The application uses two custom file formats for storing graph data:

- **`.kgc`**: A compressed file containing the basic graph structure (nodes and edges).
- **`.kge`**: An enhanced version of `.kgc` that also includes the pre-computed text embeddings for the nodes in the graph. This allows the application to load graphs much faster, as it doesn't need to regenerate the embeddings every time.

# Building and Running

To run this project, you'll need to have Python and the required dependencies installed.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application:**
    The application is a Streamlit app. To run it, use the following command:
    ```bash
    streamlit run 05-kg_semantic_explorer.py
    ```
    This will start a local web server, and you can access the application in your browser at the URL provided in the terminal.

# Development Conventions

Based on the code, here are some of the development conventions used in this project:

*   **Modular Design:** The code is organized into a main application file (`05-kg_semantic_explorer.py`) and utility modules (`UTL_kg_compressor.py`, `UTL_kg_utils.py`) for better organization and reusability.
*   **Type Hinting:** The code uses Python's type hinting to improve code clarity and maintainability.
*   **Streamlit Best Practices:** The application uses Streamlit's session state (`st.session_state`) to manage application state and caching (`st.cache_data`) to optimize performance.
*   **Clear Naming Conventions:** Functions and variables are named descriptively, making the code easier to understand.
*   **Custom File Formats:** The project uses custom file formats (`.kgc`, `.kge`) for storing graph data, with a dedicated utility for compression and decompression.