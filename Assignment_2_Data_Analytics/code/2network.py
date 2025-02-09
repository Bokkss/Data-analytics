import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Install required libraries
# pip install pandas
# pip install networkx
# pip install matplotlib

# Load the dataset
networks_df = pd.read_csv('networks_assignment.csv')

# Strip spaces and normalize column names
networks_df.columns = networks_df.columns.str.strip()
networks_df['LABELS'] = networks_df['LABELS'].str.strip()

# Debug: Print available column names
print("Available columns:", networks_df.columns.tolist())

# Define node categories
blue_nodes = ['D', 'F', 'I', 'N', 'S']
light_blue_nodes = ['BIH', 'GEO', 'ISR', 'MNE', 'SRB', 'CHE', 'TUR', 'UKR', 'GBR', 'AUS', 'HKG', 'ASU']
purple_nodes = ['AUT', 'BEL', 'BGR', 'HRV', 'CZE', 'EST', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 
                'LVA', 'LUX', 'NLD', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP']

# Combine all nodes
nodes = blue_nodes + light_blue_nodes + purple_nodes

# Function to find distances between nodes
def find_distance_networks(source, target):
    if source not in networks_df['LABELS'].values:
        print(f"Warning: Source node '{source}' not found in LABELS column.")
        return 0
    if target not in networks_df.columns:
        print(f"Warning: Target node '{target}' not found in dataset columns.")
        return 0
    return networks_df.loc[networks_df['LABELS'] == source, target].values[0]

# Function to find edges between nodes
def find_edges(collection, source, target):
    for s in source:
        for t in target:
            if s in networks_df['LABELS'].values and t in networks_df.columns:
                dist = find_distance_networks(s, t)
                if dist > 0:
                    collection.append((s, t))

# Initialize edge collections
bb_edges = []
bg_edges = []
by_edges = []

# Find edges for each category
find_edges(bb_edges, blue_nodes, blue_nodes)
find_edges(bg_edges, blue_nodes, light_blue_nodes)
find_edges(by_edges, blue_nodes, purple_nodes)

# Combine all edges
edges = bb_edges + bg_edges + by_edges

# Create the graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Define positions for the nodes
pos = nx.circular_layout(G.subgraph(light_blue_nodes + purple_nodes), center=(0, 0), scale=1.0)

# Custom positions for blue nodes
pos.update({
    'D': np.array([0, 0.5]),
    'F': np.array([0.35, 0.15]),
    'I': np.array([-0.35, 0.15]),
    'N': np.array([-0.20, -0.4]),
    'S': np.array([0.20, -0.4])
})

# Define colors for nodes
color_set = [
    (blue_nodes, '#4682b4'),  # Deep blue for main nodes
    (light_blue_nodes, '#87CEEB'),  # Light blue for outer nodes
    (purple_nodes, '#9370DB')  # Medium purple for another set of outer nodes
]

colors = [
    color for node_group, color in color_set for node in nodes if node in node_group
]

# Plot the network
plt.figure(figsize=(12, 12))

# Draw nodes, labels, and edges
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700)
nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=8, font_color='white')
nx.draw_networkx_edges(G, pos, arrows=False, edge_color='black')

# Show the plot
plt.title("Network Diagram", fontsize=16)
plt.show()
