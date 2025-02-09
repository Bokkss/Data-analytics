import pandas as pd
import plotly.graph_objects as go

# Install required libraries
# pip install pandas
# pip install plotly

# Load the dataset
sankey_data = pd.read_csv('sankey_assignment.csv')

# Define source and target categories
source_categories = ['PS', 'OMP', 'CNP', 'NRP', 'NMCCC', 'PEC', 'NCDM', 'RGS']
middle_categories = ['I', 'S', 'D', 'F', 'N']
target_categories = ['Reg', 'Aca', 'Oth']

# Extract source, middle, and target values for Sankey diagram
sources = []
targets = []
values = []

# Connect source categories to middle categories
for i, source in enumerate(source_categories):
    for j, middle in enumerate(middle_categories):
        value = sankey_data[source].sum() if source in sankey_data.columns else 0
        if value > 0:
            sources.append(i)  # Source index
            targets.append(len(source_categories) + j)  # Middle index
            values.append(value / len(middle_categories))  # Divide flow equally for simplicity

# Connect middle categories to target categories
for j, middle in enumerate(middle_categories):
    for k, target in enumerate(target_categories):
        value = sankey_data[target].sum() if target in sankey_data.columns else 0
        if value > 0:
            sources.append(len(source_categories) + j)  # Middle index
            targets.append(len(source_categories) + len(middle_categories) + k)  # Target index
            values.append(value / len(target_categories))  # Divide flow equally for simplicity

# Combine all labels
labels = source_categories + middle_categories + target_categories

# Define distinguishable shades of blue and purple for nodes
node_colors = [
    "#5B9BD5", "#6A5ACD", "#8A2BE2", "#4169E1", "#483D8B", "#6495ED", "#9370DB", "#4682B4",  # Source colors
    "#8DD3C7", "#FFB3E6", "#B39EB5", "#D8BFD8", "#98AFC7",  # Middle colors
    "#1F77B4", "#33A02C", "#6A3D9A"  # Target colors
]

# Create the Sankey diagram using Plotly
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=sources,  # Source indices
        target=targets,  # Target indices
        value=values     # Flow values
    )
)])

# Update layout to match example style
fig.update_layout(title_text="Sankey Diagram", font_size=10)

# Show the diagram
fig.show()
