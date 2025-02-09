import pandas as pd
import matplotlib.pyplot as plt

# Install required libraries
# pip install pandas
# pip install matplotlib

# Load the dataset
bar_data = pd.read_csv('bar_assignment.csv')

# Preprocess the data: Transform 1 to 'Yes' and 0 to 'No'
bar_data['COUNT'] = bar_data['COUNT'].replace({1: 'Yes', 0: 'No'})

# Aggregate data for plotting
bar_chart_data = bar_data.groupby(['LABEL', 'COUNT']).size().unstack(fill_value=0)

# Plot Bar Chart
plt.figure(figsize=(10, 6))
bar_chart_data.plot(
    kind='barh',
    stacked=True,
    color=['#6a5acd', '#4682b4'],  # Purple for No, Blue for Yes
    edgecolor='black'
)

# Customize the plot
plt.title("Bar Chart Visualization", fontsize=16)
plt.xlabel("Count", fontsize=12)
plt.ylabel("Labels", fontsize=12)
plt.legend(title="Legend", labels=['No', 'Yes'], fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
