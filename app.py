# Install if in Colab
!pip install -q pandas matplotlib seaborn

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Red Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Group quality into categories: Low (<=5), Medium (6), High (>=7)
df['quality_label'] = df['quality'].apply(lambda x: 'Low' if x <= 5 else ('High' if x >= 7 else 'Medium'))

# Select features and group by quality label
features = df.columns[:-2]  # exclude 'quality' and 'quality_label'
grouped = df.groupby('quality_label')[features].mean()

# Prepare data for radar chart
labels = list(features)
num_vars = len(labels)

# Function to create radar chart
def make_spider(row, title, color):
    values = grouped.iloc[row].values.flatten().tolist()
    values += values[:1]  # close the loop
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(1, 3, row+1, polar=True)
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_title(title, size=14, color=color, y=1.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticklabels([])

# Create 3 radar plots (Low, Medium, High)
plt.figure(figsize=(18, 6))
make_spider(0, 'Low Quality', 'red')
make_spider(1, 'Medium Quality', 'orange')
make_spider(2, 'High Quality', 'green')
plt.suptitle("Average Feature Profile by Wine Quality Group (Spider Plot)", fontsize=16)
plt.tight_layout()
plt.show()
