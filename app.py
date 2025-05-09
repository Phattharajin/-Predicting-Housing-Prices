import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Load the dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

# Categorize wine quality
def categorize_quality(df):
    df['quality_label'] = df['quality'].apply(lambda x: 'Low' if x <= 5 else ('High' if x >= 7 else 'Medium'))
    return df

# Perform KMeans clustering
def perform_clustering(df):
    features = df.columns[:-2]  # exclude 'quality' and 'quality_label'
    X = df[features]
    y = df['quality_label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['cluster'] = clusters
    return df, X_scaled, y, features

# Visualize clusters using PCA
def plot_pca(df, X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title("K-Means Clusters on Red Wine Quality (PCA Projection)")
    plt.show()

# Spider plot per quality group
def plot_spider(df, features):
    grouped = df.groupby('quality_label')[features].mean()
    labels = list(features)
    num_vars = len(labels)

    def make_spider(row, title, color):
        values = grouped.iloc[row].values.flatten().tolist()
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(1, 3, row + 1, polar=True)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_title(title, size=14, color=color, y=1.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)
        ax.set_yticklabels([])

    plt.figure(figsize=(18, 6))
    make_spider(0, 'Low Quality', 'red')
    make_spider(1, 'Medium Quality', 'orange')
    make_spider(2, 'High Quality', 'green')
    plt.suptitle("Average Feature Profile by Wine Quality Group (Spider Plot)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Main runner
def main():
    df = load_data()
    df = categorize_quality(df)
    df, X_scaled, y, features = perform_clustering(df)

    print("Confusion Matrix:")
    print(confusion_matrix(df['quality_label'], df['cluster']))

    plot_pca(df, X_scaled)
    plot_spider(df, features)

if __name__ == "__main__":
    main()
