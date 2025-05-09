# streamlit_wine_kmeans_simple.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.decomposition import PCA

# Load model and scaler
with open('kmeans_wine_model.pkl', 'rb') as f:
    loaded_model, scaler = joblib.load(f)

# Streamlit config
st.set_page_config(page_title="Wine K-Means Clustering", layout="centered")
st.title("Wine K-Means Clustering Visualizer by Phattharajin Joyjaroen")

# Load and scale wine data
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
X = df.drop(columns=['quality'])
X_scaled = scaler.transform(X)

# Predict clusters
y_kmeans = loaded_model.predict(X_scaled)

# PCA for 2D plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
centers_pca = pca.transform(loaded_model.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', label='Centroids')
ax.set_title('Wine Data Clustering (PCA Projection)')
ax.legend()
st.pyplot(fig)
