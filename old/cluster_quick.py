import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_parquet("embeddings.parquet")
X = np.vstack(df["_embedding"].to_list())

# Try a small range and pick the best silhouette
best_k, best_score = None, -1
for k in range(4, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    if score > best_score:
        best_k, best_score = k, score

km = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
labels = km.fit_predict(X)
df["_cluster"] = labels
df.to_parquet("embeddings_clustered.parquet", index=False)
print(f"Best k={best_k} (silhouette={best_score:.3f}). Saved embeddings_clustered.parquet")


