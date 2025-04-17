import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the pickle file (update the filename accordingly)
with open("structural_embeddings_0_2000.pkl", "rb") as f:
    df = pickle.load(f)

# Take the first 200 rows
df_subset = df.iloc[:200]

# Extract embeddings and labels
embeddings = df_subset["embedding"].tolist()
labels = df_subset["protein_id"].tolist()

# Apply UMAP
reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=embedding_2d[:, 0],
    y=embedding_2d[:, 1],
    hue=labels,
    palette="hsv",
    legend=False,
    s=80
)
plt.title("UMAP Projection of Protein Embeddings (First 200)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()

plt.savefig("protein_umap_plot.png", dpi=300)
plt.close()