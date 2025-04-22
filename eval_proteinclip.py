import onnxruntime as ort
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Create session options
options = ort.SessionOptions()
options.intra_op_num_threads = 2

# Load model with options
session = ort.InferenceSession("./proteinclip/project_1.onnx", sess_options=options)

# Get input details
df = pd.read_parquet("./proteinclip/protclip_embed_dataset.parquet")
df = df[2000:2100]
print('length of input embedding', len(df['proteinclip_embed'].iloc[0]))

input_name = session.get_inputs()[0].name
input_array = np.stack(df["proteinclip_embed"].to_numpy()).astype(np.float32)
# print(f"Input shape: {input_array.shape}")  # Should be (100, embed_dim)

# Run inference
outputs = session.run(None, {input_name: input_array})

# Display results
print("shape of output embedding:", [o.shape for o in outputs])

output_array = outputs[0]

# Apply UMAP
umap_model = umap.UMAP(random_state=42)
input_umap = umap_model.fit_transform(input_array)
output_umap = umap_model.fit_transform(output_array)


# Prepare for plotting
plot_df = pd.DataFrame({
    "x": np.concatenate([input_umap[:, 0], output_umap[:, 0]]),
    "y": np.concatenate([input_umap[:, 1], output_umap[:, 1]]),
    "Type": ["ProteinClip"] * len(input_umap) + ["ProteinClip+"] * len(output_umap)
})

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="x", y="y", hue="Type", alpha=0.7, palette=["#1f77b4", "#ff7f0e"])
plt.title("UMAP Projection of Embeddings (ProteinClip vs. ProteinClip+)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.tight_layout()

# Save to file
plt.savefig("umap_embedding_comparison.png", dpi=300)  # High-res PNG

# Optionally close the plot to avoid duplicate output in some environments
plt.close()