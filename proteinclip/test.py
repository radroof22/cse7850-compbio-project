from proteinclip import model_utils
import numpy as np
import torch

m = model_utils.load_proteinclip("esm", 33)  # For ESM2, 33-layer model
# Create a synthetic example
# Size corresponds to embedding dimension of "parent" protein language model
model_input = np.random.randn(1280)
# ProteinCLIP expects input to be unit-normalized
model_input /= np.linalg.norm(model_input)
x = m.predict(model_input)
print(x.shape)  # (128,)
print(np.linalg.norm(x))  # 1.0; ProteinCLIP produces unit-norm vectors