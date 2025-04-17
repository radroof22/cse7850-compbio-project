import onnxruntime as ort

# Load the ONNX model
session = ort.InferenceSession("/proteinclip_stuff/pretrained/proteinclip_esm2_6.onnx")

# Inspect output information
for output in session.get_outputs():
    print("Output name:", output.name)
    print("Output shape:", output.shape)
    print("Output type:", output.type)
