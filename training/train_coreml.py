import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from sklearn.preprocessing import StandardScaler
import os

# 1. Load Data
data_path = "data/raw/metrics.jsonl"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run collector first!")
    exit()

print("Loading data...")
df = pd.read_json(data_path, lines=True)

# Select features to train on
FEATURES = ['cpu_percent', 'mem_percent', 'disk_read_bytes', 'disk_write_bytes', 'net_sent', 'net_recv']
data = df[FEATURES].values.astype(np.float32)

# Simple Preprocessing (Standardize)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. Define PyTorch Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Compress 6 features -> 3 -> 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2) # Latent bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 3. Train (Super fast POC)
input_dim = len(FEATURES)
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Convert numpy to torch tensor
tensor_x = torch.from_numpy(data_scaled)

print("Training Autoencoder...")
for epoch in range(100):
    output = model(tensor_x)
    loss = criterion(output, tensor_x)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 4. Convert to Core ML
print("\nConverting to Core ML...")
model.eval()

# Trace with dummy input
dummy_input = torch.rand(1, input_dim)
traced_model = torch.jit.trace(model, dummy_input)

# Convert
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_features", shape=dummy_input.shape)],
    outputs=[ct.TensorType(name="reconstruction")]
)

# Add metadata (Optional but good practice)
mlmodel.short_description = "System Anomaly Autoencoder"
mlmodel.author = "Ege"
mlmodel.version = "1.0"

# Save
save_path = "models/SystemMonitor.mlpackage"
mlmodel.save(save_path)
print(f"Success! Model saved to: {save_path}")