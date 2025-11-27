import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from sklearn.preprocessing import StandardScaler
import os
import glob
import json

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/"
# Path to where the Swift App expects the model
EXPORT_PATH = "app/CoreMetric/Sources/Models/SystemMonitor.mlpackage"

# Exact keys from final_collector.py
FEATURES = [
    'cpu_percent', 
    'mem_percent', 
    'load_avg_1min',
    'net_sent_per_sec', 
    'net_recv_per_sec', 
    'disk_read_per_sec', 
    'disk_write_per_sec', 
    'ctx_switches_per_sec', 
    'thread_count'
]

# --- 1. DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Metal (MPS) acceleration.")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (Metal not detected).")

# --- 2. DATA INGESTION ---
print(f"\n📂 Scanning {RAW_DATA_DIR}...")
files = glob.glob(os.path.join(RAW_DATA_DIR, "*.jsonl"))

if not files:
    print("❌ No data found! Run the collector first.")
    exit()

print(f"   Found {len(files)} session files. Merging...")

df_list = []
for f in files:
    try:
        # Read JSONL, handle potential corrupt lines
        session_df = pd.read_json(f, lines=True)
        if not session_df.empty:
            df_list.append(session_df)
    except ValueError as e:
        print(f"   ⚠️ Skipping corrupt file: {f}")

if not df_list:
    print("❌ All data files were empty or corrupt.")
    exit()

df = pd.concat(df_list, ignore_index=True)

# Filter only selected features and handle NaNs
df = df[FEATURES].fillna(0)
print(f"   ✅ Loaded {len(df)} data points.")

# --- 3. PREPROCESSING ---
print("⚖️  Scaling data (Z-Score Normalization)...")
scaler = StandardScaler()
data_matrix = df.values.astype(np.float32)

# Fit and Transform
data_scaled = scaler.fit_transform(data_matrix)

# Convert to Tensor
tensor_x = torch.from_numpy(data_scaled).to(device)

# --- 4. MODEL ARCHITECTURE ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.Tanh(), # Tanh handles normalized negative values well
            nn.Linear(6, 3) # Latent Bottleneck (3 dims)
        )
        # Expansion
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = len(FEATURES)
model = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 5. TRAINING LOOP ---
print(f"\n🏋️  Training on {len(df)} samples...")
EPOCHS = 150
BATCH_SIZE = 64
model.train()

# Create DataLoader for batching (better stability)
dataset = torch.utils.data.TensorDataset(tensor_x, tensor_x)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_features, _ in loader:
        # Forward
        output = model(batch_features)
        loss = criterion(output, batch_features)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 25 == 0:
        avg_loss = total_loss / len(loader)
        print(f"   Epoch {epoch}: Loss {avg_loss:.6f}")

# --- 6. ANALYSIS & THRESHOLDING ---
print("\n📊 Analyzing Reconstruction Error...")
model.eval()
with torch.no_grad():
    # Run entire dataset through model
    reconstructions = model(tensor_x)
    # Calculate MSE per sample (on GPU)
    mse_per_sample = torch.mean((tensor_x - reconstructions) ** 2, dim=1)
    # Move to CPU for stats
    mse_cpu = mse_per_sample.cpu().numpy()

# Statistics
mean_error = np.mean(mse_cpu)
max_error = np.max(mse_cpu)
p99_error = np.percentile(mse_cpu, 99.9)

suggested_threshold = p99_error * 1.5  # Safety margin

print(f"   Mean Error (Normal):  {mean_error:.4f}")
print(f"   Max Error (Outlier):  {max_error:.4f}")
print(f"   99.9th Percentile:    {p99_error:.4f}")
print(f"   🎯 SUGGESTED ALERT THRESHOLD: {suggested_threshold:.4f}")

# --- 7. EXPORT TO CORE ML ---
print("\n📦 Exporting to Core ML...")

# Core ML needs CPU model
model.cpu()
dummy_input = torch.rand(1, input_dim)
traced_model = torch.jit.trace(model, dummy_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_features", shape=dummy_input.shape)],
    outputs=[ct.TensorType(name="reconstruction")]
)

# --- 8. METADATA INJECTION (THE PRO MOVE) ---
# We save the Scaler params inside the model so Swift can read them.
print("💉 Injecting Scaling Metadata...")
if scaler.mean_ is None or scaler.scale_ is None:
    print("❌ Error: Scaler was not fitted. Cannot export metadata.")
    exit()

# Convert numpy arrays to comma-separated strings
means_str = ",".join(map(str, scaler.mean_))
stds_str = ",".join(map(str, scaler.scale_))

# Core ML User Metadata
mlmodel.user_defined_metadata["feature_means"] = means_str
mlmodel.user_defined_metadata["feature_stds"] = stds_str
mlmodel.user_defined_metadata["suggested_threshold"] = str(suggested_threshold)

mlmodel.short_description = "Autoencoder for System Anomaly Detection"
mlmodel.author = "SiliconSentinel Pipeline"
mlmodel.version = "1.0 Production"

# Ensure directory exists
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
mlmodel.save(EXPORT_PATH)

print(f"✅ DONE. Model saved to: {EXPORT_PATH}")
print(f"   (Includes metadata for automatic scaling in Swift)")