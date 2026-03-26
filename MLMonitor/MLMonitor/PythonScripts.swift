//
//  PythonScripts.swift
//  CoreMetric
//
//  Embeds the Python pipeline scripts so the app is fully self-contained.
//  SetupOrchestrator writes these to Application Support before running them.
//

enum PythonScripts {

    static let requirements = """
psutil==7.1.3
pandas==2.2.3
numpy==2.0.0
scikit-learn==1.5.1
torch==2.4.1
coremltools==8.0
"""

    static let collect = #"""
import psutil
import time
import json
import os
import signal
import sys
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(DATA_DIR, f"training_data_{SESSION_ID}.jsonl")
INTERVAL = 1.0

os.makedirs(DATA_DIR, exist_ok=True)

print(f"---  CORE METRIC COLLECTOR ---")
print(f"Saving to: {OUTPUT_FILE}")
print("Leave this running in the background for 30+ minutes.")
print("Press CTRL+C to stop and save safely.\n")

class SystemMonitor:
    def __init__(self):
        self.prev_net = psutil.net_io_counters()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_ctx = psutil.cpu_stats().ctx_switches
        self.prev_time = time.time()
        psutil.cpu_percent(interval=None)

    def get_metrics(self):
        current_time = time.time()
        time_delta = current_time - self.prev_time
        if time_delta < 0.1:
            time_delta = 0.1

        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        curr_net = psutil.net_io_counters()
        curr_disk = psutil.disk_io_counters()
        curr_ctx = psutil.cpu_stats().ctx_switches

        net_sent_sec = (curr_net.bytes_sent - self.prev_net.bytes_sent) / time_delta
        net_recv_sec = (curr_net.bytes_recv - self.prev_net.bytes_recv) / time_delta
        disk_read_sec = (curr_disk.read_bytes - self.prev_disk.read_bytes) / time_delta
        disk_write_sec = (curr_disk.write_bytes - self.prev_disk.write_bytes) / time_delta
        ctx_switches_sec = (curr_ctx - self.prev_ctx) / time_delta

        self.prev_net = curr_net
        self.prev_disk = curr_disk
        self.prev_ctx = curr_ctx
        self.prev_time = current_time

        load_1, load_5, load_15 = psutil.getloadavg()

        try:
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
        except Exception:
            thread_count = 0

        try:
            process_count = len(psutil.pids())
        except Exception:
            process_count = 0

        return {
            "timestamp": current_time,
            "iso_date": datetime.now().isoformat(),
            "cpu_percent": cpu_pct,
            "mem_percent": mem.percent,
            "swap_percent": swap.percent,
            "load_avg_1min": load_1,
            "net_sent_per_sec": net_sent_sec,
            "net_recv_per_sec": net_recv_sec,
            "disk_read_per_sec": disk_read_sec,
            "disk_write_per_sec": disk_write_sec,
            "ctx_switches_per_sec": ctx_switches_sec,
            "thread_count": thread_count,
            "process_count": process_count,
        }

monitor = SystemMonitor()

def signal_handler(sig, frame):
    print("\n Collection stopped. Data saved.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    with open(OUTPUT_FILE, "a") as f:
        while True:
            start_tick = time.time()
            metrics = monitor.get_metrics()
            f.write(json.dumps(metrics) + "\n")
            if int(time.time()) % 10 == 0:
                f.flush()
            print(f"\r[ REC ] CPU: {metrics['cpu_percent']:5.1f}% | RAM: {metrics['mem_percent']:5.1f}% | Net: {metrics['net_recv_per_sec']/1024:5.0f} KB/s | Procs: {metrics['process_count']}", end="")
            time_spent = time.time() - start_tick
            sleep_time = max(0, INTERVAL - time_spent)
            time.sleep(sleep_time)

except Exception as e:
    print(f"\n Critical Error: {e}")
"""#

    static let train = #"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob
import json

# --- CONFIGURATION ---
RAW_DATA_DIR = "data/raw/"
EXPORT_PATH = "app/CoreMetric/Sources/Models/SystemMonitor.mlpackage"
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 30

# 11 features — must match SystemCollector.getTelemetry() order exactly
FEATURES = [
    'cpu_percent',
    'mem_percent',
    'swap_percent',
    'load_avg_1min',
    'net_sent_per_sec',
    'net_recv_per_sec',
    'disk_read_per_sec',
    'disk_write_per_sec',
    'ctx_switches_per_sec',
    'thread_count',
    'process_count',
]

# --- 1. DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU (Metal not detected).")

# --- 2. DATA INGESTION ---
print(f"\n Scanning {RAW_DATA_DIR}...")
files = glob.glob(os.path.join(RAW_DATA_DIR, "*.jsonl"))

if not files:
    print("No data found! Run the collector first.")
    exit()

print(f"   Found {len(files)} session files. Merging...")

df_list = []
for f in files:
    try:
        session_df = pd.read_json(f, lines=True)
        if not session_df.empty:
            df_list.append(session_df)
    except ValueError:
        print(f"   Skipping corrupt file: {f}")

if not df_list:
    print("All data files were empty or corrupt.")
    exit()

df = pd.concat(df_list, ignore_index=True)

# Fill missing columns (e.g. older data without process_count)
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

df = df[FEATURES].fillna(0)
print(f"  Loaded {len(df)} data points across {len(FEATURES)} features.")

# --- 3. PREPROCESSING ---
print("  Scaling data (Z-Score Normalization)...")
scaler = StandardScaler()
data_matrix = df.values.astype(np.float32)
data_scaled = scaler.fit_transform(data_matrix)

# 80/20 train/validation split
X_train, X_val = train_test_split(data_scaled, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train)} samples | Validation: {len(X_val)} samples")

tensor_train = torch.from_numpy(X_train).to(device)
tensor_val = torch.from_numpy(X_val).to(device)

# --- 4. MODEL ARCHITECTURE ---
# Deeper autoencoder with BatchNorm + GELU for better feature learning.
# Bottleneck of 3 captures essential system state patterns.
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Linear(8, 5),
            nn.BatchNorm1d(5),
            nn.GELU(),
            nn.Linear(5, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.BatchNorm1d(5),
            nn.GELU(),
            nn.Linear(5, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

input_dim = len(FEATURES)
model = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# --- 5. TRAINING LOOP ---
print(f"\n  Training (up to {EPOCHS} epochs, early stopping patience={PATIENCE})...")

# Clamp batch size to dataset size; drop_last avoids BatchNorm1d with batch=1
effective_batch = min(BATCH_SIZE, len(X_train))
drop_last = len(X_train) > effective_batch

dataset = torch.utils.data.TensorDataset(tensor_train, tensor_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch, shuffle=True, drop_last=drop_last)

val_dataset = torch.utils.data.TensorDataset(tensor_val, tensor_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=effective_batch, shuffle=False)

best_val_loss = float('inf')
best_state = None
patience_counter = 0
final_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_x, _ in loader:
        # Denoising autoencoder: reconstruct clean signal from noisy input
        noise = torch.randn_like(batch_x) * 0.05
        noisy = batch_x + noise
        output = model(noisy)
        loss = criterion(output, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    final_epoch = epoch + 1

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for vx, _ in val_loader:
            val_loss += criterion(model(vx), vx).item()
    avg_val = val_loss / max(1, len(val_loader))

    if epoch % 30 == 0 or epoch == EPOCHS - 1:
        avg_train = total_loss / max(1, len(loader))
        print(f"   Epoch {epoch + 1}: Train {avg_train:.6f} | Val {avg_val:.6f}")

    if avg_val < best_val_loss - 1e-7:
        best_val_loss = avg_val
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"   Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.6f})")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# --- 6. THRESHOLD ANALYSIS ---
print("\n Analyzing Reconstruction Error...")
model.eval()
tensor_all = torch.from_numpy(data_scaled).to(device)
with torch.no_grad():
    reconstructions = model(tensor_all)
    mse_per_sample = torch.mean((tensor_all - reconstructions) ** 2, dim=1)
    mse_cpu = mse_per_sample.cpu().numpy()

mean_error = np.mean(mse_cpu)
max_error  = np.max(mse_cpu)
p99_error  = np.percentile(mse_cpu, 99.9)
suggested_threshold = p99_error * 1.5

print(f"   Mean Error (Normal):  {mean_error:.4f}")
print(f"   Max Error (Outlier):  {max_error:.4f}")
print(f"   99.9th Percentile:    {p99_error:.4f}")
print(f"    SUGGESTED ALERT THRESHOLD: {suggested_threshold:.4f}")

# --- 7. EXPORT TO CORE ML ---
print("\n Exporting to Core ML...")

model.cpu()
model.eval()
dummy_input = torch.rand(1, input_dim)
traced_model = torch.jit.trace(model, dummy_input)

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_features", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="reconstruction")],
        compute_precision=ct.precision.FLOAT16,
    )
    print("   Exported with Float16 precision (faster Neural Engine)")
except Exception as e:
    print(f"   Float16 export unavailable ({e}), using default precision")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_features", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="reconstruction")],
    )

# --- 8. METADATA INJECTION ---
print(" Injecting metadata...")

mlmodel.user_defined_metadata["feature_means"]       = ",".join(map(str, scaler.mean_))
mlmodel.user_defined_metadata["feature_stds"]        = ",".join(map(str, scaler.scale_))
mlmodel.user_defined_metadata["suggested_threshold"] = str(suggested_threshold)
mlmodel.user_defined_metadata["feature_names"]       = ",".join(FEATURES)

mlmodel.short_description = "Autoencoder for System Anomaly Detection"
mlmodel.author = "CoreMetric Pipeline"
mlmodel.version = "2.0"

os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
mlmodel.save(EXPORT_PATH)

print(f" DONE. Model saved to: {EXPORT_PATH}")
print(f"   Features: {len(FEATURES)} | Epochs run: {final_epoch} | Val Loss: {best_val_loss:.6f}")
"""#
}
