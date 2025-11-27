import psutil
import time
import json
import os
import signal
import sys
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
# Create a new file for every session to avoid corruption
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(DATA_DIR, f"training_data_{SESSION_ID}.jsonl")
INTERVAL = 1.0  # Sampling rate in seconds

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

print(f"--- 🛡️ SILICON SENTINEL COLLECTOR ---")
print(f"Saving to: {OUTPUT_FILE}")
print("Leave this running in the background for 24-48 hours.")
print("Press CTRL+C to stop and save safely.\n")

class SystemMonitor:
    def __init__(self):
        # Initialize previous state for delta calculations
        self.prev_net = psutil.net_io_counters()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_ctx = psutil.cpu_stats().ctx_switches
        self.prev_time = time.time()
        
        # Warm up CPU usage (first call is always 0)
        psutil.cpu_percent(interval=None)

    def get_metrics(self):
        current_time = time.time()
        time_delta = current_time - self.prev_time
        
        # Avoid division by zero if called too fast
        if time_delta < 0.1: time_delta = 0.1

        # 1. CPU & Memory (Snapshots)
        # Using per-cpu=False because we want the system aggregate for the baseline
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 2. I/O Rates (Bytes per Second)
        # We must calculate the difference between now and last check
        curr_net = psutil.net_io_counters()
        curr_disk = psutil.disk_io_counters()
        curr_ctx = psutil.cpu_stats().ctx_switches
        
        # Calculate Deltas
        net_sent_sec = (curr_net.bytes_sent - self.prev_net.bytes_sent) / time_delta
        net_recv_sec = (curr_net.bytes_recv - self.prev_net.bytes_recv) / time_delta
        
        disk_read_sec = (curr_disk.read_bytes - self.prev_disk.read_bytes) / time_delta
        disk_write_sec = (curr_disk.write_bytes - self.prev_disk.write_bytes) / time_delta
        
        ctx_switches_sec = (curr_ctx - self.prev_ctx) / time_delta
        
        # Update State
        self.prev_net = curr_net
        self.prev_disk = curr_disk
        self.prev_ctx = curr_ctx
        self.prev_time = current_time

        # 3. System Load & Complexity
        # Load Avg is great for "is the system struggling?"
        load_1, load_5, load_15 = psutil.getloadavg()
        
        # Thread count helps detect "stuck" processes or fork bombs
        # (This is expensive to count, so we wrap in try/catch)
        try:
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
        except:
            thread_count = 0

        return {
            "timestamp": current_time,
            "iso_date": datetime.now().isoformat(),
            
            # --- Features for the Model ---
            "cpu_percent": cpu_pct,
            "mem_percent": mem.percent,
            "swap_percent": swap.percent,
            "load_avg_1min": load_1,
            "net_sent_per_sec": net_sent_sec,
            "net_recv_per_sec": net_recv_sec,
            "disk_read_per_sec": disk_read_sec,
            "disk_write_per_sec": disk_write_sec,
            "ctx_switches_per_sec": ctx_switches_sec,
            "thread_count": thread_count
        }

# --- MAIN LOOP ---
monitor = SystemMonitor()

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("\n🛑 Collection stopped. Data saved.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    with open(OUTPUT_FILE, "a") as f:
        while True:
            start_tick = time.time()
            
            metrics = monitor.get_metrics()
            
            # Write JSON Line
            f.write(json.dumps(metrics) + "\n")
            
            # Flush to disk periodically (every 10 seconds) to prevent data loss on crash
            if int(time.time()) % 10 == 0:
                f.flush()
            
            # Print a status bar so you know it's alive
            print(f"\r[ REC ] CPU: {metrics['cpu_percent']:5.1f}% | RAM: {metrics['mem_percent']:5.1f}% | Net: {metrics['net_recv_per_sec']/1024:5.0f} KB/s", end="")
            
            # Sleep precisely to maintain interval
            time_spent = time.time() - start_tick
            sleep_time = max(0, INTERVAL - time_spent)
            time.sleep(sleep_time)

except Exception as e:
    print(f"\n❌ Critical Error: {e}")