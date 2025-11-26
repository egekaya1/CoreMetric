import psutil
import time
import json
import os
from datetime import datetime
from typing import Dict, Any

# Config
INTERVAL = 1.0  # Collect every second
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "metrics.jsonl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- ML Monitor Collector ---")
print(f"Saving metrics to: {OUTPUT_FILE}")
print(f"Press CTRL+C to stop collection.\n")

def get_system_metrics() -> Dict[str, Any]:
    # Gather raw system stats
    disk_io = psutil.disk_io_counters()
    return {
        "timestamp": datetime.now().isoformat(),
        # CPU
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        # Memory
        "mem_percent": psutil.virtual_memory().percent,
        "mem_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        # Disk IO (handle None case)
        "disk_read_bytes": disk_io.read_bytes if disk_io else None,
        "disk_write_bytes": disk_io.write_bytes if disk_io else None,
        # Network
        "net_sent": psutil.net_io_counters().bytes_sent,
        "net_recv": psutil.net_io_counters().bytes_recv
    }

try:
    with open(OUTPUT_FILE, "a") as f:
        while True:
            data = get_system_metrics()
            
            # Write JSON line
            f.write(json.dumps(data) + "\n")
            f.flush()
            
            # Print status so you know it's working
            print(f"Logged: CPU {data['cpu_percent']}% | Mem {data['mem_percent']}%")
            
            time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n\nStopping collection. Data saved.")