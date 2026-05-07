import os
import json
from pathlib import Path

def get_real_training_stats(task_folder):
    processed_root = Path(f"ProcessedData/{task_folder}")
    if not processed_root.exists():
        return None
        
    stats = {}
    for split in ["Train", "Test"]:
        pcap_dir = processed_root / "pcap_data" / split
        if not pcap_dir.exists():
            continue
            
        for label_dir in pcap_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            label = label_dir.name
            if label not in stats:
                stats[label] = {"Train": 0, "Test": 0}
            
            # Count non-empty bin files
            count = 0
            for f in label_dir.glob("*.bin"):
                if f.stat().st_size > 0:
                    count += 1
            stats[label][split] = count
            
    return stats

tasks = {
    "Binary": "binary_benign_vs_malicious",
    "USTC": "ustc_multiclass",
    "MTA": "mta_multiclass",
    "MFCP": "mfcp_multiclass"
}

for task_display, task_folder in tasks.items():
    print(f"--- {task_display} ---")
    stats = get_real_training_stats(task_folder)
    if stats:
        total_train = 0
        total_test = 0
        for label in sorted(stats.keys()):
            tr = stats[label]["Train"]
            te = stats[label]["Test"]
            print(f"{label} | {tr} | {te} | {tr+te}")
            total_train += tr
            total_test += te
        print(f"总计 | {total_train} | {total_test} | {total_train+total_test}")
    else:
        print("Not found")
    print()
