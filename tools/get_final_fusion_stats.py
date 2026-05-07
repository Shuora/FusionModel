import os
import json
from pathlib import Path

def get_fusion_stats(task_folder):
    processed_root = Path(f"ProcessedData/{task_folder}")
    if not processed_root.exists():
        return None
        
    stats = {}
    for split in ["Train", "Test"]:
        img_dir = processed_root / "image_data" / split
        pcap_dir = processed_root / "pcap_data" / split
        
        if not img_dir.exists() or not pcap_dir.exists():
            continue
            
        for label_dir in img_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            label = label_dir.name
            if label not in stats:
                stats[label] = {"Train": 0, "Test": 0}
            
            # Match image filenames with bin filenames (ignoring extension)
            img_stems = {f.stem for f in label_dir.glob("*.png")}
            pcap_label_dir = pcap_dir / label
            if not pcap_label_dir.exists():
                continue
            pcap_stems = {f.stem for f in pcap_label_dir.glob("*.bin") if f.stat().st_size > 0}
            
            common = img_stems.intersection(pcap_stems)
            stats[label][split] = len(common)
            
    return stats

tasks = {
    "Binary": "binary_benign_vs_malicious",
    "USTC": "ustc_multiclass",
    "MTA": "mta_multiclass",
    "MFCP": "mfcp_multiclass"
}

for task_display, task_folder in tasks.items():
    print(f"--- {task_display} ---")
    stats = get_fusion_stats(task_folder)
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
