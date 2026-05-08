import os
import json
from pathlib import Path

def get_real_training_stats(task_name):
    manifest_path = Path(f'ProcessedData/{task_name}/metadata/manifest.json')
    if not manifest_path.exists():
        return None
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    stats = {}
    for item in data:
        label = item['label']
        split = item['split']
        bin_path_str = item['bin_path']
        if bin_path_str.startswith('/'):
            bin_path = Path(bin_path_str)
        else:
            bin_path = Path(__file__).resolve().parent.parent / bin_path_str
        if bin_path.exists() and bin_path.stat().st_size > 0:
            if label not in stats:
                stats[label] = {'Train': 0, 'Test': 0}
            stats[label][split] += 1
    return stats
tasks = {'Binary': 'binary_benign_vs_malicious', 'USTC': 'ustc_multiclass', 'MTA': 'mta_multiclass', 'MFCP': 'mfcp_multiclass'}
for (task_display, task_name) in tasks.items():
    print(f'--- {task_display} ---')
    stats = get_real_training_stats(task_name)
    if stats:
        total_train = 0
        total_test = 0
        for label in sorted(stats.keys()):
            tr = stats[label]['Train']
            te = stats[label]['Test']
            print(f'{label} | {tr} | {te} | {tr + te}')
            total_train += tr
            total_test += te
        print(f'总计 | {total_train} | {total_test} | {total_train + total_test}')
    else:
        print('Not found')
    print()