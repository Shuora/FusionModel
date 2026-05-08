import os
import json
from collections import Counter

def get_unique_stats(manifest_path):
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    unique_samples = {}
    for item in data:
        label = item['label']
        split = item['split']
        raw_path = item['raw_path']
        if label not in unique_samples:
            unique_samples[label] = {'Train': set(), 'Test': set()}
        unique_samples[label][split].add(raw_path)
    stats = {}
    for (label, splits) in unique_samples.items():
        stats[label] = {'Train': len(splits['Train']), 'Test': len(splits['Test'])}
    return stats
tasks = {'Binary': 'ProcessedData/binary_benign_vs_malicious/metadata/manifest.json', 'USTC': 'ProcessedData/ustc_multiclass/metadata/manifest.json', 'MTA': 'ProcessedData/mta_multiclass/metadata/manifest.json', 'MFCP': 'ProcessedData/mfcp_multiclass/metadata/manifest.json'}
for (task_display, path) in tasks.items():
    print(f'--- {task_display} ---')
    stats = get_unique_stats(path)
    if stats:
        total_train = 0
        total_test = 0
        total_all = 0
        for label in sorted(stats.keys()):
            tr = stats[label]['Train']
            te = stats[label]['Test']
            tot = tr + te
            print(f'{label} | {tr} | {te} | {tot}')
            total_train += tr
            total_test += te
            total_all += tot
        print(f'总计 | {total_train} | {total_test} | {total_all}')
    else:
        print('Manifest not found.')
    print()