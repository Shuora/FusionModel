import os
import json
from collections import Counter

def get_fusion_stats(image_root, pcap_root):
    if not os.path.exists(image_root) or not os.path.exists(pcap_root):
        return None
    classes = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
    stats = {}
    for cls in classes:
        img_dir = os.path.join(image_root, cls)
        pcap_dir = os.path.join(pcap_root, cls)
        if not os.path.exists(img_dir) or not os.path.exists(pcap_dir):
            continue
        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        pcap_files = {os.path.splitext(f)[0] for f in os.listdir(pcap_dir) if f.lower().endswith('.bin')}
        common = img_files.intersection(pcap_files)
        stats[cls] = len(common)
    return stats
tasks = {'Binary': 'binary_benign_vs_malicious', 'USTC': 'ustc_multiclass', 'MTA': 'mta_multiclass', 'MFCP': 'mfcp_multiclass'}
malicious_labels = {'malicious', 'Neris', 'Cridex', 'Tinba', 'Shifu', 'Geodo', 'Virut', 'Htbot', 'Zeus', 'Nsis-ay', 'Miuref', 'Qakbot', 'Ursnif', 'Hancitor', 'Trickbot', 'Dridex', 'Emotet', 'IcedID', 'Artemis', 'Cobalt', 'PUA', 'njRat'}
for (task_display, task_folder) in tasks.items():
    print(f'--- {task_display} ---')
    train_images = f'ProcessedData/{task_folder}/image_data/Train'
    train_pcaps = f'ProcessedData/{task_folder}/pcap_data/Train'
    test_images = f'ProcessedData/{task_folder}/image_data/Test'
    test_pcaps = f'ProcessedData/{task_folder}/pcap_data/Test'
    train_stats = get_fusion_stats(train_images, train_pcaps)
    test_stats = get_fusion_stats(test_images, test_pcaps)
    if train_stats is not None and test_stats is not None:
        all_labels = sorted(set(train_stats.keys()) | set(test_stats.keys()))
        if task_display == 'Binary':
            all_labels = ['malicious', 'benign']
        total_train = 0
        total_test = 0
        total_all = 0
        for label in all_labels:
            tr = train_stats.get(label, 0)
            te = test_stats.get(label, 0)
            tot = tr + te
            is_m = '恶意' if label in malicious_labels or label.lower() == 'malicious' else '非恶意'
            print(f'{is_m} | {label} | {tr} | {te} | {tot}')
            total_train += tr
            total_test += te
            total_all += tot
        print(f'总计 | | {total_train} | {total_test} | {total_all}')
    else:
        print('Data directories not found.')
    print()