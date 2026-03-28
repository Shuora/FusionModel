#!/usr/bin/env python3
"""
Sort model files, images and reports in outputs/ into subfolders by training mode keyword.
Runs on Windows; places files into outputs/<mode>/ based on filename token matching.
"""
import os
import shutil
import json
from datetime import datetime

ROOT = os.path.abspath(r"C:\Repositories\Traffic\CharBERT-MobileViT\outputs")
MODE_TOKENS = [
    'attention_all_ensembles',
    'concat_all_ensembles',
    'attention_stacking',
    'weighted_stacking',
    'concat_stacking',
    'attention_dim256',
    'attention',
    'concat',
    'weighted'
]

EXT_WHITELIST = {'.png', '.pth', '.pkl', '.md', '.txt'}
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
REPORT = {
    'root': ROOT,
    'timestamp': TIMESTAMP,
    'moved': [],
    'skipped': []
}


def find_mode_for_name(name):
    lower = name.lower()
    for tok in MODE_TOKENS:
        if tok in lower:
            return tok
    return None


def process_logs_dir():
    """Scan outputs/logs and move .log files into outputs/logs/<mode>/ folders."""
    logs_dir = os.path.join(ROOT, 'logs')
    if not os.path.isdir(logs_dir):
        return
    for entry in os.listdir(logs_dir):
        full = os.path.join(logs_dir, entry)
        # skip directories (we'll only move files at top-level of logs)
        if os.path.isdir(full):
            REPORT['skipped'].append({'name': full, 'reason': 'logs_subdir_skip'})
            continue
        # only handle .log files
        name_lower = entry.lower()
        if not name_lower.endswith('.log'):
            REPORT['skipped'].append({'name': full, 'reason': 'logs_non_log'})
            continue
        mode = find_mode_for_name(entry)
        if mode is None:
            target_folder = os.path.join(logs_dir, 'misc')
        else:
            target_folder = os.path.join(logs_dir, mode)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, entry)
        if os.path.exists(target_path):
            base, ext = os.path.splitext(entry)
            new_name = f"{base}.{TIMESTAMP}{ext}"
            target_path = os.path.join(target_folder, new_name)
        shutil.move(full, target_path)
        REPORT['moved'].append({'src': full, 'dst': target_path})


def main():
    items = os.listdir(ROOT)
    for name in items:
        path = os.path.join(ROOT, name)
        # skip logs dir and directories
        if os.path.isdir(path):
            if os.path.basename(path).lower() == 'logs':
                # we'll process logs after moving top-level outputs
                REPORT['skipped'].append({'name': name, 'reason': 'logs_dir_pending'})
            else:
                REPORT['skipped'].append({'name': name, 'reason': 'is_directory'})
            continue
        # skip the script itself and other tools
        if os.path.abspath(__file__) == os.path.abspath(path):
            REPORT['skipped'].append({'name': name, 'reason': 'script_self'})
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in EXT_WHITELIST:
            REPORT['skipped'].append({'name': name, 'reason': 'ext_not_whitelisted'})
            continue
        mode = find_mode_for_name(name)
        if mode is None:
            target_folder = os.path.join(ROOT, 'misc')
        else:
            target_folder = os.path.join(ROOT, mode)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, name)
        # if target exists, avoid overwrite by renaming (append timestamp)
        if os.path.exists(target_path):
            base, ext = os.path.splitext(name)
            new_name = f"{base}.{TIMESTAMP}{ext}"
            target_path = os.path.join(target_folder, new_name)
        shutil.move(path, target_path)
        REPORT['moved'].append({'src': path, 'dst': target_path})

    report_path = os.path.join(ROOT, f'sort_report_outputs_{TIMESTAMP}.json')
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(REPORT, rf, indent=2, ensure_ascii=False)
    print('Done. Report:', report_path)

if __name__ == '__main__':
    main()
    # now process logs/ to organize .log files into subfolders by mode
    process_logs_dir()
    # rewrite report to include logs moves
    report_path = os.path.join(ROOT, f'sort_report_outputs_{TIMESTAMP}.json')
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(REPORT, rf, indent=2, ensure_ascii=False)
    print('Logs processed (if any). Updated report:', report_path)
