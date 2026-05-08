import os
import re
import glob
import shutil
import json

def parse_relation(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    mapping = []
    for line in lines:
        if line.startswith('|') and '源文件' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 2:
                mapping.append({'source': parts[0], 'target': parts[1]})
    return mapping

def get_metrics_from_source(source_path):
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    acc = None
    if source_path.endswith('.md'):
        acc_match = re.search(r'\*\*Test Accuracy:\*\*\s*([\d\.]+)', content)
        if acc_match: acc = float(acc_match.group(1))
    else: # .log
        for line in content.split('\n'):
            if 'accuracy' in line:
                parts = line.split()
                nums = [p for p in parts if p.replace('.','',1).isdigit()]
                if nums: 
                    acc = float(nums[0])
                    break
        if acc is None:
             acc_match = re.search(r'准确率:\s*([\d\.]+)', content)
             if acc_match: 
                 acc = float(acc_match.group(1))
                 if acc > 1: acc /= 100.0
    return acc

def find_matching_raw_dir(target_acc):
    if target_acc is None: return None
    all_metrics = glob.glob("outputs/**/metrics.json", recursive=True)
    for mp in all_metrics:
        if 'outputs/final/' in mp: continue
        try:
            with open(mp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            eval_acc = data.get('eval', {}).get('acc')
            if eval_acc is None:
                 eval_acc = data.get('history', {}).get('val_acc', [-1])[-1]
            if eval_acc and abs(float(eval_acc) - target_acc) < 0.0005:
                return os.path.dirname(mp)
        except: continue
    return None

def extract_metrics(content):
    acc, f1 = None, None
    report = ""
    cm_str = ""
    report_match = re.search(r'分类报告:\s*\n(.*?)\n(?:202|\n\n|\Z)', content, re.DOTALL)
    if report_match:
        report = report_match.group(1).strip()
    cm_match = re.search(r'混淆矩阵:\s*\n(.*?)(\n202|\Z)', content, re.DOTALL)
    if cm_match:
        cm_str = cm_match.group(1).strip()
    if report:
        for line in report.split('\n'):
            if 'accuracy' in line:
                parts = line.split()
                nums = [p for p in parts if p.replace('.','',1).isdigit()]
                if nums: acc = float(nums[0])
            if 'macro avg' in line:
                parts = line.split()
                nums = [p for p in parts if p.replace('.','',1).isdigit()]
                if len(nums) >= 3: f1 = nums[2]
    return acc, f1, report, cm_str

def main():
    mapping = parse_relation('log/relation.md')
    for m in mapping:
        source = m['source']
        target_base = m['target']
        print(f"Checking {target_base} (source: {source})...")
        if os.path.exists(target_base):
            for item in os.listdir(target_base):
                path = os.path.join(target_base, item)
                if os.path.isdir(path): shutil.rmtree(path)
                else: os.remove(path)
        else:
            os.makedirs(target_base, exist_ok=True)
        target_acc = get_metrics_from_source(source)
        raw_dir = find_matching_raw_dir(target_acc)
        if raw_dir:
            print(f"  Found matching raw dir: {raw_dir} (Acc: {target_acc})")
            for item in os.listdir(raw_dir):
                s = os.path.join(raw_dir, item)
                d = os.path.join(target_base, item)
                if os.path.isdir(s): shutil.copytree(s, d)
                else: shutil.copy2(s, d)
        else:
            print(f"  No matching raw dir found for Acc {target_acc}. Target will only have log/md.")
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            source_content = f.read()
        if source.endswith('.log'):
             log_name = 'train.log'
             if 'ViT' in source: log_name = 'ViT.log'
             with open(os.path.join(target_base, log_name), 'w', encoding='utf-8') as f:
                 f.write(source_content)
             acc, f1, report, cm = extract_metrics(source_content)
             report_md = f"# 融合方式: {os.path.basename(target_base)}\n\n"
             report_md += f"**Test Accuracy:** {acc if acc is not None else 'N/A'}\n\n"
             report_md += f"**Macro F1:** {f1 if f1 is not None else 'N/A'}\n\n"
             report_md += "**分类报告:**\n\n```\n" + report + "\n```\n\n"
             report_md += "**混淆矩阵:**\n\n```\n" + cm + "\n```\n\n"
             if raw_dir:
                 report_md += "![Confusion Matrix](confusion_matrix.png)\n"
                 report_md += "![Metrics Curve](metrics_curve.png)\n"
             with open(os.path.join(target_base, 'report.md'), 'w', encoding='utf-8') as f:
                 f.write(report_md)
        else:
             with open(os.path.join(target_base, 'report.md'), 'w', encoding='utf-8') as f:
                 f.write(source_content)
             if 'soft' in source:
                  shutil.copy2(source, os.path.join(target_base, 'report_soft_voting.md'))
             if 'blender' in source:
                  shutil.copy2(source, os.path.join(target_base, 'report_two_level_blender.md'))

if __name__ == "__main__":
    main()
