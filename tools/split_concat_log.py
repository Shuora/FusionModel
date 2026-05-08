import os
import re
import shutil
import hashlib
import json
from datetime import datetime
SOURCE = os.path.abspath('C:\\Repositories\\Traffic\\CharBERT-MobileViT\\outputs\\logs\\concat_20260205_235803.log')
LOGDIR = os.path.dirname(SOURCE)
BACKUP_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
DONE_RE = re.compile('\\bdone\\b.*log=(?P<path>\\S+)', re.IGNORECASE)
START_RE = re.compile('\\bstart\\b', re.IGNORECASE)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda : f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    with open(SOURCE, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    sections = []
    in_section = False
    buf = []
    start_idx = None
    for (idx, line) in enumerate(lines):
        if not in_section and START_RE.search(line):
            in_section = True
            buf = [line]
            start_idx = idx
            continue
        if in_section:
            buf.append(line)
            m = DONE_RE.search(line)
            if m:
                target_path = m.group('path')
                target_basename = os.path.basename(target_path)
                target = os.path.join(LOGDIR, target_basename)
                sections.append({'start_idx': start_idx, 'end_idx': idx, 'lines': buf.copy(), 'target': target, 'target_basename': target_basename})
                in_section = False
                buf = []
                start_idx = None
    if in_section and buf:
        misc_target = os.path.join(LOGDIR, 'misc_split_from_concat_%s.log' % BACKUP_TS)
        sections.append({'start_idx': start_idx, 'end_idx': len(lines) - 1, 'lines': buf.copy(), 'target': misc_target, 'target_basename': os.path.basename(misc_target)})
    report = {'source': SOURCE, 'source_sha256': sha256_file(SOURCE), 'created_at': BACKUP_TS, 'sections': []}
    for (i, s) in enumerate(sections, 1):
        target = s['target']
        lines_block = s['lines']
        start_line = lines_block[0].strip() if lines_block else ''
        entry = {'section_index': i, 'start_idx': s['start_idx'], 'end_idx': s['end_idx'], 'target': target, 'target_basename': s['target_basename'], 'appended': False, 'reason': ''}
        if os.path.abspath(target) == os.path.abspath(SOURCE):
            entry['appended'] = False
            entry['reason'] = 'target_is_source_skip'
            report['sections'].append(entry)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            bak = target + '.bak.' + BACKUP_TS
            shutil.copy2(target, bak)
            entry['backup'] = os.path.basename(bak)
            with open(target, 'r', encoding='utf-8', errors='replace') as tf:
                target_text = tf.read()
            if start_line and start_line in target_text:
                entry['appended'] = False
                entry['reason'] = 'already_present_by_start_line'
                report['sections'].append(entry)
                continue
        else:
            entry['backup'] = None
        with open(target, 'a', encoding='utf-8', errors='replace') as tf:
            tf.write('\n')
            tf.writelines(lines_block)
            tf.flush()
            os.fsync(tf.fileno())
        entry['appended'] = True
        entry['reason'] = 'appended'
        entry['appended_lines'] = len(lines_block)
        report['sections'].append(entry)
    report_path = os.path.join(LOGDIR, 'split_report_concat_%s.json' % BACKUP_TS)
    with open(report_path, 'w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)
    print('Done. Sections processed: %d. Report: %s' % (len(sections), report_path))
if __name__ == '__main__':
    main()