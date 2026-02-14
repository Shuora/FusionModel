from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def evaluate_predictions(y_true: List[int], y_pred: List[int], labels: List[str]):
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "accuracy": acc,
        "report": report,
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))),
    }


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_metric_curve(history: Dict[str, List[float]], out_dir: Path, metric: str):
    plt.figure()
    plt.plot(history[metric], label=f'train_{metric}')
    if f'val_{metric}' in history:
        plt.plot(history[f'val_{metric}'], label=f'val_{metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} curve')
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'{metric}.png')
    plt.close()
