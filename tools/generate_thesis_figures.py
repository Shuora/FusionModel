import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Professional academic color palette
COLORS = {
    'blue': '#dae8fc',
    'yellow': '#fff2cc',
    'green': '#d5e8d4',
    'red': '#f8cecc',
    'purple': '#e1d5e7',
    'border_blue': '#6c8ebf',
    'border_yellow': '#d6b656',
    'border_green': '#82b366'
}

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'figure.autolayout': True
})

def plot_model_comparison(data, output_path, title="Classification Performance Comparison"):
    """
    Plots a grouped bar chart for model comparisons.
    data: DataFrame with columns ['Model', 'Metric', 'Value']
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Value", hue="Metric", data=data, palette=[COLORS['blue'], COLORS['yellow'], COLORS['green']], edgecolor='gray')
    
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=9)
                   
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def plot_robustness_curve(data, output_path, title="Robustness to Class Imbalance"):
    """
    Plots a line chart for robustness tests.
    data: DataFrame with columns ['Imbalance Ratio', 'Model', 'Macro-F1']
    """
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="Imbalance Ratio", y="Macro-F1", hue="Model", style="Model", markers=True, data=data, palette="Set1")
    
    plt.title(title)
    plt.xlabel("Imbalance Ratio (Major:Minor)")
    plt.ylabel("Macro-F1 Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def generate_placeholder_figures():
    """Generates sample figures based on typical results for illustration."""
    output_dir = Path("outputs/thesis_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model Comparison (Exp 4.3)
    comparison_df = pd.DataFrame([
        ["DeepPacket", "Accuracy", 0.92], ["DeepPacket", "Macro-F1", 0.88],
        ["LSTM", "Accuracy", 0.90], ["LSTM", "Macro-F1", 0.85],
        ["MobileViT-only", "Accuracy", 0.93], ["MobileViT-only", "Macro-F1", 0.89],
        ["Fusion (Ours)", "Accuracy", 0.97], ["Fusion (Ours)", "Macro-F1", 0.96],
    ], columns=["Model", "Metric", "Value"])
    plot_model_comparison(comparison_df, output_dir / "fig4_3_model_comparison.png")
    
    # 2. Robustness Curve (Exp 4.4)
    robustness_df = pd.DataFrame([
        [2, "DeepPacket", 0.92], [5, "DeepPacket", 0.85], [10, "DeepPacket", 0.78], [15, "DeepPacket", 0.72],
        [2, "Fusion (Ours)", 0.96], [5, "Fusion (Ours)", 0.94], [10, "Fusion (Ours)", 0.92], [15, "Fusion (Ours)", 0.90],
    ], columns=["Imbalance Ratio", "Model", "Macro-F1"])
    plot_robustness_curve(robustness_df, output_dir / "fig4_7_robustness_curve.png")

if __name__ == "__main__":
    generate_placeholder_figures()
