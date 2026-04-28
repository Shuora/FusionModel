import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.fusion_common import initialize_fusion_model
from experiments.baselines.deeppacket import DeepPacket
from experiments.baselines.lstm_baseline import LSTMClassifier
from experiments.baselines.vit_baseline import MobileViTOnly

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

def measure_latency(model, dummy_inputs, device, num_runs=100, warm_up=10):
    model.eval()
    model.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(*dummy_inputs)
            
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(*dummy_inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start_time)
            
    return np.mean(latencies) * 1000, np.std(latencies) * 1000 # ms

def main():
    parser = argparse.ArgumentParser(description="Measure model efficiency (Params, FLOPs, Latency)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    num_classes = 10
    
    models = {
        "DeepPacket (CNN)": DeepPacket(num_classes=num_classes),
        "LSTM": LSTMClassifier(num_classes=num_classes),
        "MobileViT-only": MobileViTOnly(num_classes=num_classes),
        "Fusion-Attention (Ours)": initialize_fusion_model(num_classes=num_classes, fusion_mode="attention"),
        "Fusion-Concat (Ablation)": initialize_fusion_model(num_classes=num_classes, fusion_mode="concat"),
    }
    
    print(f"{'Model':<30} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'Latency (ms)':<15}")
    print("-" * 80)
    
    for name, model in models.items():
        # Prepare dummy inputs
        dummy_img = torch.randn(1, 3, 28, 28).to(device)
        dummy_pcap = torch.randint(0, 256, (1, 784)).to(device)
        
        if isinstance(model, (DeepPacket, LSTMClassifier)):
            inputs = (dummy_pcap,)
        elif isinstance(model, MobileViTOnly):
            inputs = (dummy_img,)
        else:
            inputs = (dummy_img, dummy_pcap)
            
        # Params & FLOPs
        model.eval()
        if HAS_FVCORE:
            try:
                flops_analyzer = FlopCountAnalysis(model.to(device), inputs)
                flops_analyzer.unsupported_ops_warnings(False)
                flops = flops_analyzer.total() / 1e9 # GFLOPs
            except Exception as e:
                print(f"Warning: Could not calculate FLOPs for {name}: {e}")
                flops = 0.0
            params = sum(p.numel() for p in model.parameters()) / 1e6 # M
        else:
            params = sum(p.numel() for p in model.parameters()) / 1e6
            flops = 0.0
            
        # Latency
        avg_latency, _ = measure_latency(model, inputs, device)
        
        print(f"{name:<30} | {params:<12.3f} | {flops:<12.3f} | {avg_latency:<15.3f}")

if __name__ == "__main__":
    main()
