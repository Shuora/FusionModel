import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fusion_common import (
    add_common_args,
    build_common_kwargs,
    load_fusion_data,
    evaluate_full,
    setup_logging,
    ensure_output_dirs,
    prepare_run_output_dir,
    build_run_artifact_paths,
    plot_training_curves,
    plot_confusion,
    save_report_md,
    export_metrics_artifacts,
    log_saved
)
from experiments.baselines.deeppacket import DeepPacket
from experiments.baselines.lstm_baseline import LSTMClassifier
from experiments.baselines.vit_baseline import MobileViTOnly

def train_model(model, train_loader, val_loader, epochs, lr, device, patience, output_dir, tag):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, pcaps, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, pcaps, labels = images.to(device), pcaps.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Decide which input to use
            if isinstance(model, DeepPacket) or isinstance(model, LSTMClassifier):
                outputs = model(pcaps)
            elif isinstance(model, MobileViTOnly):
                outputs = model(images)
            else:
                outputs = model(images, pcaps)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        eval_res = evaluate_full(model, val_loader, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(eval_res["loss"])
        history["val_acc"].append(eval_res["acc"])
        history["val_f1"].append(eval_res["macro_f1"])
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Acc={eval_res['acc']:.4f}, Val F1={eval_res['macro_f1']:.4f}")
        
        if eval_res["macro_f1"] > best_val_f1:
            best_val_f1 = eval_res["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
                
    return model, history

def main():
    p = argparse.ArgumentParser(description="Train baseline models")
    add_common_args(p)
    p.add_argument("--model_type", choices=["deeppacket", "lstm", "vit"], required=True)
    args = p.parse_args()
    
    kwargs = build_common_kwargs(args)
    device = kwargs['device']
    output_dir = Path(kwargs['output_dir'])
    ensure_output_dirs(output_dir)
    
    ts = torch.utils.data.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"baseline_{args.model_type}"
    run_dir = prepare_run_output_dir(output_dir, f"{tag}_{ts}")
    artifact_paths = build_run_artifact_paths(run_dir)
    
    setup_logging(artifact_paths["train_log"], force=True)
    logger = logging.getLogger(f"baseline_{args.model_type}")
    
    train_loader, train_classes = load_fusion_data(
        kwargs['train_image_dir'], kwargs['train_pcap_dir'], 
        batch_size=kwargs['batch_size'], is_train=True, device=device
    )
    test_loader, _ = load_fusion_data(
        kwargs['test_image_dir'], kwargs['test_pcap_dir'], 
        batch_size=kwargs['batch_size'], is_train=False, device=device
    )
    
    num_classes = len(train_classes)
    if args.model_type == "deeppacket":
        model = DeepPacket(num_classes=num_classes)
    elif args.model_type == "lstm":
        model = LSTMClassifier(num_classes=num_classes)
    else:
        model = MobileViTOnly(num_classes=num_classes)
        
    model, history = train_model(
        model, train_loader, test_loader, 
        epochs=kwargs['epochs'], lr=kwargs['lr'], 
        device=device, patience=kwargs['patience'], 
        output_dir=run_dir, tag=tag
    )
    
    # Evaluation and Artifacts
    model.load_state_dict(torch.load(run_dir / "best_model.pth"))
    eval_result = evaluate_full(model, test_loader, device)
    
    plot_training_curves(history, artifact_paths["metrics_curve"], title=f"Baseline {args.model_type}")
    plot_confusion(eval_result["cm"], train_classes, artifact_paths["confusion_matrix"], f"Baseline {args.model_type}")
    
    save_report_md(
        artifact_paths["report_md"],
        title=f"Baseline: {args.model_type}",
        acc=eval_result["acc"],
        macro_f1=eval_result["macro_f1"],
        report=eval_result["report"],
        cm=eval_result["cm"],
        confusion_image=artifact_paths["confusion_matrix"].name,
        curve_image=artifact_paths["metrics_curve"].name
    )
    
    print(f"Baseline {args.model_type} training complete. Results in {run_dir}")

if __name__ == "__main__":
    main()
