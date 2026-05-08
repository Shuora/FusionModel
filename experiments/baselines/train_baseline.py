import argparse

import logging

import os

import sys

from pathlib import Path



import torch

import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm



                              

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

from experiments.baselines.vit_baseline import MalwareViT

from experiments.baselines.cnn2d_baseline import CNN2D

from experiments.baselines.mobilevit_ablation import MobileViTAblation

from experiments.baselines.charbert_ablation import CharBERTAblation



class BaselineWrapper(nn.Module):

    def __init__(self, model, model_type):

        super().__init__()

        self.model = model

        self.model_type = model_type

        

    def forward(self, images, pcaps):

        if self.model_type in ["deeppacket", "lstm", "charbert_ablation"]:

            return self.model(pcaps)

        elif self.model_type in ["vit", "cnn2d", "mobilevit_ablation"]:

            return self.model(images)

        return self.model(images, pcaps)



def train_model(model, train_loader, val_loader, epochs, lr, device, patience, output_dir, tag):

                                       

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    

    best_val_f1 = 0.0

    counter = 0

    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

    

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0.0

        correct = 0

        total = 0

        for images, pcaps, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):

            images, pcaps, labels = images.to(device), pcaps.to(device), labels.to(device)

            

            optimizer.zero_grad()

            outputs = model(images, pcaps)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            

        avg_train_loss = train_loss / len(train_loader)

        avg_train_acc = correct / total

        

                    

        eval_res = evaluate_full(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)

        history["train_acc"].append(avg_train_acc)

        history["train_f1"].append(avg_train_acc)                                                     

        history["val_loss"].append(eval_res["loss"])

        history["val_acc"].append(eval_res["acc"])

        history["val_f1"].append(eval_res["macro_f1"])

        

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={eval_res['acc']:.4f}, Val F1={eval_res['macro_f1']:.4f}")

        

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

    p.add_argument("--model_type", choices=["deeppacket", "lstm", "vit", "cnn2d", "mobilevit_ablation", "charbert_ablation"], required=True)

    args = p.parse_args()

    

    kwargs = build_common_kwargs(args)

    device = kwargs['device']

    output_dir = Path(kwargs['output_dir'])

    ensure_output_dirs(output_dir)

    

    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tag = f"baseline_{args.model_type}"

    run_dir = prepare_run_output_dir(output_dir, f"{tag}_{ts}")

    artifact_paths = build_run_artifact_paths(run_dir)

    

    setup_logging(artifact_paths["train_log"], force=True)

    logger = logging.getLogger(f"baseline_{args.model_type}")

    

    image_mode = getattr(args, "image_mode", "rgb")

    

    train_loader, train_classes = load_fusion_data(

        kwargs['train_image_dir'], kwargs['train_pcap_dir'], 

        batch_size=kwargs['batch_size'], is_train=True, image_mode=image_mode

    )

    test_loader, _ = load_fusion_data(

        kwargs['test_image_dir'], kwargs['test_pcap_dir'], 

        batch_size=kwargs['batch_size'], is_train=False, image_mode=image_mode

    )

    

    num_classes = len(train_classes)

    if args.model_type == "deeppacket":

        model = DeepPacket(num_classes=num_classes)

    elif args.model_type == "lstm":

        model = LSTMClassifier(num_classes=num_classes)

    elif args.model_type == "cnn2d":

        input_channels = 1 if image_mode == "gray" else 3

        model = CNN2D(num_classes=num_classes, input_channels=input_channels)

    elif args.model_type == "mobilevit_ablation":

        model = MobileViTAblation(num_classes=num_classes)

    elif args.model_type == "charbert_ablation":

        model = CharBERTAblation(num_classes=num_classes)

    else:

        model = MalwareViT(num_classes=num_classes)

    

    model = BaselineWrapper(model, args.model_type)

        

    model, history = train_model(

        model, train_loader, test_loader, 

        epochs=kwargs['epochs'], lr=kwargs['lr'], 

        device=device, patience=kwargs['patience'], 

        output_dir=run_dir, tag=tag

    )

    

                              

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

