import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification, MobileViTConfig
import logging

logger = logging.getLogger(__name__)

class MobileViTAblation(nn.Module):
    """
    MobileViT single-branch ablation model.
    Matches the space branch architecture in AttentionFusionModel.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Load default config for MobileViT
        mv_cfg = MobileViTConfig()
        # The fusion model uses mobilevit_feature_dim as the output of the backbone
        mobilevit_feature_dim = mv_cfg.neck_hidden_sizes[-1] if hasattr(mv_cfg, "neck_hidden_sizes") else 640
        mv_cfg.num_labels = mobilevit_feature_dim
        
        self.mobilevit = MobileViTForImageClassification(mv_cfg)
        # Match fusion model's internal projection
        self.mobilevit.classifier = nn.Linear(mobilevit_feature_dim, mobilevit_feature_dim)

        # Standard ablation head (matches fusion model's output head)
        self.classifier_head = nn.Sequential(
            nn.Linear(mobilevit_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x is (batch, channels, 28, 28)
        # MobileViT expects RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Extract features from backbone
        # MobileViTForImageClassification returns an object with 'logits'
        feats = self.mobilevit(x).logits
        
        # Pass through the ablation head
        logits = self.classifier_head(feats)
        return logits

if __name__ == "__main__":
    model = MobileViTAblation(num_classes=10)
    dummy_input = torch.randn(2, 3, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected: [2, 10]
