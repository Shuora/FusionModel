import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification, MobileViTConfig

class MobileViTOnly(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        mv_cfg = MobileViTConfig()
        mv_cfg.num_labels = num_classes
        self.mobilevit = MobileViTForImageClassification(mv_cfg)
        
    def forward(self, images, pcaps=None):
        # pcaps is ignored
        return self.mobilevit(images).logits

if __name__ == "__main__":
    model = MobileViTOnly(num_classes=10)
    dummy_img = torch.randn(2, 3, 28, 28)
    output = model(dummy_img)
    print(f"Output shape: {output.shape}")
