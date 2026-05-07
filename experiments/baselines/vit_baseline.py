import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging

logger = logging.getLogger(__name__)

class MalwareViT(nn.Module):
    """
    User-specified ViT implementation from https://github.com/Shuora/ViT
    """
    def __init__(self, num_classes=10, local_model_path="models/vit-base-patch16-224-in21k", **kwargs):
        super().__init__()
        # logger.info("初始化 MalwareViT 模型")
        # logger.info(f"类别数量: {num_classes}")
        
        if local_model_path:
            # logger.info(f"从本地路径加载 ViT 模型: {local_model_path}")
            config = ViTConfig.from_pretrained(local_model_path)
            self.vit = ViTModel.from_pretrained(local_model_path, config=config)
        else:
            # logger.info("从远程仓库加载 ViT 模型")
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        # Note: In the user's repo, they use Softmax here. 
        # But our trainer (nn.CrossEntropyLoss) expects raw logits.
        # We will keep it as raw logits for compatibility with train_baseline.py
        self.activation = nn.Identity()

    def forward(self, x):
        # x is (batch, 1, 28, 28) or (batch, 3, 28, 28)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # Gray to RGB
            
        # Interpolate to 224x224 as required by google/vit-base-patch16-224
        x = nn.functional.interpolate(x, size=(224, 224))
        
        outputs = self.vit(pixel_values=x)
        # Use pooler_output ([CLS] token representation)
        x = self.classifier(outputs.pooler_output)
        return self.activation(x)

if __name__ == "__main__":
    # Test
    model = MalwareViT(num_classes=10)
    dummy_input = torch.randn(2, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
