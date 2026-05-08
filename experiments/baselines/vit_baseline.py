import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import logging
logger = logging.getLogger(__name__)

class MalwareViT(nn.Module):

    def __init__(self, num_classes=10, local_model_path='models/vit-base-patch16-224-in21k', **kwargs):
        super().__init__()
        if local_model_path:
            config = ViTConfig.from_pretrained(local_model_path)
            self.vit = ViTModel.from_pretrained(local_model_path, config=config)
        else:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, num_classes))
        self.activation = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224))
        outputs = self.vit(pixel_values=x)
        x = self.classifier(outputs.pooler_output)
        return self.activation(x)
if __name__ == '__main__':
    model = MalwareViT(num_classes=10)
    dummy_input = torch.randn(2, 1, 28, 28)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')