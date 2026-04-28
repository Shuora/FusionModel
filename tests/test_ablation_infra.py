import torch
import torch.nn as nn
from src.fusion_common import initialize_fusion_model

def test_concat_fusion_init():
    num_classes = 10
    model = initialize_fusion_model(
        num_classes=num_classes,
        fusion_mode="concat",
        attention_dim=256
    )
    assert model.fusion_mode == "concat"
    assert isinstance(model.out[0], nn.Linear)
    # mobilevit_feature_dim (640) + attention_dim (256) = 896
    assert model.out[0].in_features == 896
    print("test_concat_fusion_init passed")

def test_concat_fusion_forward():
    num_classes = 10
    model = initialize_fusion_model(
        num_classes=num_classes,
        fusion_mode="concat",
        attention_dim=256
    )
    device = torch.device("cpu")
    model.to(device)
    
    dummy_images = torch.randn(2, 3, 28, 28)
    dummy_pcaps = torch.randint(0, 256, (2, 784))
    
    output = model(dummy_images, dummy_pcaps)
    assert output.shape == (2, num_classes)
    print("test_concat_fusion_forward passed")

if __name__ == "__main__":
    test_concat_fusion_init()
    # Skip forward test in CI environment if it takes too long or needs heavy deps
    # But for verification here it's good.
    test_concat_fusion_forward()
