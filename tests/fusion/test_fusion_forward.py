import torch


def test_fusion_forward_shapes():
    from src.fusion.models.fusion_cross_attn import FusionModel

    model = FusionModel(num_classes=10, hidden_dim=64)
    image_tensor = torch.randn(2, 3, 28, 28)
    token_ids = torch.randint(0, 1024, (2, 32))
    attn_mask = torch.ones(2, 32, dtype=torch.long)

    out = model(image_tensor, token_ids, attn_mask)
    assert out["logits_fuse"].shape == (2, 10)
