from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion_model import MobileViTETBertFusionClassifier


def test_mobilevit_etbert_fusion_model_forward_shapes_without_gate():
    model = MobileViTETBertFusionClassifier(num_classes=3, hidden_dim=64, vocab_size=4096, max_tokens=128)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 4096, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    token_type_ids = torch.zeros(2, 128, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)

    assert out["logits_fuse"].shape == (2, 3)
    assert out["logits_img"].shape == (2, 3)
    assert out["logits_tls"].shape == (2, 3)
    assert "gate" not in out


def test_mobilevit_etbert_fusion_model_handles_partial_attention_mask():
    model = MobileViTETBertFusionClassifier(num_classes=2, hidden_dim=32, vocab_size=2048, max_tokens=64)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 2048, (2, 64))
    attention_mask = torch.zeros(2, 64, dtype=torch.long)
    attention_mask[:, :8] = 1
    token_type_ids = torch.zeros(2, 64, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)
    assert out["logits_fuse"].shape == (2, 2)
    assert "gate" not in out


def test_mobilevit_etbert_fusion_model_aux_heads_use_fused_context_features():
    class DummyImageBackbone(nn.Module):
        def forward_features(self, rgb):
            batch = rgb.shape[0]
            pooled = torch.full((batch, 4), 10.0, dtype=rgb.dtype, device=rgb.device)
            tokens = torch.full((batch, 2, 4), 1.0, dtype=rgb.dtype, device=rgb.device)
            return {"tokens": tokens, "pooled": pooled}

    class DummyTextBackbone(nn.Module):
        def forward_features(self, input_ids, attention_mask, token_type_ids):
            batch = input_ids.shape[0]
            pooled = torch.full((batch, 4), 20.0, dtype=torch.float32, device=input_ids.device)
            tokens = torch.full((batch, 3, 4), 2.0, dtype=torch.float32, device=input_ids.device)
            mask = torch.ones((batch, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            return {"tokens": tokens, "mask": mask, "pooled": pooled}

    class DummyFusionEncoder(nn.Module):
        def forward(self, image_tokens, text_tokens, text_mask):
            return (
                torch.full_like(image_tokens, 3.0),
                torch.full_like(text_tokens, 4.0),
            )

    class RecorderHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            return x

    class SumHead(nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    model = MobileViTETBertFusionClassifier(num_classes=1, hidden_dim=4, vocab_size=128, max_tokens=8)
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = DummyFusionEncoder()
    model.head_img = RecorderHead()
    model.head_tls = RecorderHead()
    model.head_fuse = SumHead()
    model.fusion_proj = nn.Identity()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)

    assert torch.allclose(model.head_img.last_input, torch.full((2, 4), 3.0))
    assert torch.allclose(model.head_tls.last_input, torch.full((2, 4), 4.0))
    assert out["logits_fuse"].shape == (2, 1)


def test_mobilevit_etbert_fusion_model_warmup_bypasses_fusion_encoder_and_uses_pooled_features():
    class DummyImageBackbone(nn.Module):
        def forward_features(self, rgb):
            batch = rgb.shape[0]
            pooled = torch.full((batch, 4), 10.0, dtype=rgb.dtype, device=rgb.device)
            tokens = torch.full((batch, 2, 4), 1.0, dtype=rgb.dtype, device=rgb.device)
            return {"tokens": tokens, "pooled": pooled}

    class DummyTextBackbone(nn.Module):
        def forward_features(self, input_ids, attention_mask, token_type_ids):
            batch = input_ids.shape[0]
            pooled = torch.full((batch, 4), 20.0, dtype=torch.float32, device=input_ids.device)
            tokens = torch.full((batch, 3, 4), 2.0, dtype=torch.float32, device=input_ids.device)
            mask = torch.ones((batch, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            return {"tokens": tokens, "mask": mask, "pooled": pooled}

    class FailingFusionEncoder(nn.Module):
        def forward(self, image_tokens, text_tokens, text_mask):
            raise AssertionError("fusion encoder should not run during warmup")

    class RecorderHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            return x

    class SumHead(nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    model = MobileViTETBertFusionClassifier(num_classes=1, hidden_dim=4, vocab_size=128, max_tokens=8)
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = FailingFusionEncoder()
    model.head_img = RecorderHead()
    model.head_tls = RecorderHead()
    model.head_fuse = SumHead()
    model.fusion_proj = nn.Identity()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids, use_fusion=False)

    assert torch.allclose(model.head_img.last_input, torch.full((2, 4), 10.0))
    assert torch.allclose(model.head_tls.last_input, torch.full((2, 4), 20.0))
    assert out["logits_fuse"].shape == (2, 1)


def test_mobilevit_etbert_fusion_model_can_optionally_return_debug_tokens():
    model = MobileViTETBertFusionClassifier(num_classes=3, hidden_dim=32, vocab_size=256, max_tokens=16)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 256, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    token_type_ids = torch.zeros(2, 16, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids, return_features=True)

    assert "img_tokens" in out
    assert "txt_tokens" in out
    assert out["img_tokens"].ndim == 3
    assert out["txt_tokens"].ndim == 3
