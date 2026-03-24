from __future__ import annotations

import pytest
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


def test_mobilevit_etbert_fusion_model_aux_heads_use_prefusion_pooled_features():
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

    class ShortcutCompatibleProj(nn.Module):
        def forward(self, x):
            img_ctx, _, _, _ = torch.chunk(x, 4, dim=-1)
            return img_ctx

    model = MobileViTETBertFusionClassifier(
        num_classes=1,
        hidden_dim=4,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = DummyFusionEncoder()
    model.head_img = RecorderHead()
    model.head_tls = RecorderHead()
    model.head_fuse = SumHead()
    model.fusion_proj = ShortcutCompatibleProj()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids)

    assert torch.allclose(model.head_img.last_input, torch.full((2, 4), 10.0))
    assert torch.allclose(model.head_tls.last_input, torch.full((2, 4), 20.0))
    assert out["logits_fuse"].shape == (2, 1)


def test_mobilevit_etbert_fusion_model_fusion_head_keeps_prefusion_shortcut():
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

    class RecorderModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().clone()
            img_ctx, _, _, _ = torch.chunk(x, 4, dim=-1)
            return img_ctx

    class SumHead(nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    model = MobileViTETBertFusionClassifier(
        num_classes=1,
        hidden_dim=4,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = DummyFusionEncoder()
    model.fusion_proj = RecorderModule()
    model.head_fuse = SumHead()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    model(rgb, input_ids, attention_mask, token_type_ids)

    expected = torch.tensor([[10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0, 4.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]])
    assert torch.allclose(model.fusion_proj.last_input, expected.repeat(2, 1))


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

    class ShortcutCompatibleProj(nn.Module):
        def forward(self, x):
            img_ctx, _, _, _ = torch.chunk(x, 4, dim=-1)
            return img_ctx

    model = MobileViTETBertFusionClassifier(
        num_classes=1,
        hidden_dim=4,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = FailingFusionEncoder()
    model.head_img = RecorderHead()
    model.head_tls = RecorderHead()
    model.head_fuse = SumHead()
    model.fusion_proj = ShortcutCompatibleProj()

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


def test_mobilevit_etbert_fusion_model_residual_enhancer_keeps_image_dominant_shortcut():
    class DummyImageBackbone(nn.Module):
        def forward_features(self, rgb):
            batch = rgb.shape[0]
            pooled = torch.full((batch, 4), 10.0, dtype=rgb.dtype, device=rgb.device)
            tokens = torch.full((batch, 2, 4), 1.0, dtype=rgb.dtype, device=rgb.device)
            return {"tokens": tokens, "pooled": pooled}

    class DummyTextBackbone(nn.Module):
        def forward_features(self, input_ids, attention_mask, token_type_ids):
            batch = input_ids.shape[0]
            pooled = torch.full((batch, 4), 4.0, dtype=torch.float32, device=input_ids.device)
            tokens = torch.full((batch, 3, 4), 2.0, dtype=torch.float32, device=input_ids.device)
            mask = torch.ones((batch, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            return {"tokens": tokens, "mask": mask, "pooled": pooled}

    class DummyFusionEncoder(nn.Module):
        def forward(self, image_tokens, text_tokens, text_mask):
            return (
                torch.full_like(image_tokens, 13.0),
                torch.full_like(text_tokens, 5.0),
            )

    class ZeroResidual(nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)

    model = MobileViTETBertFusionClassifier(
        num_classes=4,
        hidden_dim=4,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = DummyFusionEncoder()
    model.fusion_proj = ZeroResidual()
    model.head_fuse = nn.Identity()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids, use_fusion=True)

    expected = torch.full((2, 4), 12.0)
    assert torch.allclose(out["logits_fuse"], expected)


def test_mobilevit_etbert_fusion_model_warmup_fusion_role_adds_context_as_residual():
    class DummyImageBackbone(nn.Module):
        def forward_features(self, rgb):
            batch = rgb.shape[0]
            pooled = torch.full((batch, 4), 10.0, dtype=rgb.dtype, device=rgb.device)
            tokens = torch.full((batch, 2, 4), 1.0, dtype=rgb.dtype, device=rgb.device)
            return {"tokens": tokens, "pooled": pooled}

    class DummyTextBackbone(nn.Module):
        def forward_features(self, input_ids, attention_mask, token_type_ids):
            batch = input_ids.shape[0]
            pooled = torch.full((batch, 4), 4.0, dtype=torch.float32, device=input_ids.device)
            tokens = torch.full((batch, 3, 4), 2.0, dtype=torch.float32, device=input_ids.device)
            mask = torch.ones((batch, 3), dtype=attention_mask.dtype, device=attention_mask.device)
            return {"tokens": tokens, "mask": mask, "pooled": pooled}

    class DummyFusionEncoder(nn.Module):
        def forward(self, image_tokens, text_tokens, text_mask):
            return (
                torch.full_like(image_tokens, 13.0),
                torch.full_like(text_tokens, 5.0),
            )

    class ContextDeltaResidual(nn.Module):
        def forward(self, x):
            img_ctx, txt_ctx, img_pre, txt_pre = torch.chunk(x, 4, dim=-1)
            return (img_ctx - img_pre) + (txt_ctx - txt_pre)

    model = MobileViTETBertFusionClassifier(
        num_classes=4,
        hidden_dim=4,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    model.image_backbone = DummyImageBackbone()
    model.text_backbone = DummyTextBackbone()
    model.fusion_encoder = DummyFusionEncoder()
    model.fusion_proj = ContextDeltaResidual()
    model.head_fuse = nn.Identity()
    model.head_img = nn.Identity()
    model.head_tls = nn.Identity()

    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 128, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    token_type_ids = torch.zeros(2, 8, dtype=torch.long)

    warmup_out = model(rgb, input_ids, attention_mask, token_type_ids, use_fusion=False)
    fusion_out = model(rgb, input_ids, attention_mask, token_type_ids, use_fusion=True)

    assert set(warmup_out.keys()) == {"logits_fuse", "logits_img", "logits_tls"}
    assert set(fusion_out.keys()) == {"logits_fuse", "logits_img", "logits_tls"}
    assert torch.allclose(warmup_out["logits_img"], torch.full((2, 4), 10.0))
    assert torch.allclose(warmup_out["logits_tls"], torch.full((2, 4), 4.0))
    assert torch.allclose(fusion_out["logits_img"], torch.full((2, 4), 10.0))
    assert torch.allclose(fusion_out["logits_tls"], torch.full((2, 4), 4.0))
    assert torch.allclose(warmup_out["logits_fuse"], torch.full((2, 4), 10.0))
    assert torch.allclose(fusion_out["logits_fuse"], torch.full((2, 4), 12.380797), atol=1e-5)


def test_mobilevit_etbert_fusion_model_legacy_mode_preserves_prefusion_fuse_path():
    model = MobileViTETBertFusionClassifier(num_classes=3, hidden_dim=16, vocab_size=256, max_tokens=16)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 256, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    token_type_ids = torch.zeros(2, 16, dtype=torch.long)

    out = model(rgb, input_ids, attention_mask, token_type_ids, use_fusion=False)

    assert out["logits_fuse"].shape == (2, 3)
    assert model.fusion_mode == "legacy"


def test_mobilevit_etbert_fusion_model_residual_mode_persists_shortcut_scale_in_state_dict():
    model = MobileViTETBertFusionClassifier(
        num_classes=2,
        hidden_dim=8,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.5,
    )
    clone = MobileViTETBertFusionClassifier(
        num_classes=2,
        hidden_dim=8,
        vocab_size=128,
        max_tokens=8,
        fusion_mode="residual_enhancer",
        text_shortcut_scale=0.1,
    )

    clone.load_state_dict(model.state_dict())

    assert float(clone.text_shortcut_scale.item()) == pytest.approx(0.5)
