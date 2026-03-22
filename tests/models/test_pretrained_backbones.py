from __future__ import annotations

import json
from pathlib import Path
import warnings

import torch

from src.models.etbert_backbone import ETBertBackbone
from src.models.mobilevit_backbone import MobileViTBackbone


def test_etbert_backbone_can_truncate_encoder_layers(tmp_path: Path):
    config_path = tmp_path / "tiny_config.json"
    config_path.write_text(
        json.dumps(
            {
                "emb_size": 128,
                "feedforward_size": 512,
                "hidden_size": 128,
                "heads_num": 2,
                "layers_num": 6,
                "max_seq_length": 256,
                "dropout": 0.1,
            }
        ),
        encoding="utf-8",
    )

    model = ETBertBackbone(vocab_size=4096, config_path=config_path, num_layers=4)
    assert model.num_layers == 4
    assert len(model.encoder.layers) == 4

    input_ids = torch.randint(0, 4096, (2, 48))
    attention_mask = torch.ones(2, 48, dtype=torch.long)
    token_type_ids = torch.zeros(2, 48, dtype=torch.long)
    output = model(input_ids, attention_mask, token_type_ids)
    assert output.shape == (2, 128)


def test_etbert_backbone_load_checkpoint_with_strict_false(tmp_path: Path):
    model = ETBertBackbone(vocab_size=256, hidden_dim=64, max_tokens=32, num_layers=2)
    checkpoint = {
        "state_dict": {
            "token_embed.weight": torch.randn_like(model.token_embed.weight),
            "unexpected.weight": torch.randn(3, 3),
        }
    }
    checkpoint_path = tmp_path / "etbert.ckpt"
    torch.save(checkpoint, checkpoint_path)

    result = model.load_pretrained_checkpoint(checkpoint_path)
    assert "missing_keys" in result
    assert "unexpected_keys" in result
    assert "unexpected.weight" in result["unexpected_keys"]
    assert any(name.startswith("encoder.") for name in result["missing_keys"])


def test_etbert_backbone_loads_bert_style_wrapped_keys(tmp_path: Path):
    model = ETBertBackbone(vocab_size=256, hidden_dim=32, max_tokens=16, num_layers=2)
    layer0 = model.encoder.layers[0]
    query_weight = torch.randn(32, 32)
    key_weight = torch.randn(32, 32)
    value_weight = torch.randn(32, 32)
    query_bias = torch.randn(32)
    key_bias = torch.randn(32)
    value_bias = torch.randn(32)
    word_embed = torch.randn_like(model.token_embed.weight)

    checkpoint = {
        "model_state_dict": {
            "model.bert.embeddings.word_embeddings.weight": word_embed,
            "model.bert.embeddings.position_embeddings.weight": torch.randn_like(model.pos_embed.weight),
            "model.bert.embeddings.token_type_embeddings.weight": torch.randn_like(model.type_embed.weight),
            "model.bert.encoder.layer.0.attention.self.query.weight": query_weight,
            "model.bert.encoder.layer.0.attention.self.key.weight": key_weight,
            "model.bert.encoder.layer.0.attention.self.value.weight": value_weight,
            "model.bert.encoder.layer.0.attention.self.query.bias": query_bias,
            "model.bert.encoder.layer.0.attention.self.key.bias": key_bias,
            "model.bert.encoder.layer.0.attention.self.value.bias": value_bias,
            "model.bert.encoder.layer.0.attention.output.dense.weight": torch.randn_like(layer0.self_attn.out_proj.weight),
            "model.bert.encoder.layer.0.attention.output.dense.bias": torch.randn_like(layer0.self_attn.out_proj.bias),
            "model.bert.encoder.layer.0.intermediate.dense.weight": torch.randn_like(layer0.linear1.weight),
            "model.bert.encoder.layer.0.intermediate.dense.bias": torch.randn_like(layer0.linear1.bias),
            "model.bert.encoder.layer.0.output.dense.weight": torch.randn_like(layer0.linear2.weight),
            "model.bert.encoder.layer.0.output.dense.bias": torch.randn_like(layer0.linear2.bias),
            "model.bert.encoder.layer.0.attention.output.LayerNorm.weight": torch.randn_like(layer0.norm1.weight),
            "model.bert.encoder.layer.0.attention.output.LayerNorm.bias": torch.randn_like(layer0.norm1.bias),
            "model.bert.encoder.layer.0.output.LayerNorm.weight": torch.randn_like(layer0.norm2.weight),
            "model.bert.encoder.layer.0.output.LayerNorm.bias": torch.randn_like(layer0.norm2.bias),
        }
    }
    checkpoint_path = tmp_path / "etbert_bert_style.ckpt"
    torch.save(checkpoint, checkpoint_path)

    result = model.load_pretrained_checkpoint(checkpoint_path)
    assert result["checkpoint_found"] is True
    assert torch.equal(model.token_embed.weight, word_embed)
    assert torch.equal(
        model.encoder.layers[0].self_attn.in_proj_weight,
        torch.cat([query_weight, key_weight, value_weight], dim=0),
    )
    assert torch.equal(
        model.encoder.layers[0].self_attn.in_proj_bias,
        torch.cat([query_bias, key_bias, value_bias], dim=0),
    )


def test_etbert_backbone_reports_missing_checkpoint_file(tmp_path: Path):
    model = ETBertBackbone(vocab_size=64, hidden_dim=32, max_tokens=16, num_layers=2)
    missing_path = tmp_path / "not_found.ckpt"

    result = model.load_pretrained_checkpoint(missing_path)
    assert result["checkpoint_found"] is False
    assert result["checkpoint_path"] == str(missing_path)
    assert "checkpoint_missing" in result["status"]
    assert model.last_checkpoint_report == result
    assert model.checkpoint_report == result


def test_etbert_backbone_reports_skipped_keys_for_shape_mismatch(tmp_path: Path):
    model = ETBertBackbone(vocab_size=64, hidden_dim=32, max_tokens=16, num_layers=2)
    checkpoint = {
        "state_dict": {
            "token_embed.weight": torch.randn(63, 32),
            "encoder.layers.0.linear1.weight": torch.randn_like(model.encoder.layers[0].linear1.weight),
        }
    }
    checkpoint_path = tmp_path / "shape_mismatch.ckpt"
    torch.save(checkpoint, checkpoint_path)

    result = model.load_pretrained_checkpoint(checkpoint_path)
    assert result["checkpoint_found"] is True
    assert result["loaded_key_count"] >= 1
    assert result["skipped_key_count"] >= 1
    assert any(item["mapped_key"] == "token_embed.weight" for item in result["skipped_keys"])
    assert any("shape_mismatch" in item["reason"] for item in result["skipped_keys"])


def test_etbert_backbone_loads_transformer_style_attention_keys(tmp_path: Path):
    model = ETBertBackbone(vocab_size=256, hidden_dim=32, max_tokens=16, num_layers=2)
    q_weight = torch.randn(32, 32)
    k_weight = torch.randn(32, 32)
    v_weight = torch.randn(32, 32)
    q_bias = torch.randn(32)
    k_bias = torch.randn(32)
    v_bias = torch.randn(32)

    checkpoint = {
        "state_dict": {
            "model.embedding.word_embedding.weight": torch.randn_like(model.token_embed.weight),
            "model.embedding.position_embedding.weight": torch.randn_like(model.pos_embed.weight),
            "model.embedding.segment_embedding.weight": torch.randn_like(model.type_embed.weight),
            "model.encoder.transformer.0.self_attn.linear_layers.0.weight": q_weight,
            "model.encoder.transformer.0.self_attn.linear_layers.1.weight": k_weight,
            "model.encoder.transformer.0.self_attn.linear_layers.2.weight": v_weight,
            "model.encoder.transformer.0.self_attn.linear_layers.0.bias": q_bias,
            "model.encoder.transformer.0.self_attn.linear_layers.1.bias": k_bias,
            "model.encoder.transformer.0.self_attn.linear_layers.2.bias": v_bias,
            "model.encoder.transformer.0.self_attn.final_linear.weight": torch.randn_like(
                model.encoder.layers[0].self_attn.out_proj.weight
            ),
            "model.encoder.transformer.0.self_attn.final_linear.bias": torch.randn_like(
                model.encoder.layers[0].self_attn.out_proj.bias
            ),
        }
    }
    checkpoint_path = tmp_path / "transformer_style.ckpt"
    torch.save(checkpoint, checkpoint_path)

    result = model.load_pretrained_checkpoint(checkpoint_path)
    assert result["checkpoint_found"] is True
    assert torch.equal(
        model.encoder.layers[0].self_attn.in_proj_weight,
        torch.cat([q_weight, k_weight, v_weight], dim=0),
    )
    assert torch.equal(
        model.encoder.layers[0].self_attn.in_proj_bias,
        torch.cat([q_bias, k_bias, v_bias], dim=0),
    )


def test_etbert_backbone_reports_incomplete_qkv_groups(tmp_path: Path):
    model = ETBertBackbone(vocab_size=128, hidden_dim=32, max_tokens=16, num_layers=2)
    checkpoint = {
        "state_dict": {
            "model.bert.encoder.layer.0.attention.self.query.weight": torch.randn(32, 32),
            "model.bert.encoder.layer.0.attention.self.key.weight": torch.randn(32, 32),
        }
    }
    checkpoint_path = tmp_path / "partial_qk.ckpt"
    torch.save(checkpoint, checkpoint_path)

    result = model.load_pretrained_checkpoint(checkpoint_path)
    assert result["checkpoint_found"] is True
    assert result["loaded_key_count"] == 0
    assert "loaded" not in result["status"]
    assert "no_compatible_keys" in result["status"]
    assert "partial" in result["status"]
    assert any(
        item["mapped_key"] == "encoder.layers.0.self_attn.in_proj_weight"
        and "query.weight" in " ".join(item["source_keys"])
        and "key.weight" in " ".join(item["source_keys"])
        and "incomplete_qkv_group" in item["reason"]
        for item in result["skipped_keys"]
    )


def test_mobilevit_backbone_projects_features():
    model = MobileViTBackbone(out_dim=96)
    x = torch.rand(1, 3, 28, 28)
    y = model(x)
    assert y.shape == (1, 96)


def test_etbert_backbone_forward_does_not_emit_nested_tensor_warning():
    model = ETBertBackbone(vocab_size=256, hidden_dim=32, max_tokens=16, num_layers=2)
    input_ids = torch.randint(0, 256, (2, 16))
    attention_mask = torch.zeros(2, 16, dtype=torch.long)
    attention_mask[:, :8] = 1
    token_type_ids = torch.zeros(2, 16, dtype=torch.long)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = model(input_ids, attention_mask, token_type_ids)

    assert output.shape == (2, 32)
    assert not any("nested tensors is in prototype stage" in str(item.message) for item in caught)


def test_etbert_backbone_disables_transformer_nested_tensor_path():
    model = ETBertBackbone(vocab_size=256, hidden_dim=32, max_tokens=16, num_layers=2)
    assert model.encoder.enable_nested_tensor is False
