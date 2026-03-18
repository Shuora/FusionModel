# MobileViT + 改动版 ET-BERT MVTBA 重构 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前项目重构为更接近 MVTBA 协议的 `MobileViT + 改动版 ET-BERT` 多模态流量分类流水线，重写旧文本预处理并跑通阶段1二分类与阶段2多分类。

**Architecture:** 保留 `session_full` 的 session 粒度和两阶段实验协议，重写特征生成与数据装载：图像侧输出 `MobileViT` 可用的 session RGB，文本侧输出 ET-BERT 对齐的 `input_ids / attention_mask / token_type_ids`。主模型替换为 `MobileViT backbone + truncated ET-BERT + gate fusion heads`，继续向训练、评估、stacking、moe 暴露 `logits_fuse / logits_img / logits_tls / gate`。

**Tech Stack:** Python 3.9、PyTorch、transformers、dpkt、numpy、pandas、Pillow、pytest。

---

### Task 1: 锁定新预处理产物格式与最小回归测试

**Files:**
- Modify: `tests/data/test_feature_encoder.py`
- Create: `tests/data/test_etbert_feature_encoder.py`
- Modify: `tests/pipeline/test_pipeline_data_protocol.py`
- Modify: `src/data/feature_encoder.py`
- Modify: `src/pipeline_data.py`

- [ ] **Step 1: 写失败测试，定义新文本产物结构**

```python
def test_save_feature_shards_writes_etbert_inputs(tmp_path):
    save_feature_shards(...)
    npz = np.load(seq_path, allow_pickle=False)
    assert {"session_id", "input_ids", "attention_mask", "token_type_ids"} <= set(npz.files)
    assert npz["input_ids"].shape == (1, 128)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/data/test_etbert_feature_encoder.py::test_save_feature_shards_writes_etbert_inputs -v`

Expected: FAIL，提示 `input_ids` 或 `token_type_ids` 不存在。

- [ ] **Step 3: 最小实现新字段骨架**

```python
np.savez_compressed(
    etbert_path,
    session_id=sid_arr,
    label=label_arr,
    input_ids=input_arr,
    attention_mask=attn_arr,
    token_type_ids=type_arr,
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/data/test_etbert_feature_encoder.py tests/data/test_feature_encoder.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/data/test_etbert_feature_encoder.py tests/data/test_feature_encoder.py tests/pipeline/test_pipeline_data_protocol.py src/data/feature_encoder.py src/pipeline_data.py
git commit -m "test+feat(data): define etbert shard output contract"
```

### Task 2: 实现 ET-BERT 对齐 tokenization 与 shard 写出

**Files:**
- Create: `src/data/etbert_tokenizer.py`
- Modify: `src/data/feature_encoder.py`
- Test: `tests/data/test_etbert_feature_encoder.py`

- [ ] **Step 1: 写失败测试，定义 tokenizer 行为**

```python
def test_encode_etbert_tokens_returns_padded_triplet():
    input_ids, attention_mask, token_type_ids = encode_etbert_tokens(session, vocab, max_len=128)
    assert input_ids.shape == (128,)
    assert attention_mask.shape == (128,)
    assert token_type_ids.shape == (128,)
    assert int(attention_mask.sum()) > 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/data/test_etbert_feature_encoder.py::test_encode_etbert_tokens_returns_padded_triplet -v`

Expected: FAIL，提示函数不存在或返回结构不匹配。

- [ ] **Step 3: 最小实现**

```python
def encode_etbert_tokens(session, vocab, max_len=128):
    tokens = build_etbert_tokens_from_payload_chunks(session["payload_chunks"])
    input_ids = tokens_to_ids(tokens, vocab, max_len=max_len)
    attention_mask = (input_ids != 0).astype(np.uint8)
    token_type_ids = np.zeros_like(input_ids, dtype=np.uint8)
    return input_ids, attention_mask, token_type_ids
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/data/test_etbert_feature_encoder.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/data/test_etbert_feature_encoder.py src/data/etbert_tokenizer.py src/data/feature_encoder.py
git commit -m "feat(data): add etbert-aligned tokenizer and shard encoding"
```

### Task 3: 重构 loader，改为读取 `rgb + etbert + manifest`

**Files:**
- Modify: `src/pipeline_data.py`
- Modify: `tests/pipeline/test_pipeline_data_protocol.py`
- Modify: `tests/pipeline/test_train_eval_report.py`

- [ ] **Step 1: 写失败测试，定义 loader 输出**

```python
def test_load_policy_multimodal_data_reads_etbert_triplet(tmp_path):
    data = load_policy_multimodal_data(root, "session_full")
    assert data["rgb"].shape[0] == 2
    assert data["input_ids"].shape == (2, 128)
    assert data["attention_mask"].shape == (2, 128)
    assert data["token_type_ids"].shape == (2, 128)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/pipeline/test_pipeline_data_protocol.py::test_load_policy_multimodal_data_reads_etbert_triplet -v`

Expected: FAIL，loader 仍返回旧 `token_ids / segment_ids`。

- [ ] **Step 3: 最小实现**

```python
return {
    "rgb": ...,
    "input_ids": ...,
    "attention_mask": ...,
    "token_type_ids": ...,
    "y": ...,
    "session_id": ...,
    "split": ...,
    "dataset": ...,
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py src/pipeline_data.py
git commit -m "feat(data): load etbert multimodal session shards"
```

### Task 4: 引入 MobileViT backbone 与 ET-BERT adapter 的模型骨架

**Files:**
- Create: `src/models/mobilevit_backbone.py`
- Create: `src/models/etbert_backbone.py`
- Modify: `src/models/fusion_model.py`
- Modify: `tests/models/test_fusion_model.py`

- [ ] **Step 1: 写失败测试，定义新模型 forward 契约**

```python
def test_fusion_model_forward_with_etbert_inputs():
    model = MobileViTETBertFusionClassifier(num_classes=3, max_tokens=128)
    rgb = torch.rand(2, 3, 28, 28)
    input_ids = torch.randint(0, 100, (2, 128))
    attention_mask = torch.ones(2, 128, dtype=torch.long)
    token_type_ids = torch.zeros(2, 128, dtype=torch.long)
    out = model(rgb, input_ids, attention_mask, token_type_ids)
    assert out["logits_fuse"].shape == (2, 3)
    assert out["gate"].shape == (2, 1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/models/test_fusion_model.py::test_fusion_model_forward_with_etbert_inputs -v`

Expected: FAIL，旧模型签名不匹配。

- [ ] **Step 3: 最小实现模型骨架**

```python
class MobileViTETBertFusionClassifier(nn.Module):
    def forward(self, rgb, input_ids, attention_mask, token_type_ids):
        img_feature = self.image_backbone(rgb)
        tls_feature = self.text_backbone(input_ids, attention_mask, token_type_ids)
        gate = self.gate(torch.cat([img_feature, tls_feature], dim=-1))
        fused = gate * img_feature + (1.0 - gate) * tls_feature
        return {
            "logits_fuse": self.head_fuse(fused),
            "logits_img": self.head_img(img_feature),
            "logits_tls": self.head_tls(tls_feature),
            "gate": gate,
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/models/test_fusion_model.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/models/test_fusion_model.py src/models/mobilevit_backbone.py src/models/etbert_backbone.py src/models/fusion_model.py
git commit -m "feat(model): add mobilevit etbert fusion model skeleton"
```

### Task 5: 接入预训练 MobileViT 与截断版 ET-BERT 权重加载

**Files:**
- Modify: `src/models/mobilevit_backbone.py`
- Modify: `src/models/etbert_backbone.py`
- Create: `tests/models/test_pretrained_backbones.py`

- [ ] **Step 1: 写失败测试，定义轻量化加载行为**

```python
def test_etbert_backbone_can_truncate_encoder_layers():
    model = ETBertBackbone(num_layers=4, hidden_size=768)
    assert model.num_layers == 4
```

```python
def test_mobilevit_backbone_projects_features():
    model = MobileViTBackbone(out_dim=768)
    x = torch.rand(1, 3, 28, 28)
    y = model(x)
    assert y.shape == (1, 768)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/models/test_pretrained_backbones.py -v`

Expected: FAIL。

- [ ] **Step 3: 最小实现预训练加载与裁剪**

```python
self.encoder_layers = pretrained_layers[:num_layers]
self.proj = nn.Linear(backbone_dim, out_dim)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/models/test_pretrained_backbones.py tests/models/test_fusion_model.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/models/test_pretrained_backbones.py src/models/mobilevit_backbone.py src/models/etbert_backbone.py
git commit -m "feat(model): load pretrained mobilevit and truncated etbert backbones"
```

### Task 6: 重写 train/evaluate 以适配新 batch 结构与模型配置

**Files:**
- Modify: `src/train.py`
- Modify: `src/evaluate.py`
- Modify: `tests/pipeline/test_train_eval_report.py`
- Modify: `tests/pipeline/test_train_stage_dispatch.py`

- [ ] **Step 1: 写失败测试，定义新训练入口输入**

```python
def test_train_main_writes_config_for_mobilevit_etbert(tmp_path):
    code = train_main([...])
    assert code == 0
    cfg = (run_dir / "config.yaml").read_text()
    assert "model_type: MobileViTETBertFusionClassifier" in cfg
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/pipeline/test_train_eval_report.py::test_train_main_writes_config_for_mobilevit_etbert -v`

Expected: FAIL，当前仍写 `TinyFusionClassifier`。

- [ ] **Step 3: 最小实现**

```python
model = MobileViTETBertFusionClassifier(...)
input_ids = data["input_ids"]
token_type_ids = data["token_type_ids"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py src/train.py src/evaluate.py
git commit -m "feat(train): switch train and evaluate to mobilevit etbert inputs"
```

### Task 7: 重新对齐阶段1二分类协议

**Files:**
- Modify: `src/experiments/stage1_binary.py`
- Modify: `tests/pipeline/test_stage1_binary_protocol.py`
- Modify: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: 写失败测试，定义阶段1基于新产物可执行**

```python
def test_stage1_protocol_executes_with_etbert_processed_root(tmp_path):
    code = stage1_main([...,"--execute"])
    assert code == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_protocol_execution.py -v`

Expected: FAIL。

- [ ] **Step 3: 最小实现**

```python
train_main([...,"--label-mode","binary", ...])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_protocol_execution.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_protocol_execution.py src/experiments/stage1_binary.py
git commit -m "feat(protocol): align stage1 binary protocol with new multimodal pipeline"
```

### Task 8: 重新对齐阶段2多分类协议

**Files:**
- Modify: `src/experiments/stage2_multiclass.py`
- Modify: `tests/pipeline/test_stage2_multiclass_protocol.py`
- Modify: `tests/pipeline/test_protocol_execution.py`

- [ ] **Step 1: 写失败测试，定义阶段2任务执行**

```python
def test_stage2_protocol_executes_multiclass_runs(tmp_path):
    code = stage2_main([...,"--execute"])
    assert code == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_protocol_execution.py -v`

Expected: FAIL。

- [ ] **Step 3: 最小实现**

```python
train_main([...,"--label-mode","multiclass", "--datasets", dataset_name])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_protocol_execution.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_protocol_execution.py src/experiments/stage2_multiclass.py
git commit -m "feat(protocol): align stage2 multiclass protocol with new multimodal pipeline"
```

### Task 9: 恢复 stacking / moe 对新模型输出的兼容

**Files:**
- Modify: `src/stacking.py`
- Modify: `src/moe.py`
- Modify: `tests/pipeline/test_stacking_pipeline.py`
- Modify: `tests/pipeline/test_moe_pipeline.py`

- [ ] **Step 1: 写失败测试，定义兼容约束**

```python
def test_stacking_pipeline_runs_with_mobilevit_etbert_base(tmp_path):
    assert stacking_main([...]) == 0
```

```python
def test_moe_pipeline_runs_with_mobilevit_etbert_base(tmp_path):
    assert moe_main([...]) == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py -v`

Expected: FAIL。

- [ ] **Step 3: 最小实现**

```python
out = model(rgb_b, input_ids_b, attention_mask_b, token_type_ids_b)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py src/stacking.py src/moe.py
git commit -m "feat(model): restore stacking and moe support for mobilevit etbert outputs"
```

### Task 10: 更新命令文档与最终验证

**Files:**
- Modify: `README.md`
- Modify: `docs/commands/session-full-experiments.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] **Step 1: 写失败测试或检查项**

Checklist:
- `README` 描述新模型与新预处理
- `session-full-experiments` 命令不再提旧文本链路
- 阶段1 / 阶段2 命令与当前代码一致

- [ ] **Step 2: 运行完整验证命令**

Run: `pytest -q tests/models/test_fusion_model.py tests/data/test_etbert_feature_encoder.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py -v`

Expected: PASS。

- [ ] **Step 3: 运行最小 smoke test**

Run: `python -m src.train --help`

Run: `python -m src.experiments.stage1_binary --help`

Run: `python -m src.experiments.stage2_multiclass --help`

Expected: 都能正常显示帮助信息。

- [ ] **Step 4: 更新文档**

```markdown
- 模型主干：MobileViT + truncated ET-BERT
- 文本输入：ET-BERT aligned tokens
- 协议：stage1 binary / stage2 multiclass
```

- [ ] **Step 5: 提交**

```bash
git add README.md docs/commands/session-full-experiments.md docs/planning-with-files/findings.md docs/planning-with-files/progress.md
git commit -m "docs: update commands and records for mobilevit etbert mvtba pipeline"
```
