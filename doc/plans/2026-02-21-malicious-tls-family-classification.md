# Malicious TLS Family Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于 `SourceData` 构建“恶意 TLS 家族多分类”端到端系统，交付主模型（RGB+TLS-BERT 融合）、增强集成（stacking，含可选 MoE）以及可复现实验报告链路。

**Architecture:** 采用“数据工程 -> 双模态表示 -> 融合训练 -> 集成增强 -> 评估报告”的分层流水线。输入保持 TLS 侧信道约束，不解密 payload；数据侧同时产出 Full-TLS 与 Leakage-reduced 两套配置，模型侧采用双向 cross-attention + gating + 辅助监督，最终由 stacking 汇总多视角预测。

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn, xgboost/lightgbm, numpy/pandas, matplotlib/seaborn, tqdm/rich, pyyaml, pytest

---

## 0. Scope 与交付定义

### 必做交付
1. 数据处理链路：TLS 过滤、会话切分、capture 分组切分、双轨（Full/Leakage-reduced）样本构建。
2. 表示构建链路：TLS-RGB（28x28x3）与 TLS-Field-BERT token 序列同时生成。
3. 模型链路：图像分支 + TLS-BERT 分支 + 双向 cross-attn + gating + 辅助头。
4. 集成链路：增强 stacking（OOF meta-features + GBDT meta-learner）。
5. 评估与报告：macroF1/macroRecall/Acc、混淆矩阵、学习曲线、per-class 图、自动 report。
6. 复现能力：配置化运行、run_id 目录、日志和图表完整落盘。

### 可选增强
1. MoE 路由层（决策级专家混合）。
2. 蒸馏到轻量 student。

### 非目标
1. 不实现 benign/malicious 二分类。
2. 不依赖明文解密或 DPI payload 特征。

---

## 1. 建议目录骨架（先搭建）

```text
configs/
  dataset_tls_full.yaml
  dataset_tls_leakage_reduced.yaml
  train_fusion.yaml
  train_stacking.yaml
  ablation.yaml
src/
  pipeline/
    pcap_reader.py
    tls_filter.py
    sessionize.py
    split_strategy.py
    leakage_control.py
    rgb_encoder.py
    token_schema.py
    token_encoder.py
    build_dataset.py
  fusion/
    datasets.py
    models/
      image_branch.py
      tls_bert_branch.py
      fusion_cross_attn.py
      heads.py
    train_stagewise.py
    stacking.py
    evaluate.py
    report.py
    run_ablation.py
  common/
    config.py
    io_utils.py
    logging_utils.py
tests/
  config/
  pipeline/
  fusion/
  integration/
doc/
  plans/
    2026-02-21-malicious-tls-family-classification.md
```

---

## 2. Milestones（完成整个任务的分段目标）

| Milestone | 必达产物 | 验收标准 |
|----------|----------|----------|
| M1 数据链路可运行 | TLS 过滤 + 会话切分 + split + 双轨 manifest | 可生成 `dataset/<name>/manifest_*.csv`，且无 capture 泄漏 |
| M2 表示链路可运行 | `image_data` 与 `pcap_data` 成对样本 | 随机抽样检查配对率 100% |
| M3 主模型可训练 | 三阶段训练脚本与 best checkpoint | 输出 `outputs/runs/<run_id>/checkpoints/best.pt` |
| M4 集成可复现 | stacking 模型与 meta-features | macroF1 不低于主模型，报告可追踪 |
| M5 报告可交付 | 指标、图表、混淆与消融结论 | 生成 `report_*.md` + 全套 figures |

---

## 3. 详细任务拆解（按执行顺序）

### Task 1: 初始化配置系统与目录协议

**Files:**
- Create: `src/common/config.py`
- Create: `configs/dataset_tls_full.yaml`
- Create: `configs/dataset_tls_leakage_reduced.yaml`
- Create: `configs/train_fusion.yaml`
- Create: `configs/train_stacking.yaml`
- Test: `tests/config/test_config_loading.py`

**Step 1: Write the failing test**
```python
def test_dataset_profile_has_required_keys():
    from src.common.config import load_yaml
    cfg = load_yaml("configs/dataset_tls_full.yaml")
    required = {"dataset_name", "source_root", "split", "tls_filter", "feature"}
    assert required.issubset(set(cfg.keys()))
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/config/test_config_loading.py -v`
Expected: FAIL（缺少模块或配置文件）

**Step 3: Write minimal implementation**
```python
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/config/test_config_loading.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/common/config.py configs/*.yaml tests/config/test_config_loading.py
git commit -m "feat: 初始化配置加载与数据训练profile"
```

---

### Task 2: TLS 过滤器（record header 规则）与 PCAP 读取

**Files:**
- Create: `src/pipeline/pcap_reader.py`
- Create: `src/pipeline/tls_filter.py`
- Test: `tests/pipeline/test_tls_filter.py`

**Step 1: Write the failing test**
```python
def test_is_tls_record_accepts_valid_header():
    from src.pipeline.tls_filter import is_tls_record
    header = bytes([22, 0x03, 0x03, 0x00, 0x2f])
    assert is_tls_record(header) is True
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_tls_filter.py -v`
Expected: FAIL（函数不存在）

**Step 3: Write minimal implementation**
```python
def is_tls_record(header: bytes) -> bool:
    if len(header) < 5:
        return False
    content_type = header[0]
    version = (header[1] << 8) | header[2]
    length = (header[3] << 8) | header[4]
    return content_type in {20, 21, 22, 23} and 0x0300 <= version <= 0x0304 and 0 < length <= 18432
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_tls_filter.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/pcap_reader.py src/pipeline/tls_filter.py tests/pipeline/test_tls_filter.py
git commit -m "feat: 实现TLS record header过滤基础能力"
```

---

### Task 3: 会话切分与分组切分（防 capture 泄漏）

**Files:**
- Create: `src/pipeline/sessionize.py`
- Create: `src/pipeline/split_strategy.py`
- Modify: `configs/dataset_tls_full.yaml`
- Test: `tests/pipeline/test_group_split.py`

**Step 1: Write the failing test**
```python
def test_group_split_no_capture_overlap():
    from src.pipeline.split_strategy import group_split_by_capture
    rows = [
        {"capture_id": "a", "label": "fam1"},
        {"capture_id": "a", "label": "fam1"},
        {"capture_id": "b", "label": "fam2"},
    ]
    train, val, test = group_split_by_capture(rows, seed=42)
    assert set(x["capture_id"] for x in train).isdisjoint(set(x["capture_id"] for x in test))
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_group_split.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def group_split_by_capture(rows, seed=42):
    # 按 capture_id 聚类后再分配集合，避免同源泄漏
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_group_split.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/sessionize.py src/pipeline/split_strategy.py configs/dataset_tls_full.yaml tests/pipeline/test_group_split.py
git commit -m "feat: 实现会话切分与capture级分组切分"
```

---

### Task 4: 泄漏控制双轨（Full-TLS vs Leakage-reduced）

**Files:**
- Create: `src/pipeline/leakage_control.py`
- Modify: `configs/dataset_tls_leakage_reduced.yaml`
- Test: `tests/pipeline/test_leakage_control.py`

**Step 1: Write the failing test**
```python
def test_sni_is_masked_in_leakage_reduced():
    from src.pipeline.leakage_control import redact_sensitive_fields
    row = {"sni": "malicious.example.com", "cert_fingerprint": "abcd"}
    out = redact_sensitive_fields(row)
    assert out["sni"] != "malicious.example.com"
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_leakage_control.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def redact_sensitive_fields(row: dict) -> dict:
    row = dict(row)
    row["sni"] = hash(row.get("sni", "")) % 2**16
    row.pop("cert_fingerprint", None)
    return row
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_leakage_control.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/leakage_control.py configs/dataset_tls_leakage_reduced.yaml tests/pipeline/test_leakage_control.py
git commit -m "feat: 增加leakage-reduced数据轨道"
```

---

### Task 5: TLS-RGB 编码器（R/G/B 语义固定）

**Files:**
- Create: `src/pipeline/rgb_encoder.py`
- Test: `tests/pipeline/test_rgb_encoder.py`

**Step 1: Write the failing test**
```python
def test_rgb_encoder_output_shape():
    import numpy as np
    from src.pipeline.rgb_encoder import encode_tls_rgb
    sample = {"records": [], "handshake": {}, "stats": {}}
    img = encode_tls_rgb(sample, image_size=28)
    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28, 3)
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_rgb_encoder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def encode_tls_rgb(sample: dict, image_size: int = 28):
    # R: record形态, G: handshake语义, B:行为统计
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_rgb_encoder.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/rgb_encoder.py tests/pipeline/test_rgb_encoder.py
git commit -m "feat: 实现TLS-RGB语义编码器"
```

---

### Task 6: TLS-Field-BERT token 生成器

**Files:**
- Create: `src/pipeline/token_schema.py`
- Create: `src/pipeline/token_encoder.py`
- Test: `tests/pipeline/test_token_encoder.py`

**Step 1: Write the failing test**
```python
def test_token_encoder_has_cls_sep():
    from src.pipeline.token_encoder import encode_tls_tokens
    ids = encode_tls_tokens({"records": [], "handshake": {}}, max_len=32)
    assert ids[0] == 101
    assert ids[-1] == 102
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_token_encoder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def encode_tls_tokens(sample: dict, max_len: int = 256):
    # [CLS] + fields + [SEP] + padding
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_token_encoder.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/token_schema.py src/pipeline/token_encoder.py tests/pipeline/test_token_encoder.py
git commit -m "feat: 实现TLS字段序列token化"
```

---

### Task 7: 数据集构建 CLI（成对样本落盘）

**Files:**
- Create: `src/pipeline/build_dataset.py`
- Create: `src/common/io_utils.py`
- Test: `tests/integration/test_build_dataset_smoke.py`

**Step 1: Write the failing test**
```python
def test_build_dataset_creates_paired_dirs(tmp_path):
    # 调用build命令后，应生成image_data与pcap_data
    ...
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/integration/test_build_dataset_smoke.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def main():
    # 读取profile，构建manifest，输出配对样本
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/integration/test_build_dataset_smoke.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/build_dataset.py src/common/io_utils.py tests/integration/test_build_dataset_smoke.py
git commit -m "feat: 完成数据构建CLI与样本落盘"
```

---

### Task 8: 融合主模型（Image + TLS-BERT + Cross-Attn + Gating）

**Files:**
- Create: `src/fusion/models/image_branch.py`
- Create: `src/fusion/models/tls_bert_branch.py`
- Create: `src/fusion/models/fusion_cross_attn.py`
- Create: `src/fusion/models/heads.py`
- Test: `tests/fusion/test_fusion_forward.py`

**Step 1: Write the failing test**
```python
def test_fusion_forward_shapes():
    from src.fusion.models.fusion_cross_attn import FusionModel
    model = FusionModel(num_classes=10, hidden_dim=256)
    out = model(image_tensor, token_ids, attn_mask)
    assert out["logits_fuse"].shape[-1] == 10
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/fusion/test_fusion_forward.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
class FusionModel(nn.Module):
    # 双向cross-attn + g * p_img + (1-g) * p_tls
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/fusion/test_fusion_forward.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/fusion/models/*.py tests/fusion/test_fusion_forward.py
git commit -m "feat: 实现跨模态融合主模型"
```

---

### Task 9: 三阶段训练器与结构化日志体系

**Files:**
- Create: `src/common/logging_utils.py`
- Create: `src/fusion/train_stagewise.py`
- Create: `src/fusion/datasets.py`
- Modify: `configs/train_fusion.yaml`
- Test: `tests/fusion/test_training_smoke.py`

**Step 1: Write the failing test**
```python
def test_training_creates_run_artifacts(tmp_path):
    # 训练后必须落盘: config, train.log, checkpoints, metrics.csv
    ...
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/fusion/test_training_smoke.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def run_train(cfg):
    # stage1: 分支热身, stage2: 融合训练, stage3: 集成前置输出
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/fusion/test_training_smoke.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/common/logging_utils.py src/fusion/train_stagewise.py src/fusion/datasets.py configs/train_fusion.yaml tests/fusion/test_training_smoke.py
git commit -m "feat: 三阶段训练与实验日志落盘"
```

---

### Task 10: 增强 Stacking（OOF meta-features + GBDT）

**Files:**
- Create: `src/fusion/stacking.py`
- Modify: `configs/train_stacking.yaml`
- Test: `tests/fusion/test_stacking_pipeline.py`

**Step 1: Write the failing test**
```python
def test_stacking_uses_oof_features():
    from src.fusion.stacking import build_meta_features
    meta = build_meta_features(pred_img, pred_tls, pred_fuse, folds=5)
    assert "entropy_fuse" in meta.columns
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/fusion/test_stacking_pipeline.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def build_meta_features(...):
    # logits + entropy + margin + gating + norm
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/fusion/test_stacking_pipeline.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/fusion/stacking.py configs/train_stacking.yaml tests/fusion/test_stacking_pipeline.py
git commit -m "feat: 增强stacking集成链路"
```

---

### Task 11: 评估、可视化与自动报告

**Files:**
- Create: `src/fusion/evaluate.py`
- Create: `src/fusion/report.py`
- Test: `tests/integration/test_report_generation.py`

**Step 1: Write the failing test**
```python
def test_report_contains_required_sections(tmp_path):
    # report中必须包含实验信息、指标、混淆、错分分析
    ...
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/integration/test_report_generation.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def generate_report(metrics, figures, output_path):
    # 生成report.md并引用图表
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/integration/test_report_generation.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/fusion/evaluate.py src/fusion/report.py tests/integration/test_report_generation.py
git commit -m "feat: 补齐评估图表与自动报告"
```

---

### Task 12: 消融实验编排与全链路验收

**Files:**
- Create: `src/fusion/run_ablation.py`
- Create: `configs/ablation.yaml`
- Create: `tests/integration/test_end_to_end_smoke.py`

**Step 1: Write the failing test**
```python
def test_ablation_runner_outputs_table(tmp_path):
    # 执行后生成ablation_summary.csv
    ...
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/integration/test_end_to_end_smoke.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
```python
def run_ablations(cfg):
    # 枚举分支/融合/RGB/集成组合并汇总指标
    ...
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/integration/test_end_to_end_smoke.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/fusion/run_ablation.py configs/ablation.yaml tests/integration/test_end_to_end_smoke.py
git commit -m "feat: 消融编排与全链路验收脚本"
```

---

### Task 13 (Optional): MoE 与蒸馏

**Files:**
- Create: `src/fusion/moe_router.py`
- Create: `src/fusion/distill.py`
- Test: `tests/fusion/test_moe_distill_smoke.py`

**Exit Criteria:**
1. MoE 相对 stacking 在 macroF1 提升 >= 0.5pt 或在同精度下显著减少推理耗时。
2. Student 模型相对 teacher 精度下降 <= 1.5pt，推理耗时下降 >= 30%。

---

## 4. 运行命令（执行阶段直接复用）

### 数据构建
```bash
python -m src.pipeline.build_dataset --config configs/dataset_tls_full.yaml
python -m src.pipeline.build_dataset --config configs/dataset_tls_leakage_reduced.yaml
```

### 主模型训练
```bash
python -m src.fusion.train_stagewise --config configs/train_fusion.yaml
```

### 集成训练
```bash
python -m src.fusion.stacking --config configs/train_stacking.yaml
```

### 评估与报告
```bash
python -m src.fusion.evaluate --run-dir outputs/runs/<run_id>
python -m src.fusion.report --run-dir outputs/runs/<run_id>
```

### 消融
```bash
python -m src.fusion.run_ablation --config configs/ablation.yaml
```

---

## 5. 验收清单（Definition of Done）

1. 数据层：`dataset/<name>/image_data` 与 `dataset/<name>/pcap_data` 成对存在，split 无 capture 泄漏。
2. 训练层：`outputs/runs/<run_id>/` 下存在 `config.yaml`, `train.log`, `metrics.csv`, `checkpoints/`, `figures/`, `report.md`。
3. 指标层：报告包含 Acc、Macro-P/R/F1、per-class metrics、混淆矩阵、学习曲线。
4. 消融层：至少完成文档中 4 组核心消融（时序分支、融合机制、RGB 通道、集成复杂度）。
5. 复现层：按 README/命令清单可在干净环境复现一次 smoke 训练并产出报告。

---

## 6. 风险与缓解

| 风险 | 触发信号 | 缓解动作 |
|------|----------|----------|
| 类别严重不平衡导致 macroF1 偏低 | 少数类 recall 长期为 0 | 使用 class-balanced focal + sampler + per-class early warning |
| 特征泄漏导致虚高指标 | Full-TLS 与 leakage-reduced 差距异常大 | 固定双轨报告并强制提交 leakage-reduced 结果 |
| 融合塌缩到单分支 | gating 长期接近 0 或 1 | 提升辅助头 loss 权重并加入分支 dropout |
| 训练不稳定（NaN/爆炸） | loss 或梯度异常 | 梯度裁剪、AMP loss-scale 检查、异常即停并写入日志 |

---

## 7. 建议排期（可按人力调整）

1. 第 1 周：Task 1-4（数据可用 + split 合规 + 双轨配置）。
2. 第 2 周：Task 5-8（表示与融合主模型打通）。
3. 第 3 周：Task 9-11（训练稳定 + 集成 + 报告自动化）。
4. 第 4 周：Task 12 + 可选 Task 13（消融收敛 + 部署优化）。

---

## 8. 执行策略建议

1. 先严格完成 Task 1-12 的主线，不要在 M3 前引入 MoE。
2. 每个 Task 完成后立即跑对应测试并提交，避免大批量未验证改动。
3. 每天结束前更新 `task_plan.md`、`findings.md`、`progress.md`，确保上下文可恢复。
