# Stage1 Accuracy Training Design

## Goal

在不修改当前 `stage1_binary` 数据协议的前提下，优先提升当前协议下的 test `accuracy`，允许 `macro_f1` 有一定波动。

## Context

- 当前最新 run `stage1-binary-195511` 的 train / val / test 都稳定在 `95.6% - 96.4%`。
- 训练端目前存在三个与“提 accuracy”直接相关的约束：
  - validation 由 train 内随机切分，不是分层切分；
  - best checkpoint 固定按 `val_macro_f1` 选择；
  - 二分类评估固定使用 `argmax`，没有做基于 validation 的阈值校准。

## Recommended Approach

本轮只做三项直接作用于当前目标的训练策略改造：

1. 将 train 内派生 validation 的逻辑改为按 label 分层切分。
2. 将 best checkpoint 的选择指标改为可配置，支持 `val_acc` 与 `val_macro_f1`。
3. 对二分类 run 增加基于 validation 的 threshold calibration，并在 evaluate 时复用该阈值。

## Alternatives Considered

### Option A: 仅改 class-weighted loss

- 优点：实现简单，通常能改善 minority 类。
- 缺点：更偏向改善宏平均，不一定直接提升 accuracy；对当前用户目标不够对齐。

### Option B: WeightedRandomSampler + class-weighted loss

- 优点：能更强地处理不平衡。
- 缺点：训练分布会被显著改写，变量太多，不适合先做“最快可验证提分”版本。

### Option C: 当前推荐方案

- 优点：直接优化 checkpoint 选择与二分类决策边界，最贴近 `accuracy` 目标。
- 缺点：`macro_f1` 可能略有回落，需要在报告中同时保留现有指标。

## Design

### 1. Stratified Validation Split

- 保留“当 manifest 未提供 `val` 时，从 train 派生 val”的现有行为。
- 但派生逻辑改为“对每个类别分别 shuffle，再按 `val_fraction` 抽样”。
- 保证：
  - 每个有样本的类别尽量在 train 和 val 中都保留；
  - 小样本类别不会被全部切进 val 或全部留在 train。

### 2. Configurable Best Metric

- `src.train` 新增参数，例如 `--best-metric {val_macro_f1,val_acc}`，默认保持现有兼容值。
- 训练日志和 `config.yaml` 记录该参数。
- 保存 `best.ckpt` 时，不再硬编码 `val_macro_f1`，而是根据配置读取当前 epoch 的对应指标。

### 3. Binary Threshold Calibration

- 仅在 `num_classes == 2` 时启用。
- 每个 epoch 在 validation 上取正类概率，搜索一组阈值候选，找到使 `val_acc` 最大的阈值。
- 当该 epoch 成为 best epoch 时，将对应阈值写入 `best.ckpt` 和 `config.yaml`。
- `src.evaluate` 在二分类且存在 `decision_threshold` 时，使用：
  - `p(class1) >= threshold -> 1`
  - 否则 -> 0
- 多分类保持现有 `argmax` 路径不变。

## Data Flow

1. `train_main` 载入数据后构造分层 train/val mask。
2. 每个 epoch 完成后，validation 侧同时计算：
   - `val_acc`
   - `val_macro_f1`
   - `val_best_threshold`（仅二分类）
3. best checkpoint 根据 `--best-metric` 决定是否更新，并附带保存阈值。
4. `evaluate_main` 读取 `best.ckpt/config.yaml` 中的阈值，在 test split 上复用。

## Error Handling

- 当某个类别样本太少，无法严格按比例切分时，优先保证 train 至少保留 1 个样本。
- 当 validation 只有单一类别或没有足够样本时，二分类 threshold calibration 自动回退到 `0.5`。
- evaluate 若未读到阈值，继续使用现有 `argmax` / `0.5` 默认行为。

## Testing Strategy

- 对训练端新增测试，覆盖：
  - 分层切分后的类分布约束；
  - `best.ckpt` 可按 `val_acc` 选取；
  - 二分类校准阈值被写入并在 evaluate 中生效。
- 保留现有 smoke test，确保不影响多分类和现有报告产物。

## Non-Goals

- 本轮不修改 `stage1_binary` 的数据协议和 capture overlap。
- 本轮不引入 sampler 或 class-weighted loss。
- 本轮不改 `stacking/moe` 的训练逻辑。
