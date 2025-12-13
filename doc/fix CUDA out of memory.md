# 优化 CUDA 内存不足

问题在于 `temporal_contrastive_loss` 的内存占用：当序列长度为 3000、批次大小为 32 时，需要计算 `(32, 6000, 6000)` 的相似度矩阵，约 4.6GB，导致显存不足。

优化方案：
1. 在损失函数中添加内存优化（分块计算或限制序列长度）
2. 在训练循环中添加显存清理
3. 减小默认的 `max_train_length` 或 `batch_size`

先优化损失函数，添加内存管理。优化损失函数，对长序列进行采样以降低内存占用。更新 `ts2vec.py`，添加内存管理并传递 `max_temporal_length` 参数。更新 `ts2vec.py`，添加内存优化参数并在训练循环中传递。在 `train.py` 中添加命令行参数。在 `train.py` 中添加命令行参数，并更新模型创建代码。更新配置，添加 `max_temporal_length` 参数。更新 `ts2vec.py` 的文档字符串，说明新参数。

## 优化内容

1. **优化 `temporal_contrastive_loss` 函数** (`models/losses.py`)：
   - 添加 `max_temporal_length` 参数（默认 1000）
   - 当序列长度超过该值时，随机采样以减少内存占用
   - 避免计算过大的相似度矩阵（如 `(32, 6000, 6000)`）

2. **更新 `hierarchical_contrastive_loss` 函数**：
   - 添加 `max_temporal_length` 参数并传递给 `temporal_contrastive_loss`

3. **更新 `TS2Vec` 类** (`ts2vec.py`)：
   - 添加 `max_temporal_length` 参数
   - 在训练循环中每 10 个 batch 清理一次显存缓存
   - 更新文档字符串

4. **更新训练脚本** (`train.py`)：
   - 添加 `--max-temporal-length` 命令行参数
   - 自动计算默认值：`min(1000, max_train_length // 2)`
   - 在配置输出中显示该参数

## 内存优化效果

- 之前：序列长度 3000 时，相似度矩阵 `(32, 6000, 6000)` 约需 4.6GB
- 现在：限制为 1000 时，相似度矩阵 `(32, 2000, 2000)` 约需 0.5GB，内存占用减少约 90%

现在可以重新运行训练命令。如果仍有内存问题，可以：
- 减小 `--batch-size`（例如从 32 降到 16 或 8）
- 减小 `--max-train-length`（例如从 3000 降到 2000）
- 手动设置 `--max-temporal-length`（例如设为 500）

这些优化应能解决 CUDA 内存不足的问题。
