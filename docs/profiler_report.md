# Profiler 性能分析报告

## 状态

待实现（Phase 5）。

## 计划

使用 `torch.profiler` 对训练过程进行性能剖析，分析以下维度：

| 维度 | 分析内容 |
|---|---|
| forward 耗时 | 各层计算时间、attention vs FFN 占比 |
| backward 耗时 | 反向计算时间、与 forward 的比例 |
| optimizer step | Adam 更新耗时 |
| communication | all-reduce / all-gather / reduce-scatter 占比 |
| dataloader | 数据加载是否成为瓶颈 |
| CUDA kernel | 哪些 kernel 占用最多 GPU 时间 |
| memory | 显存随 step 的变化曲线 |

## 导出产物

- Chrome trace JSON：在 `chrome://tracing` 中加载查看时间线
- TensorBoard 可视化：查看 op 表格和 kernel 统计
- CSV 统计表：关键 op 的 CPU/CUDA 耗时

## 使用方式

```bash
# 待实现
python train/train_single.py --config configs/single_gpu.yaml --profile
```
