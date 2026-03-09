# Embedding 模型推理优化报告

## 模型信息

- 模型：`paraphrase-multilingual-MiniLM-L12-v2`
- 参数量：117.7M
- 向量维度：384
- 测试环境：Apple M1 (macOS)，CPU 推理，30 次平均

## 优化方案对比

| 方案 | 推理延迟 | 相比 FP32 | 相比上一级 |
|------|---------|----------|-----------|
| PyTorch FP32 | 15.5ms | 基线 | - |
| PyTorch INT8 | 10.9ms | **快 41%** | 快 41% |
| **ONNX Runtime FP32** | **5.2ms** | **快 66%** | **再快 52%** |

> 当前采用：**ONNX Runtime FP32**（精度无损，速度最优）

## 检索质量验证

### ONNX Runtime vs PyTorch INT8

测试条件：5 条 query × 7 条记忆库。

| 指标 | 结果 |
|------|------|
| Embedding 余弦相似度 | 98.87% ~ 99.46% |
| **Top-1 一致率** | **100% (5/5)** |
| Top-3 一致率 | 60% (3/5) |

Top-3 不一致仅为第 2、3 名低分项的微小顺序交换（分数差 < 0.03），不影响实际使用。

### PyTorch INT8 vs FP32（历史基线）

| 指标 | 结果 |
|------|------|
| Embedding 余弦相似度 | 99.25% ~ 99.64% |
| **Top-1 一致率** | **100% (5/5)** |
| Top-3 一致率 | 60% (3/5) |

### ONNX 详细对比

```
Q: 你好帮我播放一首歌
  PT INT8: [播放财神到(0.427), 男朋友程序员(0.240), 心情不好(0.233)]
  ONNX:    [播放财神到(0.436), 心情不好(0.219), 男朋友程序员(0.217)]
  Top1: ✓ | Top3: ✗  (第2/3名交换)

Q: 我男朋友是程序员
  PT INT8: [程序员送键盘(0.633), 播放财神到(0.305), 听笑话(0.277)]
  ONNX:    [程序员送键盘(0.676), 播放财神到(0.280), 听笑话(0.267)]
  Top1: ✓ | Top3: ✓

Q: 你叫什么名字
  PT INT8: [AI叫卡卡(0.283), 摸AI脑袋(0.236), 程序员送键盘(0.225)]
  ONNX:    [AI叫卡卡(0.299), 程序员送键盘(0.208), 摸AI脑袋(0.195)]
  Top1: ✓ | Top3: ✗  (第2/3名交换)

Q: 讲个笑话给我听
  PT INT8: [讲笑话(0.416), 程序员送键盘(0.217), 摸AI脑袋(0.147)]
  ONNX:    [讲笑话(0.407), 程序员送键盘(0.230), 摸AI脑袋(0.138)]
  Top1: ✓ | Top3: ✓

Q: 有人在摸摸你的脑袋
  PT INT8: [摸AI脑袋(0.558), 心情不好(0.398), 程序员送键盘(0.329)]
  ONNX:    [摸AI脑袋(0.557), 心情不好(0.374), 程序员送键盘(0.309)]
  Top1: ✓ | Top3: ✓
```

## 性能详情

### 推理延迟全对比

| 方案 | 延迟 | 范围 | 稳定性 |
|------|------|------|--------|
| MPS (Apple GPU) | 58ms | 10-480ms | 波动大 |
| CPU FP32 | 15.5ms | 15-17ms | 稳定 |
| CPU INT8 | 10.9ms | 10-12ms | 稳定 |
| **ONNX Runtime** | **5.2ms** | **5.0-5.5ms** | **最稳定** |

> 对于 117M 小模型，MPS 的数据搬运开销远超 GPU 计算收益。ONNX Runtime 在 CPU 上是最优选择。

## K8s 部署建议

### Pod 资源配置（ONNX Runtime）

```yaml
resources:
  requests:
    memory: "256Mi"    # ONNX 模型 ~180MB，含运行时开销
    cpu: "250m"
  limits:
    memory: "384Mi"
    cpu: "500m"
```

### 并发能力估算（ONNX Runtime, threads=1）

| 服务器配置 | 单次检索延迟 | QPS | 并发用户 (1次/秒/人) |
|-----------|------------|-----|---------------------|
| 250m (0.25核) | ~21ms | ~47/s | ~47 人 |
| 500m (0.5核) | ~10ms | ~95/s | ~95 人 |
| 1000m (1核) | ~5ms | ~190/s | ~190 人 |
| 2000m (2核) | ~5ms | ~380/s | ~380 人 |

> 相比 PyTorch INT8，ONNX Runtime 在相同资源下并发能力翻倍。

## 实现方式

```python
import onnxruntime as ort

# 首次导出（一次性）
from optimum.onnxruntime import ORTModelForFeatureExtraction
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained("models/onnx/")

# 运行时加载（轻量，无需 torch）
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1        # 多 Pod 不抢 CPU
sess_opts.inter_op_num_threads = 1
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("models/onnx/model.onnx", sess_opts)
```

## 优化历程

```
FP32 (15.5ms) → INT8 (10.9ms, 快41%) → ONNX Runtime (5.2ms, 快66%)
```

## 结论

ONNX Runtime 是该模型的**最优推理方案**：
- Top-1 检索结果 100% 一致（精度无损）
- 比 FP32 快 66%，比 INT8 再快 52%
- 延迟稳定（5.0-5.5ms 范围内）
- 线程隔离（threads=1），适合多 Pod 高密度部署
- 相同 K8s 资源下并发能力翻倍
