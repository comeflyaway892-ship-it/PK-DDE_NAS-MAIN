# Table 7：分类转移核对比

入口独立于 Table 6，不修改共享配置和已有结果。六组共享 ImageNet16-120、
20 个体、15 代、temperature=7、seed=0..99、原生五种操作的相同初始种群，
每次固定 320 次候选评估（初始 20 + 15×20，重复候选也计数）。
与最新 Table 6 一致，全部使用 hp=200 的 `x-test` 搜索和报告，
各架构分数为 benchmark 可用训练种子的平均值。输出跨 100 个搜索种子的均值与样本标准差。

## 公式与实现

所有组用同一 cosine 日程及 beta 序列，构建 `Q_t=(1-beta_t)I+beta_t M`，
并通过矩阵乘法得到 `Qbar_t=Q_1 ... Q_t`。不为高斯核或距离核误用边缘核的闭式简化。
日程参数从共享配置只读加载并保存，实际 alpha_bar/beta 写入 `noise_schedule.json`。
相同 beta 表示相同噪声日程，不表示不同 M 的转移熵或混合速度相同。

| 核 | 对应公式 | 默认设置 |
|---|---|---|
| Uniform | Eq.18：每行 1/K | K=5 |
| Absorbing / mask | Eq.19：全部腐化质量指向 mask | K=6，mask ID=5，区别于 none ID=0 |
| Discrete Gaussian | Eq.20：行归一化 exp(-(i-j)^2/(2 sigma_d^2)) | 各边 sigma_d=1 |
| Distance-aware | Eq.21：行归一化 exp(-abs(i-j)/tau_d) | 各边 tau_d=1 |
| Fixed marginal | Eq.22：每行等于预设 rho_d | [0.1,0.1,0.3,0.3,0.2]，所有边相同 |
| Dynamic marginal | Eq.22：每行等于累计高分架构的操作频率 | top-6、强度 0.99（99% 知识频率 + 1% 均匀分布）、前 10 代更新 |

动态核首先使用初始种群 top-6；第 1..10 代各累计子代 top-6 并重建调度器，
新先验用于下一代。第 11..15 代固定先验。这与 Table 6 Full 一致。
fitness 映射、估计器适应度权重、一致性核、后验采样及生存选择均保留。
`--sigma` 是估计器一致性核的尺度，默认读取配置；`--gaussian-sigma` 才是 Eq.20 的转移尺度。

固定操作顺序：`none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3`。
**这些操作本身是名义类别，没有公认的序关系。** 高斯核与距离核在此只检验指定编号下的
人工邻近关系，不能据此宣称 NB201 具有有序决策语义。
固定 rho 为运行前指定的对照分布，不由 benchmark 分数估计；均匀 rho 会与 Uniform 完全等价。
`--fixed-probs` 支持 5 个共享权重或按边展开的 30 个权重，自动逐行归一化。
`--gaussian-sigma`、`--distance-tau` 各支持 1 个共享尺度或 6 个逐边尺度。

## Absorbing 核与搜索流程

默认 `--proposal-mode reverse_only`，严格保持 Table 6 的生成流程：
当前有效种群作为 x_t，估计 x0，再采样 x(t-1)。仅替换核，不额外前向腐化。
纯 absorbing 核对未被 mask 的观测 token，其反向转移只能返回原 token；
如果预测 x0 与观测 token 不同导致后验质量为零，沿用 Table 6 的 likelihood fallback，
仍返回原 token。因此这组退化成仅评估初始种群，而不是有效的 mask 去噪搜索。
它的 Validity 可以是 100%，但不同架构数不会超过初始种群。
代码和表格会明确标注这一限制，记录零质量后验次数，绝不把 mask 直接映射成 none。

可选 `--proposal-mode forward_reverse`：**六组共同**在每次生成前按 Qbar_t 腐化当前种群，
用干净种群估计 x0，采样反向一步。mask 核的残余 mask 用同一次预测的 x0 对应位置补全，
再评估有效架构。此模式需要 mask 编码适配器；不能与默认 reverse_only 的结果混在一张对比表中。
干净 x0 的预测分布永远只有五种原生操作，不含 mask。
除了评估后的 Validity，另报补全前的 `raw_validity_pct`、观测 mask 数和补全 mask 数，
防止将 mask 适配后的 100% Validity 误读为原始中间状态全部合法。

## 运行

```bash
# 检查默认设置，不加载 API
python NAS_Bench_201/ablation/run_kernel_comparison.py --dry-run

# 六种核，每组100次，自动保留日志
bash NAS_Bench_201/ablation/run_table7.sh

# 覆盖预设频率、转移尺度
bash NAS_Bench_201/ablation/run_table7.sh \
  --fixed-probs 0.1 0.1 0.3 0.3 0.2 --gaussian-sigma 1 --distance-tau 1

# 另起一组统一前向腐化的对照，不与仅替换核的结果混用
bash NAS_Bench_201/ablation/run_table7.sh --proposal-mode forward_reverse
```

可用 `PYTHON_BIN=/path/to/python bash ...` 指定解释器。
每次建立新的 `table7_run_时间_随机后缀/`，包含：

- `run.log`、`resources.txt`、`status.txt`：时间戳、进度、异常、总耗时、峰值内存和退出码。
- `results/settings.json`、`noise_schedule.json`：实际参数、token 顺序、rho、噪声日程。
- `results/runs.jsonl`：每次搜索结果、初始种群、核矩阵、动态 rho 更新轨迹、逐代指标、耗时。
- `results/summary.csv`、`summary.json`：每种核的均值、标准差、有效比例、不同架构数及 mask 诊断。
- `results/table7.md`：可复制的 Table 7，并附语义和 mask 流程说明。
- `results/execution.json`：整个批次完成后的次数和耗时。

每次搜索完成即刷新记录和汇总，输出目录必须不存在以防覆盖旧实验。
Validity 定义与 Table 6 相同：可被 API 查询的候选数量 / 全部评估候选数量，包含重复和初始种群。
测试均值针对每次搜索找到的最高测试准确率；未报告任何未实际运行的实验结果。

```bash
python -m unittest discover -s NAS_Bench_201/ablation -p 'test_kernel_comparison.py' -v
```
