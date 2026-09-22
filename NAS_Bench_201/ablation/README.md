# Table 6：PK-DDE-NAS 组件消融

独立入口，复用 `NAS_Bench_201/prior_exp` 的先验构建、D3PM 调度器和后验采样，
不修改 `config/config.py`、现有搜索入口或已有结果。仅查 NAS-Bench-201 表，不重新训练网络。

## 固定设置

- 数据集：ImageNet16-120；种群 20，迭代 15 代（另有初始种群）。
- 默认 100 个配对搜索种子 `0..99`，六组共 600 次搜索；同一种子的六组使用相同初始种群。
- 每组每种子 320 次候选评估：初始 20 + 15×20。重复和非法候选也占预算，
  不为重复项额外补采样；缓存只减少实际 API 访问，不增加可用评估次数。
- 最终结果直接使用缓存的最高测试准确率，不额外查询；报告 unique_architectures，
  不把 320 次候选评估宣称为 320 个不同架构。
- 搜索适应度为 `ReScale(2 * test_accuracy)`；全部六组使用 `x-test` 计算适应度、
  排序知识先验、选择最终架构和报告结果，不查询验证集准确率。
  指标为 hp=200 下可用训练种子的平均准确率，与旧入口直接使用测试准确率的口径一致。
  之前保存的 `search_split=x-valid` 实验结果保持原样，不能混入本次汇总，需要重新运行。
- 知识先验：初始种群选 top-6；第 1～10 代各保留子代 top-6，累计构建操作频率先验，
  更新后用于下一代；强度 λ=0.99，即知识频率占 99%、均匀分布占 1%。第 11～15 代使用第 10 代更新后的先验，不再更新。
- 所有组共享相同的父子代合并、去重、精英保留、轮盘赌及种群填充策略，
  返回与所选架构对齐的 fitness，不额外重新评估幸存种群。
- temperature 默认固定为 7.0，可通过 `--temperature` 覆盖，不修改配置文件。
- 其他共享参数只读自现有配置，在 `settings.json` 中保存实际值。

## 六组定义

| Variant | Dynamic prior | Fitness guidance | Diffusion consistency | Posterior sampling | 实现 |
|---|---|---|---|---|---|
| Full PK-DDE-NAS | Yes | Yes | Yes | Yes | 动态先验 + fitness 权重 + 扩散一致性核 + 后验采样 |
| w/o dynamic prior | No | Yes | Yes | Yes | 全程使用 uniform 转移，不使用初始或动态知识先验 |
| w/o fitness guidance | Yes | No | Yes | Yes | 估计器中的 fitness softmax 权重替换为均匀权重 |
| w/o diffusion consistency | Yes | Yes | No | Yes | 估计器中的扩散一致性核替换为常数 1 |
| w/o posterior sampling | Yes | Yes | Yes | No | 跳过后验，直接令 x(t-1) = 预测的 x0 |
| Random mutation | No | No | No | No | 每条边以 1/6 概率均匀替换为其他四种操作之一 |

`Fitness guidance` 特指生成估计器的适应度权重。关闭它不会删除共享的生存选择、
知识先验中的 top-k 排序或最终最优架构选择。Random mutation 是使用相同生存选择的
随机变异进化基线，不是独立均匀随机搜索。论文中应写明这一范围。

`Diffusion consistency` 特指 x0 估计器的核，关闭后仍保留后验中的扩散转移。
`Posterior sampling` 特指反向后验步骤；关闭后直接输出估计的 x0，x0 估计仍按原算法采样。

可选定义（应预先确定，不能看结果后挑选）：

- 默认 `--no-dynamic-mode uniform`：全程均匀先验。备选 `frozen` 仅冻结初始先验，
  不属于本次要求的消融定义。
- 默认 `--no-posterior-mode direct_x0`：绕过反向后验，直接使用估计的 x0。
  备选 `argmax` 仅去掉后验随机性，不属于本次要求的消融定义。
  不同定义不能混在同一结果汇总中。

## 指标

- **Performance (%)**：每个搜索种子在全部已评估候选（包含初始种群）中的最高测试准确率；
  跨搜索种子报告均值 ± 样本标准差（ddof=1）。
- **Validity (%)**：320 个评估候选中可被 API 索引的比例，包含初始种群及重复候选。
  五种合法操作组成的固定 NAS-Bench-201 编码通常均可索引，六组都为 100% 是可能的。
  缺失指标、数据损坏等异常直接报错，不记成无效架构。
- **connected_pct**：另行报告存在非 none 输入到输出路径的比例，不替代上述 Validity。
- **search_seconds**：单次搜索（包括初始化、初始评估和先验构建）墙钟耗时，
  不含公共 API 加载；结果直接复用缓存，不再做最后测试查询。全部组单线程 CPU 运行以减少线程开销。

## 运行与输出

在已安装项目依赖的环境下，从项目根目录运行：

```bash
# 仅检查六组设置，不加载大体积 API 文件、不启动搜索
python NAS_Bench_201/ablation/run_component_ablation.py --dry-run

# 六组，各 100 个种子；20 个体、15 代，前 10 代更新先验，自动记录日志和资源使用
bash NAS_Bench_201/ablation/run_table6.sh

# 先用一个种子完整跑六组以检查环境
bash NAS_Bench_201/ablation/run_table6.sh --runs 1
```

可用 `PYTHON_BIN=/path/to/python bash ...` 指定解释器。
脚本默认加载配置指向的 v1_1 API，且整个批次只加载一次，避免反复加载；
可用 `--api-path /absolute/path/to/benchmark.pth` 显式指定文件。
每次建立新的 `table6_run_时间_随机后缀/`：

- `run.log`：带时间戳的每组开始、迭代进度、结束、准确率和耗时；异常 traceback。
- `resources.txt`：总耗时、峰值内存等 GNU time 统计。
- `status.txt`：结束时间及退出码。
- `results/settings.json`：参数、消融定义、API 路径、预算及指标口径。
- `results/runs.jsonl`：每个已完成种子/变体的完整结果、最优架构和逐步轨迹，立即刷新。
- `results/summary.csv`：准确率、Validity、连通比例、耗时和不同架构数的均值及标准差。
- `results/table6.md`：六列组件/结果表，可用于论文；未完成的组不会伪造结果。
- `results/execution.json`：API 加载耗时及最终完成的实验数（全批完成后写出）。

相同参数再次执行会产生新目录。直接使用 Python 入口时输出目录也必须不存在，
防止覆盖或混入旧数据。不自动恢复中断的批次；已有完成结果仍然可读。

轻量测试（使用合成 API，不加载 benchmark，不是实验结果）：

```bash
python -m unittest discover -s NAS_Bench_201/ablation -p 'test_component_ablation.py' -v
```
