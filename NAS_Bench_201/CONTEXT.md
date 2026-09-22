NAS-Bench-201 目录上下文文档
==========================

目标
--
简洁、全面地描述 `NAS_Bench_201` 目录中用于搜索/评估的代码和关键接口，便于以后只传递此文档作为上下文。

总体说明
--
该目录实现了一个基于“离散扩散替代演化变异”的 NAS 搜索算法（EvoDiff）。核心入口和工作流集中在 `main.py`、`experiments.py`、`evo_diff.py`，大量功能实现放在 `utils/` 下。

主要入口
--
- `main.py`：命令行入口，解析 `--exp_type` 和 `--dataset`，调用 `experiments.py` 中的实验函数来启动搜索。
- `experiments.py`：提供多种实验/复现实验函数：
  - `exp_with_fixed_seed_in_nb201` / `exp_with_rand_seed_in_nb201`：针对 NAS-Bench-201 的复现与随机种子实验。
  - `exp_with_fixed_seed_in_meta_predictor` / `exp_with_rand_seed_in_meta_predictor`：使用元预测器（meta-predictor）进行的实验流程。
  - `topk_by_meta_enumeration`：全枚举（5^6=15625）并用元预测器打分以挑选 top-k。

核心算法
--
- `evo_diff.py`：实现 EvoDiff 搜索主循环：
  - 使用 `D3PMUniformMatrixScheduler`（去噪调度器）执行去噪步长。
  - 使用 `BayesianGenerator` 作为预测/生成器来从当前种群生成候选。
  - 使用 `select_population` 在当前与候选间选取下一代（包含精英保持与重复填充策略）。
  - 同时提供 `evo_diff`（直接基于 NB201 API 评估）和 `evo_diff_meta`（基于元代理进行预测评估）的变体。

配置与超参数
--
- `config/config.py`：统一的常量与超参数集合，包括：搜索空间常量（`nb201_vocab_size`, `nb201_num_edges` 等），D3PM / predictor / select 的默认超参，以及不同数据集的实验配置字典（`nb201_hyper_params_setting` / `meta_hyper_params_setting`）。

工具与辅助模块（`utils/` 概览）
--
- `nb201_fitness.py`：加载 NAS-Bench-201 API、将 operation_matrix 映射为 benchmark 上的准确率（`arch_fitness`、`neural_predictor`）。
- `meta_d2a.py`：元学习相关组件，包含 `MetaSurrogateUnnoisedModel`、`load_graph_config`、`load_model`、`FitnessRestorer` 等（在 meta 评估/打分中被使用）。
- `meta_fitness.py`：封装用于元预测器评分的 `meta_arch_fitness`。
- `d3pm.py`：实现 D3PM 的调度器 `D3PMUniformMatrixScheduler`（离散矩阵去噪相关）。
- `predictor.py`：实现 `BayesianGenerator` / `BayesianEstimator`（用于从带 fitness 的种群中生成/评分候选）。
- `select.py`：实现 `select_population`（合并/抽样下一代的策略：精英保留、eps 与填充策略）。
- `mapping.py`：包括 `ReScale`（把原始准确率映射到用于比较/采样的 fitness）。
- `eval_arch.py`：对候选架构做完整训练/评估的流程（用于最终复训/验证）。
- `plot.py`、`analyse.py`：绘图与统计（如 uniqueness 计算）。
- 其它：`optimizers.py`, `coreset.py`, `flop_benchmark.py` 等为可选工具/基准功能。

数据与依赖
--
- NAS-Bench-201 API 二进制：配置里默认路径为 `./nas_201_api/NAS_Bench_201-v1_1-096897.pth`（`config/config.py` 中 `nb201_api_path`）。
- 元预测器相关数据与 checkpoint：`meta_acc_predictor/data/nasbench201.pt` 和 `meta_acc_predictor/unnoised_checkpoint.pth.tar`（路径也在 `config/config.py` 指定）。

快速使用说明
--
1. 安装依赖（项目根目录）：

```bash
pip install -r requirements.txt
```

2. 运行复现实验（例如在 ImageNet16-120 上）：

```bash
python main.py --exp_type reproduce --dataset imagenet
```

3. 运行全枚举 top-k（使用元预测器）：在 Python 里调用 `experiments.topk_by_meta_enumeration(...)` 或运行 `main.py` 并在 `experiments.test_top10` 中查看用法。

关键注意事项
--
- 如果使用 meta 相关功能，确保 `meta_acc_predictor` 的数据与 checkpoint 已经准备并且 `config` 中路径正确。
- `nb201_fitness.load_nb201_api` 在代码中使用了局部的 pth 路径（`./nas_201_api/NAS_Bench_201-v1_0-e61699.pth`），若报错请根据本地文件调整为 `config.config.nb201_api_path` 的实际文件名。
- 本目录的运行依赖 GPU（可选），以及较多内存时在全枚举阶段（`topk_by_meta_enumeration`）可能会需要分批处理。

定位常用符号（快速索引）
--
- 实验入口： `main.py` -> `experiments.py`
- 搜索循环： `evo_diff.py:evo_diff` / `evo_diff.py:evo_diff_meta`
- 元代理加载： `utils/meta_d2a.py` / 配置 `config/config.py`
- NB201 映射/打分： `utils/nb201_fitness.py`

文件位置：NAS_Bench_201/CONTEXT.md
