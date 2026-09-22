prior_Konwleage_top20_l1_dyn30：四个数据集重训练统计（准确率 %）

统计口径：使用 max_valid_acc（训练期间最高验证准确率）。每个架构对重训练种子
777/888/999 分别计算 Max、Mean、Min；将搜索种子 0/1/2 各自的三个架构合并，
对九个架构的 Max、Mean、Min 分别计算均值 ± 样本标准差（ddof=1）。
按 (search_seed, arch_idx) 计九个候选，不跨搜索种子去重。这里的 ± 为标准差，不是置信区间。

| Dataset | Stats. | Prior top20 λ=1 dyn30 |
|---|---|---:|
| CIFAR-10 | Max | 97.52 ± 0.10 |
| CIFAR-10 | Mean | 97.48 ± 0.09 |
| CIFAR-10 | Min | 97.43 ± 0.08 |
| CIFAR-100 | Max | 86.28 ± 0.22 |
| CIFAR-100 | Mean | 86.16 ± 0.17 |
| CIFAR-100 | Min | 86.07 ± 0.17 |
| Aircraft | Max | 82.12 ± 0.43 |
| Aircraft | Mean | 81.82 ± 0.49 |
| Aircraft | Min | 81.51 ± 0.67 |
| Oxford-IIIT Pets | Max | 95.41 ± 0.34 |
| Oxford-IIIT Pets | Mean | 95.18 ± 0.28 |
| Oxford-IIIT Pets | Min | 94.93 ± 0.37 |
