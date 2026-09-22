from nas_201_api import NASBench201API

import os

pth = "./nas_201_api/NAS_Bench_201-v1_1-096897.pth"
print("cwd =", os.getcwd())
print("exists =", os.path.exists(pth))
print("abs path =", os.path.abspath(pth))
# 你下载的 NAS-Bench-201-v1_1-096897.pth 路径
api = NASBench201API(pth)

# 1) 最常见口径：在 ImageNet16-120 上，用 valid 选最优（可改成 'test'）
best_idx, best_acc = api.find_best(dataset="ImageNet16-120", metric_on_set="test", hp="200")

print("best_idx =", best_idx)
print("best_acc =", best_acc)
print("best_arch =", api.arch(best_idx))  # 这行输出的就是你要的“最优架构”
