# 提供的可选数据集
nb201_dataset_list = ["cifar10", "cifar100", "ImageNet16-120"]  # NAS_Bench_201
meta_dataset_list = ["aircraft", "pets"]  # MetaD2A

# ==============================
# NAS-Bench-201 搜索空间编码常量
# （op 取值范围与 operation_matrix 的边数）
# ==============================
nb201_vocab_size = 5  # 每个边上的离散操作 token 数（对应 x 的取值范围 0..4）
nb201_num_edges = 6  # operation_matrix 的边数（每个 genotype 展开后为 6）
nb201_api_path = "./nas_201_api/NAS_Bench_201-v1_1-096897.pth"  # NAS-Bench-201 API 本地 pth 文件路径
nb201_total_archs = nb201_vocab_size ** nb201_num_edges  # 枚举空间大小：vocab_size^num_edges（默认 5^6=15625）

# 用于 test_top10 的固定随机种子（避免每次实验不可复现）
test_top10_seed = 233

# ==============================
# Meta accuracy predictor 相关配置
# （用于：top-k 枚举/训练代理模型 fitness_restorer 之类）
# ==============================
meta_predictor_num_sample = 20  # MetaSurrogate/测试集/fitness_restorer 的采样数
meta_predictor_nvt = 7  # 图结构节点相关参数（传给 load_graph_config/MetaSurrogate）
meta_predictor_hs = 512  # MetaSurrogate hidden size
meta_predictor_nz = 56  # MetaSurrogate latent size

meta_predictor_meta_test_data_path = "./meta_acc_predictor/data/meta_predictor_dataset/"  # MetaTestDataset 数据根目录
meta_predictor_nasbench201_pt_path = "meta_acc_predictor/data/nasbench201.pt"  # nasbench201.pt 路径
meta_predictor_ckpt_path = "meta_acc_predictor/unnoised_checkpoint.pth.tar"  # unnoised checkpoint

# ==============================
# 评估相关可配置参数
# ==============================
eval_repeat_times = 3  # 对同一架构重复训练并取最好精度（eval_architectures 默认值）

# ==============================
# top-k 枚举/评估相关可配置参数
# ==============================
topk_k = 10  # top-k 元学习打分取 top-k 的 k
topk_batch_size = 64  # top-k 枚举时 meta_arch_fitness 的 batch_size

# ==============================
# D3PM / Predictor / Select 工作流超参数
# （这些参数原先在 utils 中以默认值存在，当前补到 config 以便统一配置）
# ==============================

# D3PMUniformMatrixScheduler
d3pm_eps = 1e-5  # 对 alpha_bar 的数值下界/稳定性 clamp
d3pm_schedule = "cosine"  # "linear" 或 "cosine"
d3pm_cosine_s = 0.008  # cosine schedule 常用 s

# BayesianEstimator / BayesianGenerator
predictor_estimator_eps = 1e-12  # BayesianEstimator.__init__ 里的 eps（用于数值稳定）
predictor_temperature = 0.4  # BayesianEstimator.estimate() 里的 temperature（影响 fitness softmax 锋利程度）
predictor_sigma = 1.0  # BayesianEstimator.estimate() 里的 sigma（取默认值1，这里实际上sigma并未发挥作用,用于控制kernel的平滑程度）

# select_population
select_elite_frac = 0.3  # 精英个体比例（当需要从去重池抽样时）
select_eps = 1e-20  # fitness 转概率时的 eps，避免除零/权重退化
select_fill_uniform = True  # True: 随机复制补齐（不看 fitness 加权）

# 实验超参数设置
nb201_hyper_params_setting = {
    "cifar10": {
        "num_step": 50,
        "population_num": 20,
        "rand_exp_num": 20,
        "save_dir": "./results/nb201_benchmark/cifar10/",
        "seed": [
            0,# 94.37    94.37
            1,# 94.37    94.37
            2,# 94.37    94.37
            3,# 94.37    94.37
            4,# 94.37    94.37
            5,# 94.21    94.37
            6,# 94.34    94.37
            7,# 94.31    94.37
            8,# 94.37    94.37
            9,# 94.37    94.37
            # 1731578139,  # max_acc@1: 94.37
            # 1731578141,  # max_acc@1: 94.37
            # 1731578146,  # max_acc@1: 94.37
            # 1731578150,  # max_acc@1: 94.37
            # 1731578154,  # max_acc@1: 94.37

        ],
    },
    "cifar100": {
        "num_step": 60,
        "population_num": 50,
        "rand_exp_num": 20,
        "save_dir": "./results/nb201_benchmark/cifar100/",
        "seed": [
            0,  #73.51   73.51
            1,  #73.26   73.51
            2,  #73.51   73.51
            3,  #73.51   73.51
            4,  #73.51   73.26
            5,  #73.51   73.51
            6,  #73.17   73.51
            7,  #73.26   73.51
            8,  #71.96   73.51
            9,  #73.00   73.51
            # 1731578242,  # max_acc@1: 73.51
            # 1731578247,  # max_acc@1: 73.51
            # 1731578254,  # max_acc@1: 73.51
            # 1731578256,  # max_acc@1: 73.51
            # 1731578258,  # max_acc@1: 73.51
        ],
    },
    "ImageNet16-120": {
        "num_step": 60,
        "population_num": 50,
        "rand_exp_num": 20,
        "save_dir": "./results/nb201_benchmark/imagenet16_120/",
        "seed": [
            0, #46.93    47.31
            1, #47.03    46.83
            2, #46.10    47.31
            3, #46.90    47.31
            4, #47.31    46.83
            5, #47,31    47.31
            6, #46.90    47.31
            7, #47.31    47.31
            8, #47.31    47.31
            9, #47.03    47.31
            10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
            # 1731578516,  # max_acc@1: 47.31
            # 1731578531,  # max_acc@1: 47.31
            # 1731578554,  # max_acc@1: 47.31
            # 1731578556,  # max_acc@1: 47.31
            # 1731578650,  # max_acc@1: 47.31
        ],
    },
}
meta_hyper_params_setting = {
    "aircraft": {
        "num_step": 60,
        "population_num": 50,
        "rand_exp_num": 5,
        "save_dir": "./results/meta/aircraft/",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 3,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
            # Current Benchmark, max_acc@3: 59.15+-0.58
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
            ,21,22,23,24,25,26,27,28,29,30
        ],
    },
    "pets": {
        "num_step": 60,
        "population_num": 50,
        "rand_exp_num": 5,
        "save_dir": "./results/meta/pets/",
        "eta_min": 0.0,
        "epochs": 200,
        "warmup": 10,
        "LR": 0.1,
        "decay": 0.0005,
        "momentum": 0.9,
        "nesterov": True,
        "batch_size": 256,
        "image_cutout": 5,
        "topk": 2,
        "early_stop": False,
        "multi_thread": False,
        "seed": [
        #    999,  # max_acc@1: 41.90, max_acc@2: 46.37
        #    88,  # max_acc@1: 46.84, max_acc@2: 43.70
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
            # # Current Benchmark: 41.80+-3.82
            # 66,  # max_acc@1: 45.50, max_acc@2: 46.21
            # 77,  # max_acc@1: 46.21, max_acc@2: 45.66
            # 99,  # max_acc@1: 38.47, max_acc@2: 43.31
            # 777,  # max_acc@1: 44.41, max_acc@2: 44.80
            # 7890,  # max_acc@1: 45.35, max_acc@2: 47.46
            # 3456,  # max_acc@1: 46.75, max_acc@2: 43.39
            # 111,
            # 222,
            # 444,
            # 333,
            # 1234,
            # 2345,
            # 9012,
            # 555,
            # 666,
            # 888,
            # 4567,
            # 6789,
            # 8901,
            # 1001,
            # 2002,
            # 3003,
            # 4004,
            # 5005,
            # 6006,
            # 7007,
            # 8008,
            # 9009,
            # 42,
            # 78,
            # 63,
        ],
    },
}
