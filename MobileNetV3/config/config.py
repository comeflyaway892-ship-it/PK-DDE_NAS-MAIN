import os


MOBILENETV3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(MOBILENETV3_ROOT)
NAS_BENCH_201_ROOT = os.path.join(REPO_ROOT, "NAS_Bench_201")


def _m(*parts):
    return os.path.join(MOBILENETV3_ROOT, *parts)


def _n(*parts):
    return os.path.join(NAS_BENCH_201_ROOT, *parts)


# Available meta-predictor datasets
meta_dataset_list = ["cifar10", "cifar100", "aircraft", "pets"]

# MobileNetV3/OFA 25-token encoding:
# [D1..D5, O1,1..O5,4], with D in {0,1,2} -> depth {2,3,4}
# and O in {0..8} -> kernel/expand operation.
vocab_size = 9
num_edges = 25
num_depth_tokens = 5
num_op_tokens = 20

# OFA/MetaD2A proxy predictor dimensions.
ofa_proxy_num_sample = 20
ofa_proxy_nvt = 27
ofa_proxy_hs = 512
ofa_proxy_nz = 56
ofa_proxy_eval_repeats = 10
ofa_proxy_acc_mean = 55.0
ofa_proxy_acc_std = 10.0

# Legacy NAS201 meta accuracy predictor paths and dimensions.
meta_predictor_num_sample = 20
meta_predictor_nvt = 7
meta_predictor_hs = 512
meta_predictor_nz = 56

meta_predictor_meta_test_data_path = _n("meta_acc_predictor", "data", "meta_predictor_dataset")
meta_predictor_nasbench201_pt_path = _n("meta_acc_predictor", "data", "nasbench201.pt")
meta_predictor_ckpt_path = _n("meta_acc_predictor", "unnoised_checkpoint.pth.tar")
mobile_retrain_data_root = _n("data")

# D3PM / Predictor / Select workflow hyperparameters.
d3pm_eps = 1e-5
d3pm_schedule = "cosine"
d3pm_cosine_s = 0.008

predictor_estimator_eps = 1e-12
predictor_temperature = 0.4
predictor_sigma = 1.0

select_elite_frac = 0.3
select_eps = 1e-20
select_fill_uniform = True

# MobileNet/OFA retraining settings.
mobile_retrain_gpu = os.environ.get("DDE_NAG_GPU", os.environ.get("EDNAG_GPU", "0"))
mobile_retrain_topk = 3
mobile_retrain_seeds = [777, 888, 999]
mobile_retrain_workers = 8
mobile_retrain_epochs = 20
mobile_retrain_batch_size = 96
mobile_retrain_lr = 0.01
mobile_retrain_momentum = 0.9
mobile_retrain_weight_decay = 4e-5
mobile_retrain_report_freq = 50
mobile_retrain_grad_clip = 5
mobile_retrain_cutout = True
mobile_retrain_cutout_length = 16
mobile_retrain_autoaugment = True
mobile_retrain_drop = 0.2
mobile_retrain_drop_path = 0.2
mobile_retrain_img_size = 224

meta_hyper_params_setting = {
    "cifar10": {
        "num_step": 50,
        "population_num": 20,
        "save_dir": _m("results", "meta", "cifar10") + os.sep,
        "seed": list(range(10)),
    },
    "cifar100": {
        "num_step": 50,
        "population_num": 20,
        "save_dir": _m("results", "meta", "cifar100") + os.sep,
        "seed": list(range(10)),
    },
    "aircraft": {
        "num_step": 50,
        "population_num": 20,
        "save_dir": _m("results", "meta", "aircraft") + os.sep,
        "seed": list(range(10)),
    },
    "pets": {
        "num_step": 50,
        "population_num": 20,
        "save_dir": _m("results", "meta", "pets") + os.sep,
        "seed": list(range(10)),
    },
}
