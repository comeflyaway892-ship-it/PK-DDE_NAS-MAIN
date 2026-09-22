import os


MOBILENETV3_ABLATION_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MOBILENETV3_ABLATION_ROOT)
ORIGINAL_MOBILENETV3_ROOT = os.environ.get(
    "MOBILENETV3_ORIGINAL_ROOT",
    os.path.join(REPO_ROOT, "MobileNetV3"),
)


def _m(*parts):
    return os.path.join(ORIGINAL_MOBILENETV3_ROOT, *parts)


RAW_DATA_PATH = os.path.join(REPO_ROOT, "NAS_Bench_201", "data")
PROCESSED_DATA_PATH = _m("data", "ofa", "data_transfer_nag")
SCORE_MODEL_DATA_PATH = _m("data", "ofa", "data_score_model", "ofa_database_500000.pt")
SCORE_MODEL_DATA_IDX_PATH = _m("data", "ofa", "data_score_model", "ridx-500000.pt")

NOISE_META_PREDICTOR_CKPT_PATH = _m("checkpoints", "ofa", "noise_aware_meta_surrogate", "model_best.pth.tar")
SCORE_MODEL_CKPT_PATH = _m("checkpoints", "ofa", "score_model", "model_best.pth.tar")
UNNOISE_META_PREDICTOR_CKPT_PATH = _m("checkpoints", "ofa", "unnoised_meta_surrogate_from_metad2a")
OFA_SUPERNET_CKPT_PATH = _m("checkpoints", "ofa", "ofa_net", "ofa_mbv3_d234_e346_k357_w1.0")
CONFIG_PATH = _m("configs", "transfer_nag_ofa.pt")
