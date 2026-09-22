"""Configuration for surrogate-only NAS-Bench-301 search."""

# A cell is encoded as:
# [o1, o2, p3, o3, o4, p4, o5, o6, p5, o7, o8]
# Normal and reduction cell encodings are concatenated into 22 tokens.
nb301_cell_vocab_sizes = (7, 7, 3, 7, 7, 6, 7, 7, 10, 7, 7)
nb301_vocab_sizes = nb301_cell_vocab_sizes * 2
nb301_num_tokens = len(nb301_vocab_sizes)

# Official NAS-Bench-301 operation order used by the token representation.
nb301_operations = (
    "avg_pool_3x3",
    "max_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
)

# Download the official xgb_v1.0 ensemble and put it at this path. The path is
# resolved relative to NAS_Bench_301 when main.py is used.
nb301_surrogate_path = "./nb_models/xgb_v1.0"
nb301_with_noise = False

# D3PM / predictor / selection parameters.
d3pm_eps = 1e-5
d3pm_schedule = "cosine"
d3pm_cosine_s = 0.008
predictor_estimator_eps = 1e-12
predictor_temperature = 0.4
predictor_sigma = 1.0
select_elite_frac = 0.3
select_eps = 1e-20
select_fill_uniform = True

nb301_hyper_params_setting = {
    "num_step": 100,
    "population_num": 50,
    "rand_exp_num": 20,
    "save_dir": "./results/nb301_surrogate/cifar10/",
    "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
