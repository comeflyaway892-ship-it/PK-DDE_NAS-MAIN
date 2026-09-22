# 实验超参数设置   #genotype修改为macro4 micro6
global_params=[]

# D3PM / predictor / selection settings shared by prior experiments.
d3pm_eps = 1e-5
d3pm_schedule = "cosine"
d3pm_cosine_s = 0.008
predictor_estimator_eps = 1e-12
predictor_temperature = 0.4
predictor_sigma = 1.0
select_elite_frac = 0.3
select_eps = 1e-20
select_fill_uniform = True

hyper_params_setting = {
    "macro": {
        "class_scene": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/class_scene",
            "seed": [
                0,
                1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                17,18,19,20,21,22,23,24,25,26,27,28,29,30
            ],
        },
        "class_object": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/class_object",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "room_layout": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/room_layout",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "jigsaw": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/jigsaw",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "segmentsemantic": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/segmentsemantic",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "normal": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/normal",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "autoencoder": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/macro/autoencoder",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
    },
    "micro": {
        "class_scene": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/class_scene",
            "seed": [
                0,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            ],
        },
        "class_object": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/class_object",
            "seed": [
                0,
                1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                17,18,19,20,21,22,23,24,25,26,27,28,29,30
            ],
        },
        "room_layout": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/room_layout",
            "seed": [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "jigsaw": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/jigsaw",
            "seed": [
                0,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "segmentsemantic": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/segmentsemantic",
            "seed": [
                0,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "normal": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/normal",
            "seed": [
                0,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            ],
        },
        "autoencoder": {
            "num_step": 60,
            "population_num": 50,
            "save_dir": "./results/micro/autoencoder",
            "seed": [
                    0,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        }
    }
}
