import torch
import time
import os
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from utils.nb201_fitness import load_nb201_api, get_nb201_arch_str
from config.config import (
    nb201_hyper_params_setting,
    nb201_dataset_list,
    meta_hyper_params_setting,
    meta_dataset_list,
    nb201_vocab_size,
    nb201_num_edges,
    nb201_api_path,
    nb201_total_archs,
    meta_predictor_num_sample,
    meta_predictor_nvt,
    meta_predictor_hs,
    meta_predictor_nz,
    meta_predictor_meta_test_data_path,
    meta_predictor_nasbench201_pt_path,
    meta_predictor_ckpt_path,
    eval_repeat_times,
    topk_k,
    topk_batch_size,
    test_top10_seed,
    # d3pm / predictor / select workflow hyperparameters
    d3pm_eps,
    d3pm_schedule,
    d3pm_cosine_s,
    predictor_estimator_eps,
    predictor_temperature,
    predictor_sigma,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from evo_diff import evo_diff, evo_diff_meta
from utils.eval_arch import eval_architectures
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import (
    MetaTestDataset,
    MetaSurrogateUnnoisedModel,
    load_graph_config,
    load_model,
)
from utils.meta_d2a import FitnessRestorer


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_topk_archs(x: torch.Tensor, dataset: str, k: int, api, seed):
    print(f">>> Selecting top-{k} architectures from the population...")
    arch_str_list = []
    population = x.shape[0]
    for i in range(population):
        arch_matrix = x[i].view(nb201_num_edges)
        arch_str = get_nb201_arch_str(arch_matrix)
        arch_str_list.append(arch_str)

    # 准备元学习预测器
    test_dataset = MetaTestDataset(
        data_path=meta_predictor_meta_test_data_path,
        data_name=dataset,
        num_sample=meta_predictor_num_sample,
    )
    graph_config = load_graph_config(
        graph_data_name="nasbench201",
        nvt=meta_predictor_nvt,
        data_path=meta_predictor_nasbench201_pt_path,
    )
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
        nvt=meta_predictor_nvt,
        hs=meta_predictor_hs,
        nz=meta_predictor_nz,
        num_sample=meta_predictor_num_sample,
        graph_config=graph_config,
    )
    meta_surrogate_unnoised_model = load_model(
        model=meta_surrogate_unnoised_model,
        ckpt_path=meta_predictor_ckpt_path,
    )
    nasbench201 = torch.load(meta_predictor_nasbench201_pt_path)
    fitness_restorer = FitnessRestorer(
        dataset_name=dataset,
        num_sample=meta_predictor_num_sample,
        seed=seed,
    )

    _, fitness, _ = meta_arch_fitness(
        operation_matrix=x,
        api=api,
        dataset=dataset,
        test_dataset=test_dataset,
        meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
        nasbench201=nasbench201,
        fitness_restorer=fitness_restorer,
    )

    sorted_indices = torch.argsort(fitness, descending=True)
    unique_arch_str = set()
    topk_indices = []

    # 保证topk_indices中的架构arch_str不重复
    for idx in sorted_indices:
        if len(topk_indices) >= k:
            break
        arch_str = arch_str_list[idx]
        if arch_str not in unique_arch_str:
            unique_arch_str.add(arch_str)
            topk_indices.append(idx)

    # 如果架构不足k个，则只返回unique的架构
    if len(topk_indices) < k:
        print(
            f">>> Only found {len(topk_indices)} unique architectures in the top-{k} architectures."
        )

    topk_archs = x[torch.tensor(topk_indices)]
    print(f">>> Top-{k} architectures selected.")
    return topk_archs


def exp_with_rand_seed_in_nb201(dataset: str):
    """
    Experiment for random test using NAS_Bench_201.
    """
    assert dataset in nb201_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)
    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # Show Benchmark API
    total_acc = []
    for i in range(nb201_total_archs):
        acc = api.query_test_acc_by_index(i, dataset)
        total_acc.append(acc)
    print(f"For {dataset}, avg acc: {np.mean(total_acc)}, max acc: {np.max(total_acc)}")

    # 导入Hyper-parameters设置
    args = nb201_hyper_params_setting[dataset]

    # 随机种子实验
    avg_max_acc = torch.tensor(0.0)
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for exp in range(args["rand_exp_num"]):
        seed = int(time.time())
        set_random_seed(seed)
        print(f"\n>>> Exp {exp}: Running on {dataset} dataset with seed {seed}...")
        max_acc, duration, uniq_rate, _ = evo_diff(
            dataset=dataset,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=args["save_dir"] + "rand_exp/",
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        avg_max_acc += max_acc
        avg_duration += duration
        avg_uniq_rate += uniq_rate


    rand_exp_num = args['rand_exp_num']
    avg_max_acc_final = avg_max_acc / rand_exp_num
    avg_duration_final = avg_duration / rand_exp_num
    avg_uniq_rate_final = avg_uniq_rate / rand_exp_num

    print(
        f'>>> In {rand_exp_num} random seed experiment on {dataset} dataset, average max accuracy is {avg_max_acc_final:.2f}, average duration is {avg_duration_final:.2f} seconds, average uniqueness rate is {avg_uniq_rate_final:.2f}.\n'
    )


def exp_with_rand_seed_in_meta_predictor(dataset: str):
    """
    Experiment for random test using meta-predictor.
    """
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # 导入Hyper-parameters设置
    args = meta_hyper_params_setting[dataset]

    # 随机种子实验
    avg_max_acc = torch.tensor(0.0)
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for exp in range(args["rand_exp_num"]):
        seed = int(time.time())
        set_random_seed(seed)
        print(f"\n>>> Exp {exp}: Running on {dataset} dataset with seed {seed}...")
        max_acc, duration, uniq_rate, x = evo_diff_meta(
            dataset=dataset,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=args["save_dir"] + "rand_exp/",
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        # 暂时不做重训练  单纯看搜索结果
        # x = get_topk_archs(x=x, dataset=dataset, k=args["topk"], api=api, seed=seed)
        # max_acc, acc_list = eval_architectures(
        #     x=x.cpu(),
        #     api=api,
        #     dataset_name=dataset,
        #     image_cutout=args["image_cutout"],
        #     batch_size=args["batch_size"],
        #     device="cuda" if torch.cuda.is_available() else "cpu",
        #     lr=args["LR"],
        #     momentum=args["momentum"],
        #     decay=args["decay"],
        #     nesterov=args["nesterov"],
        #     train_epochs=args["epochs"],
        #     warmup_epoch=args["warmup"],
        #     eta_min=args["eta_min"],
        #     multi_thread=args["multi_thread"],
        #     early_stop=args["early_stop"],
        # )
        # avg_max_acc += max_acc
        # avg_duration += duration
        # avg_uniq_rate += uniq_rate
    # 提前取出args中的关键值，避免在f-string内使用[]访问字典
    rand_exp_num = args['rand_exp_num']
    # 计算最终平均值（提前计算，让f-string更简洁）
    avg_max_accuracy = avg_max_acc / rand_exp_num
    avg_total_duration = avg_duration / rand_exp_num
    avg_final_uniq_rate = avg_uniq_rate / rand_exp_num
    #
    # # 打印结果（f-string中仅使用临时变量，无字典[]操作）
    # print(
    #     f'>>> In {rand_exp_num} random seed experiment on {dataset} dataset, average max accuracy is {avg_max_accuracy:.2f}, average duration is {avg_total_duration:.2f} seconds, average uniqueness rate is {avg_final_uniq_rate:.2f}.\n'
    # )
    print(
        f'>>> In {rand_exp_num} random seed experiment on {dataset} dataset, average max accuracy is {avg_max_acc:.2f}, average duration is {avg_duration:.2f} seconds, average uniqueness rate is {avg_uniq_rate:.2f}.\n'
    )

def exp_with_fixed_seed_in_nb201(dataset: str):
    """
    Experiment for reproducibility using NAS_Bench_201.
    """
    assert dataset in nb201_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # Show Benchmark API
    total_acc = []
    for i in range(nb201_total_archs):
        acc = api.query_test_acc_by_index(i, dataset)
        total_acc.append(acc)
    print(
        f">>> For {dataset}, avg acc: {np.mean(total_acc)}, max acc: {np.max(total_acc)}"
    )

    # 导入Hyper-parameters设置
    args = nb201_hyper_params_setting[dataset]
    seed_list = args["seed"]

    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for seed in seed_list:
        # 复现最佳结果
        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed}...")
        _, duration, uniq_rate, _ = evo_diff(
            dataset=dataset,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=args["save_dir"] + "reproduce_exp/",
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        avg_duration += duration
        avg_uniq_rate += uniq_rate
    print(
        f">>> For dataset {dataset}, average search duration is {avg_duration / len(seed_list):.2f} seconds, average uniqueness rate is {avg_uniq_rate / len(seed_list):.2f}.\n"
    )


def exp_with_fixed_seed_in_meta_predictor(dataset: str):
    """
    Experiment for reproducibility using meta-predictor.
    """
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)


    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # 导入Hyper-parameters设置
    args = meta_hyper_params_setting[dataset]
    seed_list = args["seed"]
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for seed in seed_list:
        if os.path.exists(f"results/search_log/{dataset}_search.pth"):
            result_log = torch.load(f"results/search_log/{dataset}_search.pth",)
        else:
            result_log = {}
        if seed in result_log:
            print(
                ">>> Log already exists, pass this seed. Searching Log: ",
                result_log[seed],
            )
        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed}...")
        max_acc, duration, uniq_rate, x = evo_diff_meta(
            dataset=dataset,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=args["save_dir"] + "reproduce_exp/",
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        print(f"max_pred_acc:{max_acc:.2f}\n")
        x = get_topk_archs(x=x, dataset=dataset, k=args["topk"], api=api, seed=seed)

        max_acc, acc_list = eval_architectures(
            x=x.cpu(),
            api=api,
            dataset_name=dataset,
            image_cutout=args["image_cutout"],
            batch_size=args["batch_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args["LR"],
            momentum=args["momentum"],
            decay=args["decay"],
            nesterov=args["nesterov"],
            train_epochs=args["epochs"],
            warmup_epoch=args["warmup"],
            eta_min=args["eta_min"],
            multi_thread=args["multi_thread"],
            early_stop=args["early_stop"],
            repeat_times=eval_repeat_times,
        )
        avg_duration += duration
        avg_uniq_rate += uniq_rate
        result_log[seed] = {
            "max_acc": max_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "acc_list": acc_list,
        }
        os.makedirs("results/search_log", exist_ok=True)
        torch.save(result_log, f"results/search_log/{dataset}_search.pth")
    print(
        f">>> For dataset {dataset}, average search duration is {avg_duration / len(seed_list):.2f} seconds, average uniqueness rate is {avg_uniq_rate / len(seed_list):.2f}.\n"
    )

# 这些部分用来测试top10

import time
import torch

from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import MetaSurrogateUnnoisedModel, load_graph_config, load_model, FitnessRestorer

def build_meta_components(dataset: str, seed: int):
    """
    按你的 evo_diff_meta 里的方式加载 meta surrogate + nasbench201 + fitness_restorer
    """
    nasbench201 = torch.load(meta_predictor_nasbench201_pt_path)
    graph_config = load_graph_config(
        graph_data_name="nasbench201",
        nvt=meta_predictor_nvt,
        data_path=meta_predictor_nasbench201_pt_path,
    )
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
        nvt=meta_predictor_nvt,
        hs=meta_predictor_hs,
        nz=meta_predictor_nz,
        num_sample=meta_predictor_num_sample,
        graph_config=graph_config,
    )
    meta_surrogate_unnoised_model = load_model(
        model=meta_surrogate_unnoised_model,
        ckpt_path=meta_predictor_ckpt_path,
    )
    fitness_restorer = FitnessRestorer(
        dataset_name=dataset,
        num_sample=meta_predictor_num_sample,
        seed=seed,
    )
    return nasbench201, meta_surrogate_unnoised_model, fitness_restorer

def enumerate_all_nb201_ops(vocab_size: int = 5, num_edges: int = 6) -> torch.Tensor:
    """
    枚举所有 operation_matrix: (15625, 6)，每个元素 in [0, vocab_size-1]
    """
    grids = [torch.arange(vocab_size) for _ in range(num_edges)]
    # cartesian_prod 输出 shape: (vocab_size**num_edges, num_edges)
    return torch.cartesian_prod(*grids).long()

@torch.no_grad()
def topk_by_meta_enumeration(
    dataset: str,
    api,
    seed: int,
    k: int = 10,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    全枚举 NAS-Bench-201 (5^6=15625)，用 meta_arch_fitness 打分，
    返回：x_topk (k,6)、topk_pred_acc、耗时、valid_rate(整体均值)
    """
    print("=" * 80)
    print(f"[DEBUG] Enter topk_by_meta_enumeration")
    print(f"[DEBUG] dataset={dataset} seed={seed} k={k} batch_size={batch_size} device={device}")

    # 1) build meta components
    t_build = time.time()
    nasbench201, meta_model, fitness_restorer = build_meta_components(dataset, seed)
    print(f"[DEBUG] build_meta_components done, time={time.time() - t_build:.3f}s")
    print(f"[DEBUG] meta_model type={type(meta_model)}  fitness_restorer type={type(fitness_restorer)}")

    # 2) move model to device
    meta_model = meta_model.to(device).eval()
    print(f"[DEBUG] meta_model moved to {device} and set eval()")

    # 3) enumerate full space
    t_enum = time.time()
    x_all = enumerate_all_nb201_ops(vocab_size=nb201_vocab_size, num_edges=nb201_num_edges)
    N = x_all.size(0)
    print(f"[DEBUG] enumerate_all_nb201_ops done, time={time.time() - t_enum:.3f}s, x_all.shape={tuple(x_all.shape)}")
    print(f"[DEBUG] total architectures N={N}")

    preds = []
    valid_rates = []

    # 4) loop batches
    t0 = time.time()
    num_batches = (N + batch_size - 1) // batch_size
    print(f"[DEBUG] start scoring loop, num_batches={num_batches}")

    for bi, s in enumerate(range(0, N, batch_size), start=1):
        b_start = s
        b_end = min(s + batch_size, N)
        t_batch = time.time()

        xb = x_all[b_start:b_end].to(device)

        # 可选：打印显存占用（只在cuda时有效）
        if device.startswith("cuda"):
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(
                f"[DEBUG][batch {bi}/{num_batches}] xb range=[{b_start}:{b_end}] xb.shape={tuple(xb.shape)} "
                f"cuda_mem_alloc={mem_alloc:.1f}MB reserved={mem_reserved:.1f}MB"
            )
        else:
            print(
                f"[DEBUG][batch {bi}/{num_batches}] xb range=[{b_start}:{b_end}] xb.shape={tuple(xb.shape)}"
            )

        accurancy, fitness, valid_rate = meta_arch_fitness(
            operation_matrix=xb,
            api=api,
            dataset=dataset,
            test_dataset=None,
            meta_surrogate_unnoised_model=meta_model,
            nasbench201=nasbench201,
            fitness_restorer=fitness_restorer,
        )

        # accurancy: (B,) 或 (B,1)
        acc_flat = accurancy.view(-1).detach().cpu()
        preds.append(acc_flat)
        valid_rates.append(torch.tensor(float(valid_rate)))

        # 打印本 batch 的统计信息
        batch_time = time.time() - t_batch
        print(
            f"[DEBUG][batch {bi}/{num_batches}] meta_arch_fitness done, time={batch_time:.3f}s, "
            f"acc stats: min={acc_flat.min().item():.4f} mean={acc_flat.mean().item():.4f} max={acc_flat.max().item():.4f}, "
            f"valid_rate={float(valid_rate):.4f}"
        )

    duration = time.time() - t0
    print(f"[DEBUG] scoring loop finished, total_time={duration:.3f}s")

    # 5) concat preds
    t_cat = time.time()
    preds = torch.cat(preds, dim=0)  # (15625,)
    print(f"[DEBUG] torch.cat preds done, time={time.time() - t_cat:.3f}s, preds.shape={tuple(preds.shape)}")

    # sanity check
    if preds.numel() != N:
        print(f"[WARN] preds.numel()={preds.numel()} != N={N}")

    # 6) topk
    t_topk = time.time()
    topv, topi = torch.topk(preds, k=k, largest=True)
    x_topk = x_all[topi].cpu()
    print(f"[DEBUG] torch.topk done, time={time.time() - t_topk:.3f}s")

    print(f"[DEBUG] top{k} predicted scores: {topv.tolist()}")
    print(f"[DEBUG] top{k} indices: {topi.tolist()}")
    print(f"[DEBUG] top{k} operation matrices (first few rows):\n{x_topk}")

    # 7) avg valid rate
    avg_valid_rate = torch.stack(valid_rates).mean().item()
    print(f"[DEBUG] avg_valid_rate over batches = {avg_valid_rate:.4f}")

    print(f"[DEBUG] return x_topk.shape={tuple(x_topk.shape)}, topv.shape={tuple(topv.shape)}, duration={duration:.3f}s")
    print("=" * 80)

    return x_topk, topv, duration, avg_valid_rate

def test_top10(dataset: str):
    """
    Experiment for reproducibility using meta-predictor.
    """
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    # 导入Hyper-parameters设置
    args = meta_hyper_params_setting[dataset]
    avg_duration = torch.tensor(0.0)


    if os.path.exists(f"results/search_log/{dataset}_search.pth"):
        result_log = torch.load(f"results/search_log/{dataset}_search.pth",)
    else:
        result_log = {}

    x, pred_scores, duration, avg_valid_rate = topk_by_meta_enumeration(
        dataset=dataset,
        api=api,
        seed=test_top10_seed,
        k=topk_k,
        batch_size=topk_batch_size,
    )

    max_acc, acc_list = eval_architectures(
        x=x.cpu(),
        api=api,
        dataset_name=dataset,
        image_cutout=args["image_cutout"],
        batch_size=args["batch_size"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=args["LR"],
        momentum=args["momentum"],
        decay=args["decay"],
        nesterov=args["nesterov"],
        train_epochs=args["epochs"],
        warmup_epoch=args["warmup"],
        eta_min=args["eta_min"],
        multi_thread=args["multi_thread"],
        early_stop=args["early_stop"],
        repeat_times=eval_repeat_times,
    )
    avg_duration += duration

    os.makedirs("results/search_log", exist_ok=True)
    torch.save(result_log, f"results/search_log/{dataset}_search.pth")
