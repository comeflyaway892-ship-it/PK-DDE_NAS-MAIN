import torch
import tqdm
import time
import copy

from config.config import (
    nb201_vocab_size,
    nb201_num_edges,
    meta_predictor_nasbench201_pt_path,
    meta_predictor_nvt,
    meta_predictor_hs,
    meta_predictor_nz,
    meta_predictor_num_sample,
    meta_predictor_ckpt_path,
)
from utils.plot_cn import plot_denoise
from utils.select import select_population
# from utils.predictor import BayesianGenerator
from utils.predictor import BayesianGenerator
# from utils.ddim import DDIMSchedulerCosine
from utils.d3pm import D3PMUniformMatrixScheduler
from utils.nb201_fitness import arch_fitness
from utils.meta_fitness import meta_arch_fitness
from utils.meta_d2a import FitnessRestorer
from utils.analyse import compute_uniqueness
from utils.meta_d2a import MetaSurrogateUnnoisedModel, load_graph_config, load_model


def evo_diff(
    dataset,
    api,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
    *,
    d3pm_eps: float,
    d3pm_schedule: str,
    d3pm_cosine_s: float,
    predictor_estimator_eps: float,
    predictor_temperature: float,
    predictor_sigma: float,
    select_elite_frac: float,
    select_eps: float,
    select_fill_uniform: bool,
):
    start_time = time.time()

    # 随机初始化种群样本
    x = torch.randint(low=0, high=nb201_vocab_size, size=(population_num, nb201_num_edges))
    x_prev = copy.deepcopy(x)

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []


    scheduler = D3PMUniformMatrixScheduler(
        num_steps=num_step,
        vocab_size=nb201_vocab_size,
        eps=d3pm_eps,
        schedule=d3pm_schedule,
        cosine_s=d3pm_cosine_s,
    )
    # mapping_fn = Energy(temperature=temperature)

    # 计算适应度值
    accurancy, fitness, valid_rate = arch_fitness(
        operation_matrix=x, api=api, dataset=dataset
    )

    # 迭代去噪
    bar = tqdm.tqdm(range(num_step), ncols=120)
    for t in bar:
        # Predictor
        generator = BayesianGenerator(
            x=x,
            fitness=fitness,
            q=scheduler.Q,
            qbar=scheduler.Qbar,
            t=num_step - t,
            estimator_eps=predictor_estimator_eps,
            sigma=predictor_sigma,
            temperature=predictor_temperature,
        )
        x_next = generator.generate()
        uniq_rate = compute_uniqueness(arch_op_matrices=x_next)
        accurancy, fitness_next, valid_rate = arch_fitness(
            operation_matrix=x_next, api=api, dataset=dataset
        )
        max_acc = accurancy.max().item()
        avg_acc = accurancy.mean().item()


        # 保存记录
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        valid_rate_trace.append(valid_rate)
        uniq_rate_trace.append(uniq_rate)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                "valid_rate": f"{valid_rate:.2f}",
                "uniq_rate": f"{uniq_rate:.2f}",
            }
        )

        x = select_population(
            x,
            fitness,
            x_next,
            fitness_next,
            population_num,
            elite_frac=select_elite_frac,
            eps=select_eps,
            fill_uniform=select_fill_uniform,
        )
        # x=x_next
        fitness=fitness_next

    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )
    end_time = time.time()
    print(f"max:{max(max_acc_trace):.2f}\n")
    return max(max_acc_trace), end_time - start_time, uniq_rate, x


def evo_diff_meta(
    dataset,
    api,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
    *,
    d3pm_eps: float,
    d3pm_schedule: str,
    d3pm_cosine_s: float,
    predictor_estimator_eps: float,
    predictor_temperature: float,
    predictor_sigma: float,
    select_elite_frac: float,
    select_eps: float,
    select_fill_uniform: bool,
):
    # 随机初始化种群样本
    x = torch.randint(low=0, high=nb201_vocab_size, size=(population_num, nb201_num_edges))

    # 记录迭代中的种群样本和适应度值变化
    # 把搜索到的所有个体保存下来  用来选取最优结果
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []


    scheduler = D3PMUniformMatrixScheduler(
        num_steps=num_step,
        vocab_size=nb201_vocab_size,
        eps=d3pm_eps,
        schedule=d3pm_schedule,
        cosine_s=d3pm_cosine_s,
    )



    ####
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

    accurancy, fitness, valid_rate = meta_arch_fitness(
        operation_matrix=x,
        api=api,
        dataset=dataset,
        test_dataset=None,
        meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
        nasbench201=nasbench201,
        fitness_restorer=fitness_restorer,
    )
    all_x=[]
    # 迭代去噪
    start_time = time.time()
    bar = tqdm.tqdm(range(num_step), ncols=120)
    for t in bar:

        # Predictor
        generator = BayesianGenerator(
            x=x,
            fitness=fitness,
            q=scheduler.Q,
            qbar=scheduler.Qbar,
            t=num_step - t,
            estimator_eps=predictor_estimator_eps,
            sigma=predictor_sigma,
            temperature=predictor_temperature,
        )
        x_next = generator.generate()
        uniq_rate = compute_uniqueness(arch_op_matrices=x_next)

        # 计算适应度值
        accurancy, fitness_next, valid_rate = meta_arch_fitness(
            operation_matrix=x_next,
            api=api,
            dataset=dataset,
            test_dataset=None,
            meta_surrogate_unnoised_model=meta_surrogate_unnoised_model,
            nasbench201=nasbench201,
            fitness_restorer=fitness_restorer,
        )

        max_acc = accurancy.max().item()
        avg_acc = accurancy.mean().item()

        # 保存记录
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        valid_rate_trace.append(valid_rate)
        uniq_rate_trace.append(uniq_rate)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                "valid_rate": f"{valid_rate:.2f}",
                "uniq_rate": f"{uniq_rate:.2f}",
            }
        )

        x = select_population(
            x,
            fitness,
            x_next,
            fitness_next,
            population_num,
            elite_frac=select_elite_frac,
            eps=select_eps,
            fill_uniform=select_fill_uniform,
        )
        # x=x_next
        fitness = fitness_next
        all_x.append(x)


    end_time = time.time()
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )


    return max(max_acc_trace), end_time - start_time, uniq_rate, torch.cat(all_x,dim=0)
