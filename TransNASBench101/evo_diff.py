from sched import scheduler

import torch
import tqdm
import time
import math
import copy

from utils.d3pm_preset_prior_scheduler import D3PMPresetPriorMatrixScheduler
from utils.d3pm import D3PMUniformMatrixScheduler
from utils.correct import select_population,random_mutation
from utils.mapping import Power, Energy, Identity
from utils.plot import plot_denoise
from utils.predictor import BayesianGenerator
from utils.transnasbench101_fitness import arch_fitness
from utils.population_similarity import population_similarity


def evo_diff(
    task,
    search_space,
    api,
    num_step,
    population_num,
    plot_results,
    save_dir,
    seed,
):
    start_time = time.time()
    if search_space is "micro":
        geno_size=6
    else:
        geno_size=7

    # 随机初始化种群样本
    x = torch.randint(low=0,high=4,size=(population_num,geno_size))

    # 记录迭代中的种群样本和适应度值变化
    avg_acc_trace = []
    max_acc_trace = []
    # sim_trace = []
    scheduler = D3PMUniformMatrixScheduler(num_steps=num_step, vocab_size=4)

    # scheduler = D3PMPresetPriorMatrixScheduler(num_steps=num_step,vocab_size=4,prior_probs=[57,77,69,97])
    # scheduler = D3PMPresetPriorMatrixScheduler(num_steps=num_step, vocab_size=4,prior_probs=[85,109,51,23])
    # 计算适应度值
    accurancy, fitness = arch_fitness(
        operation_matrix=x, api=api, task=task , search_space=search_space
    )

    # 迭代去噪
    bar = tqdm.tqdm(range(num_step),ncols=120)
    for t in bar:
        generator = BayesianGenerator(x=x, fitness=fitness, q=scheduler.Q, qbar=scheduler.Qbar, t=num_step - t)
        x_next = generator.generate()

        # x_next = random_mutation(x_next, mutation_rate=0.1, vocab_size=4)
        # x_next = random_mutation(x, mutation_rate=0.1, vocab_size=4)
        # 计算适应度值
        accurancy, fitness_next = arch_fitness(
            operation_matrix=x_next, api=api, task=task, search_space=search_space
        )

        max_acc = accurancy.max().item()
        if task == "room_layout":
            max_acc = accurancy.min().item()
        avg_acc = accurancy.mean().item()

        # sim_score = population_similarity(x, x_next)["average_pairwise_similarity"]

        # 保存记录
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        # sim_trace.append(sim_score)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                # "sim": f"{sim_score:.3f}",
            }
        )

        x=select_population(x,fitness,x_next,fitness_next,population_num)
        # x=x_next
        fitness=fitness_next


    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset=task,
        )
    end_time = time.time()

    if task == "room_layout":
        accurancy, fitness = arch_fitness(
            operation_matrix=x, api=api, task=task, search_space=search_space
        )
        max_acc = min(accurancy.min().item(), max_acc)
    else:
        accurancy, fitness = arch_fitness(
            operation_matrix=x, api=api, task=task, search_space=search_space
        )
        max_acc = max(accurancy.max().item(), max_acc)

    return max_acc, end_time - start_time, x
