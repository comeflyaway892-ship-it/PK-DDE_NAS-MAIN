import warnings
from TransNASBench101.api import TransNASBenchAPI as API
from experiments import main_exp


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path2nas_bench_file = "transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    task_list = (
        api.task_list
    )  # ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
    search_space_list = api.search_spaces  # ['macro', 'micro']
    for task in task_list:
        # for search_space in search_space_list:
        main_exp(
            task=task,
            search_space='macro',
        )

    # main_exp(
    #     task="class_scene",
    #     search_space="micro",
    # )