from utils.mapping import ReScale


def meta_arch_fitness(operation_matrix, dataset, ofa_proxy):
    assert dataset in ["cifar10", "cifar100", "aircraft", "pets"], f"Unsupported dataset: {dataset}"
    pred_acc, valid_rate = ofa_proxy.predict(operation_matrix)
    fitness = ReScale()(pred_acc)
    return pred_acc, fitness, valid_rate
