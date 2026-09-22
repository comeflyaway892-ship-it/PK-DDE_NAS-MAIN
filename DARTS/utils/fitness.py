"""Surrogate-only fitness evaluation for complete NAS-Bench-301 genotypes."""

import torch
from utils.NB301 import validate_tokens


def neural_predictor(operation_matrix, predictor):
    operation_matrix = operation_matrix.long()
    if operation_matrix.ndim != 2 or operation_matrix.shape[1] != 22:
        raise ValueError(
            "operation_matrix must have shape (population, 22), got {}".format(
                tuple(operation_matrix.shape)
            )
        )

    accuracies = []
    invalid = 0
    for architecture in operation_matrix:
        try:
            validate_tokens(architecture)
            accuracies.append(predictor.predict_tokens(architecture))
        except (TypeError, ValueError):
            # Invalid encodings should not be produced by the variable-vocab
            # D3PM, but keep a useful valid-rate diagnostic at this boundary.
            accuracies.append(0.0)
            invalid += 1
    accuracy_tensor = torch.tensor(accuracies, dtype=torch.float32)
    valid_rate = 1.0 - invalid / float(operation_matrix.shape[0])
    return accuracy_tensor, valid_rate


def arch_fitness(operation_matrix, predictor):
    accuracies, valid_rate = neural_predictor(operation_matrix, predictor)
    # NAS-Bench-301 surrogate scores are already accuracy percentages.  Do not
    # apply the piecewise ReScale function inherited from NAS-Bench-201: its
    # large slopes around 90--100 make tiny prediction differences dominate
    # the Bayesian temperature and population selection.
    fitness = accuracies.clone()
    return accuracies, fitness, valid_rate
