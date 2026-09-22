"""Small search diagnostics."""

import torch


def compute_uniqueness(architectures):
    if architectures.shape[0] == 0:
        return 0.0
    unique_count = torch.unique(architectures.long(), dim=0).shape[0]
    return float(unique_count) / float(architectures.shape[0])

