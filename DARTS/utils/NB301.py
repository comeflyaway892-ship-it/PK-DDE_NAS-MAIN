"""NAS-Bench-301 22-token encoding and official surrogate adapter."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from config.config import (
    nb301_cell_vocab_sizes,
    nb301_num_tokens,
    nb301_operations,
    nb301_vocab_sizes,
)


Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

P3_PAIRS = ((0, 1), (0, 2), (1, 2))
P4_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
P5_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)

_CELL_PAIR_SPECS = (
    (2, 3, 4, P3_PAIRS),
    (5, 6, 7, P4_PAIRS),
    (8, 9, 10, P5_PAIRS),
)
_OP_TO_INDEX = {name: index for index, name in enumerate(nb301_operations)}


def _as_integer_list(tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().reshape(-1).tolist()
    elif isinstance(tokens, np.ndarray):
        tokens = tokens.reshape(-1).tolist()
    else:
        tokens = list(tokens)

    if len(tokens) != nb301_num_tokens:
        raise ValueError(
            "A NAS-Bench-301 architecture must contain exactly "
            "{} tokens, got {}".format(nb301_num_tokens, len(tokens))
        )

    result = []
    for position, (value, vocab_size) in enumerate(zip(tokens, nb301_vocab_sizes)):
        integer = int(value)
        if integer != value:
            raise ValueError("Token {} is not an integer: {!r}".format(position, value))
        if not 0 <= integer < vocab_size:
            raise ValueError(
                "Token {} must be in [0, {}], got {}".format(
                    position, vocab_size - 1, integer
                )
            )
        result.append(integer)
    return result


def validate_tokens(tokens):
    """Validate the 22-token representation and return a 1-D long tensor."""
    return torch.tensor(_as_integer_list(tokens), dtype=torch.long)


def _cell_tokens_to_numeric_edges(cell_tokens):
    """Convert one 11-token cell to eight ``(predecessor, op_index)`` edges."""
    if len(cell_tokens) != len(nb301_cell_vocab_sizes):
        raise ValueError("A cell must contain exactly 11 tokens")

    edges = [(0, cell_tokens[0]), (1, cell_tokens[1])]
    for pair_pos, first_op_pos, second_op_pos, pair_table in _CELL_PAIR_SPECS:
        first_predecessor, second_predecessor = pair_table[cell_tokens[pair_pos]]
        edges.append((first_predecessor, cell_tokens[first_op_pos]))
        edges.append((second_predecessor, cell_tokens[second_op_pos]))
    return tuple(edges)


def tokens_to_numeric_architecture(tokens):
    """Return ``(normal_edges, reduction_edges)`` in the JSON numeric format."""
    values = _as_integer_list(tokens)
    return (
        _cell_tokens_to_numeric_edges(values[:11]),
        _cell_tokens_to_numeric_edges(values[11:]),
    )


def _numeric_cell_to_tokens(edges):
    if len(edges) != 8:
        raise ValueError("A DARTS cell must contain exactly eight edges")

    normalized = [(int(predecessor), int(operation)) for predecessor, operation in edges]
    if tuple(predecessor for predecessor, _ in normalized[:2]) != (0, 1):
        raise ValueError("Node 2 predecessors must be ordered as (0, 1)")

    cell_tokens = [normalized[0][1], normalized[1][1]]
    offset = 2
    for pair_table in (P3_PAIRS, P4_PAIRS, P5_PAIRS):
        first, second = normalized[offset : offset + 2]
        pair = (first[0], second[0])
        try:
            pair_token = pair_table.index(pair)
        except ValueError:
            raise ValueError(
                "Invalid or unsorted predecessor pair {} at edge offset {}".format(
                    pair, offset
                )
            )
        cell_tokens.extend((pair_token, first[1], second[1]))
        offset += 2
    return cell_tokens


def numeric_architecture_to_tokens(architecture):
    """Convert two numeric DARTS cells into the canonical 22-token tensor."""
    if len(architecture) != 2:
        raise ValueError("An architecture must contain normal and reduction cells")
    values = _numeric_cell_to_tokens(architecture[0])
    values.extend(_numeric_cell_to_tokens(architecture[1]))
    return validate_tokens(values)


def _numeric_edges_to_genotype_edges(edges):
    return [(nb301_operations[operation], predecessor) for predecessor, operation in edges]


def tokens_to_genotype(tokens):
    """Convert tokens to the Genotype accepted by the official NB301 API."""
    normal, reduce = tokens_to_numeric_architecture(tokens)
    return Genotype(
        normal=_numeric_edges_to_genotype_edges(normal),
        normal_concat=[2, 3, 4, 5],
        reduce=_numeric_edges_to_genotype_edges(reduce),
        reduce_concat=[2, 3, 4, 5],
    )


def _genotype_cell_to_numeric_edges(edges):
    if len(edges) != 8:
        raise ValueError("A DARTS genotype cell must contain exactly eight edges")
    numeric = []
    for operation, predecessor in edges:
        if operation not in _OP_TO_INDEX:
            raise ValueError("Unsupported NAS-Bench-301 operation: {}".format(operation))
        numeric.append((int(predecessor), _OP_TO_INDEX[operation]))
    return tuple(numeric)


def genotype_to_tokens(genotype):
    """Convert an official DARTS Genotype to the canonical 22-token tensor."""
    architecture = (
        _genotype_cell_to_numeric_edges(genotype.normal),
        _genotype_cell_to_numeric_edges(genotype.reduce),
    )
    return numeric_architecture_to_tokens(architecture)


def genotype_to_dict(genotype):
    """Return a JSON-serializable, lossless representation of a genotype."""
    return {
        "normal": [[operation, int(predecessor)] for operation, predecessor in genotype.normal],
        "normal_concat": [int(index) for index in genotype.normal_concat],
        "reduce": [[operation, int(predecessor)] for operation, predecessor in genotype.reduce],
        "reduce_concat": [int(index) for index in genotype.reduce_concat],
    }


class NB301Surrogate:
    """Thin cached wrapper around the official NAS-Bench-301 surrogate API."""

    def __init__(self, model, with_noise=False):
        self.model = model
        self.with_noise = bool(with_noise)
        self._cache = {}

    def predict_tokens(self, tokens):
        values = tuple(_as_integer_list(tokens))
        if not self.with_noise and values in self._cache:
            return self._cache[values]

        prediction = self.model.predict(
            config=tokens_to_genotype(values),
            representation="genotype",
            with_noise=self.with_noise,
        )
        prediction = float(np.asarray(prediction).reshape(-1)[0])
        if not self.with_noise:
            self._cache[values] = prediction
        return prediction


def load_nb301_surrogate(model_path, with_noise=False):
    """Load an official NB301 ensemble from ``model_path``.

    This function deliberately does not download models. Search only performs
    surrogate inference; installing the official ``nasbench301`` package and
    placing the pretrained ensemble on disk are explicit environment steps.
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            "NAS-Bench-301 surrogate directory does not exist: {}".format(model_path)
        )
    if not any(model_path.rglob("*surrogate_model.model")):
        raise FileNotFoundError(
            "No '*surrogate_model.model' ensemble members found under {}".format(
                model_path
            )
        )

    try:
        import nasbench301 as nb301
    except ImportError as error:
        raise RuntimeError(
            "The official 'nasbench301' package is required for surrogate search"
        ) from error

    return NB301Surrogate(
        nb301.load_ensemble(str(model_path)), with_noise=with_noise
    )


# Backward-compatible name used by the old entry point.
get_api = load_nb301_surrogate
