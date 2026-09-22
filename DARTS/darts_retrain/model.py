"""Evaluation networks from the official DARTS convolutional code.

This is a current-PyTorch port of ``quark0/darts/cnn/model.py``. The cell,
stems, auxiliary heads, and reduction locations match the official source.
"""

import torch
import torch.nn as nn

from .operations import FactorizedReduce, Identity, OPS, ReLUConvBN


def drop_path(inputs, drop_prob):
    """Per-example path dropout, equivalent to official DARTS ``drop_path``."""
    if drop_prob <= 0.0:
        return inputs
    keep_prob = 1.0 - drop_prob
    mask = inputs.new_empty(inputs.size(0), 1, 1, 1).bernoulli_(keep_prob)
    return inputs.div(keep_prob).mul(mask)


class Cell(nn.Module):
    def __init__(
        self, genotype, c_prev_prev, c_prev, c, reduction, reduction_prev
    ):
        super().__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(c, op_names, indices, concat, reduction)

    def _compile(self, c, op_names, indices, concat, reduction):
        if len(op_names) != len(indices) or len(op_names) % 2 != 0:
            raise ValueError("A DARTS cell must contain pairs of operation edges")
        self._steps = len(op_names) // 2
        self._concat = tuple(concat)
        self.multiplier = len(self._concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            if name not in OPS:
                raise ValueError("Unsupported DARTS operation: {}".format(name))
            stride = 2 if reduction and index < 2 else 1
            self._ops.append(OPS[name](c, stride, True))
        self._indices = tuple(indices)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for step in range(self._steps):
            h1 = states[self._indices[2 * step]]
            h2 = states[self._indices[2 * step + 1]]
            op1 = self._ops[2 * step]
            op2 = self._ops[2 * step + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            states.append(h1 + h2)
        return torch.cat([states[index] for index in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    """Official auxiliary classifier for an 8x8 CIFAR feature map."""

    def __init__(self, c, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(c, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs):
        outputs = self.features(inputs)
        return self.classifier(outputs.view(outputs.size(0), -1))


class AuxiliaryHeadImageNet(nn.Module):
    """Official auxiliary classifier for a 14x14 ImageNet feature map."""

    def __init__(self, c, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(c, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # Intentionally no BatchNorm here: the official paper experiments
            # omitted it, and the official source preserves that behavior.
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs):
        outputs = self.features(inputs)
        return self.classifier(outputs.view(outputs.size(0), -1))


class NetworkCIFAR(nn.Module):
    def __init__(self, c, num_classes, layers, auxiliary, genotype):
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0

        c_curr = 3 * c
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        reduction_prev = False
        for layer in range(layers):
            if layer in (layers // 3, 2 * layers // 3):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype,
                c_prev_prev,
                c_prev,
                c_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr
            if layer == 2 * layers // 3:
                c_to_auxiliary = c_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(c_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, inputs):
        logits_aux = None
        s0 = s1 = self.stem(inputs)
        for layer, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if layer == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        outputs = self.global_pooling(s1)
        logits = self.classifier(outputs.view(outputs.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):
    def __init__(self, c, num_classes, layers, auxiliary, genotype):
        super().__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, c // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )

        c_prev_prev, c_prev, c_curr = c, c, c
        self.cells = nn.ModuleList()
        reduction_prev = True
        for layer in range(layers):
            if layer in (layers // 3, 2 * layers // 3):
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype,
                c_prev_prev,
                c_prev,
                c_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr
            if layer == 2 * layers // 3:
                c_to_auxiliary = c_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(c_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, inputs):
        logits_aux = None
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        for layer, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if layer == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        outputs = self.global_pooling(s1)
        logits = self.classifier(outputs.view(outputs.size(0), -1))
        return logits, logits_aux
