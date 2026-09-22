"""DARTS convolutional operations.

Adapted for current PyTorch from the official DARTS implementation:
https://github.com/quark0/darts/blob/master/cnn/operations.py

The layer definitions are intentionally unchanged because changing details such
as pooling padding, BatchNorm placement, or FactorizedReduce changes the model
being evaluated.
"""

import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, inputs):
        return self.op(inputs)


class DilConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine=True,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, inputs):
        return self.op(inputs)


class SepConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, inputs):
        return self.op(inputs)


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        if self.stride == 1:
            return inputs.mul(0.0)
        return inputs[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in, c_out, affine=True):
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError("FactorizedReduce requires an even output channel count")
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(
            c_in, c_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.conv_2 = nn.Conv2d(
            c_in, c_out // 2, 1, stride=2, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, inputs):
        inputs = self.relu(inputs)
        outputs = torch.cat(
            [self.conv_1(inputs), self.conv_2(inputs[:, :, 1:, 1:])], dim=1
        )
        return self.bn(outputs)


OPS = {
    "none": lambda c, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda c, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda c, stride, affine: nn.MaxPool2d(
        3, stride=stride, padding=1
    ),
    "skip_connect": lambda c, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(c, c, affine=affine)
    ),
    "sep_conv_3x3": lambda c, stride, affine: SepConv(
        c, c, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda c, stride, affine: SepConv(
        c, c, 5, stride, 2, affine=affine
    ),
    "dil_conv_3x3": lambda c, stride, affine: DilConv(
        c, c, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda c, stride, affine: DilConv(
        c, c, 5, stride, 4, 2, affine=affine
    ),
}
