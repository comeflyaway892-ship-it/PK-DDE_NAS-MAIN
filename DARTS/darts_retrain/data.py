"""Data pipelines matching the official DARTS evaluation scripts."""

from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Cutout:
    """Official DARTS Cutout transform."""

    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        height, width = image.size(1), image.size(2)
        mask = np.ones((height, width), np.float32)
        y = np.random.randint(height)
        x = np.random.randint(width)
        y1 = np.clip(y - self.length // 2, 0, height)
        y2 = np.clip(y + self.length // 2, 0, height)
        x1 = np.clip(x - self.length // 2, 0, width)
        x2 = np.clip(x + self.length // 2, 0, width)
        mask[y1:y2, x1:x2] = 0.0
        image *= torch.from_numpy(mask).expand_as(image)
        return image


def cifar10_transforms(cutout_length=16):
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
            Cutout(cutout_length),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    return train_transform, valid_transform


def imagenet_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, valid_transform


def build_loaders(protocol, data_path, workers=None, batch_size=None):
    """Build official DARTS train/test or train/val loaders."""
    data_path = Path(data_path).expanduser().resolve()
    worker_count = protocol.workers if workers is None else workers
    effective_batch_size = protocol.batch_size if batch_size is None else int(batch_size)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if worker_count < 0:
        raise ValueError("workers must be non-negative")

    if protocol.dataset == "cifar10":
        train_transform, valid_transform = cifar10_transforms(
            protocol.cutout_length
        )
        train_data = datasets.CIFAR10(
            root=str(data_path), train=True, download=True, transform=train_transform
        )
        valid_data = datasets.CIFAR10(
            root=str(data_path), train=False, download=True, transform=valid_transform
        )
    else:
        train_dir = data_path / "train"
        valid_dir = data_path / "val"
        if not train_dir.is_dir() or not valid_dir.is_dir():
            raise FileNotFoundError(
                "ImageNet must use ImageFolder layout with '{}' and '{}'".format(
                    train_dir, valid_dir
                )
            )
        train_transform, valid_transform = imagenet_transforms()
        train_data = datasets.ImageFolder(str(train_dir), train_transform)
        valid_data = datasets.ImageFolder(str(valid_dir), valid_transform)
        if len(train_data.classes) != protocol.num_classes:
            raise ValueError(
                "Expected 1000 ImageNet train classes, found {}".format(
                    len(train_data.classes)
                )
            )
        if train_data.class_to_idx != valid_data.class_to_idx:
            raise ValueError("ImageNet train/val class mappings do not match")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=effective_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=worker_count,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=effective_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=worker_count,
    )
    return train_loader, valid_loader
