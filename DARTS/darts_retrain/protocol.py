"""Locked hyperparameters from the official DARTS evaluation scripts."""

from dataclasses import asdict, dataclass


PROTOCOL_NAME = "darts_official_2018"
OFFICIAL_REPOSITORY = "https://github.com/quark0/darts"
IMAGENET_PHYSICAL_BATCH_SIZE = 512
IMAGENET_GRADIENT_ACCUMULATION_STEPS = 2


@dataclass(frozen=True)
class DARTSProtocol:
    name: str
    dataset: str
    num_classes: int
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    init_channels: int
    layers: int
    auxiliary: bool
    auxiliary_weight: float
    drop_path_prob: float
    grad_clip: float
    workers: int
    scheduler: str
    scheduler_gamma: float = 0.0
    scheduler_period: int = 0
    cutout: bool = False
    cutout_length: int = 0
    label_smoothing: float = 0.0
    warmup_epochs: int = 0

    def to_dict(self):
        return asdict(self)


CIFAR10_PROTOCOL = DARTSProtocol(
    name=PROTOCOL_NAME,
    dataset="cifar10",
    num_classes=10,
    epochs=600,
    batch_size=96,
    learning_rate=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    init_channels=36,
    layers=20,
    auxiliary=True,
    auxiliary_weight=0.4,
    drop_path_prob=0.2,
    grad_clip=5.0,
    workers=2,
    scheduler="cosine_to_zero",
    cutout=True,
    cutout_length=16,
)


IMAGENET_PROTOCOL = DARTSProtocol(
    name="pc_darts_2020",
    dataset="imagenet",
    num_classes=1000,
    epochs=250,
    batch_size=1024,
    learning_rate=0.5,
    momentum=0.9,
    weight_decay=3e-5,
    init_channels=48,
    layers=14,
    auxiliary=True,
    auxiliary_weight=0.4,
    drop_path_prob=0.0,
    grad_clip=5.0,
    workers=4,
    scheduler="linear_warmup_to_zero",
    label_smoothing=0.1,
    warmup_epochs=5,
)


def get_protocol(dataset):
    normalized = dataset.lower().replace("-", "")
    if normalized == "cifar10":
        return CIFAR10_PROTOCOL
    if normalized in ("imagenet", "imagenet1k"):
        return IMAGENET_PROTOCOL
    raise ValueError("dataset must be 'cifar10' or 'imagenet'")
