"""Training engine for the locked official DARTS evaluation protocols."""

import json
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .data import build_loaders
from .model import NetworkCIFAR, NetworkImageNet
from utils.NB301 import genotype_to_dict, validate_tokens


WORKER_RETRY_DELAY_SECONDS = 5
WORKER_FAILURES_BEFORE_REDUCTION = 2
WORKER_RECOVERY_EPOCHS = 5


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, count=1):
        self.sum += float(value) * count
        self.count += count
        self.avg = self.sum / self.count


class CrossEntropyLabelSmooth(nn.Module):
    """Exact label-smoothed cross entropy used by official DARTS ImageNet."""

    def __init__(self, num_classes, epsilon):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        smoothed = (
            (1.0 - self.epsilon) * one_hot + self.epsilon / self.num_classes
        )
        return (-smoothed * log_probs).mean(0).sum()


def accuracy(outputs, targets, topk=(1,)):
    max_k = max(topk)
    batch_size = targets.size(0)
    _, predictions = outputs.topk(max_k, 1, True, True)
    correct = predictions.t().eq(targets.reshape(1, -1).expand(max_k, -1))
    return [
        correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size)
        for k in topk
    ]


def count_parameters(model, exclude_auxiliary=False):
    total = 0
    for name, parameter in model.named_parameters():
        if exclude_auxiliary and "auxiliary" in name:
            continue
        total += parameter.numel()
    return total


def set_random_seed(seed, gpu):
    """Use the official cuDNN policy and seed all augmentation RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(gpu)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_model(protocol, genotype):
    network = NetworkCIFAR if protocol.dataset == "cifar10" else NetworkImageNet
    return network(
        protocol.init_channels,
        protocol.num_classes,
        protocol.layers,
        protocol.auxiliary,
        genotype,
    )


def build_scheduler(protocol, optimizer):
    if protocol.scheduler == "cosine_to_zero":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=protocol.epochs, eta_min=0.0
        )
    if protocol.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=protocol.scheduler_period,
            gamma=protocol.scheduler_gamma,
        )
    if protocol.scheduler == "linear_warmup_to_zero":
        return LinearWarmupLinearDecay(
            optimizer,
            total_epochs=protocol.epochs,
            warmup_epochs=protocol.warmup_epochs,
        )
    raise ValueError("Unknown scheduler: {}".format(protocol.scheduler))


class LinearWarmupLinearDecay(torch.optim.lr_scheduler._LRScheduler):
    """PC-DARTS ImageNet warmup followed by linear decay to zero."""

    def __init__(self, optimizer, total_epochs, warmup_epochs, last_epoch=-1):
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        if self.total_epochs <= 0 or not 0 < self.warmup_epochs < self.total_epochs:
            raise ValueError("warmup_epochs must be between 0 and total_epochs")
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            scale = float(epoch + 1) / self.warmup_epochs
        else:
            scale = max(
                0.0,
                float(self.total_epochs - epoch)
                / (self.total_epochs - self.warmup_epochs),
            )
        return [base_lr * scale for base_lr in self.base_lrs]


def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _log(message, log_file):
    line = "{} {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _save_checkpoint(state, output_dir):
    """Atomically retain exactly one resumable checkpoint for a run.

    ``last.pt.tmp`` is only an in-progress write.  Replacing ``last.pt`` is
    atomic on the same filesystem, so an interruption leaves either the old
    complete checkpoint or the new complete checkpoint, never a partial one.
    """
    checkpoint_path = output_dir / "last.pt"
    temporary = output_dir / "last.pt.tmp"
    torch.save(state, str(temporary))
    os.replace(str(temporary), str(checkpoint_path))
    stale_best = output_dir / "best.pt"
    if stale_best.exists():
        stale_best.unlink()


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not torch.is_tensor(torch_state):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    # Checkpoints are loaded with map_location=cuda for the model.  The
    # default generator, however, requires its RNG state on CPU.
    torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    torch.set_rng_state(torch_state)
    if "cuda" in state:
        cuda_states = []
        for cuda_state in state["cuda"]:
            if not torch.is_tensor(cuda_state):
                cuda_state = torch.as_tensor(cuda_state, dtype=torch.uint8)
            cuda_states.append(
                cuda_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _run_epoch(
    loader,
    model,
    criterion,
    device,
    optimizer=None,
    auxiliary_weight=0.0,
    grad_clip=5.0,
    report_freq=50,
    log_file=None,
    phase="train",
    amp=False,
    grad_accumulation_steps=1,
    scaler=None,
):
    training = optimizer is not None
    if grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be positive")
    if training and amp and scaler is None:
        raise ValueError("AMP training requires a GradScaler")
    model.train(training)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    context = torch.enable_grad() if training else torch.no_grad()
    if training:
        optimizer.zero_grad()
    with context:
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits, logits_aux = model(inputs)
                loss = criterion(logits, targets)
                if training and logits_aux is not None:
                    loss = loss + auxiliary_weight * criterion(logits_aux, targets)
            if training:
                # Average gradients over the current accumulation window.  The
                # final window may contain fewer mini-batches when the dataset
                # size is not divisible by the effective batch size.
                window_size = min(
                    grad_accumulation_steps,
                    len(loader) - (step // grad_accumulation_steps) * grad_accumulation_steps,
                )
                scaled_loss = loss / float(window_size)
                if amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_update_step = (
                    (step + 1) % grad_accumulation_steps == 0
                    or step + 1 == len(loader)
                )
                if is_update_step:
                    if amp:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    if amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            precision1, precision5 = accuracy(logits, targets, topk=(1, 5))
            count = inputs.size(0)
            losses.update(loss.item(), count)
            top1.update(precision1.item(), count)
            top5.update(precision5.item(), count)
            if log_file is not None and step % report_freq == 0:
                _log(
                    "{} step={:04d} loss={:.6f} top1={:.4f} top5={:.4f}".format(
                        phase, step, losses.avg, top1.avg, top5.avg
                    ),
                    log_file,
                )
    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}


def _is_dataloader_worker_failure(error):
    """Recognize worker/shared-memory failures that can be retried safely."""
    message = str(error).lower()
    markers = (
        "dataloader worker",
        "no space left on device",
        "unable to write to file </torch_",
        "shared memory",
        "worker process",
    )
    return any(marker in message for marker in markers)


class AdaptiveWorkerController:
    """Reduce problematic loader workers and probe recovery later."""

    def __init__(self, initial_workers):
        if initial_workers < 0:
            raise ValueError("initial_workers must be non-negative")
        self.initial_workers = int(initial_workers)
        self.current_workers = int(initial_workers)
        self.consecutive_failures = 0
        self.successful_epochs = 0

    def register_failure(self):
        self.successful_epochs = 0
        self.consecutive_failures += 1
        if self.consecutive_failures < WORKER_FAILURES_BEFORE_REDUCTION:
            return self.current_workers
        self.consecutive_failures = 0
        self.current_workers = max(1, self.current_workers // 2)
        return self.current_workers

    def register_successful_epoch(self):
        self.consecutive_failures = 0
        self.successful_epochs += 1
        if (
            self.current_workers < self.initial_workers
            and self.successful_epochs >= WORKER_RECOVERY_EPOCHS
        ):
            self.successful_epochs = 0
            self.current_workers = min(
                self.initial_workers, max(1, self.current_workers * 2)
            )
        return self.current_workers


def train_official_protocol(
    protocol,
    genotype,
    tokens,
    data_path,
    output_dir,
    seed=0,
    gpu=0,
    parallel=False,
    workers=None,
    resume=None,
    allow_existing=False,
    restart=False,
):
    """Train a genotype with the locked official protocol."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official DARTS retraining requires CUDA; no CUDA device is visible"
        )
    if parallel and gpu != 0:
        raise ValueError("--parallel requires --gpu 0 as the DataParallel primary")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if restart and resume is not None:
        raise ValueError("restart and resume cannot be used together")
    if restart:
        # A requested restart is intentionally destructive only inside this
        # one run directory; architecture files and other seed runs remain.
        for stale_path in (
            output_dir / "last.pt",
            output_dir / "best.pt",
            output_dir / "last.pt.tmp",
            output_dir / "metrics.jsonl",
            output_dir / "train.log",
            output_dir / "run.json",
        ):
            if stale_path.exists():
                stale_path.unlink()
    if resume is None and not allow_existing and any(output_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty; use a new directory or --resume"
        )
    # These artifacts are never valid resume inputs.  Remove leftovers from
    # older versions/runs so each run retains only last.pt.
    for stale_path in (output_dir / "best.pt", output_dir / "last.pt.tmp"):
        if stale_path.exists():
            stale_path.unlink()
    log_file = output_dir / "train.log"
    metrics_file = output_dir / "metrics.jsonl"
    device = torch.device("cuda:{}".format(gpu))
    set_random_seed(seed, gpu)

    model = build_model(protocol, genotype)
    model = model.to(device)
    if parallel:
        model = nn.DataParallel(model)

    evaluation_criterion = nn.CrossEntropyLoss().to(device)
    if protocol.dataset == "imagenet":
        training_criterion = CrossEntropyLabelSmooth(
            protocol.num_classes, protocol.label_smoothing
        ).to(device)
    else:
        training_criterion = evaluation_criterion

    optimizer = torch.optim.SGD(
        model.parameters(),
        protocol.learning_rate,
        momentum=protocol.momentum,
        weight_decay=protocol.weight_decay,
    )
    scheduler = build_scheduler(protocol, optimizer)
    global_batch_size = protocol.batch_size
    amp_enabled = protocol.dataset == "imagenet"
    if amp_enabled:
        from .protocol import (
            IMAGENET_GRADIENT_ACCUMULATION_STEPS,
            IMAGENET_PHYSICAL_BATCH_SIZE,
        )

        physical_batch_size = IMAGENET_PHYSICAL_BATCH_SIZE
        grad_accumulation_steps = IMAGENET_GRADIENT_ACCUMULATION_STEPS
    else:
        physical_batch_size = global_batch_size
        grad_accumulation_steps = 1
    device_count = torch.cuda.device_count() if parallel else 1
    requested_workers = protocol.workers if workers is None else int(workers)
    if protocol.dataset == "imagenet" and requested_workers < 1:
        raise ValueError(
            "ImageNet DataLoader workers must be at least 1; use adaptive workers "
            "instead of setting --imagenet-workers 0"
        )
    if physical_batch_size % device_count != 0:
        raise ValueError(
            "Physical batch size {} must be divisible by {} GPUs".format(
                physical_batch_size, device_count
            )
        )
    loader_batch_size = physical_batch_size // device_count
    effective_batch_size = (
        loader_batch_size * device_count * grad_accumulation_steps
    )
    if effective_batch_size != global_batch_size:
        raise ValueError(
            "Effective batch size {} does not match protocol global batch size {}".format(
                effective_batch_size, global_batch_size
            )
        )
    train_loader, valid_loader = build_loaders(
        protocol, data_path, requested_workers, batch_size=loader_batch_size
    )
    worker_controller = AdaptiveWorkerController(requested_workers)
    loader_workers = requested_workers

    token_values = validate_tokens(tokens).tolist()
    genotype_encoding = genotype_to_dict(genotype)
    run_metadata = {
        "protocol": protocol.to_dict(),
        "tokens": token_values,
        "genotype": genotype_encoding,
        "seed": seed,
        "data": str(Path(data_path).expanduser().resolve()),
        "parallel": bool(parallel),
        "gpu": gpu,
        "workers": requested_workers,
        "workers_adaptive": bool(protocol.dataset == "imagenet"),
        "global_batch_size": global_batch_size,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "loader_batch_size_per_device": loader_batch_size,
        "device_count": device_count,
        "amp": amp_enabled,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(gpu),
    }
    metadata_path = output_dir / "run.json"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_top1 = 0.0
    if resume is not None:
        checkpoint = torch.load(
            str(Path(resume).expanduser().resolve()), map_location=device
        )
        if checkpoint.get("tokens") != token_values:
            raise ValueError("Resume checkpoint architecture tokens do not match")
        if checkpoint.get("genotype") != genotype_encoding:
            raise ValueError("Resume checkpoint genotype does not match tokens")
        if checkpoint.get("protocol") != protocol.to_dict():
            raise ValueError("Resume checkpoint protocol does not match")
        _base_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp_enabled and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_top1 = float(checkpoint["best_top1"])
        _restore_rng_state(checkpoint.get("rng_state"))

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        if existing_metadata.get("tokens") != token_values:
            raise ValueError("Output directory run.json tokens do not match")
        if existing_metadata.get("genotype") != genotype_encoding:
            raise ValueError("Output directory run.json genotype does not match tokens")
        if existing_metadata.get("protocol") != protocol.to_dict():
            raise ValueError("Output directory run.json protocol does not match")
        # These fields were added for the ImageNet AMP/micro-batch setup.
        # CIFAR checkpoints created before that setup remain valid and should
        # be resumable without rewriting their historical metadata.
        config_keys = (
            (
                "amp",
                "physical_batch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "loader_batch_size_per_device",
                "device_count",
            )
            if amp_enabled
            else ()
        )
        config_mismatch = [
            key
            for key in config_keys
            if existing_metadata.get(key) != run_metadata.get(key)
        ]
        if config_mismatch:
            if (output_dir / "last.pt").exists():
                raise ValueError(
                    "Output directory training configuration does not match: {}".format(
                        ", ".join(config_mismatch)
                    )
                )
            _write_json(metadata_path, run_metadata)
        elif existing_metadata.get("workers") != run_metadata["workers"]:
            # Changing workers is a loader/runtime change, not a model or
            # optimizer incompatibility.  Record the new requested value when
            # resuming from an existing checkpoint.
            _write_json(metadata_path, run_metadata)
    else:
        _write_json(metadata_path, run_metadata)

    comparable_parameters = count_parameters(_base_model(model), True)
    _log(
        "protocol={} dataset={} params_no_aux={:.3f}M start_epoch={}".format(
            protocol.name,
            protocol.dataset,
            comparable_parameters / 1e6,
            start_epoch,
        ),
        log_file,
    )

    report_freq = 50 if protocol.dataset == "cifar10" else 100

    def rebuild_loaders():
        nonlocal train_loader, valid_loader, loader_workers
        train_loader = None
        valid_loader = None
        gc.collect()
        train_loader, valid_loader = build_loaders(
            protocol, data_path, loader_workers, batch_size=loader_batch_size
        )

    def run_epoch_with_worker_recovery(loader_name, **kwargs):
        nonlocal loader_workers, best_top1
        while True:
            loader = train_loader if loader_name == "train" else valid_loader
            try:
                return _run_epoch(loader, **kwargs)
            except Exception as error:
                if loader_workers < 1 or not _is_dataloader_worker_failure(error):
                    raise
                previous_workers = loader_workers
                time.sleep(WORKER_RETRY_DELAY_SECONDS)
                loader = None
                optimizer_for_retry = kwargs.get("optimizer")
                if optimizer_for_retry is not None:
                    optimizer_for_retry.zero_grad()
                scaler_for_retry = kwargs.get("scaler")
                if scaler_for_retry is not None and hasattr(
                    scaler_for_retry, "_per_optimizer_states"
                ):
                    # A worker can fail after unscale_() but before update().
                    # Clear that per-optimizer bookkeeping before retrying.
                    scaler_for_retry._per_optimizer_states.clear()

                # Retry from the last completed epoch rather than applying
                # part of the failed epoch twice.  No checkpoint exists only
                # for a failure during epoch 0, where the initial state is the
                # best available recovery point.
                retry_checkpoint = output_dir / "last.pt"
                if retry_checkpoint.is_file() and optimizer_for_retry is not None:
                    checkpoint = torch.load(
                        str(retry_checkpoint), map_location=device
                    )
                    _base_model(model).load_state_dict(checkpoint["model"])
                    optimizer_for_retry.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    if (
                        scaler_for_retry is not None
                        and checkpoint.get("scaler") is not None
                    ):
                        scaler_for_retry.load_state_dict(checkpoint["scaler"])
                    _restore_rng_state(checkpoint.get("rng_state"))
                    best_top1 = float(checkpoint["best_top1"])
                    if protocol.scheduler == "linear_warmup_to_zero":
                        scheduler.step(epoch)
                loader_workers = worker_controller.register_failure()
                if loader_workers < previous_workers:
                    _log(
                        "{} worker failure persisted; reducing DataLoader workers "
                        "from {} to {} and retrying".format(
                            loader_name, previous_workers, loader_workers
                        ),
                        log_file,
                    )
                else:
                    _log(
                        "{} DataLoader worker failure; waiting and retrying with "
                        "{} workers".format(loader_name, loader_workers),
                        log_file,
                    )
                rebuild_loaders()

    for epoch in range(start_epoch, protocol.epochs):
        if protocol.scheduler == "linear_warmup_to_zero":
            scheduler.step(epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        _base_model(model).drop_path_prob = (
            protocol.drop_path_prob * epoch / protocol.epochs
        )
        _log(
            "epoch={:03d}/{} lr={:.8f} drop_path={:.6f}".format(
                epoch, protocol.epochs - 1, learning_rate,
                _base_model(model).drop_path_prob
            ),
            log_file,
        )
        train_metrics = run_epoch_with_worker_recovery(
            "train",
            model=model,
            criterion=training_criterion,
            device=device,
            optimizer=optimizer,
            auxiliary_weight=protocol.auxiliary_weight,
            grad_clip=protocol.grad_clip,
            report_freq=report_freq,
            log_file=log_file,
            phase="train",
            amp=amp_enabled,
            grad_accumulation_steps=grad_accumulation_steps,
            scaler=scaler,
        )
        valid_metrics = run_epoch_with_worker_recovery(
            "valid",
            model=model,
            criterion=evaluation_criterion,
            device=device,
            report_freq=report_freq,
            log_file=log_file,
            phase="valid",
            amp=amp_enabled,
        )
        recovered_workers = worker_controller.register_successful_epoch()
        if recovered_workers != loader_workers:
            _log(
                "DataLoader recovered after {} successful epochs; probing with "
                "{} workers".format(WORKER_RECOVERY_EPOCHS, recovered_workers),
                log_file,
            )
            loader_workers = recovered_workers
            rebuild_loaders()
        if protocol.scheduler != "linear_warmup_to_zero":
            scheduler.step()
        best_top1 = max(best_top1, valid_metrics["top1"])

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "drop_path_prob": _base_model(model).drop_path_prob,
            "train": train_metrics,
            "valid": valid_metrics,
            "best_top1": best_top1,
        }
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        _save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": _base_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "best_top1": best_top1,
                "protocol": protocol.to_dict(),
                "tokens": token_values,
                "genotype": genotype_encoding,
                "rng_state": _rng_state(),
            },
            output_dir,
        )
        _log(
            "epoch={:03d} train_top1={:.4f} valid_top1={:.4f} "
            "valid_top5={:.4f} best_top1={:.4f}".format(
                epoch,
                train_metrics["top1"],
                valid_metrics["top1"],
                valid_metrics["top5"],
                best_top1,
            ),
            log_file,
        )
    return best_top1
