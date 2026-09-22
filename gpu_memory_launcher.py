#!/usr/bin/env python3
"""Wait for GPU memory to become available, then run a PyTorch script in-process.

The target must be a Python file. Running it in the same interpreter is
intentional: CUDA memory cached by this launcher can then be reused by the
target's PyTorch allocator. A separate child process cannot reuse that cache.
"""

import argparse
import gc
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time


MIB = 1024 * 1024
RESERVATION_HEADROOM = 32 * MIB


class ReservationUnavailable(RuntimeError):
    """The requested cache size cannot currently be allocated."""


def _log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _split_arguments(argv):
    if "--" not in argv:
        raise ValueError(
            "missing '--' separator; expected: gpu_memory_launcher.py [options] -- script.py [args]"
        )
    separator = argv.index("--")
    launcher_argv = argv[:separator]
    target_argv = argv[separator + 1 :]
    if not target_argv:
        raise ValueError("a target Python script is required after '--'")
    return launcher_argv, target_argv


def _build_parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --gpu GPU [options] -- script.py [script args]",
        description=(
            "Poll GPU memory, reserve a reusable PyTorch CUDA cache, and run a "
            "Python script in the same process."
        )
    )
    parser.add_argument("--gpu", required=True, help="physical GPU index or UUID")
    parser.add_argument(
        "--start-below",
        type=float,
        default=50.0,
        help="start only when global GPU memory usage is below this percentage (default: 50)",
    )
    parser.add_argument(
        "--reserve-at-least",
        type=float,
        default=90.0,
        help="keep this process's reusable PyTorch reservation at or above this percentage (default: 90)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="polling and maintenance interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=256,
        help="maximum size of each cache-priming allocation in MiB (default: 256)",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="working directory for the target script (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the launch configuration without querying or allocating a GPU",
    )
    return parser


def _validate_args(parser, args):
    if not 0.0 < args.start_below < 100.0:
        parser.error("--start-below must be between 0 and 100")
    if not 0.0 < args.reserve_at_least < 100.0:
        parser.error("--reserve-at-least must be between 0 and 100")
    if args.reserve_at_least <= args.start_below:
        parser.error("--reserve-at-least must be greater than --start-below")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")

    args.cwd = args.cwd.expanduser().resolve()
    if not args.cwd.is_dir():
        parser.error(f"--cwd is not a directory: {args.cwd}")


def _resolve_script(parser, cwd, target_argv):
    script = Path(target_argv[0]).expanduser()
    if not script.is_absolute():
        script = cwd / script
    script = script.resolve()
    if not script.is_file():
        parser.error(f"target script does not exist: {script}")
    if script.suffix.lower() != ".py":
        parser.error("target must be a .py file so it can share the PyTorch CUDA cache")
    return script


def _query_gpu_memory(gpu):
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"nvidia-smi failed for GPU {gpu}: {detail}") from exc

    line = result.stdout.strip().splitlines()
    if len(line) != 1:
        raise RuntimeError(f"unexpected nvidia-smi output for GPU {gpu}: {result.stdout!r}")
    try:
        used_mib, total_mib = (float(value.strip()) for value in line[0].split(","))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot parse nvidia-smi output: {line[0]!r}") from exc
    if total_mib <= 0:
        raise RuntimeError(f"invalid total memory reported for GPU {gpu}: {total_mib} MiB")
    return used_mib, total_mib


def _wait_until_available(gpu, threshold, interval):
    while True:
        used_mib, total_mib = _query_gpu_memory(gpu)
        percent = used_mib * 100.0 / total_mib
        _log(
            f"GPU {gpu}: {used_mib:.0f}/{total_mib:.0f} MiB used "
            f"({percent:.2f}%); waiting for < {threshold:g}%"
        )
        if percent < threshold:
            return
        time.sleep(interval)


def _cuda_memory(torch):
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    reserved_bytes = torch.cuda.memory_reserved(0)
    return free_bytes, total_bytes, reserved_bytes


def _prime_torch_cache(torch, reserve_percent, chunk_bytes):
    """Reserve the target fraction, then turn live tensors into reusable cache."""
    held_tensors = []
    try:
        while True:
            free_bytes, total_bytes, reserved_bytes = _cuda_memory(torch)
            target_bytes = math.ceil(total_bytes * reserve_percent / 100.0)
            missing_bytes = target_bytes - reserved_bytes
            if missing_bytes <= 0:
                break

            # Add a small margin so allocator rounding cannot leave the result
            # fractionally below the requested percentage.
            allocation_bytes = min(chunk_bytes, missing_bytes + MIB)
            if allocation_bytes > free_bytes:
                raise ReservationUnavailable(
                    f"only {free_bytes / MIB:.0f} MiB is free, but another "
                    f"{allocation_bytes / MIB:.0f} MiB allocation is required"
                )
            held_tensors.append(
                torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")
            )
        torch.cuda.synchronize(0)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raise ReservationUnavailable(
            f"could not raise this process's GPU reservation to {reserve_percent:g}%; "
            "another process may have allocated memory after the availability check"
        ) from exc
    finally:
        # Dropping live references makes these allocator blocks available to
        # the target while PyTorch keeps the CUDA reservation for this process.
        held_tensors.clear()
        gc.collect()

    free_bytes, total_bytes, reserved_bytes = _cuda_memory(torch)
    reserved_percent = reserved_bytes * 100.0 / total_bytes
    if reserved_percent + 1e-6 < reserve_percent:
        raise RuntimeError(
            f"PyTorch did not retain the requested CUDA cache: process reservation is "
            f"{reserved_percent:.2f}%, requested at least {reserve_percent:g}%"
        )
    global_used_bytes = total_bytes - free_bytes
    return reserved_bytes, global_used_bytes, total_bytes


def _wait_until_reservable(torch, reserve_percent, interval):
    """Wait until the process can reserve the requested fraction of the GPU."""
    while True:
        free_bytes, total_bytes, reserved_bytes = _cuda_memory(torch)
        target_bytes = math.ceil(total_bytes * reserve_percent / 100.0)
        missing_bytes = max(target_bytes - reserved_bytes, 0)
        global_used_percent = (total_bytes - free_bytes) * 100.0 / total_bytes
        if free_bytes >= missing_bytes + RESERVATION_HEADROOM:
            return
        _log(
            f"GPU has passed the start threshold, but {free_bytes / MIB:.0f} MiB "
            f"is free and this process still needs {missing_bytes / MIB:.0f} MiB "
            f"to reserve {reserve_percent:g}% (global usage {global_used_percent:.2f}%); "
            f"checking again in {interval:g}s"
        )
        time.sleep(interval)


def _maintain_cache(torch, reserve_percent, chunk_bytes, interval, stop_event):
    while not stop_event.wait(interval):
        try:
            free_bytes, total_bytes, reserved_bytes = _cuda_memory(torch)
            reserved_percent = reserved_bytes * 100.0 / total_bytes
            global_used_bytes = total_bytes - free_bytes
            if reserved_percent < reserve_percent:
                _log(
                    f"process reservation fell to {reserved_percent:.2f}%; replenishing reusable "
                    f"cache to >= {reserve_percent:g}%"
                )
                reserved_bytes, global_used_bytes, total_bytes = _prime_torch_cache(
                    torch,
                    reserve_percent,
                    chunk_bytes,
                )
                reserved_percent = reserved_bytes * 100.0 / total_bytes
            global_used_percent = global_used_bytes * 100.0 / total_bytes
            _log(
                f"GPU cache guard: process reserved {reserved_percent:.2f}%, "
                f"global usage {global_used_percent:.2f}%"
            )
        except Exception as exc:  # Keep the target alive so it can save/exit cleanly.
            _log(f"WARNING: GPU cache guard could not replenish memory: {exc}")


def _run_target(script, target_args, cwd):
    os.chdir(str(cwd))
    sys.argv = [str(script)] + list(target_args)
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name="__main__")


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    if raw_argv in (["-h"], ["--help"]):
        parser.parse_args(raw_argv)
    try:
        launcher_argv, target_argv = _split_arguments(raw_argv)
    except ValueError as exc:
        parser.error(str(exc))

    args = parser.parse_args(launcher_argv)
    _validate_args(parser, args)
    script = _resolve_script(parser, args.cwd, target_argv)
    target_args = target_argv[1:]

    _log(
        f"target={script} gpu={args.gpu} start_below={args.start_below:g}% "
        f"reserve_at_least={args.reserve_at_least:g}% interval={args.interval:g}s"
    )
    if args.dry_run:
        _log("dry-run complete; no GPU was queried or allocated")
        return

    _wait_until_available(args.gpu, args.start_below, args.interval)

    # This must happen before importing torch so logical cuda:0 maps to the
    # requested physical device for the lifetime of this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["DDE_NAG_GPU"] = "0"
    os.environ["EDNAG_GPU"] = "0"
    os.environ["DDE_NAG_PHYSICAL_GPU"] = str(args.gpu)
    os.environ["DDE_NAG_GPU_MEMORY_LAUNCHER"] = "1"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available after selecting physical GPU {args.gpu}")
    torch.cuda.set_device(0)

    while True:
        _wait_until_reservable(torch, args.reserve_at_least, args.interval)
        try:
            reserved_bytes, global_used_bytes, total_bytes = _prime_torch_cache(
                torch,
                args.reserve_at_least,
                args.chunk_mib * MIB,
            )
            break
        except ReservationUnavailable as exc:
            # The target is not loaded yet, so it is safe to release a partial
            # reservation and retry if another process won the allocation race.
            torch.cuda.empty_cache()
            _log(f"reservation race: {exc}; retrying in {args.interval:g}s")
            time.sleep(args.interval)
    _log(
        f"GPU {args.gpu} ready: this process reserved "
        f"{reserved_bytes * 100.0 / total_bytes:.2f}% "
        f"({reserved_bytes / MIB:.0f} MiB) as reusable PyTorch cache; "
        f"global usage is {global_used_bytes * 100.0 / total_bytes:.2f}%"
    )

    # Prevent target code from explicitly returning the shared cache to CUDA.
    # Live tensors can still be freed and their blocks immediately reused.
    original_empty_cache = torch.cuda.empty_cache

    def _keep_cache_reserved():
        _log("ignored torch.cuda.empty_cache() to keep the reusable reservation")

    torch.cuda.empty_cache = _keep_cache_reserved
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_maintain_cache,
        args=(
            torch,
            args.reserve_at_least,
            args.chunk_mib * MIB,
            args.interval,
            stop_event,
        ),
        name="gpu-cache-guard",
        daemon=True,
    )
    monitor.start()

    _log(f"starting target script with args: {target_args}")
    try:
        _run_target(script, target_args, args.cwd)
    finally:
        stop_event.set()
        monitor.join(timeout=max(args.interval, 1.0) + 5.0)
        torch.cuda.empty_cache = original_empty_cache
        _log("target finished; CUDA reservation will be released when this process exits")


if __name__ == "__main__":
    main()
