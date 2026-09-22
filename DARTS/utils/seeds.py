"""Shared CLI parsing for explicit experiment seed lists."""

import argparse


DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"


def parse_seed_list(value):
    """Parse an English-comma-separated seed list such as ``0,1,2``."""
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("--seeds must be a non-empty comma list")
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--seeds must use non-empty values separated by English commas"
        )
    try:
        seeds = [int(part.strip()) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("every seed must be an integer") from error
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds
