###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from .torch_tools import to_numpy


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: Optional[int] = 0,
):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add filter for rank
    logger.addFilter(lambda _: rank == 0)

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(lambda record: record.levelno >= logging.DEBUG)
        logger.addHandler(fh_debug)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))

def get_atomic_number_table_from_zs_data(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z.item())
    return AtomicNumberTable(sorted(list(z_set)))



def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")


# pylint: disable=abstract-method, arguments-differ
class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None
