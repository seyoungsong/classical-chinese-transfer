import multiprocessing
import os
import random
import sys
from typing import Callable, Collection, TypeVar

from loguru import logger
from tqdm import tqdm

_PX = TypeVar("_PX")
_PY = TypeVar("_PY")

_CPU_COUNT = int(os.cpu_count())  # type: ignore


def _is_interactive() -> bool:
    # source: https://stackoverflow.com/a/64523765
    return hasattr(sys, "ps1")


def _shuffle_list(lx: list[_PX], seed: int = 42) -> list[_PX]:
    lx2 = random.Random(x=seed).sample(lx, len(lx))
    return lx2


def pool_map(
    func: Callable[[_PX], _PY],
    xs: Collection[_PX],
    desc: str = "pool",
    n_proc: int = _CPU_COUNT,
) -> list[_PY]:
    output: list[_PY] = []
    if _is_interactive():
        logger.debug("thread=1")
        for x in tqdm(xs, desc=f"{desc}_1"):
            y = func(x)
            output.append(y)
        output = _shuffle_list(output)
    else:
        logger.debug(f"thread={n_proc}")
        with multiprocessing.Pool(n_proc) as pool:
            with tqdm(total=len(xs), desc=f"{desc}_{n_proc}") as pbar:
                for y in pool.imap_unordered(func, xs):
                    pbar.update()
                    output.append(y)
    return output
