import hashlib
import sys
from importlib import reload

import numpy as np
import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty

from src import utils


def uid2split(uid: str, ratio: tuple[float, float, float]) -> str:
    uid_with_seed = f"{uid}/42"
    hex_hash = hashlib.sha256(uid_with_seed.encode()).hexdigest()
    int_hash = int(hex_hash, 16)
    rng = np.random.default_rng(int_hash)
    split: str = rng.choice(["train", "valid", "test"], p=ratio)
    return split


def deprecated_tag_subset_by_uid(
    df: pd.DataFrame, ratio: tuple[float, float, float]
) -> None:
    # check
    assert sum(ratio) == 1
    assert all(r > 0 for r in ratio)

    # split
    pandarallel.initialize(progress_bar=True)
    df["subset"] = df["uid"].parallel_apply(lambda x: uid2split(uid=x, ratio=ratio))

    # check
    assert df["subset"].isnull().sum() == 0


def main() -> None:
    pass


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            typer.run(main)
