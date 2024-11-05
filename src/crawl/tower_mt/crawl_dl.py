import sys
from importlib import reload

import typer
from datasets import load_dataset
from loguru import logger
from rich import pretty

import src.crawl.tower_mt.root as sroot
from src import utils


def gen_hface_file() -> None:
    # read
    ds0 = load_dataset("Unbabel/TowerBlocks-v0.2")
    ds = ds0["train"]

    # check
    ds.shuffle()[0]

    # write
    sroot.HFACE_PQ.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(sroot.HFACE_PQ, compression="gzip")


def main() -> None:
    # download from hface and save to file
    gen_hface_file()
    utils.log_written(sroot.HFACE_PQ)  # 439.3M, 5e73cad4


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.tower_mt.crawl_dl
            typer.run(main)
