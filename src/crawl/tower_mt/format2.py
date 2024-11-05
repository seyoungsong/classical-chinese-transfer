import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.tower_mt.root as sroot
from src import utils


def gen_format2_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # add col
    df["meta.data_id"] = df["key"]

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.groupby(["meta.data_id"]).size().value_counts()
    df.sort_values(by=["meta.data_id"], inplace=True, ignore_index=True)

    # write
    utils.write_df2(sroot.FORMAT2_PQ, df)


def main() -> None:
    # rename and sort cols
    gen_format2_file()  # 397.3M, 3f177506, 373525


if __name__ == "__main__":
    tqdm.pandas()
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.tower_mt.format2
            typer.run(main)
