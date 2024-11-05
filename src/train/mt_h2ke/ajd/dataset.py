import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.dataset.mt_h2ke.root
import src.train.mt_h2ke.ajd.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read file
    df = utils.read_df(src.dataset.mt_h2ke.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # check: stat
    df.info()
    df.groupby(["meta.corpus", "lang.src", "lang.tgt", "split"], dropna=False).size()

    # filter: only ajd
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["ajd"])
    df = df[idx].reset_index(drop=True)
    logger.debug(df["meta.corpus"].value_counts())

    # check: no empty text
    df = utils.replace_blank_to_none(df)
    cols = [c for c in df.columns if not c.startswith("meta")]
    assert df[cols].isnull().sum().sum() == 0, "no empty text"
    df.dropna(axis=1, how="all", inplace=True)

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 282.5M, 0365e134, 416819


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_h2ke.ajd.dataset
            typer.run(main)
