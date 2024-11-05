import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.ner.root
import src.train.ner.ajd_cc.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read file
    df = utils.read_df(src.dataset.ner.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # check: stat
    df.info()
    df.groupby(["meta.corpus", "split"], dropna=False).size()

    # filter: only ajd+cc & plain split
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["ajd", "wyweb_ner"]) & df["split"].isin(
        ["train", "valid", "test"]
    )
    idx.mean() * 100
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
    gen_dataset_file()  # 275.1M, b8000bb0, 388726


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.ner.ajd_cc.dataset
            typer.run(main)
