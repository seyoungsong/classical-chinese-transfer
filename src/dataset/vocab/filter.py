import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.vocab.root as sroot
from src import utils

DATA_COLS = ["text"]
DEDUP_COLS = ["meta.corpus", "lang"]


def dedup_each_corpus_lang(df: pd.DataFrame) -> pd.DataFrame:
    df_list = [
        df_group.drop_duplicates(subset=DATA_COLS)
        for _, df_group in df.groupby(["meta.corpus", "lang"])
    ]
    df1 = pd.concat(df_list, ignore_index=True)
    df1.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    vc1 = df1["meta.corpus"].value_counts().sort_index()
    vc0 = df["meta.corpus"].value_counts().sort_index()
    logger.debug((vc1 / vc0 * 100).sort_index().round(1))

    return df1


def gen_filter_file() -> None:
    # read
    df = utils.read_df(sroot.CONCAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # dedup each corpus and lang
    df = dedup_each_corpus_lang(df)

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup, prevent leakage
    gen_filter_file()  # 3.6G, a77b7873, 5423067


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
            # python -m src.dataset.vocab.filter
            typer.run(main)
