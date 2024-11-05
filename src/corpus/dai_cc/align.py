import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.dai_cc.root as sroot
import src.crawl.dai_cc.root
from src import utils


def clean_mt(df0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # rename
    {k: k for k in df.columns}
    rcols = {
        "text": "text.cc",
        "url": "meta.url",
        "meta.book_title": "meta.book_title.cc",
        "meta.data_id": "meta.data_id.cc",
    }
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["meta.data_id.cc"].is_unique
    df.sort_values(by="meta.data_id.cc", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df0 = utils.read_df(src.crawl.dai_cc.root.FORMAT2_PQ)

    # check
    df0.sample(1).iloc[0].to_dict()
    len(df0)  # 266514

    # get cc
    df = clean_mt(df0=df0)
    df.sample(1).iloc[0].to_dict()

    # check
    temp1 = df.isna().sum()
    temp1[temp1 > 0]
    df.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df)


def main() -> None:
    # align samples
    gen_align_file()  # 2.5G, 8f3841e4, 15694


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
            # python -m src.corpus.dai_cc.align
            typer.run(main)
