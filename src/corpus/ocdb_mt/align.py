import json
import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ocdb_mt.root as sroot
import src.crawl.ocdb_cc_ko.root
from src import utils


def clean_mt(df_mt0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_mt0.copy()
    df.sample(1).iloc[0].to_dict()

    # filter by count
    x = df.sample(1).iloc[0].to_dict()
    len(json.loads(x["text.cc"]))
    df["len.cc"] = df["text.cc"].progress_apply(lambda x: len(json.loads(x)))
    df["len.ko"] = df["text.ko"].progress_apply(lambda x: len(json.loads(x)))
    idx1 = df["len.cc"] > 0
    idx2 = df["len.ko"] > 0
    idx3 = df["len.cc"] == df["len.ko"]
    idx = idx1 & idx2 & idx3
    idx.mean()  # 0.85
    df = df[idx].reset_index(drop=True)
    df.drop(columns=["len.cc", "len.ko"], inplace=True)

    # concat text
    df["text.cc"] = df["text.cc"].progress_apply(lambda x: " ".join(json.loads(x)))
    df["text.ko"] = df["text.ko"].progress_apply(lambda x: " ".join(json.loads(x)))

    # empty
    df = utils.replace_blank_to_none(df)
    df.info()
    idx = df["text.cc"].isna() | df["text.ko"].isna()
    idx.mean() * 100  # 0.3%
    df = df[~idx].reset_index(drop=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["meta.data_id"].is_unique
    df.sort_values(by="meta.data_id", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df_mt0 = utils.read_df(src.crawl.ocdb_cc_ko.root.FORMAT2_PQ)

    # check
    df_mt0.sample(1).iloc[0].to_dict()  # ITKC_ST_U0_A03_08A_27A_00130
    len(df_mt0)  # 28341

    # get ko
    df_mt = clean_mt(df_mt0=df_mt0)
    df_mt.sample(1).iloc[0].to_dict()

    # check
    temp1 = df_mt.isna().sum()
    temp1[temp1 > 0]
    df_mt.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_mt)


def main() -> None:
    # align samples
    gen_align_file()  # 22.9M, 742e719e, 23940


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
            # python -m src.corpus.ocdb_mt.align
            typer.run(main)
