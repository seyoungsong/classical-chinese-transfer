import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root
import src.corpus.klc_mt.root
import src.corpus.wyweb_punc.root
import src.dataset.punc.root as sroot
from src import utils


def get_ajd() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.ajd.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.hj"].notna()
    idx.mean() * 100  # 100.0
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    rcols = {"text.hj": "text"}
    df.rename(columns=rcols, inplace=True)

    # drop cols
    [c for c in df.columns if not c.startswith("meta")]
    dcols = [
        "text.cko",
        "text.en",
        "text.oko",
        "text_xml.en",
        "text_xml.hj",
        "text_xml.oko",
    ]
    df.drop(columns=dcols, inplace=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "ajd"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.hj"]
    assert df["key2"].is_unique
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    if 0:
        df.sample(1).iloc[0].to_dict()
        cols = [c for c in df.columns if "xml" in c]
        df.drop(columns=cols, inplace=True)

    return df


def get_klc_mt() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.klc_mt.root.FILTER2_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.hj"].notna() & df["is_punc.hj"]
    idx.mean() * 100  # 26.2
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    [c for c in df.columns if not c.startswith("meta")]
    dcols = ["is_punc.hj", "text.ko", "text_xml.hj", "text_xml.ko"]
    df.drop(columns=dcols, inplace=True)

    # rename cols
    rcols = {"text.hj": "text"}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # add key2 for unique sorting
    df["meta.corpus"] = "klc"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.hj"]
    assert df["key2"].is_unique
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    if 0:
        df.sample(1).iloc[0].to_dict()
        cols = [c for c in df.columns if "xml" in c]
        df.drop(columns=cols, inplace=True)

    # check
    if 0:
        df["meta.book_extra.hj"].value_counts()
        df["meta.data_id.hj"].str[:7].value_counts()
        df["split"].value_counts().sort_index()

    return df


def get_wyweb_punc() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.wyweb_punc.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.cc"].notna()
    idx.mean() * 100  # 100
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    if 0:
        [c for c in df.columns if not c.startswith("meta")]
        dcols = ["is_punc.hj", "text.ko", "text_xml.hj", "text_xml.ko"]
        df.drop(columns=dcols, inplace=True)

    # rename cols
    rcols = {"text.cc": "text", "text_xml.cc": "meta.text_xml.cc"}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # add key2 for unique sorting
    df["meta.corpus"] = "wyweb_punc"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.cc"]
    assert df["key2"].is_unique
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    if 0:
        df.sample(1).iloc[0].to_dict()
        cols = [c for c in df.columns if "xml" in c]
        df.drop(columns=cols, inplace=True)

    return df


def gen_concat_file() -> None:
    # concat
    df_cat = pd.concat(
        [get_ajd(), get_klc_mt(), get_wyweb_punc()], axis=0, ignore_index=True
    )
    assert df_cat["key2"].is_unique, "key2 is not unique"
    df_cat.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # check
    size = df_cat.groupby(["split", "meta.corpus"]).size()
    size
    logger.debug(size)
    """
split   meta.corpus
test    ajd             41008
        klc              1885
        wyweb_punc      30839
test2   klc              2302
train   ajd            330930
        klc             14450
        wyweb_punc      71495
train2  klc             18405
valid   ajd             41385
        klc              1799
        wyweb_punc      32800
valid2  klc              2247
    """

    # check
    temp1 = df_cat.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df_cat)


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 206.7M, e8dc7c70, 589545


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
            # python -m src.dataset.punc.concat
            typer.run(main)
