import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd_ner.root
import src.corpus.klc_ner.root
import src.corpus.wyweb_ner.root
import src.dataset.ner.root as sroot
from src import utils


def get_ajd_ner() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.ajd_ner.root.FORMAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # move text to meta
    cols = [c for c in df.columns if "text" in c and "xml" not in c]
    cols += ["text_xml.oko"]
    rcols = {c: f"meta.{c}" for c in cols}
    df.rename(columns=rcols, inplace=True)

    # filter
    idx = df["text_xml.hj"].notna()
    idx.mean() * 100  # 5.3
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename and drop cols
    [c for c in df.columns if not c.startswith("meta")]
    rcols = {"text_xml.hj": "text_xml"}
    df.rename(columns=rcols, inplace=True)
    df.drop(columns=["meta.text_xml.oko", "meta.text.hj"], inplace=True)
    df.sample(1).iloc[0].to_dict()

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

    return df


def get_klc_ner() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.klc_ner.root.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text_xml.hj"].notna()
    idx.mean() * 100
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename and drop cols
    [c for c in df.columns if not c.startswith("meta")]
    rcols = {
        "text_xml.hj": "text_xml",
        "is_punc.hj": "meta.is_punc.hj",
        "text_xml.ko": "meta.text_xml.ko",
    }
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

    return df


def get_wyweb_ner() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.wyweb_ner.root.FILTER_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text_xml.cc"].notna()
    idx.mean() * 100
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename and drop cols
    [c for c in df.columns if not c.startswith("meta")]
    rcols = {"text_xml.cc": "text_xml"}
    df.rename(columns=rcols, inplace=True)
    df.drop(columns=["text.cc"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # add key2 for unique sorting
    df["meta.corpus"] = "wyweb_ner"
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

    return df


def gen_concat_file() -> None:
    # concat
    df_cat = pd.concat(
        [get_ajd_ner(), get_klc_ner(), get_wyweb_ner()], axis=0, ignore_index=True
    )
    assert df_cat["key2"].is_unique, "key2 is not unique"
    df_cat.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # check
    size = df_cat.groupby(["split", "meta.corpus"]).size()
    size
    logger.debug(size)
    """
split   meta.corpus
test    ajd             40993
        klc              1035
        wyweb_ner        2000
test2   klc              1154
train   ajd            330799
        klc              8036
        wyweb_ner       14762
train2  klc              9297
valid   ajd             41373
        klc               995
        wyweb_ner        2000
valid2  klc              1140
    """

    # check
    temp1 = df_cat.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df_cat)


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 339.5M, 7b8e1410, 453584


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
            # python -m src.dataset.ner.concat
            typer.run(main)
