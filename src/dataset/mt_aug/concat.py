import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.niu_mt.root
import src.corpus.wyweb_mt.root
import src.dataset.mt_aug.root as sroot
from src import utils


def get_niu_mt() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.niu_mt.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # split: cc-zh

    # cc-zh
    idx = df["text.cc"].notna() & df["text.zh"].notna()
    idx.mean() * 100  # 100.0
    df_cz = df[idx].reset_index(drop=True)
    df_cz.sample(1).iloc[0].to_dict()
    #
    [c for c in df_cz.columns if not c.startswith("meta")]
    rcols = {"text.cc": "text.src", "text.zh": "text.tgt"}
    df_cz.rename(columns=rcols, inplace=True)
    #
    df_cz["lang.src"] = "cc"
    df_cz["lang.tgt"] = "zh"
    df_cz.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_cz], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "niu_mt"
    df["key2"] = (
        df["meta.corpus"]
        + "|"
        + df["meta.data_id.cc"]
        + "|"
        + df["lang.src"]
        + "|"
        + df["lang.tgt"]
    )
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
    cols = [c for c in df.columns if "xml" in c]
    df.drop(columns=cols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    return df


def get_wyweb_mt() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.wyweb_mt.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # split: cc-zh

    # cc-zh
    idx = df["text.cc"].notna() & df["text.zh"].notna()
    idx.mean() * 100  # 100.0
    df_cz = df[idx].reset_index(drop=True)
    df_cz.sample(1).iloc[0].to_dict()
    #
    [c for c in df_cz.columns if not c.startswith("meta")]
    rcols = {"text.cc": "text.src", "text.zh": "text.tgt"}
    df_cz.rename(columns=rcols, inplace=True)
    #
    df_cz["lang.src"] = "cc"
    df_cz["lang.tgt"] = "zh"
    df_cz.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_cz], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "wyweb_mt"
    df["key2"] = (
        df["meta.corpus"]
        + "|"
        + df["meta.data_id.cc"]
        + "|"
        + df["lang.src"]
        + "|"
        + df["lang.tgt"]
    )
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
    cols = [c for c in df.columns if "xml" in c]
    df.drop(columns=cols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    return df


def gen_concat_file() -> None:
    # get
    df_niu = get_niu_mt()
    df_niu.sample(1).iloc[0].to_dict()
    df_niu.groupby(["split", "lang.src", "lang.tgt"]).size()

    # get
    df_wyw = get_wyweb_mt()
    df_wyw.sample(1).iloc[0].to_dict()
    df_wyw.groupby(["split", "lang.src", "lang.tgt"]).size()

    # concat
    df_cat = pd.concat([df_niu, df_wyw], axis=0, ignore_index=True)
    assert df_cat["key2"].is_unique, "key2 is not unique"
    df_cat.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # check
    size = df_cat.groupby(["split", "lang.src", "lang.tgt", "meta.corpus"]).size()
    size
    logger.debug(size)
    """
split  lang.src  lang.tgt  meta.corpus
test   cc        zh        niu_mt          97231
                           wyweb_mt        19985
train  cc        zh        niu_mt         777838
                           wyweb_mt       226548
valid  cc        zh        niu_mt          97398
                           wyweb_mt        19981
    """

    # check
    temp1 = df_cat.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df_cat)


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 124.1M, 46151700, 1238981


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
            # python -m src.dataset.mt_aug.concat
            typer.run(main)
