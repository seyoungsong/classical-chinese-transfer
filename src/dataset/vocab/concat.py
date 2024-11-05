import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root
import src.corpus.dai_cc.root
import src.corpus.drri.root
import src.corpus.drs.root
import src.corpus.klc_hj.root
import src.corpus.niu_mt.root
import src.corpus.wyweb_mt.root
import src.dataset.vocab.root as sroot
from src import utils


def get_ajd() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.ajd.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.hj"].notna()
    idx.mean() * 100  # 100.0
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    rcols = {"text.hj": "text"}
    df.rename(columns=rcols, inplace=True)

    # add lang
    df["lang"] = "hj"

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

    return df


def get_drs() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.drs.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.hj"].notna()
    idx.mean() * 100  # 99.99
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    rcols = {"text.hj": "text"}
    df.rename(columns=rcols, inplace=True)

    # add lang
    df["lang"] = "hj"

    # drop cols
    [c for c in df.columns if not c.startswith("meta")]
    dcols = ["text.ko", "text_xml.hj"]
    df.drop(columns=dcols, inplace=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "drs"
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


def get_drri() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.drri.root.FILTER_PQ)
    df0.sample(1).iloc[0].to_dict()

    # hj (body)
    df = df0.copy()
    rcols = {"text_body.hj": "text"}
    df.rename(columns=rcols, inplace=True)
    df.drop(columns=["text_title.hj", "text_title.ko", "text_body.ko"], inplace=True)
    df["text.type"] = "body"
    #
    idx = df["text"].notna()
    round(idx.mean() * 100, 1)  # 76.0
    df_hkb = df[idx].reset_index(drop=True)
    df_hkb.sample(1).iloc[0].to_dict()

    # hj (title)
    df = df0.copy()
    rcols = {"text_title.hj": "text"}
    df.rename(columns=rcols, inplace=True)
    df.drop(columns=["text_body.hj", "text_body.ko", "text_title.ko"], inplace=True)
    df["text.type"] = "title"
    #
    idx = df["text"].notna()
    round(idx.mean() * 100, 1)  # 92.1
    df_hkt = df[idx].reset_index(drop=True)
    df_hkt.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_hkb, df_hkt], axis=0, ignore_index=True)

    # filter
    idx = df["text"].notna()
    idx.mean() * 100  # 99.99
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # drop cols
    [c for c in df.columns if not c.startswith("meta")]
    dcols = ["text_body_xml.hj", "text_title_xml.hj"]
    df.drop(columns=dcols, inplace=True)

    # add lang
    df["lang"] = "hj"

    # add key2 for unique sorting
    df["meta.corpus"] = "drri"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.hj"] + "|" + df["text.type"]
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


def get_klc_hj() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.klc_hj.root.FILTER2_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter
    idx = df["text.hj"].notna()
    idx.mean() * 100
    df = df[idx].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # rename cols
    rcols = {"text.hj": "text"}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # add lang
    df["lang"] = "hj"

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


def get_niu_mt() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.niu_mt.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # get cc
    df1 = df[df["text.cc"].notna()].reset_index(drop=True)
    df1.sample(1).iloc[0].to_dict()
    df1.drop(columns=["text.zh"], inplace=True)
    df1.rename(columns={"text.cc": "text"}, inplace=True)
    df1["lang"] = "cc"

    # get zh
    df2 = df[df["text.zh"].notna()].reset_index(drop=True)
    df2.sample(1).iloc[0].to_dict()
    df2.drop(columns=["text.cc"], inplace=True)
    df2.rename(columns={"text.zh": "text"}, inplace=True)
    df2["lang"] = "zh"

    # merge
    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "niu_mt"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.cc"] + "|" + df["lang"]
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


def get_wyweb_mt() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.wyweb_mt.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # get cc
    df1 = df[df["text.cc"].notna()].reset_index(drop=True)
    df1.sample(1).iloc[0].to_dict()
    df1.drop(columns=["text.zh"], inplace=True)
    df1.rename(columns={"text.cc": "text"}, inplace=True)
    df1["lang"] = "cc"

    # get zh
    df2 = df[df["text.zh"].notna()].reset_index(drop=True)
    df2.sample(1).iloc[0].to_dict()
    df2.drop(columns=["text.cc"], inplace=True)
    df2.rename(columns={"text.zh": "text"}, inplace=True)
    df2["lang"] = "zh"

    # merge
    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "wyweb_mt"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.cc"] + "|" + df["lang"]
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


def get_dai_cc() -> pd.DataFrame:
    # read
    df = utils.read_df(src.corpus.dai_cc.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # get cc
    df = df[df["text.cc"].notna()].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()
    df.rename(columns={"text.cc": "text"}, inplace=True)
    df["lang"] = "cc"

    # add key2 for unique sorting
    df["meta.corpus"] = "dai_cc"
    df["key2"] = df["meta.corpus"] + "|" + df["meta.data_id.cc"] + "|" + df["lang"]
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
    df = pd.concat(
        [
            get_ajd(),
            get_drs(),
            get_drri(),
            get_klc_hj(),
            get_niu_mt(),
            get_wyweb_mt(),
            get_dai_cc(),
        ],
        axis=0,
        ignore_index=True,
    )
    assert df["key2"].is_unique, "key2 is not unique"
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # check
    size = df.groupby(["meta.corpus", "lang"])["text"].apply(
        lambda x: x.str.len().sum()
    )
    size
    logger.debug(size)
    """
meta.corpus  lang
ajd          hj        71506930
dai_cc       cc      1689419112
drri         hj        50048623
drs          hj       291493884
klc          hj       219989761
niu_mt       cc        21818491
             zh        32475043
wyweb_mt     cc         5824228
             zh         9419409
    """

    # check
    temp1 = df.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df)


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 3.6G, 04a0e230, 5963484


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
            # python -m src.dataset.vocab.concat
            typer.run(main)
