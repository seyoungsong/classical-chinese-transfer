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
import src.corpus.niu_mt_ko.root
import src.corpus.wyweb_mt_ko.root
import src.dataset.mt_h2ke.root as sroot
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

    # gen ko by overwrite oko with cko if possible
    df["meta.lang.ko"] = "oko"
    df.rename(columns={"text.oko": "text.ko"}, inplace=True)
    idx = df["text.cko"].notna()
    if 0:
        idx.mean() * 100  # 12.3%
        df[idx].sample(1).iloc[0].to_dict()
        [c for c in df.columns if c.endswith("cko")]
    df.loc[idx, "text.ko"] = df.loc[idx, "text.cko"]
    df.loc[idx, "meta.lang.ko"] = "cko"
    df.drop(columns=["text.cko"], inplace=True)
    # check
    df.sample(1).iloc[0].to_dict()
    df["meta.lang.ko"].value_counts() / len(df) * 100  # 12.3%
    df[df["meta.lang.ko"] == "cko"].sample(1).iloc[0].to_dict()

    # move text_xml to meta
    cols = [c for c in df.columns if "xml" in c]
    rcols = {c: f"meta.{c}" for c in cols}
    df.rename(columns=rcols, inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # split: hj-ko, hj-en, ko-en

    # hj-ko
    idx = df["text.hj"].notna() & df["text.ko"].notna()
    idx.mean() * 100  # 99.9
    df_hk = df[idx].reset_index(drop=True)
    df_hk.sample(1).iloc[0].to_dict()
    #
    [c for c in df_hk.columns if not c.startswith("meta")]
    rcols = {"text.hj": "text.src", "text.ko": "text.tgt"}
    df_hk.rename(columns=rcols, inplace=True)
    df_hk.drop(columns=["text.en"], inplace=True)
    #
    df_hk["lang.src"] = "hj"
    df_hk["lang.tgt"] = "ko"
    df_hk.sample(1).iloc[0].to_dict()

    # hj-en
    idx = df["text.hj"].notna() & df["text.en"].notna()
    idx.mean() * 100  # 5.3
    df_he = df[idx].reset_index(drop=True)
    df_he.sample(1).iloc[0].to_dict()
    #
    [c for c in df_he.columns if not c.startswith("meta")]
    rcols = {"text.hj": "text.src", "text.en": "text.tgt"}
    df_he.rename(columns=rcols, inplace=True)
    df_he.drop(columns=["text.ko"], inplace=True)
    #
    df_he["lang.src"] = "hj"
    df_he["lang.tgt"] = "en"
    df_he.sample(1).iloc[0].to_dict()

    # ko-en
    idx = df["text.ko"].notna() & df["text.en"].notna()
    idx.mean() * 100  # 5.3
    df_ke = df[idx].reset_index(drop=True)
    df_ke.sample(1).iloc[0].to_dict()
    #
    [c for c in df_ke.columns if not c.startswith("meta")]
    rcols = {"text.ko": "text.src", "text.en": "text.tgt"}
    df_ke.rename(columns=rcols, inplace=True)
    df_ke.drop(columns=["text.hj"], inplace=True)
    #
    df_ke["lang.src"] = "ko"
    df_ke["lang.tgt"] = "en"
    df_ke.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_hk, df_he, df_ke], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "ajd"
    df["key2"] = (
        df["meta.corpus"]
        + "|"
        + df["meta.data_id.hj"]
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
    df.sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if "xml" in c]
    df.drop(columns=cols, inplace=True)

    return df


def get_klc() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.klc_mt.root.FILTER2_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # filter: drop train2/valid2/test2
    idx = df["split"].isin(["train", "valid", "test"])
    idx.mean() * 100  # 42.6
    df = df[idx].reset_index(drop=True)

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()

    # move text_xml to meta
    cols = [c for c in df.columns if "xml" in c]
    rcols = {c: f"meta.{c}" for c in cols}
    df.rename(columns=rcols, inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # hj-ko
    idx = df["text.hj"].notna() & df["text.ko"].notna()
    idx.mean() * 100  # 100.0
    df_hk = df[idx].reset_index(drop=True)
    df_hk.sample(1).iloc[0].to_dict()
    #
    [c for c in df_hk.columns if not c.startswith("meta")]
    rcols = {
        "text.hj": "text.src",
        "text.ko": "text.tgt",
        "is_punc.hj": "meta.is_punc.hj",
    }
    df_hk.rename(columns=rcols, inplace=True)
    #
    df_hk["lang.src"] = "hj"
    df_hk["lang.tgt"] = "ko"
    df_hk.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_hk], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "klc"
    df["key2"] = (
        df["meta.corpus"]
        + "|"
        + df["meta.data_id.hj"]
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

    # check is_punc
    if 0:
        df["meta.is_punc.hj"].value_counts() / len(df) * 100  # 26.2%
        df.groupby(["meta.is_punc.hj", "split"]).size()
        df.groupby(["meta.is_punc.hj"]).sample(1).to_dict()

    return df


def get_niu() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.niu_mt_ko.root.FORMAT_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()
        df["split"].value_counts() / len(df) * 100

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # cc-ko
    idx = df["text.cc"].notna() & df["text.ko"].notna()
    idx.mean() * 100  # 99.94
    df_hk = df[idx].reset_index(drop=True)
    df_hk.sample(1).iloc[0].to_dict()
    #
    [c for c in df_hk.columns if not c.startswith("meta")]
    rcols = {
        "text.cc": "text.src",
        "text.ko": "text.tgt",
        "text.zh": "meta.text.zh",
    }
    df_hk.rename(columns=rcols, inplace=True)
    #
    df_hk["lang.src"] = "cc"
    df_hk["lang.tgt"] = "ko"
    df_hk.sample(1).iloc[0].to_dict()

    # cc-zh
    idx = df["text.cc"].notna() & df["text.zh"].notna()
    idx.mean() * 100
    df_cz = df[idx].reset_index(drop=True)
    df_cz.sample(1).iloc[0].to_dict()
    #
    [c for c in df_cz.columns if not c.startswith("meta")]
    rcols = {
        "text.cc": "text.src",
        "text.zh": "text.tgt",
        "text.ko": "meta.text.ko",
    }
    df_cz.rename(columns=rcols, inplace=True)
    #
    df_cz["lang.src"] = "cc"
    df_cz["lang.tgt"] = "zh"
    df_cz.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_hk, df_cz], axis=0, ignore_index=True)

    # add key2 for unique sorting
    df["meta.corpus"] = "niu"
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


def get_wyweb() -> pd.DataFrame:
    # read
    df0 = utils.read_df(src.corpus.wyweb_mt_ko.root.FORMAT_PQ)
    df0.sample(1).iloc[0].to_dict()

    # prepare
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # check
    if 0:
        df1 = df.dropna()
        df1.sample(1).iloc[0].to_dict()
        df["split"].value_counts() / len(df) * 100

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # cc-ko
    idx = df["text.cc"].notna() & df["text.ko"].notna()
    idx.mean() * 100  # 99.69
    df_hk = df[idx].reset_index(drop=True)
    df_hk.sample(1).iloc[0].to_dict()
    #
    [c for c in df_hk.columns if not c.startswith("meta")]
    rcols = {
        "text.cc": "text.src",
        "text.ko": "text.tgt",
        "text.zh": "meta.text.zh",
    }
    df_hk.rename(columns=rcols, inplace=True)
    #
    df_hk["lang.src"] = "cc"
    df_hk["lang.tgt"] = "ko"
    df_hk.sample(1).iloc[0].to_dict()

    # cc-zh
    idx = df["text.cc"].notna() & df["text.zh"].notna()
    idx.mean() * 100  # 99.69
    df_cz = df[idx].reset_index(drop=True)
    df_cz.sample(1).iloc[0].to_dict()
    #
    [c for c in df_cz.columns if not c.startswith("meta")]
    rcols = {
        "text.cc": "text.src",
        "text.zh": "text.tgt",
        "text.ko": "meta.text.ko",
    }
    df_cz.rename(columns=rcols, inplace=True)
    #
    df_cz["lang.src"] = "cc"
    df_cz["lang.tgt"] = "zh"
    df_cz.sample(1).iloc[0].to_dict()

    # concat
    df = pd.concat([df_hk, df_cz], axis=0, ignore_index=True)

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
    # concat
    df = pd.concat(
        [get_ajd(), get_klc(), get_niu(), get_wyweb()], axis=0, ignore_index=True
    )

    # sort rows
    assert df["key2"].is_unique, "key2 is not unique"
    df.sort_values(by=["key2"], inplace=True, ignore_index=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # check
    temp1 = df.isna().sum()
    temp1[temp1 > 0]

    # save
    utils.write_df2(sroot.CONCAT_PQ, df)

    # check
    size = df.groupby(["split", "lang.src", "lang.tgt", "meta.corpus"]).size()
    sroot.CONCAT_STAT_TXT.parent.mkdir(parents=True, exist_ok=True)
    utils.write_str(sroot.CONCAT_STAT_TXT, str(size))


def main() -> None:
    # concat samples, drop cols, format
    gen_concat_file()  # 800.8M, 1258563f, 3000138


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
            # python -m src.dataset.mt_h2ke.concat
            typer.run(main)
