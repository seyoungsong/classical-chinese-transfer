import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_mt.root as sroot
import src.crawl.klc_hj_ko.root
from src import utils


def clean_ko(df_mt0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_mt0.copy()
    df.sample(1).iloc[0].to_dict()
    df["lang"].value_counts()
    df = df[df["lang"] == "ko"].reset_index(drop=True)
    df.drop(columns=["lang", "punc_type"], inplace=True)

    # get ko
    rcols = {k: f"{k}.ko" for k in df.columns}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()
    #
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    return df


def clean_hj(df_mt0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_mt0.copy()
    df.sample(1).iloc[0].to_dict()
    df = df[df["lang"] == "hj"].reset_index(drop=True)

    #
    df["lang"].value_counts()
    df.drop(columns=["lang"], inplace=True)

    #
    df["punc_type"].value_counts()
    df["is_punc"] = df["punc_type"] == "punc"
    df.drop(columns=["punc_type"], inplace=True)

    # get ko
    rcols = {k: f"{k}.hj" for k in df.columns}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()
    #
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    return df


def merge_ko(df_hj: pd.DataFrame, df_ko: pd.DataFrame) -> pd.DataFrame:
    # parse dates, gen temp_id with gisa order, merge by temp_id, replace misalignments

    # check
    df_hj.sample(1).iloc[0].to_dict()
    df_ko.sample(1).iloc[0].to_dict()

    # get hj
    df1 = df_hj.copy()
    df1.sample(1).iloc[0].to_dict()
    assert df1["meta.data_id.hj"].is_unique

    # get ko
    df2 = df_ko.copy()
    df2.sample(1).iloc[0].to_dict()
    assert df2["meta.data_id.ko"].is_unique

    # merge ko to hj
    assert (
        len(set(df2.columns).intersection(set(df1.columns))) == 0
    ), "duplicate columns"
    if 0:
        df1.sample(1).iloc[0].to_dict()
        df2.sample(1).iloc[0].to_dict()
        df1["meta.data_id.hj"].isin(df2["meta.data_id.ko"]).mean() * 100  # 100.0
        df2["meta.data_id.ko"].isin(df1["meta.data_id.hj"]).mean() * 100  # 100.0
    df3 = pd.merge(
        df1, df2, left_on="meta.data_id.hj", right_on="meta.data_id.ko", how="left"
    )
    df3.sample(1).iloc[0].to_dict()
    df3[df3["meta.data_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # clean
    df_out = df3.copy()

    # check & drop
    if 0:
        df_temp = df_out[df_out["meta.data_id.ko"].notna()].reset_index(drop=True)
        df_temp.sample(1).iloc[0].to_dict()
        assert (df_temp["meta.data_id.ko"] == df_temp["meta.elem_id.ko"]).all()
        assert (df_temp["meta.data_id.hj"] == df_temp["meta.elem_id.hj"]).all()
        df_out.drop(columns=["meta.data_id.ko", "meta.data_id.hj"], inplace=True)

    # sort columns
    c1 = [c for c in df_out.columns if not c.startswith("meta")]
    c2 = [c for c in df_out.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df_out = df_out[cols].reset_index(drop=True)
    df_out.sample(1).iloc[0].to_dict()
    df_out[df_out["meta.url.ko"].notna()].sample(1).iloc[0].to_dict()

    # drop same columns
    cols = [c for c in df_out.columns if c.endswith(".ko")]
    pairs = [(c, c[:-2] + "hj") for c in cols]
    pairs = [
        (c1, c2) for c1, c2 in pairs if c1 in df_out.columns and c2 in df_out.columns
    ]
    cols = [c2 for c1, c2 in pairs if df_out[c1].equals(df_out[c2])]
    df_out.drop(columns=cols, inplace=True)
    df_out.sample(1).iloc[0].to_dict()

    return df_out


def gen_align_file() -> None:
    # read
    df_mt0 = utils.read_df(src.crawl.klc_hj_ko.root.FORMAT2_PQ)

    # check
    df_mt0.sample(1).iloc[0].to_dict()  # ITKC_ST_U0_A03_08A_27A_00130
    len(df_mt0)  # 314404

    # get ko
    df_ko = clean_ko(df_mt0=df_mt0)
    df_ko.sample(1).iloc[0].to_dict()

    # get hj
    df_hj = clean_hj(df_mt0=df_mt0)
    df_hj.sample(1).iloc[0].to_dict()

    # merge ko
    df_mt = merge_ko(df_hj=df_hj, df_ko=df_ko)
    df_mt.sample(1).iloc[0].to_dict()
    df_mt[df_mt["meta.elem_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # check loss
    df_ko["meta.elem_id.ko"].isin(df_mt["meta.elem_id.ko"]).mean() * 100  # 100
    df_mt["meta.elem_id.ko"].isin(df_ko["meta.elem_id.ko"]).mean() * 100  # 100

    # check
    temp1 = df_mt.isna().sum()
    temp1[temp1 > 0]
    df_mt.sample(1).iloc[0].to_dict()
    df_mt[df_mt["meta.elem_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_mt)


def main() -> None:
    # align samples
    gen_align_file()  # 619.9M, 4e8121a3, 157202


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
            # python -m src.corpus.klc_mt.align
            typer.run(main)
