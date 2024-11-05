import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root as sroot
import src.crawl.ajd_cko.root
import src.crawl.ajd_en.root
import src.crawl.ajd_hj_oko.root
import src.crawl.ajd_link.root
from src import utils


def clean_link(df_link: pd.DataFrame) -> pd.DataFrame:
    # get mapping
    df_lk = df_link.copy()
    df_lk.rename(columns={"data_id": "ko", "label": "hj"}, inplace=True)
    df_lk.sample(1).iloc[0].to_dict()

    # check
    df_lk["hj"].value_counts().value_counts()
    df_lk["ko"].value_counts().value_counts()
    df_lk["hj"].str.len().value_counts()
    df_lk["ko"].str.len().value_counts()

    # drop empty
    df_lk["hj"].str.len().value_counts()
    df_lk = df_lk[df_lk["hj"].str.len() > 0].reset_index(drop=True)

    # rename w to k
    df_lk["hj"].str[:4].value_counts()
    df_lk["hj"] = df_lk["hj"].apply(lambda x: "k" + x[1:])

    # drop bad
    vc = df_lk["hj"].value_counts()
    bad_hj = vc[vc > 1].index
    if 0:
        len(bad_hj)
        df_lk[df_lk["hj"].isin(bad_hj)].sample(1).iloc[0].to_dict()
    df_lk = df_lk[~df_lk["hj"].isin(bad_hj)].reset_index(drop=True)

    return df_lk


def merge_oko(df_hj: pd.DataFrame) -> pd.DataFrame:
    # copy
    df_both = df_hj.copy()

    # check
    df_both.sample(1).iloc[0].to_dict()
    df_both["lang"].value_counts()

    # deduplicate columns
    assert df_both["meta.data_id"].equals(df_both["meta.elem_id"])
    df_both.drop(columns=["meta.elem_id"], inplace=True)

    # get hj
    df1 = df_both[df_both["lang"] == "hj"].reset_index(drop=True)
    rcols = {k: f"{k}.hj" for k in df1.columns}
    df1.rename(columns=rcols, inplace=True)
    df1.sample(1).iloc[0].to_dict()

    # get oko
    df2 = df_both[df_both["lang"] == "ko"].reset_index(drop=True)
    rcols = {k: f"{k}.oko" for k in df2.columns}
    df2.rename(columns=rcols, inplace=True)
    df2.sample(1).iloc[0].to_dict()

    # merge
    assert (
        len(set(df1.columns).intersection(set(df2.columns))) == 0
    ), "duplicate columns"
    df3 = pd.merge(
        df1, df2, left_on="meta.data_id.hj", right_on="meta.data_id.oko", how="left"
    )

    # sort columns
    c1 = [c for c in df3.columns if not c.startswith("meta")]
    c2 = [c for c in df3.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df3 = df3[cols].reset_index(drop=True)
    df3.sample(1).iloc[0].to_dict()

    # deduplicate columns
    for col0 in sorted(df_both.columns):
        col1 = f"{col0}.hj"
        col2 = f"{col0}.oko"
        if col1 in df3.columns and col2 in df3.columns:
            if df3[col1].equals(df3[col2]):
                df3.drop(columns=[col2], inplace=True)
    df3.sample(1).iloc[0].to_dict()

    # drop lang
    dcols = ["lang.hj", "lang.oko"]
    dcols = [c for c in dcols if c in df3.columns]
    df3.drop(columns=dcols, inplace=True)

    # check
    df3["meta.data_id.hj"].value_counts().value_counts()

    return df3


def check_cko(df_al: pd.DataFrame, df_ko: pd.DataFrame, df_lk: pd.DataFrame) -> None:
    # sample
    df_al.sample(1).iloc[0].to_dict()
    df_ko.sample(1).iloc[0].to_dict()
    df_lk.sample(1).iloc[0].to_dict()

    # compare lk-cko
    df_ko["meta.elem_id"].isin(df_lk["ko"]).mean()
    df_lk["ko"].isin(df_ko["meta.elem_id"]).mean()

    # compare hj-lk
    df_al["meta.data_id.hj"].isin(df_lk["hj"]).mean()
    df_lk["hj"].isin(df_al["meta.data_id.hj"]).mean()

    # compare hj-cko
    df_ko.sample(1).iloc[0].to_dict()
    df_ko["meta.elem_id"].str[:11].value_counts().sort_index()
    temp1 = df_ko["meta.elem_id"].str[8:9].value_counts().sort_index().reset_index()
    temp1["temp_id"] = temp1["meta.elem_id"].apply(lambda x: f"k{str(x).lower()}a")
    #
    df_al["meta.data_id.hj"].str[:3].value_counts().sort_index()
    df_al_co = df_al[
        df_al["meta.data_id.hj"].str[:3].isin(temp1["temp_id"])
    ].reset_index(drop=True)
    #
    vc1 = df_al_co["meta.data_id.hj"].str[:3].value_counts().sort_index()
    vc1a = vc1.reset_index(name="count_hj")
    vc2 = df_ko["meta.elem_id"].str[:10].value_counts().sort_index()
    vc2a = vc2.reset_index(name="count_ko")
    #
    vc3a = pd.concat([vc1a, vc2a], axis=1, sort=False)
    vc3a["ratio"] = vc3a["count_ko"] / vc3a["count_hj"] * 100


def merge_cko(
    df_al: pd.DataFrame, df_ko: pd.DataFrame, df_lk: pd.DataFrame
) -> pd.DataFrame:
    # check
    df_al.sample(1).iloc[0].to_dict()
    df_ko.sample(1).iloc[0].to_dict()
    df_lk.sample(1).iloc[0].to_dict()

    # get hj
    df1 = df_al.copy()
    df1.sample(1).iloc[0].to_dict()

    # get link
    df2 = df_lk.copy()
    df2.drop(columns=["meta.url"], inplace=True)
    rcols = {"ko": "temp.cko", "hj": "temp.hj"}
    df2.rename(columns=rcols, inplace=True)
    df2.sample(1).iloc[0].to_dict()

    # get cko
    df3 = df_ko.copy()
    rcols = {k: f"{k}.cko" for k in df3.columns}
    df3.rename(columns=rcols, inplace=True)
    df3.sample(1).iloc[0].to_dict()

    # merge hj and link
    assert (
        len(set(df1.columns).intersection(set(df2.columns))) == 0
    ), "duplicate columns"
    df4 = pd.merge(df1, df2, left_on="meta.data_id.hj", right_on="temp.hj", how="left")
    df4.sample(1).iloc[0].to_dict()
    df4[df4["temp.cko"].notna()].sample(1).iloc[0].to_dict()

    # merge hj and cko
    assert (
        len(set(df4.columns).intersection(set(df3.columns))) == 0
    ), "duplicate columns"
    df5 = pd.merge(
        df4, df3, left_on="temp.cko", right_on="meta.elem_id.cko", how="left"
    )
    df5.sample(1).iloc[0].to_dict()
    df5[df5["meta.elem_id.cko"].notna()].sample(1).iloc[0].to_dict()

    # clean
    df_out = df5.copy()

    # sort columns
    c1 = [c for c in df_out.columns if not c.startswith("meta")]
    c2 = [c for c in df_out.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df_out = df_out[cols].reset_index(drop=True)
    df_out.sample(1).iloc[0].to_dict()
    df_out[df_out["meta.elem_id.cko"].notna()].sample(1).iloc[0].to_dict()

    # drop columns
    dcols = ["temp.cko", "temp.hj"]
    dcols = [c for c in dcols if c in df_out.columns]
    df_out.drop(columns=dcols, inplace=True)

    return df_out


def check_en(df_al: pd.DataFrame, df_en: pd.DataFrame) -> None:
    # sample
    df_al.sample(1).iloc[0].to_dict()
    df_en.sample(1).iloc[0].to_dict()

    # compare hj-en
    df_al["meta.data_id.hj"].str[:4].value_counts().sort_index()
    df_en["meta.data_id"].str[:4].value_counts().sort_index()
    df_en["temp_id"] = df_en["meta.data_id"].apply(lambda x: "k" + x[1:])
    #
    df_al_co = df_al[
        df_al["meta.data_id.hj"].str[:4].isin(df_en["temp_id"].str[:4])
    ].reset_index(drop=True)
    # 70% (19년까지만 번역됨)
    df_al_co["meta.data_id.hj"].isin(df_en["temp_id"]).mean()
    df_en["temp_id"].isin(df_al_co["meta.data_id.hj"]).mean()

    df_en.drop(columns=["temp_id"], inplace=True)


def merge_en(df_al: pd.DataFrame, df_en: pd.DataFrame) -> pd.DataFrame:
    # check
    df_al.sample(1).iloc[0].to_dict()
    df_en.sample(1).iloc[0].to_dict()

    # get hj
    df1 = df_al.copy()
    df1.sample(1).iloc[0].to_dict()

    # get en
    df2 = df_en.copy()
    rcols = {k: f"{k}.en" for k in df2.columns}
    df2.rename(columns=rcols, inplace=True)
    df2.sample(1).iloc[0].to_dict()

    # drop non-en
    df2["lang.en"].value_counts()
    df2 = df2[df2["lang.en"] == "en"].reset_index(drop=True)
    df2["meta.data_id.en"].value_counts().value_counts()

    # gen temp_id
    df2["meta.data_id.en"].str[:4].value_counts()
    df2["temp_id.en"] = df2["meta.data_id.en"].apply(lambda x: "k" + x[1:])

    # merge
    assert (
        len(set(df1.columns).intersection(set(df2.columns))) == 0
    ), "duplicate columns"
    df3 = pd.merge(
        df1, df2, left_on="meta.data_id.hj", right_on="temp_id.en", how="left"
    )
    df3.sample(1).iloc[0].to_dict()
    df3[df3["meta.data_id.en"].notna()].sample(1).iloc[0].to_dict()

    # clean
    df_out = df3.copy()

    # sort columns
    c1 = [c for c in df_out.columns if not c.startswith("meta")]
    c2 = [c for c in df_out.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df_out = df_out[cols].reset_index(drop=True)
    df_out.sample(1).iloc[0].to_dict()
    df_out[df_out["meta.data_id.en"].notna()].sample(1).iloc[0].to_dict()

    # drop columns
    dcols = ["lang.en", "temp_id.en"]
    dcols = [c for c in dcols if c in df_out.columns]
    df_out.drop(columns=dcols, inplace=True)

    return df_out


def gen_align_file() -> None:
    # read
    df_hj = utils.read_df(src.crawl.ajd_hj_oko.root.FORMAT2_PQ)
    df_ko = utils.read_df(src.crawl.ajd_cko.root.FORMAT2_PQ)
    df_en = utils.read_df(src.crawl.ajd_en.root.FORMAT2_PQ)
    df_link = utils.read_df(src.crawl.ajd_link.root.FORMAT_PQ)

    # check
    df_hj.sample(1).iloc[0].to_dict()  # kua_10902027_001
    df_ko.sample(1).iloc[0].to_dict()  # ITKC_JR_D0_A07_02A_19A
    df_en.sample(1).iloc[0].to_dict()  # eda_10404018_002
    df_link.sample(1).iloc[0].to_dict()
    # ITKC_JR_D0_A03_09A_13A_00020 -> wda_10309013_002

    # convert link
    df_lk = clean_link(df_link=df_link)
    df_lk.sample(1).iloc[0].to_dict()

    # merge hj-oko
    df_al = merge_oko(df_hj=df_hj)
    df_al.sample(1).iloc[0].to_dict()

    # merge cko
    check_cko(df_al=df_al, df_ko=df_ko, df_lk=df_lk)
    df_al = merge_cko(df_al=df_al, df_ko=df_ko, df_lk=df_lk)
    df_al.sample(1).iloc[0].to_dict()
    df_al[df_al["meta.elem_id.cko"].notna()].sample(1).iloc[0].to_dict()

    # merge en
    check_en(df_al=df_al, df_en=df_en)
    df_al = merge_en(df_al=df_al, df_en=df_en)
    df_al.sample(1).iloc[0].to_dict()
    df_al[df_al["meta.data_id.en"].notna()].sample(1).iloc[0].to_dict()

    # check loss
    df_ko["meta.elem_id"].isin(df_al["meta.elem_id.cko"]).mean() * 100  # 99.48
    df_al["meta.elem_id.cko"].isin(df_ko["meta.elem_id"]).mean() * 100  # 12.36
    #
    df_en["meta.data_id"].isin(df_al["meta.data_id.en"]).mean() * 100  # 100
    df_al["meta.data_id.en"].isin(df_en["meta.data_id"]).mean() * 100  # 5.28

    # check
    temp1 = df_al.isna().sum()
    temp1[temp1 > 0]
    df_al.sample(1).iloc[0].to_dict()
    df_al[df_al["meta.elem_id.cko"].notna()].sample(1).iloc[0].to_dict()
    df_al[df_al["meta.data_id.en"].notna()].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_al)


def main() -> None:
    # align samples
    gen_align_file()  # 557.4M, 348d8057, 413323


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
            # python -m src.corpus.ajd.align
            typer.run(main)
