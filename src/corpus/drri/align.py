import re
import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drri.root as sroot
import src.crawl.drri_hj.root
import src.crawl.drri_ko.root
import src.crawl.drri_link.root
from src import utils


def clean_link(df_lk0: pd.DataFrame) -> pd.DataFrame:
    # get mapping
    df = df_lk0.copy()
    {k: k for k in df.columns}
    rcols = {
        "data_id": "ko",
        "label.day": "hj.day",
        "label.month": "hj.month",
        "label.py_day_ganji": "hj.day_ganji",
        "label.py_king_nm": "hj.king",
        "label.py_yun": "hj.yun",
        "label.year": "hj.year",
        "meta.url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # gen hj
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        f"{x['hj.king']}_{x['hj.year']}년_{x['hj.month']}월(윤{x['hj.yun']})_{x['hj.day']}일({x['hj.day_ganji']})"
    df["hj"] = df.apply(
        lambda x: f"{x['hj.king']}_{x['hj.year']}년_{x['hj.month']}월(윤{x['hj.yun']})_{x['hj.day']}일({x['hj.day_ganji']})",
        axis=1,
    )
    df.drop(columns=[c for c in df.columns if c.startswith("hj.")], inplace=True)

    # check
    df["hj"].value_counts().value_counts()
    df["ko"].value_counts().value_counts()
    df["hj"].str.len().value_counts()
    df["ko"].str.len().value_counts()

    # sample
    df["hj_len"] = df["hj"].str.len()
    df.groupby("hj_len").sample(2).to_dict(orient="records")
    df.drop(columns=["hj_len"], inplace=True)

    # drop bad_len
    df["hj"].str.len().value_counts()
    df = df[df["hj"].str.len() == 24].reset_index(drop=True)

    # drop bad
    if 0:
        vc = df["hj"].value_counts()
        bad_hj = vc[vc > 1].index
        if 0:
            len(bad_hj)
            df[df["hj"].isin(bad_hj)].sample(1).iloc[0].to_dict()
        df = df[~df["hj"].isin(bad_hj)].reset_index(drop=True)

    # drop columns
    df.drop(columns=["meta.url"], inplace=True)

    return df


def parse_date_ko(s1: str) -> str:
    pat = re.compile(
        r"^(?P<king>[^\d원즉위]+)[\d원즉위]+년[^(]+\((?P<year>\d+)\)(?P<yun>윤?)(?P<month>\d+)월(?P<day>\d+일|미상)"
    )
    d1 = pat.search(s1).groupdict()  # type: ignore

    # year, month
    for k in ["year", "month"]:
        d1[k] = int(d1[k])

    # yun_month
    d1["yun"] = len(d1["yun"])
    assert d1["yun"] in [0, 1], "bad yun"

    # day
    d1["day"] = "??" if d1["day"] == "미상" else int(d1["day"].replace("일", ""))

    date_id = f"{d1['king']}_{d1['year']}년_{d1['month']:02}월(윤{d1['yun']})_{d1['day']:02}일"
    return date_id


def clean_ko(df_ko0: pd.DataFrame) -> pd.DataFrame:
    # get ko
    df = df_ko0.copy()
    rcols = {k: f"{k}.ko" for k in df.columns}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()
    #
    {k: k for k in df.columns}
    rcols = {"text.ko": "text_body.ko", "meta.elem_title.ko": "text_title.ko"}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()
    #
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # parse date_id
    df["temp1"] = df["meta.data_title.ko"].apply(utils.remove_whites)
    if 0:
        t1 = "\n".join(df["temp1"].sample(100).to_list())
        utils.write_str(utils.TEMP_TXT, t1)
        s1 = df["temp1"].sample(1).iloc[0]
        parse_date_ko(s1)
    df["date_id"] = df["temp1"].progress_apply(parse_date_ko)
    df.drop(columns=["temp1"], inplace=True)
    if 0:
        for s1 in tqdm(df["temp1"]):
            _ = parse_date_ko(s1)

    # check
    df["date_id"].str.len().value_counts()
    df["date_id"].str[:2].value_counts().sort_index()
    df["date_id"].str[:8].value_counts().sort_index()
    df.sample(1).iloc[0].to_dict()

    # sort
    df.sort_values(["date_id", "meta.data_id.ko"], inplace=True, ignore_index=True)
    df.sample(1).iloc[0].to_dict()
    if 0:
        utils.write_df(utils.TEMP_JSON, df[:100])

    # add date_id2
    df["order"] = df.groupby("date_id").cumcount() + 1  # Start counting from 1
    digit = len(str(df["order"].max()))
    df["total"] = df.groupby("date_id")["date_id"].transform("count")
    df["date_id2"] = (
        df["date_id"]
        + "_"
        + df["order"].apply(lambda x: f"{x:0{digit}}")
        + "/"
        + df["total"].apply(lambda x: f"{x:0{digit}}기사")
    )
    df = df.drop(["order", "total"], axis=1)

    return df


def parse_date_hj(s1: str) -> str:
    pat = re.compile(
        r"^(?P<king>\D+)\d+\((?P<year>\d+)\)년(?P<yun>윤?)(?P<month>\d+)월(?P<day>\d+일)"
    )
    d1 = pat.search(s1).groupdict()  # type: ignore

    # year, month
    for k in ["year", "month"]:
        d1[k] = int(d1[k])

    # yun_month
    d1["yun"] = len(d1["yun"])
    assert d1["yun"] in [0, 1], "bad yun"

    # day
    d1["day"] = "??" if d1["day"] == "미상" else int(d1["day"].replace("일", ""))

    date_id = f"{d1['king']}_{d1['year']}년_{d1['month']:02}월(윤{d1['yun']})_{d1['day']:02}일"
    return date_id


def clean_hj(df_hj0: pd.DataFrame) -> pd.DataFrame:
    df = df_hj0.copy()
    rcols = {k: f"{k}.hj" for k in df.columns}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # parse date_id
    df["temp1"] = df["meta.data_date.hj"].apply(utils.remove_whites)
    if 0:
        t1 = "\n".join(df["temp1"].sample(500).to_list())
        utils.write_str(utils.TEMP_TXT, t1)
        s1 = df["temp1"].sample(1).iloc[0]
        parse_date_hj(s1)
    df["date_id"] = df["temp1"].progress_apply(parse_date_hj)
    df.drop(columns=["temp1"], inplace=True)
    if 0:
        for s1 in tqdm(df["temp1"]):
            _ = parse_date_hj(s1)

    # check
    df["date_id"].str.len().value_counts()
    df["date_id"].str[:4].value_counts().sort_index()
    df["date_id"].str[:8].value_counts().sort_index()
    df.sample(1).iloc[0].to_dict()

    # sort
    df.sort_values(["date_id", "meta.elem_idx.hj"], inplace=True, ignore_index=True)
    df.sample(1).iloc[0].to_dict()
    if 0:
        utils.write_df(utils.TEMP_JSON, df[:100])

    # add date_id2
    df["order"] = df.groupby("date_id").cumcount() + 1  # Start counting from 1
    digit = len(str(df["order"].max()))
    df["total"] = df.groupby("date_id")["date_id"].transform("count")
    df["date_id2"] = (
        df["date_id"]
        + "_"
        + df["order"].apply(lambda x: f"{x:0{digit}}")
        + "/"
        + df["total"].apply(lambda x: f"{x:0{digit}}기사")
    )
    df = df.drop(["order", "total"], axis=1)

    return df


def merge_ko(
    df_hj: pd.DataFrame, df_ko: pd.DataFrame, df_lk: pd.DataFrame
) -> pd.DataFrame:
    # parse dates, gen temp_id with gisa order, merge by temp_id, replace misalignments

    # check
    df_hj.sample(1).iloc[0].to_dict()
    df_ko.sample(1).iloc[0].to_dict()
    df_lk.sample(1).iloc[0].to_dict()

    # get hj
    df1 = df_hj.copy()
    df1.rename(
        columns={"date_id": "date_id.hj", "date_id2": "date_id2.hj"}, inplace=True
    )
    df1.sample(1).iloc[0].to_dict()

    # get ko
    df2 = df_ko.copy()
    df2.rename(
        columns={"date_id": "date_id.ko", "date_id2": "date_id2.ko"}, inplace=True
    )
    df2.sample(1).iloc[0].to_dict()

    # merge hj and ko
    assert (
        len(set(df1.columns).intersection(set(df2.columns))) == 0
    ), "duplicate columns"
    df3 = pd.merge(df1, df2, left_on="date_id2.hj", right_on="date_id2.ko", how="left")
    df3.sample(1).iloc[0].to_dict()
    df3[df3["date_id2.ko"].notna()].sample(1).iloc[0].to_dict()

    # clean
    df_out = df3.copy()

    # check & drop
    df_temp = df_out[df_out["date_id2.ko"].notna()].reset_index(drop=True)
    df_temp.sample(1).iloc[0].to_dict()
    assert (df_temp["date_id2.ko"] == df_temp["date_id2.hj"]).all()
    assert (df_temp["date_id.ko"] == df_temp["date_id.hj"]).all()
    df_out.drop(columns=["date_id2.ko", "date_id.ko"], inplace=True)

    # rename
    df_out.rename(
        columns={"date_id2.hj": "meta.date_id2.hj", "date_id.hj": "meta.date_id.hj"},
        inplace=True,
    )

    # sort columns
    c1 = [c for c in df_out.columns if not c.startswith("meta")]
    c2 = [c for c in df_out.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df_out = df_out[cols].reset_index(drop=True)
    df_out.sample(1).iloc[0].to_dict()
    df_out[df_out["meta.url.ko"].notna()].sample(1).iloc[0].to_dict()

    return df_out


def gen_align_file() -> None:
    # read
    df_hj0 = utils.read_df(src.crawl.drri_hj.root.FORMAT2_PQ)
    df_ko0 = utils.read_df(src.crawl.drri_ko.root.FORMAT2_PQ)
    df_lk0 = utils.read_df(src.crawl.drri_link.root.FORMAT_PQ)

    # check
    df_hj0.sample(1).iloc[0].to_dict()  # kua_10902027_001
    df_ko0.sample(1).iloc[0].to_dict()  # ITKC_JR_D0_A07_02A_19A
    df_lk0.sample(1).iloc[0].to_dict()
    # ITKC_JR_D0_A03_09A_13A_00020 -> wda_10309013_002

    # clean link
    df_lk = clean_link(df_lk0=df_lk0)
    df_lk.sample(1).iloc[0].to_dict()

    # clean ko
    df_ko = clean_ko(df_ko0=df_ko0)
    df_ko.sample(1).iloc[0].to_dict()

    # clean hj
    df_hj = clean_hj(df_hj0=df_hj0)
    df_hj.sample(1).iloc[0].to_dict()

    # merge ko
    df_hj = merge_ko(df_hj=df_hj, df_ko=df_ko, df_lk=df_lk)
    df_hj.sample(1).iloc[0].to_dict()
    df_hj[df_hj["meta.elem_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # check loss
    df_ko["meta.elem_id.ko"].isin(df_hj["meta.elem_id.ko"]).mean() * 100  # 96.61
    df_hj["meta.elem_id.ko"].isin(df_ko["meta.elem_id.ko"]).mean() * 100  # 29.51

    # check
    temp1 = df_hj.isna().sum()
    temp1[temp1 > 0]
    df_hj.sample(1).iloc[0].to_dict()
    df_hj[df_hj["meta.elem_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_hj)


def main() -> None:
    # align samples
    gen_align_file()  # 181.2M, 06ac4271, 367124


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
            # python -m src.corpus.drri.align
            typer.run(main)
