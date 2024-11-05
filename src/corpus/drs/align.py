import re
import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drs.root as sroot
import src.crawl.drs_hj.root
import src.crawl.drs_ko.root
import src.crawl.drs_link.root
from src import utils


def clean_link(df_lk0: pd.DataFrame) -> pd.DataFrame:
    # get mapping
    df = df_lk0.copy()
    {k: k for k in df.columns}
    rcols = {
        "data_id": "ko",
        "label": "hj",
        "meta.url": "meta.url",
    }
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # temp url
    df.drop(columns=["meta.url"], inplace=True)
    df["url.hj"] = df["hj"].apply(lambda x: f"https://sjw.history.go.kr/id/{x}")
    df["url.ko"] = df["ko"].apply(
        lambda x: f"https://db.itkc.or.kr/dir/item?itemId=ST#/dir/node?dataId={x}"
    )

    # check
    df["hj"].value_counts().value_counts().sort_index()
    df["hj"].str.len().value_counts()  # 19-기사,13-요목,4-없음
    df["hj_len"] = df["hj"].str.len()
    df.groupby("hj_len").sample(2).to_dict(orient="records")

    # check
    df["ko"].value_counts().value_counts()
    df["ko"].str.len().value_counts()  # 28-기사, 22-요목/일기청관원(=hj기사)
    df["ko_len"] = df["ko"].str.len()
    df.groupby("ko_len").sample(2).to_dict(orient="records")

    # drop
    df.drop(columns=["hj_len", "ko_len"], inplace=True)

    # drop bad_len (not 기사)
    df["hj"].str.len().value_counts()
    df = df[df["hj"].str.len() == 19].reset_index(drop=True)

    # drop bad_len (not 기사)
    df["ko"].str.len().value_counts()
    df = df[df["ko"].str.len() == 28].reset_index(drop=True)

    # drop bad (not 1 to 1)
    df["ko"].value_counts().value_counts().sort_index()
    df["hj"].value_counts().value_counts().sort_index()
    vc = df["hj"].value_counts()
    bad_hj = vc[vc > 1].index
    if 0:
        len(bad_hj)
        df[df["hj"].isin(bad_hj)].sample(1).iloc[0].to_dict()
    df = df[~df["hj"].isin(bad_hj)].reset_index(drop=True)

    # drop columns
    if 0:
        df.drop(columns=["meta.url.hj", "meta.url.ko"], inplace=True)

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
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    if 0:
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

    if 0:
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
    df1.sample(1).iloc[0].to_dict()

    # get ko
    df2 = df_ko.copy()
    df2.sample(1).iloc[0].to_dict()

    # get link
    df3 = df_lk.copy()
    rcols = {k: f"temp.{k}" for k in df3.columns}
    df3.rename(columns=rcols, inplace=True)
    df3.sample(1).iloc[0].to_dict()

    # merge lk to hj
    assert (
        len(set(df1.columns).intersection(set(df3.columns))) == 0
    ), "duplicate columns"
    if 0:
        df1.sample(1).iloc[0].to_dict()
        df3.sample(1).iloc[0].to_dict()
        df1["meta.data_id.hj"].isin(df3["temp.hj"]).mean() * 100  # 25.61
        df3["temp.hj"].isin(df1["meta.data_id.hj"]).mean() * 100  # 93.01
        idx = df3["temp.hj"].isin(df1["meta.data_id.hj"])
        df3[~idx].sample(1).iloc[0].to_dict()  # 좌목 등
    df4 = pd.merge(df1, df3, left_on="meta.data_id.hj", right_on="temp.hj", how="left")
    df4.sample(1).iloc[0].to_dict()
    df4[df4["temp.hj"].notna()].sample(1).iloc[0].to_dict()

    # merge ko to hj
    assert (
        len(set(df2.columns).intersection(set(df4.columns))) == 0
    ), "duplicate columns"
    if 0:
        df4.sample(1).iloc[0].to_dict()
        df2.sample(1).iloc[0].to_dict()
        df4["temp.ko"].isin(df2["meta.elem_id.ko"]).mean() * 100  # 25.61
        df2["meta.elem_id.ko"].isin(df4["temp.ko"]).mean() * 100  # 82.78
    df5 = pd.merge(df4, df2, left_on="temp.ko", right_on="meta.elem_id.ko", how="left")
    df5.sample(1).iloc[0].to_dict()
    df5[df5["temp.ko"].notna()].sample(1).iloc[0].to_dict()

    # clean
    df_out = df5.copy()

    # check & drop
    df_temp = df_out[df_out["temp.ko"].notna()].reset_index(drop=True)
    df_temp.sample(1).iloc[0].to_dict()
    assert (df_temp["temp.ko"] == df_temp["meta.elem_id.ko"]).all()
    assert (df_temp["temp.hj"] == df_temp["meta.data_id.hj"]).all()
    df_out.drop(columns=["temp.ko", "temp.hj"], inplace=True)

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
    df_hj0 = utils.read_df(src.crawl.drs_hj.root.FORMAT2_PQ)
    df_ko0 = utils.read_df(src.crawl.drs_ko.root.FORMAT2_PQ)
    df_lk0 = utils.read_df(src.crawl.drs_link.root.FORMAT_PQ)

    # check
    df_hj0.sample(1).iloc[0].to_dict()  # SJW-H29060220-01000
    df_ko0.sample(1).iloc[0].to_dict()  # ITKC_ST_U0_A03_08A_27A_00130
    df_lk0.sample(1).iloc[0].to_dict()
    # ITKC_ST_Z0_A21_04A_27A_00040 -> SJW-K21040270-00300

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
    df_ko["meta.elem_id.ko"].isin(df_hj["meta.elem_id.ko"]).mean() * 100  # 82.78
    df_hj["meta.elem_id.ko"].isin(df_ko["meta.elem_id.ko"]).mean() * 100  # 25.61

    # check
    temp1 = df_hj.isna().sum()
    temp1[temp1 > 0]
    df_hj.sample(1).iloc[0].to_dict()
    df_hj[df_hj["meta.elem_id.ko"].notna()].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_hj)


def main() -> None:
    # align samples
    gen_align_file()  # 859.9M, 0589a10d, 1787007


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
            # python -m src.corpus.drs.align
            typer.run(main)
