import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.drs.root as sroot
from src import utils


# Function to check for mixed null and non-null values
def check_mixed_nulls(group: pd.Series) -> bool:  # type: ignore
    return group.isnull().any() and group.notnull().any()


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.SPLIT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # blank
    df = utils.replace_blank_to_none(df)

    # rule: meta.data_id.ko는 meta.link_id.hj와 같아야 함
    idx1 = df["meta.link_id.hj"] != df["meta.data_id.ko"]
    idx2 = df["meta.link_id.hj"].notna() & df["meta.data_id.ko"].notna()
    idx = idx1 & idx2
    idx.sum()  # 723
    if 0:
        df[idx].sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None

    # rule: meta.link_id.hj는 unique해야 함
    vc = df["meta.link_id.hj"].value_counts()
    bad_vals = vc[vc > 1].index.tolist()
    idx = df["meta.link_id.hj"].isin(bad_vals)
    idx.mean() * 100  # 0.8
    if 0:
        df[idx]["meta.data_id.ko"].notnull().mean() * 100  # 23
        idx2 = idx & df["meta.data_id.ko"].notna()
        idx2.sum()  # 3497
        df[idx2].sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None

    # rule: text.ko에 '원문 빠짐'이 포함된 경우, None으로 처리
    idx = df["text.ko"].str.contains("원문 빠짐") | df["text.ko"].str.contains(
        "원문 결락"
    )
    idx.sum()  # 10820
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None
    if 0:
        df[idx].sample(1).iloc[0].to_dict()
        df3 = df[idx].reset_index(drop=True)
        vc = df3["text.ko"].value_counts()
        df2 = vc[vc > 1].reset_index(name="count")
        utils.write_json(utils.TEMP_JSON, df2.to_dict(orient="records"))

    # rule: text.ko가 null인 경우, 전체 ko를 null 처리
    temp1 = df.isna().sum()
    temp2 = temp1[temp1 > 0]
    temp2.reset_index(name="count").groupby("count").value_counts()
    #
    idx = df["text.ko"].isna() != df["meta.data_id.ko"].isna()
    if 0:
        df[idx].sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None

    # rule: book_title이 다를 경우, null 처리 (인조 -> 효종인 경우는 유효하지만 그래도 삭제)
    df["temp1"] = df["meta.book_title.hj"].apply(utils.remove_whites)
    df["temp1"].value_counts(dropna=False)
    df["meta.book_title.ko"].value_counts(dropna=False)
    idx1 = df["meta.book_title.ko"] != df["temp1"]
    idx2 = df["meta.book_title.ko"].notna() & df["temp1"].notna()
    idx = idx1 & idx2
    if 0:
        df[idx].sample(1).iloc[0].to_dict()
    cols = [c for c in df.columns if c.endswith(".ko")]
    df.loc[idx, cols] = None
    df.drop(columns=["temp1"], inplace=True)

    # sample
    if 0:
        df1 = df.dropna()
        x1 = df1.sample(1).iloc[0].to_dict()
        utils.write_json(utils.TEMP_JSON, x1)
        utils.open_code(utils.TEMP_JSON)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    if 0:
        temp3 = df.groupby("meta.book_title.hj")["text.ko"].apply(
            lambda x: x.notnull().mean() * 100
        )
        temp3[temp3 > 0]  # 고종, 인조, 순종, 영조35%

        df = utils.read_df(sroot.FILTER_PQ)
        df["meta.data_id.ko"].notna().mean() * 100  # 24.77
        df["meta.data_id.ko"].notna().sum()  # 442584

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # filter poorly aligned data
    gen_filter_file()  # 840.9M, ba9d1229, 1787007


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
            # python -m src.corpus.drs.filter
            typer.run(main)
