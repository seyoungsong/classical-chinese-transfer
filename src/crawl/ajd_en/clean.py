import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.ajd_en.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.LV4_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # check error
    if 0:
        # e.g.: https://sillok.history.go.kr/id/kra_11012008_001
        df1 = df[df["error"].notnull()].reset_index(drop=True)
        df1["error"].value_counts()
        df1["id_len"] = df1["x1.data_id"].str[:6]
        df1.groupby(["id_len", "error"]).size()

        # 영역본은 세종 32년 중 19년까지밖에 없음
        df["error"].value_counts(dropna=False)
        df["id_len"] = df["x1.data_id"].str[:7]
        df.groupby(["id_len", "error"], dropna=False).size()

        idx = df1["id_len"] < 27
        df1.loc[idx, "error"].value_counts()
        df1 = df1[~idx].reset_index(drop=True)

        df1.sample(1).iloc[0].to_dict()

        idx = df1["error"].apply(lambda x: "data_id not" in x)
        df1[idx].sample(1).iloc[0].to_dict()

    # drop
    if "error" in df.columns:
        df = df[df["error"].isnull()].reset_index(drop=True)
        df.drop(columns=["error"], inplace=True)

    # check
    df.info()
    df.sample().iloc[0].to_dict()

    # drop
    dcols = ["x1.temp_id", "x1.fname", "x1.fname2", "x1.size", "x1.temp_len"]
    df.drop(columns=[c for c in dcols if c in df.columns], inplace=True)

    # rename
    {k: f"meta.{k}" for k in df.columns}
    rcols = {
        "body_html": "meta.elem_body_html",
        "body_text": "meta.elem_body_text",
        "data_id": "meta.elem_id",
        "elem_idx": "meta.elem_col",
        "page_date": "meta.page_date",
        "title_html": "meta.elem_title_html",
        "title_text": "meta.elem_title_text",
        "url": "meta.url",
        "x1.book_id": "meta.book_id",
        "x1.book_title": "meta.book_title",
        "x1.data_id": "meta.data_id",
        "x1.page_title": "meta.mokcha_title",
        "x1.row_idx": "meta.mokcha_row",
        "x1.url": "meta.elem_url",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    df = df[sorted(df.columns)].reset_index(drop=True)

    # drop cols
    df.sample().iloc[0].to_dict()
    assert (df["meta.elem_id"] == df["meta.data_id"]).all(), "elem_id != data_id"
    if 0:
        idx = df["meta.elem_id"] != df["meta.data_id"]
        df[idx].sample(1).iloc[0].to_dict()
    df["meta.mokcha_row"].unique()

    dcols = [
        "meta.book_id",
        "meta.book_title",
        "meta.elem_id",
        "meta.mokcha_row",
        "meta.mokcha_title",
    ]
    df.drop(columns=dcols, inplace=True)

    # fix types
    df["meta.elem_col"] = df["meta.elem_col"].astype(int)
    df.sample().iloc[0].to_dict()

    # sort rows
    df["meta.elem_col"].value_counts()
    cols = ["meta.data_id", "meta.elem_col"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 40.6M, 019c27b2


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_en.clean
            typer.run(main)
