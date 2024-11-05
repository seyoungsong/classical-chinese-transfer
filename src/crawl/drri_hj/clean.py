import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.drri_hj.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # check error
    if 0:
        # e.g.: https://sillok.history.go.kr/id/kra_11012008_001
        df1 = df[df["error"].notnull()].reset_index(drop=True)
        df1["error"].value_counts()
        df1["id_len"] = df1["x1.data_id"].str.len()
        df1.groupby(["id_len", "error"]).size()

        idx = df1["id_len"] < 27
        df1.loc[idx, "error"].value_counts()
        df1 = df1[~idx].reset_index(drop=True)

        df1.sample(1).iloc[0].to_dict()

        idx = df1["error"].apply(lambda x: "data_id not" in x)
        df1[idx].sample(1).iloc[0].to_dict()

    # drop
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
        "elem_idx": "meta.elem_idx_lv2",
        "item_idx": "meta.elem_idx_lv1",
        "page_date": "meta.page_date",
        "title_html": "meta.elem_title_html",
        "title_text": "meta.elem_title_text",
        "url2": "meta.url",
        "x1.book_cd": "meta.book_id",
        "x1.book_id": "_meta.book_id",
        "x1.book_title": "_meta.book_title",
        "x1.king_year": "_meta.king_year",
        "x1.py_day_ganji": "_meta.py_day_ganji",
        "x1.py_day": "_meta.py_day",
        "x1.py_king_nm": "_meta.py_king_nm",
        "x1.py_king_year": "_meta.py_king_year",
        "x1.py_month": "_meta.py_month",
        "x1.py_year": "_meta.py_year",
        "x1.py_yun": "_meta.py_yun",
        "x1.row_idx": "_meta.row_idx",
        "x1.title": "_meta.title",
        "x1.url": "meta.data_url",
        "x1.vol_no": "meta.book_id_vol",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop cols
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    df = df[sorted(df.columns)].reset_index(drop=True)

    # fix types
    df["meta.elem_idx_lv1"] = df["meta.elem_idx_lv1"].astype(int)
    df["meta.elem_idx_lv2"] = df["meta.elem_idx_lv2"].astype(int)
    df.sample().iloc[0].to_dict()

    # sort rows
    df.duplicated().sum()  # 3317
    df.drop_duplicates(inplace=True, ignore_index=True)
    cols = ["meta.data_url", "meta.elem_idx_lv1", "meta.elem_idx_lv2"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 164.6M, cc867f4a


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.clean
            typer.run(main)
