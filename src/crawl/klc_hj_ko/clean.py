import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.klc_hj_ko.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # check error
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
    dcols = ["x1.fname", "x1.fname2", "x1.size"]
    df.drop(columns=[c for c in dcols if c in df.columns], inplace=True)

    # rename
    {k: f"meta.{k}" for k in df.columns}
    rcols = {
        "body_html": "meta.elem_body_html",
        "body_text": "meta.elem_body_text",
        "copyright_html": "meta.elem_copyright_html",
        "copyright_text": "meta.elem_copyright_text",
        "data_id": "meta.elem_id",
        "dci_html": "meta.elem_dci_html",
        "dci_text": "meta.elem_dci_text",
        "elem_idx": "meta.elem_col",
        "page_path": "meta.page_path",
        "page_title": "meta.page_title",
        "title_html": "meta.elem_title_html",
        "title_text": "meta.elem_title_text",
        "url": "meta.elem_url",
        "x1.author": "meta.author",
        "x1.book_id": "meta.book_id",
        "x1.category": "meta.book_category",
        "x1.data_id": "meta.data_id",
        "x1.extra_type": "meta.book_extra_type",
        "x1.extra": "meta.book_extra",
        "x1.page_title": "meta.mokcha_title",
        "x1.publisher": "meta.publisher",
        "x1.row_idx": "meta.mokcha_row",
        "x1.temp_id": "_meta.temp_id",
        "x1.title": "meta.book_title",
        "x1.translator": "meta.translator",
        "x1.url": "meta.url",
        "x1.url2": "meta.url2",
        "x1.year_str": "meta.year",
        "x1.year1": "_meta.year1",
        "x1.year2": "_meta.year2",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop cols
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    df = df[sorted(df.columns)].reset_index(drop=True)

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
    gen_clean_file()  # 693.3M, 55e2c41d


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj_ko.clean
            typer.run(main)
