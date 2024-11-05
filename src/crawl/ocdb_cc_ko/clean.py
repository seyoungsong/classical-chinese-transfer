import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.LV2A_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # check
    if "error" in df.columns:
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
    else:
        logger.info("No error column")

    # drop
    if "error" in df.columns:
        df = df[df["error"].isnull()].reset_index(drop=True)
        df.drop(columns=["error"], inplace=True)

    # check
    df.info()
    df.sample().iloc[0].to_dict()

    # rename
    {k: f"meta.{k.replace('x.', '')}" for k in df.columns}
    rcols = {
        "body.html": "body.html",
        "bookname": "meta.bookname",
        "data_id": "meta.data_id",
        "title.html": "title.html",
        "title.text": "title.text",
        "x.data_id": "_1",
        "x.row_idx": "_2",
        "x.title": "_3",
        "x.url": "meta.data_url",
        "x.url2": "meta.url",
        "x.x.book_title": "_4",
        "x.x.data_id": "_5",
        "x.x.url": "_6",
        "x.x.url2": "_7",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop cols
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    df["meta.data_id"].value_counts().value_counts()
    cols = ["meta.data_id"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 60.3M, a621e3a9, 28341


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.clean
            typer.run(main)
