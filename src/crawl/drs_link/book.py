import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.drs_ko.root
import src.crawl.drs_link.root as sroot
from src import utils


def gen_book() -> None:
    # read
    df = utils.read_df(src.crawl.drs_ko.root.FORMAT2_PQ)
    df.sample(1).iloc[0].to_dict()

    # get
    cols = ["meta.elem_id", "meta.elem_url"]
    df1 = df[cols].reset_index(drop=True)

    # rename
    rcols = {"meta.elem_id": "data_id", "meta.elem_url": "url"}
    df1.rename(columns=rcols, inplace=True)
    df1.sample(1).iloc[0].to_dict()

    # check
    df1["data_id"].value_counts().value_counts()

    # sort
    df1.sort_values("data_id", inplace=True, ignore_index=True)
    df1 = df1[sorted(df1.columns)].reset_index(drop=True)

    # save
    sroot.BOOK_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.BOOK_PQ, df1)
    logger.debug(len(df1))


def main() -> None:
    # get list of books (+ id, meta)
    gen_book()  # 1.9M, a123c7f0, 552965


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_link.book
            typer.run(main)
