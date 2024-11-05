import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.drri_hj.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    url = "https://kyudb.snu.ac.kr/series/main.do?item_cd=ILS"
    html = utils.get_httpx(url)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("a"))
    tags = [t for t in tags if "goNextMenuK" in str(t)]
    len(tags)
    if 0:
        t = tags[0]
        t.attrs

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t.text.strip()

    df["tag_title"] = df["tag"]
    df["data_id"] = df["tag_title"].apply(lambda x: x.text.strip())
    df["book_title"] = df["tag_title"].apply(lambda x: str(x.text).strip())

    # clean
    df.sample(1).iloc[0].to_dict()
    df.drop(columns=["tag", "tag_title"], inplace=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df)


def gen_book2_jsonl() -> None:
    # read
    df = utils.read_df(sroot.BOOK_JSONL)

    # check
    assert df["data_id"].is_unique

    # sort cols
    df = df[sorted(df.columns)]

    # save
    sroot.BOOK2_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK2_JSONL, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # get list of books (+ id, meta)
    gen_book_jsonl()
    gen_book2_jsonl()  # 446B, 41faab1e, 10


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.book
            typer.run(main)
