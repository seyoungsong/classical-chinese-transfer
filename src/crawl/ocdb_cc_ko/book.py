import random
import re
import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.ocdb_cc_ko.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    # http://db.cyberseodang.or.kr/front/main/mainmenu.do?tab=tab1_02
    # http://mdb.cyberseodang.or.kr/mobile2/bookList/bookList.do
    url = "http://mdb.cyberseodang.or.kr/mobile2/bookList/bookList.do"
    html = utils.get_httpx(url, mobile=True)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("a"))
    tags = [t for t in tags if "href" in t.attrs and "bnCode=" in t.attrs["href"]]
    if 0:
        t = random.choice(tags)
        t.attrs

    pattern = re.compile(r"bnCode=(.+)")

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t["href"]
        pattern.search(str(t["href"])).group(1)
        t.text

    df["book_title"] = df["tag"].apply(lambda x: str(x.text).strip())
    df["data_id"] = df["tag"].apply(
        lambda x: pattern.search(x.attrs["href"]).group(1)  # type: ignore
    )

    # clean
    df.sample(1).iloc[0].to_dict()
    df.drop(columns=["tag"], inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True, ignore_index=True)

    # sort
    df.sort_values(by=["data_id", "book_title"], inplace=True, ignore_index=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df)


def gen_book2_jsonl() -> None:
    # read
    df = utils.read_df(sroot.BOOK_JSONL)

    # check
    assert df["data_id"].is_unique
    assert df["book_title"].is_unique

    # sort cols
    df = df[sorted(df.columns)]

    # save
    sroot.BOOK2_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK2_JSONL, df)


def main() -> None:
    # get list of books and their ids
    gen_book_jsonl()  # 3.7K, 0754beca, 67
    gen_book2_jsonl()  # 3.7K, 0754beca, 67


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ocdb_cc_ko.book
            typer.run(main)
