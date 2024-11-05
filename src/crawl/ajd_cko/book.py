import re
import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.ajd_cko.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    url = "https://db.itkc.or.kr/dir/item?itemId=JR#dir/list?itemId=JR&gubun=book&pageIndex=1&pageUnit=50"
    html = utils.get_httpx(url)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("a, li"))
    tags = [
        t
        for t in tags
        if "dataid" in ",".join(list(t.attrs.keys())).lower() and "ITKC_" in str(t)
    ]
    len(tags)
    if 0:
        t = tags[1]
        t.attrs

    pattern = re.compile(r"ITKC_[A-Z\d_]+")

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t.attrs["data-dataid"]
        pattern.search(str(t.attrs["data-dataid"])).group(0)

        [
            a.text
            for a in t.select("a")
            if "href" in a.attrs and "view" in a.attrs["href"]
        ]
        t.select_one("span").text

    df["tag_title"] = df["tag"]
    df["data_id"] = df["tag_title"].apply(
        lambda x: pattern.search(x.attrs["data-dataid"]).group(0)  # type: ignore
    )
    df["book_title"] = df["tag_title"].apply(
        lambda x: str(x.select_one("span").text).strip()
    )

    # clean
    df.sample(1).iloc[0].to_dict()
    df.drop(columns=["tag", "tag_title"], inplace=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df)


def gen_book2_jsonl() -> None:
    # read
    df = utils.read_df(sroot.BOOK_JSONL)

    # rename
    if 0:
        {k: k for k in df.columns}
        rcols = {
            "서명": "title",
            "저자": "author",
            "간행년": "year_str",
            "집수명": "jisu",
            "data_id": "data_id",
            "book_title": "book_title",
            "extra": "extra",
        }
        df.rename(columns=rcols, inplace=True)
        df.sample(1).iloc[0].to_dict()

        # drop
        assert df["title"].equals(df["book_title"])
        df.drop(columns=["book_title"], inplace=True)

    # check
    assert df["data_id"].is_unique

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # drop: not yearly records
    idx = df["data_id"].apply(lambda x: str(x).split("_")[2].startswith("0"))
    df = df[~idx].reset_index(drop=True)

    # save
    sroot.BOOK2_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK2_JSONL, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # get list of books (+ id, meta)
    gen_book_jsonl()
    gen_book2_jsonl()  # 5


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_cko.book
            typer.run(main)
