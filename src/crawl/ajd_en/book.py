import re
import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.ajd_en.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    url = "http://esillok.history.go.kr/record/recordView.do?id=eda_10008014_001&yearViewType=byAD&sillokViewType=EngKor&lang=ko"
    html = utils.get_httpx(url)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("li.byAD_li"))
    len(tags)
    if 0:
        t = tags[0]
        t.attrs
        [t.text.strip() for t in tags]

    pattern = re.compile(r"\d+")

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t.attrs["href"]
        pattern.search(str(t.text)).group(0)

        [
            a.text
            for a in t.select("a")
            if "href" in a.attrs and "view" in a.attrs["href"]
        ]

    df["tag_title"] = df["tag"]
    df["data_id"] = df["tag_title"].apply(
        lambda x: int(pattern.search(x.text).group(0))  # type: ignore
    )
    df["book_title"] = df["tag_title"].apply(lambda x: str(x.text).strip())

    # clean
    df.sample(1).iloc[0].to_dict()
    df.drop(columns=["tag", "tag_title"], inplace=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df)


def main() -> None:
    # get list of books (+ id, meta)
    gen_book_jsonl()  # 820B, 3b9478c6


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.ajd_en.book
            typer.run(main)
