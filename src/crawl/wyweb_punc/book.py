import re
import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.wyweb_punc.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    # https://sjw.history.go.kr/search/inspectionList.do
    url = "https://sjw.history.go.kr/search/inspectionList.do"
    html = utils.get_httpx(url)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # df1
    df1_list = pd.read_html(html)
    assert len(df1_list) == 2
    df1_list = df1_list[:1]  # drop 은대조례
    df1 = pd.concat(df1_list, ignore_index=True).reset_index(drop=True)
    df1.info()
    assert len(df1) == 12

    # clean
    df1.nunique()
    if 0:
        df1.drop(columns=["원본이미지"], inplace=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df1)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("a"))
    tags = [t for t in tags if "searchMonthList" in str(t)]
    tags = [t for t in tags if "true" not in str(t)]
    len(tags)
    if 0:
        t = tags[0]
        t.attrs

    pattern = re.compile(r"SJW-[^\'\"\(\)]+")

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t.attrs["href"]
        pattern.search(str(t.attrs["href"])).group(0)

    df["tag_title"] = df["tag"]
    df["data_id"] = df["tag_title"].apply(
        lambda x: pattern.search(x.attrs["href"]).group(0)  # type: ignore
    )
    df["book_title"] = df["tag_title"].apply(lambda x: str(x.text).strip())

    # clean
    df.sample(1).iloc[0].to_dict()
    df.drop(columns=["tag", "tag_title"], inplace=True)

    # merge (side by side)
    df1 = df1.reset_index(drop=True)
    df2 = pd.concat([df1, df], axis=1)

    # save
    utils.write_df(sroot.BOOK_JSONL, df2)


def gen_book2_jsonl() -> None:
    # read
    df = utils.read_df(sroot.BOOK_JSONL)

    # rename
    {k: k for k in df.columns}
    rcols = {
        "왕명": "title",
        "재위기간": "year_str",
        "data_id": "data_id",
        "book_title": "book_title",
    }
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()

    # drop
    assert df["title"].equals(df["book_title"])
    df.drop(columns=["book_title"], inplace=True)

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
    gen_book2_jsonl()  # 12


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.wyweb_punc.book
            typer.run(main)
