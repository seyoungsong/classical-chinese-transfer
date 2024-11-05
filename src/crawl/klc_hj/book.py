import re
import sys
from importlib import reload

import pandas as pd
import typer
from bs4 import BeautifulSoup, Tag
from loguru import logger
from rich import pretty

import src.crawl.klc_hj.root as sroot
from src import utils


def gen_book_jsonl() -> None:
    # get
    # https://db.itkc.or.kr/dir/item?itemId=MO#dir/list?itemId=MO&gubun=book&pageIndex=1&pageUnit=50
    url = (
        "https://db.itkc.or.kr/dir/list?itemId=MO&pageIndex=1&pageUnit=2000&gubun=book"
    )
    html = utils.get_httpx(url)
    utils.write_str(utils.TEMP_HTML, html)
    if 0:
        html = utils.read_str(utils.TEMP_HTML)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # df1
    df1 = pd.read_html(html)[0]
    df1.info()
    assert len(df1) == 1259

    # clean
    df1.nunique()
    df1.drop(columns=["연계정보", "부가정보"], inplace=True)

    # save
    sroot.BOOK_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK_JSONL, df1)

    # soup
    soup = BeautifulSoup(html, "lxml")

    # rows
    tags = list(soup.select("tr"))
    tags = [t for t in tags if "dataId=" in str(t)]
    len(tags)
    if 0:
        t = tags[0]
        t.attrs

    pattern = re.compile(r"ITKC_[A-Z\d_]+")

    # df
    df = pd.DataFrame()
    df["tag"] = tags

    if 0:
        t = df["tag"].sample(1).iloc[0]
        assert isinstance(t, Tag)
        t.select("a")[0]["href"]
        pattern.search(str(t.select("a")[0]["href"])).group(0)

        [
            a.text
            for a in t.select("a")
            if "href" in a.attrs and "view" in a.attrs["href"]
        ]

    df["tag_title"] = df["tag"].apply(lambda x: x.select("a")[0])
    df["data_id"] = df["tag_title"].apply(
        lambda x: pattern.search(x.attrs["href"]).group(0)  # type: ignore
    )
    df["book_title"] = df["tag_title"].apply(lambda x: str(x.text).strip())
    df["extra"] = df["tag"].apply(
        lambda t: ",".join(
            [
                a.text
                for a in t.select("a")
                if "href" in a.attrs and "view" in a.attrs["href"]
            ]
        )
    )

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
    df = df[sorted(df.columns)]

    # save
    sroot.BOOK2_JSONL.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df(sroot.BOOK2_JSONL, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # get list of books (+ id, meta)
    gen_book_jsonl()  # 227.9K, 3a3bc83f
    gen_book2_jsonl()  # 172.2K, eff86330, 1259


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.klc_hj.book
            typer.run(main)
