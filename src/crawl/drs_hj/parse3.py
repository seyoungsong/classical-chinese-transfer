import re
import sys
from importlib import reload
from typing import Any

import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.drs_hj.root as sroot
import src.tool.corpus as ctool
from src import utils


def parse_elem_btn_ko_html(x: dict[str, Any]) -> str:
    html: str = x["meta.elem_btn_ko_html"]
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)
        utils.open_url(x["meta.url"])

    tags = list(soup.select("a"))
    tags = [t for t in tags if "ITKC_" in str(t)]
    tags = [t for t in tags if "href" in t.attrs]
    hrefs = [t.attrs["href"] for t in tags]
    pat = re.compile(r"ITKC_[A-Z\d_]+")
    ids = [pat.search(h).group(0) for h in hrefs]  # type: ignore
    all_id = ";".join(sorted(set(ids)))

    return all_id


def gen_parse3_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # meta.elem_copyright_text (nothing to do)
    if "meta.elem_copyright_text" in df.columns:
        utils.write_json(
            utils.TEMP_JSON, df["meta.elem_copyright_text"].value_counts().to_dict()
        )

    # fillna
    df.fillna("", inplace=True)

    # meta.elem_btn_ko_text
    if 0:
        idx = df["meta.elem_id"] == "ITKC_BT_1461A_0010_000_0010"
        temp1 = df["meta.elem_btn_ko_html"].apply(len)
        temp1.value_counts()
        idx = temp1 == temp1.max()
        x = df[idx].iloc[0].to_dict()
        x = df.sample(1).iloc[0].to_dict()
        parse_elem_btn_ko_html(x)
    if "meta.elem_btn_ko_html" in df.columns:
        df["meta.elem_btn_ko_text"] = df.parallel_apply(parse_elem_btn_ko_html, axis=1)  # type: ignore
        df["meta.elem_btn_ko_text"].apply(len).value_counts()

    # meta.page_title: nothing to do
    vc1 = df["meta.page_title"].value_counts()
    vc2 = vc1[vc1 > 1].reset_index()
    utils.write_df(utils.TEMP_JSON, vc2)
    if 0:
        # drop 입직
        idx = df["meta.page_title"] == "입직"
        logger.debug(f"{idx.mean():.1%}")  # 6.5%
        if 0:
            df[idx].sample(1).iloc[0].to_dict()
        df = df[~idx].reset_index(drop=True)

    # meta.elem_body_text
    vc1 = df["meta.elem_body_text"].value_counts()
    vc2 = vc1[vc1 > 1].reset_index()
    utils.write_df(utils.TEMP_JSON, vc2)
    if 0:
        # replace explicit EMPTY to None
        idx = df["meta.elem_body_text"].apply(lambda x: "해당 국역이 없습니다." in x)
        logger.debug(f"{idx.mean():.1%}, n={idx.sum()}")  # 158
        df.loc[idx, "meta.elem_body_text"] = ""

    # drop useless columns
    dcols = [c for c in df.columns if "html" in c]
    df.drop(columns=dcols, inplace=True)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # sort rows
    kcols = ["meta.data_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # replace
    df = utils.replace_blank_to_none(df)
    df.isnull().sum()[df.isnull().sum() > 0]

    # save
    utils.write_df2(sroot.PARSE3_PQ, df)
    logger.debug(f"len(df)={len(df)}")


def main() -> None:
    # parse etc html to text

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_parse3_file()  # 368.1M, bd02c815, 1787007


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(ctool)
        #
        tqdm.pandas()
        pandarallel.initialize(progress_bar=True)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drs_hj.parse3
            typer.run(main)
