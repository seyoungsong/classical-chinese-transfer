import sys
from importlib import reload
from typing import Any

import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_hj.root as sroot
import src.tool.corpus as ctool
from src import utils


def parse_elem_dci_html(x: dict[str, Any]) -> str:
    html: str = x["meta.elem_dci_html"]
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    tags = list(soup.select("a"))
    tags = [t for t in tags if "DCI" in t.text]
    assert len(tags) <= 1, f"tags bad, x={x}, {x['meta.elem_id']}"
    if len(tags) == 0:
        return ""  # e.g. ITKC_BT_1461A_0010_000_0010
    t = tags[0]
    data_dci = t.attrs["data-dci-copy"].strip()
    return data_dci


def gen_parse3_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE2_PQ)
    df.sample(1).iloc[0].to_dict()

    # fillna
    df.fillna("", inplace=True)

    # meta.page_title: nothing to do
    vc1 = df["meta.elem_title_text"].value_counts()
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

    # merge cols
    digit1 = len(str(df["meta.elem_idx_lv1"].max()))
    digit2 = len(str(df["meta.elem_idx_lv2"].max()))
    temp1 = df["meta.elem_idx_lv1"].apply(lambda x: f"{x:0{digit1}d}")
    temp2 = df["meta.elem_idx_lv2"].apply(lambda x: f"{x:0{digit2}d}")
    df["meta.elem_idx"] = temp1 + "-" + temp2
    df.drop(columns=["meta.elem_idx_lv1", "meta.elem_idx_lv2"], inplace=True)

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
    gen_parse3_file()  # 82.4M, 834aed97, 367124


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
            # python -m src.crawl.drri_hj.parse3
            typer.run(main)
