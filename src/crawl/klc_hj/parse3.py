import sys
from importlib import reload
from typing import Any

import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.klc_hj.root as sroot
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

    # meta.elem_copyright_text (nothing to do)
    utils.write_json(
        utils.TEMP_JSON, df["meta.elem_copyright_text"].value_counts().to_dict()
    )

    # meta.elem_dci_text
    if 0:
        idx = df["meta.elem_id"] == "ITKC_BT_1461A_0010_000_0010"
        x = df[idx].iloc[0].to_dict()
        x = df.sample(1).iloc[0].to_dict()
        parse_elem_dci_html(x)
    df["meta.elem_dci_text"] = df.parallel_apply(parse_elem_dci_html, axis=1)  # type: ignore
    df["meta.elem_dci_text"].apply(len).value_counts()

    # meta.elem_title_text: nothing to do
    df["meta.elem_title_text"].value_counts()

    # drop useless columns
    dcols = [c for c in df.columns if "html" in c]
    df.drop(columns=dcols, inplace=True)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # sort rows
    kcols = ["meta.data_id", "meta.elem_id"]
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
    gen_parse3_file()  # 395.7M, 17270b6a, 653758


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
            # python -m src.crawl.klc_hj.parse3
            typer.run(main)
