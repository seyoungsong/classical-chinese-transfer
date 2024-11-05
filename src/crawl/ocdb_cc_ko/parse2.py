import json
import sys
from importlib import reload
from typing import Any

import pandas as pd
import typer
from bs4 import BeautifulSoup
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.ocdb_cc_ko.root as sroot
import src.tool.corpus as ctool
from src import utils


def parse_body_html(x: dict[str, Any]) -> str:  # noqa: C901
    html: str = x["body.html"]
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, str(soup))
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)
        utils.open_url(x["meta.url"])
        utils.open_url(str(x["meta.data_url"]).replace("Inside", ""))

    # decompose
    tags = list(
        soup.select(
            "div.bxb, div.bxb_trans, a.ico_exp, span.exp, div.ph, span.wju, a.ico_wju"
        )
    )
    for t in tags:
        t.decompose()

    # decompose
    tags = list(soup.select("span._chi"))
    for t in tags:
        t.decompose()

    # decompose
    tags = list(soup.select("div.juso_org span.sm"))
    for t in tags:
        t.decompose()

    # unwrap
    tags = list(soup.select("div._gakju_layer"))
    for t in tags:
        _ = t.unwrap()

    # replace
    if 0:
        tags = list(soup.select("br"))
        for t in tags:
            _ = t.replace_with("\n" + t.text.strip())

    # body_ko
    tags = list(soup.select("div.juso_trans"))
    body_ko = [t.text.strip() for t in tags]

    # body_cc
    tags = list(soup.select("div.juso_org"))
    body_cc = [t.text.strip() for t in tags]

    # body
    body_json = {"cc": body_cc, "ko": body_ko}
    body_text = json.dumps(body_json, ensure_ascii=False, indent=2)

    return body_text


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


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN2_PQ)
    df.sample(1).iloc[0].to_dict()

    # sort by size
    if 0:
        df["size"] = df["body.html"].str.len()
        df["size"].describe()
        df.sort_values("size", inplace=True, ignore_index=True, ascending=False)
        df.drop(columns=["size"], inplace=True)

    # sample
    if 0:
        x = df[:100].sample(1).iloc[0].to_dict()  # large
        html = x["body.html"]
        ctool.find_special_tags(html=html)
        utils.open_url(x["meta.url"])
        #
        result1 = df["title.html"][:100].parallel_apply(ctool.find_special_tags)
        result2 = df["body.html"][:100].parallel_apply(ctool.find_special_tags)
        result: utils.SeriesType = pd.concat([result1, result2], ignore_index=True)
        found_tags = sorted(
            set([s.strip() for s in ";".join(result.unique()).split(";")])
        )
        found_tags = [t for t in found_tags if len(t) >= 1]
        logger.debug(", ".join(found_tags))
        # img, span.jusok, span.xsl_img_open, span.xsl_tbl_open, span.xsl_wonju, table
        x = df[-100:].sample(1).iloc[0].to_dict()  # small
        x = df[:100].sample(1).iloc[0].to_dict()  # large
        #
        idx = df["meta.elem_id"] == "ITKC_MP_0184A_0030_010_0010"
        idx = df["meta.elem_id_type"].isin(["MP", "KP"]) & df["body.html"].str.contains(
            "color:"
        )
        idx = df["meta.elem_id_type"].isin(["MP", "KP"]) & df["body.html"].str.contains(
            "jusok"
        )
        idx = (
            df["body.html"].str.contains("5fb636")
            & df["body.html"].str.contains("xsl_wonju")
            & df["meta.elem_id_type"].isin(["MP", "KP"])
        )
        x = df[idx].sample(1).iloc[0].to_dict()

    # parse
    if 0:
        x = df.sample(1).iloc[0].to_dict()  # random
        print(parse_body_html(x=x))
        utils.open_url(x["meta.url"])
        utils.open_url(x["meta.data_url"])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["meta.body.text"] = df.parallel_apply(parse_body_html, axis=1)  # type: ignore
    df = utils.replace_blank_to_none(df)

    # drop cols
    cols = [c for c in df.columns if "html" in c]
    df.drop(columns=cols, inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    kcols = ["meta.data_id"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # replace
    df = utils.replace_blank_to_none(df)

    # save
    df.info()
    df.isnull().sum()
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    # parse html to text
    gen_parse2_file()  # 31.6M, a66dc385, 28341


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
            # python -m src.crawl.ocdb_cc_ko.parse2
            typer.run(main)
