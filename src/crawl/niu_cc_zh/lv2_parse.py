import random
import re
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty

import src.crawl.niu_cc_zh.root as sroot
from src import utils


def parse1(x1: dict[str, Any]) -> dict[Any, Any]:
    # read
    fname1 = x1["fname"]
    html = Path(fname1).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, html)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # tags
    tags = list(soup.select("a"))
    tags = [t for t in tags if "searchTree" in str(t)]
    len(tags)
    if 0:
        t = random.choice(tags)
        t = tags[0]
        t.attrs

    # df
    df1 = pd.DataFrame()
    df1["tag"] = tags

    # parse
    pattern = re.compile(r"SJW-[^\'\"\(\)]+")
    df1["data_id"] = df1["tag"].apply(lambda x: pattern.search(x.attrs["href"]).group(0))  # type: ignore
    df1["page_title"] = df1["tag"].apply(lambda x: str(x.text).strip())
    df1["fname"] = str(fname1)
    df1["row_idx"] = df1.index + 1

    # clean
    df1.drop(columns=["tag"], inplace=True)

    # convert
    d1 = df1.to_dict(orient="records")
    x2 = {"pred": d1}

    return x2


def gen_lv2_file() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.LV2_TASK_PQ)

    # test
    if 0:
        ds = ds.map(
            lambda x1: {"size": Path(x1["fname"]).stat().st_size}, num_proc=num_proc
        )
        ds = ds.sort("size")
        x1 = ds[0]
        x1 = ds.shuffle()[0]
        parse1(x1=x1)

    # parse
    ds = ds.shuffle(seed=42)
    ds = ds.map(parse1, batched=False, load_from_cache_file=False, num_proc=num_proc)
    logger.success("gen_parse_dir done")

    # convert
    if 0:
        x1 = ds.shuffle()[0]
        ds["pred"][0]
    pred_list: list[Any] = sum(ds["pred"], [])

    # merge
    df2 = pd.DataFrame(pred_list)
    df2.sample(1).iloc[0].to_dict()

    # sort rows
    cols = ["fname", "row_idx"]
    df2.groupby(cols).size().value_counts()  # 38011
    df2.sort_values(cols, inplace=True, ignore_index=True)

    # sort columns
    df2 = df2[sorted(df2.columns)]

    # save
    sroot.LV2_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV2_PQ, df2)
    logger.debug(f"len: {len(df2)}")


def gen_lv2a_file() -> None:
    # read
    df = utils.read_df(sroot.LV2_PQ)
    df["temp_id"] = df["fname"].apply(lambda x: Path(str(x)).stem)
    df.drop(columns=["fname"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # read
    df1 = utils.read_df(sroot.LV2_TASK_PQ)
    df1["temp_id"] = df1["fname"].apply(lambda x: Path(str(x)).stem)
    df1.drop(columns=["fname"], inplace=True)
    df1.sample(1).iloc[0].to_dict()
    df1.drop(columns=["data_id", "page_title", "row_idx"], inplace=True)

    # join
    assert set(df1.columns).intersection(df.columns) == {"temp_id"}
    df = df.merge(df1, on="temp_id", how="left")
    df.sample(1).iloc[0].to_dict()

    # drop old cols
    df.drop(columns=["url", "temp_id"], inplace=True)

    # sort cols
    df = df[sorted(df.columns)]
    df.sample(1).iloc[0].to_dict()

    # save
    sroot.LV2A_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV2A_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # parse index files
    gen_lv2_file()
    gen_lv2a_file()  # 61.3K, 6030054c, 7799


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.lv2_parse
            typer.run(main)
