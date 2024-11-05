import json
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import typer
from loguru import logger
from rich import pretty

import src.crawl.drri_hj.root as sroot
from src import utils


def parse1(x1: dict[str, Any]) -> dict[Any, Any]:
    # read
    fname1 = x1["fname"]
    html = Path(fname1).read_text(encoding="utf-8")
    d1 = json.loads(html)
    assert isinstance(d1, dict), "d1 not dict"
    d2 = utils.notnull_collection(d1)
    if 0:
        utils.write_json(utils.TEMP_JSON, d2)
        utils.open_code(utils.TEMP_JSON)
    df1 = pd.DataFrame(d2["bodyList"])  # type: ignore
    df1.sample(1).iloc[0].to_dict()
    if 0:
        utils.write_str(utils.TEMP_HTML, html)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # parse
    df1["fname"] = str(fname1)
    df1["row_idx"] = df1.index + 1

    # clean
    df1.sample(1).iloc[0].to_dict()
    dcols = ["id", "ord"]
    df1.drop(columns=[c for c in dcols if c in df1.columns], inplace=True)

    # all str
    for c in list(df1.columns):
        df1[c] = df1[c].astype(str)

    # convert
    d1 = df1.to_dict(orient="records")
    x2 = {"pred": d1}

    return x2


def gen_lv3_file() -> None:
    # init
    num_proc = psutil.cpu_count() - 1

    # read
    ds = utils.read_ds(sroot.LV3_TASK_PQ)

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
    sroot.LV3_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV3_PQ, df2)
    logger.debug(f"len: {len(df2)}")


def gen_lv3a_file() -> None:
    # read
    df = utils.read_df(sroot.LV3_PQ)
    df["temp_id"] = df["fname"].apply(lambda x: Path(str(x)).stem)
    df.drop(columns=["fname"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # read
    df1 = utils.read_df(sroot.LV3_TASK_PQ)
    df1["temp_id"] = df1["fname"].apply(lambda x: Path(str(x)).stem)
    df1.drop(columns=["fname"], inplace=True)
    df1.sample(1).iloc[0].to_dict()
    dcols = sorted(set(df1.columns).intersection(df.columns) - {"temp_id"})
    df1.drop(columns=dcols, inplace=True)

    # join
    assert set(df1.columns).intersection(df.columns) == {"temp_id"}
    df = df.merge(df1, on="temp_id", how="left")
    df.sample(1).iloc[0].to_dict()

    # drop old cols
    df.drop(columns=["url", "temp_id"], inplace=True)

    # sort cols
    df = df[sorted(df.columns)]
    df.sample(1).iloc[0].to_dict()

    # drop trivial
    nu = df.nunique()
    df[list(nu[nu <= 1].index)].value_counts()
    df = df[list(nu[nu > 1].index)].reset_index(drop=True)

    # save
    sroot.LV3A_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV3A_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # parse index files
    gen_lv3_file()
    gen_lv3a_file()  # 71.7K, 7af2daf1, 56178 days (=154 years)


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_hj.lv3_parse
            typer.run(main)
