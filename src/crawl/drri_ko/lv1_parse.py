import random
import re
import sys
from importlib import reload
from pathlib import Path

import pandas as pd
import typer
from bs4 import BeautifulSoup
from loguru import logger
from rich import pretty
from tqdm import tqdm

import src.crawl.drri_ko.root as sroot
from src import utils


def parse1(fname1: str | Path) -> pd.DataFrame:
    # read
    html = Path(fname1).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    if 0:
        utils.write_str(utils.TEMP_HTML, html)
        utils.open_file(utils.TEMP_HTML)
        utils.open_code(utils.TEMP_HTML)

    # tags
    tags = list(soup.select("a"))
    tags = [t for t in tags if "href" in t.attrs]
    tags = [t for t in tags if "ITKC_" in t.attrs["href"]]
    len(tags)
    if 0:
        t = random.choice(tags)
        t = tags[0]
        t.attrs

    # df
    df1 = pd.DataFrame()
    df1["tag"] = tags

    # parse
    pattern = re.compile(r"ITKC_[A-Z\d_]+")
    df1["data_id"] = df1["tag"].progress_apply(lambda x: pattern.search(x.attrs["href"]).group(0))  # type: ignore
    df1["page_title"] = df1["tag"].progress_apply(lambda x: str(x.text).strip())
    df1["fname"] = str(fname1)
    df1["row_idx"] = df1.index + 1

    # clean
    df1.drop(columns=["tag"], inplace=True)

    return df1


def gen_lv1_file() -> None:
    # read
    df = utils.read_df(sroot.LV1_TASK_PQ)

    # parse
    df_list = []
    fname1 = df["fname"].sample(1).iloc[0]
    for fname1 in tqdm(df["fname"]):
        df1 = parse1(fname1=fname1)
        df_list.append(df1)

    # merge
    df2 = pd.concat(df_list, ignore_index=True)
    df2.sample(1).iloc[0].to_dict()
    len(df2)  # 587407

    # sort rows
    cols = ["fname", "row_idx"]
    df2.groupby(cols).size().value_counts()
    df2.sort_values(cols, inplace=True, ignore_index=True)

    # sort columns
    df2 = df2[sorted(df2.columns)]

    # save
    sroot.LV1_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1_PQ, df2)
    logger.debug(f"len: {len(df2)}")


def gen_lv1a_file() -> None:
    # read
    df = utils.read_df(sroot.LV1_PQ)
    df["temp_id"] = df["fname"].apply(lambda x: Path(str(x)).stem)
    df.drop(columns=["fname"], inplace=True)
    df.sample(1).iloc[0].to_dict()

    # read
    df1 = utils.read_df(sroot.LV1_TASK_PQ)
    df1["temp_id"] = df1["fname"].apply(lambda x: Path(str(x)).stem)
    df1.drop(columns=["fname"], inplace=True)
    df1.sample(1).iloc[0].to_dict()
    df1.rename(columns={"data_id": "book_id"}, inplace=True)

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
    sroot.LV1A_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.LV1A_PQ, df)
    logger.debug(f"len: {len(df)}")


def main() -> None:
    # parse index files
    gen_lv1_file()  # 4.6M, 4ea44972, 136487
    gen_lv1a_file()  # 4.6M, 76997878, 136487


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.drri_ko.lv1_parse
            typer.run(main)
