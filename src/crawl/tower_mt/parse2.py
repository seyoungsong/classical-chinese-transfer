import sys
from importlib import reload

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.tower_mt.root as sroot
import src.tool.corpus as ctool
from src import utils


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    df["meta.lang"].value_counts()
    sorted(set(utils.flatten([str(s).split("-") for s in df["meta.lang"].unique()])))
    _ = ["de", "en", "es", "fr", "it", "ko", "mixed", "nl", "pl", "pt", "ru", "zh"]

    # filter by lang (ko, en, zh, mixed)
    langs = {"ko", "en", "zh", "mixed"}
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        x = x1["meta.lang"]
        str(x).split("-")
        langs.issuperset(str(x).split("-"))
    idx: utils.SeriesType = df["meta.lang"].progress_apply(
        lambda x: langs.issuperset(str(x).split("-"))
    )
    idx.mean() * 100  # 58.6%
    df = df[idx].reset_index(drop=True)
    # check
    df["meta.lang"].value_counts()
    df.groupby(["meta.lang", "meta.task"]).size()

    # filter by task
    # (skip because nmt is majority)
    if 0:
        df["meta.task"].value_counts()
        tasks = [
            c
            for c in df["meta.task"].unique()
            if "translation" in c or "mt_" in c or "chat" in c
        ]
        [c for c in df["meta.task"].unique() if c not in tasks]
        idx = df["meta.task"].isin(tasks)
        idx.mean() * 100  # 100%

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    cols = ["key"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    # parse body html to text
    gen_parse2_file()  # 397.0M, a561595b, 373525


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        reload(ctool)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.tower_mt.parse2
            typer.run(main)
