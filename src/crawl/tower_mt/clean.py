import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.tower_mt.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # convert
    df["split"].value_counts()
    df["split"].replace({"dev": "valid"}, inplace=True)

    # convert
    df["lang"].value_counts()
    rule = {k: str(k).replace("_", "-") for k in list(df["lang"].unique())}
    rule = {k: v for k, v in rule.items() if k != v}
    df["lang"].replace(rule, inplace=True)

    # rename
    {k: f"meta.{k}" for k in df.columns}
    rcols = {
        "conversations": "conversations",
        "dataset": "meta.dataset",
        "lang": "meta.lang",
        "split": "split",
        "task": "meta.task",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # add cols
    df["meta.idx"] = df.index + 1
    digit = len(str(len(df)))
    df["key"] = df["meta.idx"].apply(lambda x: f"L{x:0{digit}d}")
    df.drop(columns=["meta.idx"], inplace=True)

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["key"].is_unique
    df.sort_values("key", inplace=True, ignore_index=True)

    # empty to nan
    df = utils.replace_blank_to_none(df)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]
    if 0:
        df.fillna("", inplace=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 469.2M, 27b87134, 637563


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.tower_mt.clean
            typer.run(main)
