import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.crawl.niu_cc_zh.root as sroot
from src import utils


def gen_clean_file() -> None:
    # read
    df = utils.read_df(sroot.PARSE_PQ)
    df.sample(1).iloc[0].to_dict()
    df.info()

    # drop
    dcols = ["x.temp_id", "x.fname2", "x.size", "x.temp_len"]
    df.drop(columns=[c for c in dcols if c in df.columns], inplace=True)

    # rename
    {k: f"meta.{k}" for k in df.columns}
    rcols = {
        "cc": "text.cc",
        "zh": "text.zh",
        "row_idx": "meta.row_idx",
        "x.fname": "meta.fname",
    }
    assert len(rcols.values()) == len(set(rcols.values()))
    df.rename(columns=rcols, inplace=True)

    # drop cols
    dcols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=dcols, inplace=True)

    # sort columns
    df = df[sorted(df.columns)].reset_index(drop=True)

    # fix types
    df["meta.row_idx"] = df["meta.row_idx"].astype(int)
    df.sample().iloc[0].to_dict()

    # sort rows
    cols = ["meta.fname", "meta.row_idx"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.CLEAN_PQ, df)


def main() -> None:
    # remove errors and drop unnecessary columns
    gen_clean_file()  # 89.7M, 753e2137


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.clean
            typer.run(main)
