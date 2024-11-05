import sys
from importlib import reload
from pathlib import Path

import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.niu_cc_zh.root as sroot
from src import utils


def gen_path(fname: str) -> str:
    p1 = Path(fname).resolve()
    p2 = p1.relative_to(sroot.CRAWL_DL_DIR)
    path1 = p2.parent
    return str(path1)


def gen_clean2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)
    df.sample(1).iloc[0].to_dict()

    # check
    print(", ".join(df.columns))
    # meta.fname, meta.row_idx, text.cc, text.zh

    # add: path
    if 0:
        x1 = df.sample(1).iloc[0].to_dict()
        gen_path(fname=x1["meta.fname"])
    df["meta.path"] = df["meta.fname"].parallel_apply(gen_path)
    df.drop(columns=["meta.fname"], inplace=True)

    # empty to nan
    df.replace(r"^\s*$", None, regex=True, inplace=True)
    df.info()
    df.isna().sum()[df.isna().sum() > 0]

    # ok to be empty for now
    df.fillna("", inplace=True)

    # sample
    df.sample(1).iloc[0].to_dict()

    # sort rows
    kcols = ["meta.path", "meta.row_idx"]
    df.groupby(kcols).size().value_counts()
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # save
    df.info()
    utils.write_df2(sroot.CLEAN2_PQ, df)


def main() -> None:
    # drop some rows, and add some columns
    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_clean2_file()  # 89.7M, c2d07f05


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
        #
        tqdm.pandas()
        pandarallel.initialize(progress_bar=True)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.crawl.niu_cc_zh.clean2
            typer.run(main)
