import sys
from importlib import reload

import hanja
import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.niu_cc_zh.root as sroot
import src.tool.corpus as ctool
from src import utils


def parse_meta_path(s: str) -> str:
    s2 = str(hanja.translate(text=s, mode="combination-text-reversed")).strip()
    return s2


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN2_PQ)
    df.sample(1).iloc[0].to_dict()

    # fix
    df["meta.path"].apply(lambda x: str(x)[:5]).value_counts()
    df["meta.path"] = df["meta.path"].apply(lambda x: str(x)[5:])

    # fix
    df["text.cc"].apply(lambda x: str(x)[:3]).value_counts()
    df["text.cc"] = df["text.cc"].apply(lambda x: str(x)[3:])

    # fix
    df["text.zh"].apply(lambda x: str(x)[:4]).value_counts()
    df["text.zh"] = df["text.zh"].apply(lambda x: str(x)[4:])

    # sort by size
    if 0:
        df["size"] = df["text.cc"].progress_apply(len)
        df["size"].describe()
        df.sort_values("size", inplace=True, ignore_index=True, ascending=False)
        df.drop(columns=["size"], inplace=True)

    # sample
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        print(parse_meta_path(s=x["meta.path"]))

    # parse
    df["meta.path"].nunique()
    path_list = df["meta.path"].unique().tolist()
    df2 = pd.DataFrame(path_list, columns=["meta.path"])
    df2["meta.path2"] = df2["meta.path"].progress_apply(parse_meta_path)

    # merge
    df = pd.merge(df, df2, on="meta.path", how="left")
    df.sample(1).iloc[0].to_dict()

    # sort cols
    df = df[sorted(df.columns)].reset_index(drop=True)

    # sort rows
    kcols = ["meta.path", "meta.row_idx"]
    logger.debug(df.groupby(kcols).size().value_counts())
    df.sort_values(kcols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    # parse body html to text

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_parse2_file()  # 89.7M, eb887c2a


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
            # python -m src.crawl.niu_cc_zh.parse2
            typer.run(main)
