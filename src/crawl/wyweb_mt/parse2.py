import sys
from importlib import reload

import hanja
import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.crawl.wyweb_mt.root as sroot
import src.tool.corpus as ctool
from src import utils


def parse_meta_path(s: str) -> str:
    s2 = str(hanja.translate(text=s, mode="combination-text-reversed")).strip()
    return s2


def gen_parse2_file() -> None:
    # read
    df = utils.read_df(sroot.CLEAN_PQ)
    df.sample(1).iloc[0].to_dict()

    # fix
    if 0:
        df["meta.path"].apply(lambda x: str(x)[:5]).value_counts()
        df["meta.path"] = df["meta.path"].apply(lambda x: str(x)[5:])

    # fix
    if 0:
        df["text_cc"].apply(lambda x: str(x)[:3]).value_counts()
        df["text_cc"] = df["text_cc"].apply(lambda x: str(x)[3:])

    # fix
    if 0:
        df["text_zh"].apply(lambda x: str(x)[:4]).value_counts()
        df["text_zh"] = df["text_zh"].apply(lambda x: str(x)[4:])

    # sort by size
    if 0:
        df["size"] = df["text_cc"].progress_apply(len)
        df["size"].describe()
        df.sort_values("size", inplace=True, ignore_index=True, ascending=False)
        df.drop(columns=["size"], inplace=True)

    # sample
    if 0:
        x = df.sample(1).iloc[0].to_dict()
        print(parse_meta_path(s=x["meta.path"]))

    # parse
    if 0:
        df["meta.path"].nunique()
        path_list = df["meta.path"].unique().tolist()
        df2 = pd.DataFrame(path_list, columns=["meta.path"])
        df2["meta.path2"] = df2["meta.path"].progress_apply(parse_meta_path)

        # merge
        df = pd.merge(df, df2, on="meta.path", how="left")
        df.sample(1).iloc[0].to_dict()

    # sort columns
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    cols = ["meta.split", "meta.row_idx"]
    df.groupby(cols).size().value_counts()
    df.sort_values(cols, inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.PARSE2_PQ, df)


def main() -> None:
    # parse body html to text

    # init
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    gen_parse2_file()  # 29.8M, 53d522f1


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
            # python -m src.crawl.wyweb_mt.parse2
            typer.run(main)
