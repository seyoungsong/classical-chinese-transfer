import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_hj.root as sroot
import src.crawl.klc_hj.root
from src import utils


def clean_hj(df_hj0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_hj0.copy()
    df.sample(1).iloc[0].to_dict()

    #
    df["lang"].value_counts()
    df.drop(columns=["lang"], inplace=True)

    #
    df["punc_type"].value_counts()
    df.drop(columns=["punc_type"], inplace=True)

    # get ko
    rcols = {k: f"{k}.hj" for k in df.columns}
    df.rename(columns=rcols, inplace=True)
    df.sample(1).iloc[0].to_dict()
    #
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    return df


def gen_align_file() -> None:
    # read
    df_hj0 = utils.read_df(src.crawl.klc_hj.root.FORMAT2_PQ)

    # check
    df_hj0.sample(1).iloc[0].to_dict()  # ITKC_ST_U0_A03_08A_27A_00130
    len(df_hj0)  # 653386

    # get hj
    df_hj = clean_hj(df_hj0=df_hj0)
    df_hj.sample(1).iloc[0].to_dict()

    # check
    temp1 = df_hj.isna().sum()
    temp1[temp1 > 0]
    df_hj.sample(1).iloc[0].to_dict()
    df_hj[df_hj["meta.elem_id.hj"].notna()].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_hj)


def main() -> None:
    # align samples
    gen_align_file()  # 395.6M, 85f32b41, 653386


if __name__ == "__main__":
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    #
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.corpus.klc_hj.align
            typer.run(main)
