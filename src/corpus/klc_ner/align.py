import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.klc_mt.root
import src.corpus.klc_ner.root as sroot
from src import utils


def clean_ner(df_klc0: pd.DataFrame) -> pd.DataFrame:
    # copy
    df = df_klc0.copy()
    df.sample(1).iloc[0].to_dict()

    # filter
    idx1 = df["text.hj"].notna() & df["text_xml.hj"].notna()
    idx2 = df["text.ko"].notna() & df["text_xml.ko"].notna()
    idx = idx1 & idx2
    df = df[idx].reset_index(drop=True)

    # drop
    df.drop(columns=["text.hj", "text.ko"], inplace=True)

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["key"].is_unique
    df.sort_values(by="key", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df_klc0 = utils.read_df(src.corpus.klc_mt.root.FILTER2_PQ)

    # check
    df_klc0.sample(1).iloc[0].to_dict()
    len(df_klc0)  # 156836

    # get ko
    df = clean_ner(df_klc0=df_klc0)
    df.sample(1).iloc[0].to_dict()

    # check
    temp1 = df.isna().sum()
    temp1[temp1 > 0]
    df.sample(1).iloc[0].to_dict()
    if 0:
        idx = df["text.oko"].isna()
        df[idx].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df)


def main() -> None:
    # align samples
    gen_align_file()  # 314.7M, e533078d, 156836


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
            # python -m src.corpus.klc_ner.align
            typer.run(main)
