import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.ajd.root
import src.corpus.ajd_ner.root as sroot
from src import utils


def clean_ner(df_ajd0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_ajd0.copy()
    df.sample(1).iloc[0].to_dict()

    # drop cko and en
    cols = [c for c in df.columns if c.endswith(".cko") or c.endswith(".en")]
    df.drop(columns=cols, inplace=True)

    # drop if text.oko is None
    idx = df["text.oko"].notna()
    df = df[idx].reset_index(drop=True)

    # rename
    if 0:
        {k: k for k in df.columns}
        rcols = {
            "text": "text.hj",
            "text_xml": "text_xml.hj",
            "meta.data_id": "meta.data_id.hj",
            "meta.king_idx": "meta.king_idx",
            "meta.split": "split",
        }
        df.rename(columns=rcols, inplace=True)
        df.sample(1).iloc[0].to_dict()

    # sort cols
    c1 = [c for c in df.columns if not c.startswith("meta")]
    c2 = [c for c in df.columns if c.startswith("meta")]
    cols = sorted(c1) + sorted(c2)
    df = df[cols].reset_index(drop=True)
    df.sample(1).iloc[0].to_dict()

    # sort rows
    assert df["meta.data_id.hj"].is_unique
    df.sort_values(by="meta.data_id.hj", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df_ajd0 = utils.read_df(src.corpus.ajd.root.FILTER_PQ)

    # check
    df_ajd0.sample(1).iloc[0].to_dict()
    len(df_ajd0)  # 412710

    # get ko
    df_ner = clean_ner(df_ajd0=df_ajd0)
    df_ner.sample(1).iloc[0].to_dict()

    # check
    temp1 = df_ner.isna().sum()
    temp1[temp1 > 0]
    df_ner.sample(1).iloc[0].to_dict()
    if 0:
        idx = df_ner["text.oko"].isna()
        df_ner[idx].sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_ner)


def main() -> None:
    # align samples
    gen_align_file()  # 520.6M, 6d727174, 413165


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
            # python -m src.corpus.ajd_ner.align
            typer.run(main)
