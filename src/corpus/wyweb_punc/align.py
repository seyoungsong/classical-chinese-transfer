import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.corpus.wyweb_punc.root as sroot
import src.crawl.wyweb_punc.root
from src import utils


def clean_punc(df_punc0: pd.DataFrame) -> pd.DataFrame:
    # filter
    df = df_punc0.copy()
    df.sample(1).iloc[0].to_dict()

    # rename
    {k: k for k in df.columns}
    rcols = {
        "text": "text.cc",
        "text_xml": "text_xml.cc",
        "meta.data_id": "meta.data_id.cc",
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
    assert df["meta.data_id.cc"].is_unique
    df.sort_values(by="meta.data_id.cc", inplace=True)

    return df


def gen_align_file() -> None:
    # read
    df_punc0 = utils.read_df(src.crawl.wyweb_punc.root.FORMAT2_PQ)

    # check
    df_punc0.sample(1).iloc[0].to_dict()
    len(df_punc0)  # 135134

    # get ko
    df_punc = clean_punc(df_punc0=df_punc0)
    df_punc.sample(1).iloc[0].to_dict()

    # check
    temp1 = df_punc.isna().sum()
    temp1[temp1 > 0]
    df_punc.sample(1).iloc[0].to_dict()

    # save
    utils.write_df2(sroot.ALIGN_PQ, df_punc)


def main() -> None:
    # align samples
    gen_align_file()  # 60.1M, 2d9c3bcc, 135134


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
            # python -m src.corpus.wyweb_punc.align
            typer.run(main)
