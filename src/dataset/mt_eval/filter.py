import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_eval.root as sroot
from src import utils


def dedup_each_corpus_split(df: pd.DataFrame) -> pd.DataFrame:
    df_list = [
        df_group.drop_duplicates(subset=["text.src", "text.tgt"])
        for _, df_group in df.groupby(["meta.corpus", "split"])
    ]
    df1 = pd.concat(df_list, ignore_index=True)
    df1.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    vc1 = df1["split"].value_counts().sort_index()
    vc0 = df["split"].value_counts().sort_index()
    logger.debug((vc1 / vc0 * 100).sort_index().round(1))

    return df1


def gen_filter_file() -> None:
    # read
    df = utils.read_df(sroot.CONCAT_PQ)
    df.sample(1).iloc[0].to_dict()

    # klc fix: test -> test0, train2+valid2+test2 -> test
    idx = df["meta.corpus"] == "klc"
    df.loc[idx, "split"].value_counts()
    rvals = {"test": "test0"}
    df.loc[idx, "split"] = df.loc[idx, "split"].replace(rvals)
    rvals = {"train2": "test", "valid2": "test", "test2": "test"}
    df.loc[idx, "split"] = df.loc[idx, "split"].replace(rvals)

    # ocdb override: train+valid+test -> test
    idx = df["meta.corpus"] == "ocdb"
    df.loc[idx, "split"].value_counts()
    rvals = {"train": "test", "valid": "test"}
    df.loc[idx, "split"] = df.loc[idx, "split"].replace(rvals)

    # drop train, valid (+ test0)
    df["split"].value_counts()
    idx = df["split"].isin(["train", "valid", "test0"])
    df = df[~idx].reset_index(drop=True)

    # dedup each corpus-split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = dedup_each_corpus_split(df)
    df.sort_values(by="key2", inplace=True, ignore_index=True)

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup, prevent leakage
    gen_filter_file()  # 294.1M, d4d710d6, 433463


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
            # python -m src.dataset.mt_eval.filter
            typer.run(main)
