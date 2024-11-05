import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from pandarallel import pandarallel
from rich import pretty
from tqdm import tqdm

import src.dataset.mt_aug.root as sroot
from src import utils


def dedup_all(df: pd.DataFrame) -> pd.DataFrame:
    if 0:
        df_list = [
            df_group.drop_duplicates(subset=["text.src", "text.tgt"])
            for _, df_group in df.groupby("split")
        ]
        df1 = pd.concat(df_list, ignore_index=True)

    # shuffle
    df1 = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # dedup
    df1.drop_duplicates(
        subset=["text.src", "text.tgt"], inplace=True, ignore_index=True
    )

    # sort
    df1.sort_values(by="key2", inplace=True, ignore_index=True)

    # check
    vc1 = df1["split"].value_counts().sort_index()
    vc0 = df["split"].value_counts().sort_index()
    logger.debug((vc1 / vc0 * 100).sort_index().round(1))

    return df1


def gen_filter_file() -> None:
    # read
    df0 = utils.read_df(sroot.CONCAT_PQ)
    df = df0.copy()
    df.sample(1).iloc[0].to_dict()

    # dedup each split, remove test from train & valid, remove valid from train
    df = dedup_all(df)

    # check
    df0["split"].value_counts() / len(df0) * 100
    df["split"].value_counts() / len(df) * 100
    df["split"].value_counts() / df0["split"].value_counts() * 100
    round(len(df) / len(df0) * 100, 1)  # 90.0%

    # save
    utils.write_df2(sroot.FILTER_PQ, df)


def main() -> None:
    # dedup all (since our purpose is to augment, not train)
    gen_filter_file()  # 114.3M, defefb92, 1115091


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
            # python -m src.dataset.mt_aug.filter
            typer.run(main)
