import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.dataset.mt_aug.root
import src.train.mt_api.gpt4.root as sroot
from src import utils

NUM_SAMPLES = 10000


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.mt_aug.root.FILTER_PQ)
    df.sample(1).iloc[0].to_dict()

    # check: stat
    df.info()
    df.groupby(["meta.corpus", "lang.src", "lang.tgt"], dropna=False).size()

    # filter: sample random per corpus and lang (to reduce train time)
    df1 = (
        df.groupby(["meta.corpus", "lang.src", "lang.tgt"], dropna=False)
        .apply(lambda x: x.sample(n=min(len(x), NUM_SAMPLES), random_state=42))
        .reset_index(drop=True)
    )
    df1.groupby(["meta.corpus", "lang.src", "lang.tgt"], dropna=False).size()
    logger.debug(f"{len(df1) / len(df):.1%}")
    df = df1

    # filter
    logger.debug(df["meta.corpus"].value_counts())
    if 0:
        idx = df["meta.corpus"].isin(["ajd", "klc"])
        df = df[idx].reset_index(drop=True)
        logger.debug(df["meta.corpus"].value_counts())

    # check: no empty text
    if 0:
        df = utils.replace_blank_to_none(df)
        cols = [c for c in df.columns if not c.startswith("meta")]
        assert df[cols].isnull().sum().sum() == 0, "no empty text"
        df.dropna(axis=1, how="all", inplace=True)

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 3.0M, 449acbfe, 20000


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_api.gpt4.dataset
            typer.run(main)
