import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.dataset.mt_llm.root
import src.train.mt_llm.cc.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.mt_llm.root.TRAIN_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: ajd+klc (of ajd, klc, niu, wyweb_mt)
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["niu", "wyweb_mt"])
    df = df[idx].reset_index(drop=True)
    logger.debug(df["meta.corpus"].value_counts())

    # filter: lang
    idx = df["lang"] == "cc-zh"
    df = df[~idx].reset_index(drop=True)
    logger.debug(df.groupby(["meta.corpus", "lang"]).size())

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 1.5M, 7c2870b6, 10000


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_llm.cc.dataset
            typer.run(main)
