import sys
from importlib import reload

import typer
from loguru import logger
from rich import pretty

import src.dataset.mt_llm.root
import src.train.mt_llm.ajd_klc_cc.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.mt_llm.root.TRAIN_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: ajd+klc (of ajd, klc, niu, wyweb_mt)
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["ajd", "klc", "niu", "wyweb_mt"])
    df = df[idx].reset_index(drop=True)
    logger.debug(df["meta.corpus"].value_counts())

    # filter: lang
    idx = df["lang"] == "cc-zh"
    df = df[~idx].reset_index(drop=True)
    logger.debug(df.groupby(["meta.corpus", "lang"]).size())

    # check
    if 0:
        df0 = utils.read_df(src.dataset.mt_llm.root.FILTER_PQ)
        df0.groupby(["meta.corpus", "lang", "split"]).size()

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 18.7M, 9aa72044, 50000


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_llm.ajd_klc_cc.dataset
            typer.run(main)
