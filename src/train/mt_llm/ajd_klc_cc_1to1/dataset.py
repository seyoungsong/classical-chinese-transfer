import sys
from importlib import reload

import pandas as pd
import typer
from loguru import logger
from rich import pretty

import src.dataset.mt_llm.root
import src.train.mt_llm.ajd_klc_cc_1to1.root as sroot
from src import utils


def gen_dataset_file() -> None:
    # read
    df = utils.read_df(src.dataset.mt_llm.root.TRAIN_PQ)
    df.sample(1).iloc[0].to_dict()

    # filter: ajd+klc+cc (of ajd, klc, niu, wyweb_mt)
    logger.debug(df["meta.corpus"].value_counts())
    idx = df["meta.corpus"].isin(["ajd", "klc", "niu", "wyweb_mt"])
    df = df[idx].reset_index(drop=True)
    logger.debug(df["meta.corpus"].value_counts())

    # filter: lang
    idx = df["lang"] == "cc-zh"
    df = df[~idx].reset_index(drop=True)
    logger.debug(df.groupby(["meta.corpus", "lang"]).size())

    # sample: low-resource setting for Hanja
    # Train Data MT (1 : 1) (0.5 : 1) (0.1 : 1)
    # AJD 30,000 5,000 2,500 500
    # KLC 10,000 5,000 2,500 500
    # CC 10,000 10,000 10,000 10,000
    num_ajd = 5000
    num_klc = 5000
    df_ajd = df[df["meta.corpus"] == "ajd"].sample(n=num_ajd, random_state=42)
    df_klc = df[df["meta.corpus"] == "klc"].sample(n=num_klc, random_state=42)
    df_etc = df[~df["meta.corpus"].isin(["ajd", "klc"])].reset_index(drop=True)
    df = pd.concat([df_ajd, df_klc, df_etc], ignore_index=True)
    assert df["key2"].is_unique
    df.sort_values("key2", inplace=True, ignore_index=True)
    logger.debug(df.groupby(["meta.corpus", "lang"]).size())

    # check
    if 0:
        df0 = utils.read_df(src.dataset.mt_llm.root.FILTER_PQ)
        df0.groupby(["meta.corpus", "lang", "split"]).size()

    # save
    sroot.DATASET_PQ.parent.mkdir(exist_ok=True, parents=True)
    utils.write_df2(sroot.DATASET_PQ, df)


def main() -> None:
    gen_dataset_file()  # 6.7M, fe27c023, 20000


if __name__ == "__main__":
    if hasattr(sys, "ps1"):
        pretty.install()
        reload(utils)
        reload(sroot)
    else:
        with logger.catch(onerror=lambda _: sys.exit(1)):
            # python -m src.train.mt_llm.ajd_klc_cc_1to1.dataset
            typer.run(main)
